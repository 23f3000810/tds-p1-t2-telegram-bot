import os
import sys
import json
import logging
import subprocess
import datetime
import tempfile
import time
import urllib.request
import urllib.parse
from google.cloud import storage
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GCS_KEY_FILE = "key-1.json"
GCS_BUCKET_NAME = "q1-ae7586717277a99"
LOG_FILENAME = "run.jsonl"

# State tracking: chat_id -> list of message objects: {"role": "user"|"model", "text": "..."}
chat_histories = {}

def call_gemini(contents, system_instruction=None, retries=3):
    """Calls Gemini 3.6-flash API with retries and exponential backoff."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": contents
    }
    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
        
    data = json.dumps(payload).encode('utf-8')
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as res:
                res_data = json.loads(res.read().decode('utf-8'))
                
                # Check candidates
                candidates = res_data.get('candidates', [])
                if not candidates:
                    raise Exception("No candidates returned from Gemini API")
                    
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if not parts:
                    # check if blocked
                    finish_reason = candidates[0].get('finishReason', 'UNKNOWN')
                    raise Exception(f"Empty parts. Finish reason: {finish_reason}")
                    
                text = parts[0].get('text', '')
                if not text:
                    raise Exception("Candidate parts contains no text content")
                    
                return text
                
        except Exception as e:
            wait_time = 2 ** (attempt + 1)
            logger.warning(f"Gemini API call attempt {attempt+1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
    raise Exception(f"Gemini API call failed after {retries} attempts.")

def upload_log_to_gcs():
    """Uploads the local run.jsonl file to the public GCS bucket."""
    try:
        if not os.path.exists(LOG_FILENAME):
            return
        client = storage.Client.from_service_account_json(GCS_KEY_FILE)
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(LOG_FILENAME)
        blob.upload_from_filename(LOG_FILENAME, content_type="text/plain")
        try:
            blob.make_public()
        except Exception as e:
            logger.warning(f"Failed blob.make_public(): {e}")
        logger.info("Log uploaded successfully to GCS.")
    except Exception as e:
        logger.error(f"Error uploading log to GCS: {e}")

def append_to_log(log_entry):
    """Appends a log entry to run.jsonl and triggers upload."""
    try:
        with open(LOG_FILENAME, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        upload_log_to_gcs()
    except Exception as e:
        logger.error(f"Error writing local log: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
        
    chat_id = update.message.chat_id
    user_text = update.message.text
    logger.info(f"Received message from chat {chat_id}: {user_text}")
    
    # Update context history
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    
    chat_histories[chat_id].append({"role": "user", "text": user_text})
    
    # System instruction for ReAct loop
    system_instruction = (
        "You are an expert Data Analyst Agent. The user has sent a request. You should write a python script to solve it "
        "if it requires calculation, data fetching, or page parsing. "
        "To write a python script, output ONLY a python code block enclosed in ```python ... ``` and nothing else.\n"
        "Your code should download datasets using urllib.request and process them. Do NOT use shell commands like curl or wget.\n"
        "If you do not need code (e.g., simple reasoning, or you already have the final result or search output), "
        "output the final answer to the user's question. The final answer MUST be a raw JSON object matching the exact shape "
        "the user requested, and absolutely nothing else. Do not include markdown fences, prose, or quotes around the JSON."
    )
    
    agent_steps = []
    
    # Initialize the contents trace for the current question
    contents = []
    for turn in chat_histories[chat_id][:-1]:
        contents.append({
            "role": turn["role"],
            "parts": [{"text": turn["text"]}]
        })
    
    contents.append({
        "role": "user",
        "parts": [{"text": user_text}]
    })
    
    max_iterations = 3
    final_response = ""
    
    for i in range(max_iterations):
        logger.info(f"ReAct Loop Iteration {i+1}/{max_iterations}...")
        try:
            llm_response = call_gemini(contents, system_instruction)
            logger.info(f"LLM Response:\n{llm_response}")
            
            # Check if code block was generated
            python_code = ""
            if "```python" in llm_response:
                try:
                    start_idx = llm_response.find("```python") + len("```python")
                    end_idx = llm_response.find("```", start_idx)
                    python_code = llm_response[start_idx:end_idx].strip()
                except Exception as e:
                    logger.error(f"Failed to extract python code: {e}")
            
            if python_code:
                logger.info(f"Executing python code...")
                with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                    temp_file.write(python_code.encode('utf-8'))
                    temp_file_path = temp_file.name
                
                try:
                    res = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True, timeout=30)
                    stdout = res.stdout
                    stderr = res.stderr
                    exit_code = res.returncode
                except subprocess.TimeoutExpired:
                    stdout = ""
                    stderr = "Timeout expired after 30 seconds"
                    exit_code = -1
                except Exception as e:
                    stdout = ""
                    stderr = f"Error during execution: {e}"
                    exit_code = -1
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                
                logger.info(f"Exit code: {exit_code}, stdout: {stdout.strip()}, stderr: {stderr.strip()}")
                agent_steps.append({
                    "iteration": i+1,
                    "code": python_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code
                })
                
                # Update Gemini contents with the turn
                contents.append({"role": "model", "parts": [{"text": llm_response}]})
                execution_feedback = (
                    f"Code execution output:\n"
                    f"STDOUT:\n{stdout}\n"
                    f"STDERR:\n{stderr}\n"
                    f"EXIT CODE: {exit_code}\n\n"
                    f"Please analyze these results. If you need to run another script, output the code block. "
                    f"Otherwise, output the final answer as the raw JSON object requested by the user."
                )
                contents.append({"role": "user", "parts": [{"text": execution_feedback}]})
            else:
                final_response = llm_response
                agent_steps.append({
                    "iteration": i+1,
                    "direct_response": llm_response
                })
                break
        except Exception as e:
            logger.error(f"Error in ReAct loop: {e}", exc_info=True)
            final_response = f'{{"error": "Agent loop error: {str(e)}"}}'
            break
    else:
        # Loop finished without a direct answer, force format
        logger.info("Forcing final formatting after max iterations...")
        contents.append({"role": "user", "parts": [{"text": "Format your final answer as the raw JSON object requested by the user. Output ONLY the JSON."}]})
        try:
            final_response = call_gemini(contents, system_instruction)
            agent_steps.append({"type": "forced_final_formatting", "content": final_response})
        except Exception as e:
            final_response = f'{{"error": "Forced formatting error: {str(e)}"}}'
            
    # Extract JSON object from final_response
    parsed_answer = None
    start_idx = final_response.find("{")
    end_idx = final_response.rfind("}")
    if start_idx != -1 and end_idx != -1:
        json_str = final_response[start_idx:end_idx+1]
        try:
            parsed_answer = json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed parsing JSON from: {json_str}. Error: {e}")
            
    if parsed_answer is None:
        parsed_answer = {"error": "Failed to parse final answer", "raw_response": final_response}
        
    # Construct final Telegram reply JSON
    log_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{LOG_FILENAME}"
    reply_body = {
        "answer": parsed_answer,
        "log_url": log_url
    }
    
    reply_json = json.dumps(reply_body)
    logger.info(f"Final Telegram Reply JSON: {reply_json}")
    
    # Save to chat history
    chat_histories[chat_id].append({"role": "model", "text": reply_json})
    
    # Write to local run.jsonl and upload
    log_entry = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "chat_id": chat_id,
        "message": user_text,
        "agent_steps": agent_steps,
        "reply": reply_body
    }
    append_to_log(log_entry)
    
    # Send reply back
    await update.message.reply_text(reply_json)

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
        sys.exit(1)
        
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
        
    if not os.path.exists(GCS_KEY_FILE):
        print(f"Error: GCP credentials key file '{GCS_KEY_FILE}' not found in current folder.")
        sys.exit(1)
        
    print("Starting Telegram Bot long-polling server...")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == '__main__':
    main()
