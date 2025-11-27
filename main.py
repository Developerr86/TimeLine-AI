from flask import Flask, render_template, jsonify, request, send_from_directory
import threading
import time
import json
import os
from pathlib import Path
from datetime import datetime
import base64
from PIL import ImageGrab, Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import requests
import ollama
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

SCREENSHOTS_DIR = Path("screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)
CONFIG_FILE = "config.json"
RESPONSES_FILE = "responses.json"

DEFAULT_CONFIG = {
    "interval": 5,
    "model_type": "ollama",
    "ollama_model": os.getenv('OLLAMA_MODEL', 'qwen2.5vl:3b'),
    "gemini_model": os.getenv('GEMINI_MODEL', 'gemini-2.5-pro'),
    "Apple_FastVLM": os.getenv('REMOTE_URL', 'http://localhost:5001/predict'),
    "similarity_threshold": 0.95,
    "notes_history_limit": 5,
    "notes_model_provider": "gemini",
    "notes_ollama_model": "llama3",
    "enabled": False
}

capture_thread = None
stop_capture = threading.Event()
latest_activity = {"timestamp": None, "title": None, "screenshot_taken": False}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def load_responses():
    if os.path.exists(RESPONSES_FILE):
        try:
            with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_response(response_data):
    responses = load_responses()
    responses.append(response_data)
    with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

def calculate_ssim(image1_path, image2_path):
    img1 = cv2.imread(str(image1_path))
    img2 = cv2.imread(str(image2_path))
    
    if img1 is None or img2 is None:
        return 0.0
    
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

def get_last_screenshot():
    screenshots = sorted(SCREENSHOTS_DIR.glob("*.png"))
    return screenshots[-1] if screenshots else None

def take_screenshot():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = SCREENSHOTS_DIR / f"screenshot_{timestamp}.png"
    
    screenshot = ImageGrab.grab()
    screenshot.save(filename)
    
    return filename

def analyze_screenshot_ollama(image_path, model_name):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """
        You are an AI assistant analyzing a user's computer activity from a screenshot.
        Your task is to describe what is happening on the screen and then create a summary of the activity.

        1.  **Analyze the Screen:** Look closely at the applications, websites, and any visible text on the screen.
            Be specific and factual. For example, instead of "coding", say "writing a Python function in VS Code".

        2.  **Generate a JSON Summary:** Based on your analysis, create a title and a brief summary for this activity.
            The title should be conversational and 5-8 words long.
            The summary should be 1-2 sentences describing the main task.

        Respond with ONLY a valid JSON object in the following format:
        {
          "title": "A short, conversational title of the activity",
          "summary": "A 1-2 sentence summary of what the user is doing."
        }
        """
        
        client = ollama.Client()
        response = client.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [base64_image]
            }],
            options={'verbose': True}
        )
        
        token_info = {
            'prompt_tokens': response.get('prompt_eval_count', 0),
            'completion_tokens': response.get('eval_count', 0),
            'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
        }
        
        response_content = response['message']['content']
        
        if response_content.strip().startswith("```json"):
            json_str = response_content.strip()[7:-3].strip()
        else:
            json_str = response_content
        
        summary_data = json.loads(json_str)
        return {
            'title': summary_data.get('title', 'No Title'),
            'summary': summary_data.get('summary', 'No Summary'),
            'raw_response': response_content,
            'token_usage': token_info
        }
    except Exception as e:
        return {
            'title': 'Error',
            'summary': f'Analysis failed: {str(e)}',
            'error': str(e)
        }

def analyze_screenshot_gemini(image_path, model_name):
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        prompt = """
        You are an AI assistant analyzing a user's computer activity from a screenshot.
        Your task is to describe what is happening on the screen and then create a summary of the activity.

        1.  **Analyze the Screen:** Look closely at the applications, websites, and any visible text on the screen.
            Be specific and factual. For example, instead of "coding", say "writing a Python function in VS Code".

        2.  **Generate a JSON Summary:** Based on your analysis, create a title and a brief summary for this activity.
            The title should be conversational and 5-8 words long.
            The summary should be 1-2 sentences describing the main task.

        Respond with ONLY a valid JSON object in the following format:
        {
          "title": "A short, conversational title of the activity",
          "summary": "A 1-2 sentence summary of what the user is doing."
        }
        """
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": image_data}
        ])
        
        token_info = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            token_info = {
                'prompt_tokens': getattr(usage, 'prompt_token_count', 0),
                'completion_tokens': getattr(usage, 'candidates_token_count', 0),
                'total_tokens': getattr(usage, 'total_token_count', 0)
            }
        
        response_content = response.text
        
        if response_content.strip().startswith("```json"):
            json_str = response_content.strip()[7:-3].strip()
        elif response_content.strip().startswith("```"):
            lines = response_content.strip().split('\n')
            json_str = '\n'.join(lines[1:-1]).strip()
        else:
            json_str = response_content
        
        summary_data = json.loads(json_str)
        return {
            'title': summary_data.get('title', 'No Title'),
            'summary': summary_data.get('summary', 'No Summary'),
            'raw_response': response_content,
            'token_usage': token_info
        }
    except Exception as e:
        return {
            'title': 'Error',
            'summary': f'Analysis failed: {str(e)}',
            'error': str(e)
        }

def analyze_screenshot_remote(image_path, url):
    try:
        prompt = """
        You are an AI assistant analyzing a user's computer activity from a screenshot.
        Your task is to describe what is happening on the screen and then create a summary of the activity.

        1.  **Analyze the Screen:** Look closely at the applications, websites, and any visible text on the screen.
            Be specific and factual. For example, instead of "coding", say "writing a Python function in VS Code".

        2.  **Generate a JSON Summary:** Based on your analysis, create a title and a brief summary for this activity.
            The title should be conversational and 5-8 words long.
            The summary should be 1-2 sentences describing the main task.

        Respond with ONLY a valid JSON object in the following format:
        {
          "title": "A short, conversational title of the activity",
          "summary": "A 1-2 sentence summary of what the user is doing."
        }
        """

        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {'prompt': prompt}
            
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                # Assuming the remote server follows the structure implied by test_client.py
                # which expects response.json().get('response')
                response_json = response.json()
                response_content = response_json.get('response')
                
                if not response_content:
                    # Fallback if the key 'response' is missing, maybe the whole body is the content
                    if isinstance(response_json, dict) and 'title' in response_json:
                         # It returned the JSON object directly
                         return {
                            'title': response_json.get('title', 'No Title'),
                            'summary': response_json.get('summary', 'No Summary'),
                            'raw_response': json.dumps(response_json),
                            'token_usage': {}
                        }
                    response_content = str(response_json)
                
                # Parse the content which is expected to be the JSON string
                if response_content.strip().startswith("```json"):
                    json_str = response_content.strip()[7:-3].strip()
                elif response_content.strip().startswith("```"):
                    lines = response_content.strip().split('\n')
                    json_str = '\n'.join(lines[1:-1]).strip()
                else:
                    json_str = response_content
                
                try:
                    summary_data = json.loads(json_str)
                    return {
                        'title': summary_data.get('title', 'No Title'),
                        'summary': summary_data.get('summary', 'No Summary'),
                        'raw_response': response_content,
                        'token_usage': {}
                    }
                except json.JSONDecodeError:
                    # If it's not JSON, treat the whole text as summary
                    return {
                        'title': 'Remote Analysis',
                        'summary': response_content[:500],
                        'raw_response': response_content,
                        'token_usage': {}
                    }
            else:
                return {
                    'title': 'Error',
                    'summary': f'Remote server error: {response.status_code}',
                    'error': response.text
                }
    except Exception as e:
        return {
            'title': 'Error',
            'summary': f'Analysis failed: {str(e)}',
            'error': str(e)
        }

def capture_loop():
    global stop_capture, latest_activity
    config = load_config()
    
    print("⏳ Waiting 10 seconds before starting capture...")
    time.sleep(10)
    print("✅ Starting screenshot capture now!")
    
    while not stop_capture.is_set():
        try:
            screenshot_path = take_screenshot()
            print(f"📸 Screenshot taken: {screenshot_path.name}")
            
            latest_activity["screenshot_taken"] = True
            latest_activity["timestamp"] = datetime.now().isoformat()
            
            last_screenshot = get_last_screenshot()
            should_analyze = True
            
            if last_screenshot and last_screenshot != screenshot_path:
                similarity = calculate_ssim(last_screenshot, screenshot_path)
                print(f"📊 Similarity: {similarity:.2%}")
                
                if similarity >= config['similarity_threshold']:
                    print(f"❌ Removing similar screenshot: {screenshot_path.name}")
                    screenshot_path.unlink()
                    should_analyze = False
            
            if should_analyze:
                print(f"🧠 Analyzing screenshot...")
                
                if config['model_type'] == 'ollama':
                    result = analyze_screenshot_ollama(screenshot_path, config['ollama_model'])
                    model_name = config['ollama_model']
                elif config['model_type'] == 'remote':
                    result = analyze_screenshot_remote(screenshot_path, config['remote_url'])
                    model_name = config['remote_url']
                else:
                    result = analyze_screenshot_gemini(screenshot_path, config['gemini_model'])
                    model_name = config['gemini_model']
                
                response_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": config['model_type'],
                    "model_name": model_name,
                    "image_path": str(screenshot_path),
                    "title": result.get('title', 'No Title'),
                    "summary": result.get('summary', 'No Summary'),
                    "raw_response": result.get('raw_response', ''),
                    "token_usage": result.get('token_usage', {})
                }
                
                if 'error' in result:
                    response_entry['error'] = result['error']
                
                save_response(response_entry)
                print(f"✅ Analysis complete: {result['title']}")
                
                latest_activity["title"] = result['title']
                latest_activity["new_analysis"] = True
            
            time.sleep(config['interval'])
        except Exception as e:
            print(f"❌ Error in capture loop: {e}")
            time.sleep(5)

@app.route('/')
def index():
    responses = load_responses()
    responses.reverse()
    return render_template('index.html', responses=responses)

@app.route('/config')
def config_page():
    config = load_config()
    return render_template('config.html', config=config)

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        data = request.json
        config = load_config()
        
        config['interval'] = int(data.get('interval', 5))
        config['model_type'] = data.get('model_type', 'ollama')
        config['ollama_model'] = data.get('ollama_model', config['ollama_model'])
        config['gemini_model'] = data.get('gemini_model', config['gemini_model'])
        config['remote_url'] = data.get('remote_url', config.get('remote_url', 'http://localhost:5001/predict'))
        config['similarity_threshold'] = float(data.get('similarity_threshold', 0.95))
        config['notes_history_limit'] = int(data.get('notes_history_limit', 5))
        config['notes_model_provider'] = data.get('notes_model_provider', 'gemini')
        config['notes_ollama_model'] = data.get('notes_ollama_model', 'llama3')
        
        save_config(config)
        return jsonify({'status': 'success', 'config': config})
    else:
        return jsonify(load_config())

@app.route('/api/start', methods=['POST'])
def start_capture():
    global capture_thread, stop_capture
    
    config = load_config()
    if config.get('enabled', False):
        return jsonify({'status': 'already_running'})
    
    config['enabled'] = True
    save_config(config)
    
    stop_capture.clear()
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_capture_route():
    global stop_capture
    
    config = load_config()
    config['enabled'] = False
    save_config(config)
    
    stop_capture.set()
    
    return jsonify({'status': 'stopped'})

@app.route('/api/status')
def status():
    config = load_config()
    responses = load_responses()
    return jsonify({
        'enabled': config.get('enabled', False),
        'total_responses': len(responses),
        'config': config
    })

@app.route('/api/responses')
def api_responses():
    responses = load_responses()
    responses.reverse()
    return jsonify(responses)

@app.route('/api/latest_activity')
def api_latest_activity():
    global latest_activity
    activity = latest_activity.copy()
    if activity.get("screenshot_taken"):
        latest_activity["screenshot_taken"] = False
    if activity.get("new_analysis"):
        latest_activity["new_analysis"] = False
    return jsonify(activity)

@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    return send_from_directory(SCREENSHOTS_DIR, filename)

@app.route('/api/generate_notes', methods=['POST'])
def generate_notes():
    try:
        config = load_config()
        responses = load_responses()
        
        # Get the limit from config, default to 5 if not set
        limit = config.get('notes_history_limit', 5)
        
        # Get the last N responses
        recent_responses = responses[-limit:] if responses else []
        
        if not recent_responses:
            return jsonify({
                'status': 'error', 
                'message': 'No recent activities to generate notes from.'
            }), 400
            
        # Prepare the prompt
        activities_text = ""
        for i, resp in enumerate(recent_responses):
            timestamp = resp.get('timestamp', 'Unknown time')
            title = resp.get('title', 'No Title')
            summary = resp.get('summary', 'No Summary')
            activities_text += f"{i+1}. [{timestamp}] {title}: {summary}\n"
            
        prompt = f"""
        You are an intelligent assistant helping a user review their recent computer activity.
        Here are the user's last {len(recent_responses)} recorded activities:
        
        {activities_text}
        
        Based on these activities, please generate a concise set of notes.
        - Summarize the main themes or tasks the user was working on.
        - Highlight any potential distractions if apparent.
        - Estimate roughly how much time was spent on different contexts (coding, browsing, etc.) if possible.
        - Keep the tone professional and helpful.
        - Format the output with Markdown (bullet points, bold text, etc.).
        """
        
        provider = config.get('notes_model_provider', 'gemini')
        
        if provider == 'ollama':
             # Use Ollama
            model_name = config.get('notes_ollama_model', 'llama3')
            client = ollama.Client()
            response = client.chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }]
            )
            response_text = response['message']['content']
        else:
            # Use Gemini (Default)
            # Check for Gemini API key
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return jsonify({
                    'status': 'error',
                    'message': 'GEMINI_API_KEY not set. Please configure it in .env file.'
                }), 400
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config.get('gemini_model', 'gemini-2.5-pro'))
            response = model.generate_content(prompt)
            response_text = response.text
        
        return jsonify({
            'status': 'success',
            'notes': response_text,
            'activity_count': len(recent_responses)
        })
        
    except Exception as e:
        print(f"❌ Error generating notes: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    try:
        config = load_config()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"uploaded_{timestamp}.png"
        filepath = SCREENSHOTS_DIR / filename
        
        file.save(filepath)
        
        print(f"📤 Uploaded image: {filename}")
        print(f"🧠 Analyzing uploaded image...")
        
        if config['model_type'] == 'ollama':
            result = analyze_screenshot_ollama(filepath, config['ollama_model'])
            model_name = config['ollama_model']
        elif config['model_type'] == 'remote':
            result = analyze_screenshot_remote(filepath, config['remote_url'])
            model_name = config['remote_url']
        else:
            result = analyze_screenshot_gemini(filepath, config['gemini_model'])
            model_name = config['gemini_model']
        
        response_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": config['model_type'],
            "model_name": model_name,
            "image_path": str(filepath),
            "title": result.get('title', 'No Title'),
            "summary": result.get('summary', 'No Summary'),
            "raw_response": result.get('raw_response', ''),
            "token_usage": result.get('token_usage', {}),
            "uploaded": True
        }
        
        if 'error' in result:
            response_entry['error'] = result['error']
        
        save_response(response_entry)
        print(f"✅ Upload analysis complete: {result['title']}")
        
        return jsonify({
            'status': 'success',
            'title': result.get('title', 'No Title'),
            'summary': result.get('summary', 'No Summary'),
            'image_path': str(filepath)
        })
    
    except Exception as e:
        print(f"❌ Error processing upload: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/delete_responses', methods=['POST'])
def delete_responses():
    try:
        data = request.json
        timestamps_to_delete = data.get('timestamps', [])
        
        if not timestamps_to_delete:
            return jsonify({'status': 'error', 'message': 'No items selected'}), 400
            
        responses = load_responses()
        new_responses = []
        deleted_count = 0
        
        for resp in responses:
            if resp['timestamp'] in timestamps_to_delete:
                # Delete the image file
                if 'image_path' in resp and resp['image_path']:
                    try:
                        # Handle both string path and Path object (though JSON usually has string)
                        image_path = Path(resp['image_path'])
                        if image_path.exists():
                            image_path.unlink()
                    except Exception as e:
                        print(f"⚠️ Failed to delete image {resp['image_path']}: {e}")
                deleted_count += 1
            else:
                new_responses.append(resp)
        
        # Save the filtered list
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_responses, f, indent=2, ensure_ascii=False)
            
        return jsonify({
            'status': 'success', 
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        print(f"❌ Error deleting responses: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/clear_context', methods=['POST'])
def clear_context():
    try:
        # Stop capture if running
        config = load_config()
        was_running = config.get('enabled', False)
        
        if was_running:
            stop_capture.set()
            # Update config to disabled
            config['enabled'] = False
            save_config(config)
            
        # Clear responses.json
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
            
        # Clear screenshots directory
        if SCREENSHOTS_DIR.exists():
            for file_path in SCREENSHOTS_DIR.glob('*'):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"⚠️ Failed to delete {file_path}: {e}")
                    
        # Clear latest activity
        global latest_activity
        latest_activity = {"timestamp": None, "title": None, "screenshot_taken": False}
        
        return jsonify({
            'status': 'success', 
            'message': 'Context cleared successfully',
            'was_running': was_running
        })
    except Exception as e:
        print(f"❌ Error clearing context: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    config = load_config()
    
    if config.get('enabled', False):
        stop_capture.clear()
        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()
        print("🚀 Auto-capture started")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
