import ollama
import base64
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Load configuration from environment variables
OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL', 'qwen2.5vl:3b')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
REMOTE_URL = os.getenv('REMOTE_URL', 'http://localhost:5001/predict')
RESPONSES_FILE = "responses.json"

def encode_image_to_base64(image_path: Path) -> str:
    """Encodes the image at the given path to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"❌ Error: The file was not found at {image_path}")
        exit(1)
    except Exception as e:
        print(f"❌ An error occurred while reading the image: {e}")
        exit(1)

def load_responses():
    """Load existing responses from the JSON file."""
    if os.path.exists(RESPONSES_FILE):
        try:
            with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def ensure_responses_file():
    """Create responses.json file if it doesn't exist."""
    if not os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        print(f"📁 Created {RESPONSES_FILE} file")

def print_config_info():
    """Print current configuration information."""
    print("🔧 Configuration (loaded from .env):")
    print(f"   Ollama Model: {OLLAMA_MODEL_NAME}")
    print(f"   Gemini Model: {GEMINI_MODEL_NAME}")
    print(f"   Gemini API Key: {'✅ Set' if GEMINI_API_KEY else '❌ Not set'}")
    print(f"   Responses File: {RESPONSES_FILE}")
    print()

def save_response(response_data):
    """Append a new response to the responses.json file."""
    responses = load_responses()
    responses.append(response_data)
    
    with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

def generate_activity_summary_ollama(image_path: str):
    """
    Analyzes a screenshot using a local Ollama vision model and generates a summary.
    This logic is inspired by the multi-step prompt process in the Dayflow Swift app.
    """
    screenshot_path = Path(image_path)
    if not screenshot_path.is_file():
        print(f"❌ Error: Invalid file path provided: {image_path}")
        return

    print(f"Analysing Screenshot with Ollama: {screenshot_path.name}...")

    base64_image = encode_image_to_base64(screenshot_path)

    # This prompt is a simplified version of the prompts found in the
    # `OllamaProvider.swift` file, combining frame description and summary generation.
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

    try:
        # Using the ollama library with verbose output
        client = ollama.Client()

        print(f"🧠 Prompting model: ({OLLAMA_MODEL_NAME})...")

        # Make request with verbose output to get token information
        response = client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [base64_image]
                }
            ],
            options={
                'verbose': True
            }
        )

        print("✅ Model has extracted activity")
        
        # Extract token information if available
        token_info = {}
        if 'prompt_eval_count' in response:
            token_info['prompt_tokens'] = response.get('prompt_eval_count', 0)
        if 'eval_count' in response:
            token_info['completion_tokens'] = response.get('eval_count', 0)
        if 'prompt_eval_duration' in response:
            token_info['prompt_eval_duration_ns'] = response.get('prompt_eval_duration', 0)
        if 'eval_duration' in response:
            token_info['eval_duration_ns'] = response.get('eval_duration', 0)
        if 'total_duration' in response:
            token_info['total_duration_ns'] = response.get('total_duration', 0)
        
        # Calculate total tokens if available
        if 'prompt_tokens' in token_info and 'completion_tokens' in token_info:
            token_info['total_tokens'] = token_info['prompt_tokens'] + token_info['completion_tokens']
        
        # Print token information
        if token_info:
            print(f"📊 Token Usage:")
            if 'prompt_tokens' in token_info:
                print(f"   Prompt tokens: {token_info['prompt_tokens']}")
            if 'completion_tokens' in token_info:
                print(f"   Completion tokens: {token_info['completion_tokens']}")
            if 'total_tokens' in token_info:
                print(f"   Total tokens: {token_info['total_tokens']}")
            if 'total_duration_ns' in token_info:
                duration_ms = token_info['total_duration_ns'] / 1_000_000
                print(f"   Total duration: {duration_ms:.2f}ms")

        # Extract the content and parse the JSON
        response_content = response['message']['content']
        
        # Clean the response to ensure it's valid JSON
        # Models sometimes wrap their JSON response in markdown ```json ... ```
        if response_content.strip().startswith("```json"):
            json_str = response_content.strip()[7:-3].strip()
        else:
            json_str = response_content

        try:
            summary_data = json.loads(json_str)
            title = summary_data.get("title", "No Title Provided")
            summary = summary_data.get("summary", "No Summary Provided")

            print("\n--- Activity Summary ---")
            print(f"🏷️  **Title:** {title}")
            print(f"📝 **Summary:** {summary}")
            print("----------------------\n")

            # Save response to file
            response_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": "ollama",
                "model_name": OLLAMA_MODEL_NAME,
                "image_path": str(screenshot_path),
                "title": title,
                "summary": summary,
                "raw_response": response_content,
                "token_usage": token_info
            }
            save_response(response_entry)
            print(f"✅ Response saved to {RESPONSES_FILE}")

        except json.JSONDecodeError:
            print("❌ Error: Failed to parse JSON from the model's response.")
            print("Raw Response:")
            print(response_content)
            
            # Save error response to file
            response_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": "ollama",
                "model_name": OLLAMA_MODEL_NAME,
                "image_path": str(screenshot_path),
                "error": "JSON parsing failed",
                "raw_response": response_content,
                "token_usage": token_info
            }
            save_response(response_entry)


    except Exception as e:
        print(f"❌ An error occurred while communicating with Ollama: {e}")
        print(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL_NAME}' is installed (`ollama pull {OLLAMA_MODEL_NAME}`).")
        
        # Save error response to file
        response_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": "ollama",
            "model_name": OLLAMA_MODEL_NAME,
            "image_path": str(screenshot_path),
            "error": str(e)
        }
        save_response(response_entry)


def generate_activity_summary_gemini(image_path: str):
    """
    Analyzes a screenshot using Google's Gemini vision model and generates a summary.
    """
    screenshot_path = Path(image_path)
    if not screenshot_path.is_file():
        print(f"❌ Error: Invalid file path provided: {image_path}")
        return

    print(f"Analysing Screenshot with Gemini: {screenshot_path.name}...")

    # Check for API key
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        print("Please set your Gemini API key: set GEMINI_API_KEY=your_api_key_here")
        return

    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        # Read the image
        with open(screenshot_path, 'rb') as image_file:
            image_data = image_file.read()

        # Create the prompt
        prompt = """
        You are an AI assistant analyzing a user's computer activity from a screenshot.
        Your task is to describe what is happening on the screen and then create a summary of the activity.

        1.  **Analyze the Screen:** Look closely at the applications, websites, and any visible text on the screen.
            Be specific and factual. For example, instead of "coding", say "writing a Python function in VS Code".

        2.  **Generate a JSON Summary:** Based on your analysis, create a title and a brief summary for this activity.
            The title should be conversational and 5-8 words long.
            The summary should be 3 sentences describing the main task.

        Respond with ONLY a valid JSON object in the following format:
        {
          "title": "A short, conversational title of the activity",
          "summary": "A 1-2 sentence summary of what the user is doing."
        }
        """

        print(f"🧠 Prompting model: ({GEMINI_MODEL_NAME})...")

        # Generate content with image
        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": image_data}
        ])

        print("✅ Model has extracted activity")

        # Extract token usage information if available
        token_info = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            if hasattr(usage, 'prompt_token_count'):
                token_info['prompt_tokens'] = usage.prompt_token_count
            if hasattr(usage, 'candidates_token_count'):
                token_info['completion_tokens'] = usage.candidates_token_count
            if hasattr(usage, 'total_token_count'):
                token_info['total_tokens'] = usage.total_token_count
        
        # Print token information
        if token_info:
            print(f"📊 Token Usage:")
            if 'prompt_tokens' in token_info:
                print(f"   Prompt tokens: {token_info['prompt_tokens']}")
            if 'completion_tokens' in token_info:
                print(f"   Completion tokens: {token_info['completion_tokens']}")
            if 'total_tokens' in token_info:
                print(f"   Total tokens: {token_info['total_tokens']}")

        # Extract the content and parse the JSON
        response_content = response.text
        
        # Clean the response to ensure it's valid JSON
        if response_content.strip().startswith("```json"):
            json_str = response_content.strip()[7:-3].strip()
        elif response_content.strip().startswith("```"):
            # Handle generic code blocks
            lines = response_content.strip().split('\n')
            json_str = '\n'.join(lines[1:-1]).strip()
        else:
            json_str = response_content

        try:
            summary_data = json.loads(json_str)
            title = summary_data.get("title", "No Title Provided")
            summary = summary_data.get("summary", "No Summary Provided")

            print("\n--- Activity Summary ---")
            print(f"🏷️  **Title:** {title}")
            print(f"📝 **Summary:** {summary}")
            print("----------------------\n")

            # Save response to file
            response_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": "gemini",
                "model_name": GEMINI_MODEL_NAME,
                "image_path": str(screenshot_path),
                "title": title,
                "summary": summary,
                "raw_response": response_content,
                "token_usage": token_info
            }
            save_response(response_entry)
            print(f"✅ Response saved to {RESPONSES_FILE}")

        except json.JSONDecodeError:
            print("❌ Error: Failed to parse JSON from the model's response.")
            print("Raw Response:")
            print(response_content)
            
            # Save error response to file
            response_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": "gemini",
                "model_name": GEMINI_MODEL_NAME,
                "image_path": str(screenshot_path),
                "error": "JSON parsing failed",
                "raw_response": response_content,
                "token_usage": token_info
            }
            save_response(response_entry)

    except Exception as e:
        print(f"❌ An error occurred while communicating with Gemini: {e}")
        print("Please ensure your GEMINI_API_KEY is valid and you have internet connectivity.")
        
        # Save error response to file
        response_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": "gemini",
            "model_name": GEMINI_MODEL_NAME,
            "image_path": str(screenshot_path),
            "error": str(e)
        }
        save_response(response_entry)


def generate_activity_summary_remote(image_path: str, url: str):
    """
    Analyzes a screenshot using a remote server and generates a summary.
    """
    screenshot_path = Path(image_path)
    if not screenshot_path.is_file():
        print(f"❌ Error: Invalid file path provided: {image_path}")
        return

    print(f"Analysing Screenshot with Remote Server ({url}): {screenshot_path.name}...")

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

        with open(screenshot_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {'prompt': prompt}
            
            print(f"🧠 Sending request to {url}...")
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                print("✅ Remote server responded")
                
                response_json = response.json()
                response_content = response_json.get('response')
                
                if not response_content:
                     if isinstance(response_json, dict) and 'title' in response_json:
                         response_content = json.dumps(response_json)
                     else:
                         response_content = str(response_json)

                if response_content.strip().startswith("```json"):
                    json_str = response_content.strip()[7:-3].strip()
                elif response_content.strip().startswith("```"):
                    lines = response_content.strip().split('\n')
                    json_str = '\n'.join(lines[1:-1]).strip()
                else:
                    json_str = response_content
                
                try:
                    summary_data = json.loads(json_str)
                    title = summary_data.get("title", "No Title Provided")
                    summary = summary_data.get("summary", "No Summary Provided")
                except json.JSONDecodeError:
                    title = "Remote Analysis"
                    summary = response_content[:500]

                print("\n--- Activity Summary ---")
                print(f"🏷️  **Title:** {title}")
                print(f"📝 **Summary:** {summary}")
                print("----------------------\n")

                response_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": "remote",
                    "model_name": url,
                    "image_path": str(screenshot_path),
                    "title": title,
                    "summary": summary,
                    "raw_response": response_content,
                    "token_usage": {}
                }
                save_response(response_entry)
                print(f"✅ Response saved to {RESPONSES_FILE}")

            else:
                print(f"❌ Remote server error: {response.status_code}")
                print(response.text)
                
                response_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": "remote",
                    "model_name": url,
                    "image_path": str(screenshot_path),
                    "error": f"Remote error: {response.status_code}",
                    "raw_response": response.text
                }
                save_response(response_entry)

    except Exception as e:
        print(f"❌ An error occurred while communicating with remote server: {e}")
        response_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": "remote",
            "model_name": url,
            "image_path": str(screenshot_path),
            "error": str(e)
        }
        save_response(response_entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze screenshots using AI vision models")
    parser.add_argument("image_path", help="Path to the screenshot image")
    parser.add_argument("-g", "--gemini", action="store_true", 
                       help="Use Google Gemini model instead of local Ollama")
    parser.add_argument("-r", "--remote", action="store_true",
                       help="Use a remote server (e.g. ml-fastvlm)")
    parser.add_argument("--url", type=str, default=REMOTE_URL,
                       help=f"URL for remote server (default: {REMOTE_URL})")
    
    args = parser.parse_args()
    
    # Initialize
    ensure_responses_file()
    print_config_info()
    
    if args.gemini:
        generate_activity_summary_gemini(args.image_path)
    elif args.remote:
        generate_activity_summary_remote(args.image_path, args.url)
    else:
        generate_activity_summary_ollama(args.image_path)