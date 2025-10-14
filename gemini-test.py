import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure your API key from environment variable
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in environment variables.")
    print("Please create a .env file with: GEMINI_API_KEY=your_api_key_here")
    exit(1)

genai.configure(api_key=api_key)

def test_gemini_api():
    """
    Tests the Gemini API by listing available models and generating content.
    """
    try:
        # 1. List available models
        print("Available models:")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f"  - {m.name}")

        # 2. Get a specific model (e.g., 'gemini-pro')
        model = genai.GenerativeModel("gemini-2.0-flash")

        # 3. Generate content
        print("\nGenerating content...")
        prompt = "Write a short, inspiring poem about the beauty of nature."
        response = model.generate_content(prompt)

        print("\n--- Generated Content ---")
        print(response.text)
        print("-------------------------")

        # 4. (Optional) Test a multi-turn conversation (chat)
        print("\nStarting a chat session...")
        chat = model.start_chat(history=[])
        chat_response1 = chat.send_message("Hello, tell me a fun fact about outer space.")
        print(f"\nUser: Hello, tell me a fun fact about outer space.")
        print(f"Gemini: {chat_response1.text}")

        chat_response2 = chat.send_message("That's interesting! What about black holes?")
        print(f"\nUser: That's interesting! What about black holes?")
        print(f"Gemini: {chat_response2.text}")

        # 5. (Optional) Test image generation (if using a vision-capable model)
        # Note: 'gemini-pro-vision' is required for image input.
        # You'll need to install Pillow: pip install Pillow
        # from PIL import Image
        #
        # print("\nTesting image generation (requires 'gemini-pro-vision')...")
        # vision_model = genai.GenerativeModel('gemini-pro-vision')
        # img = Image.open('path/to/your/image.jpg') # Replace with a path to an actual image
        #
        # vision_response = vision_model.generate_content(["Describe this image:", img])
        # print(f"\nImage description: {vision_response.text}")


    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your API key is correct and you have network access.")
        print("You can get an API key from: https://makersuite.google.com/keys")

if __name__ == "__main__":
    print("Running Gemini API test script...")
    test_gemini_api()
    print("\nTest script finished.")