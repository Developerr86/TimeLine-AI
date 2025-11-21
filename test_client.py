import requests
import argparse
import os

def test_predict(image_path, prompt="Describe the image.", url="http://localhost:5001/predict"):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    print(f"Sending request to {url}...")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")

    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {'prompt': prompt}
        
        try:
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                print("\nResponse:")
                print(response.json().get('response'))
            else:
                print(f"\nError: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.ConnectionError:
            print("\nError: Could not connect to server. Is it running?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="screenshots/screenshot_20251114_120811_514018.png")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--url", type=str, default="http://localhost:5001/predict")
    args = parser.parse_args()

    test_predict(args.image, args.prompt, args.url)
