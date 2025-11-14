#!/usr/bin/env python3
"""
TimeLine AI Setup Script
Checks for Ollama installation, downloads required models, and installs dependencies.
"""

import subprocess
import sys
import os
import platform
import webbrowser
from pathlib import Path

def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════╗
    ║                                               ║
    ║           TimeLine AI - Setup Script          ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝
    """
    print(banner)

def print_step(step_number, message):
    print(f"\n{'='*60}")
    print(f"  STEP {step_number}: {message}")
    print(f"{'='*60}")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def print_warning(message):
    print(f"⚠️  {message}")

def check_ollama_installed():
    """Check if Ollama is installed on the system."""
    print_step(1, "Checking for Ollama installation")
    
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Ollama is installed: {version}")
            return True
        else:
            return False
    except FileNotFoundError:
        print_error("Ollama is not installed on this system")
        return False
    except subprocess.TimeoutExpired:
        print_error("Ollama check timed out")
        return False
    except Exception as e:
        print_error(f"Error checking Ollama: {e}")
        return False

def get_ollama_download_url():
    """Get the appropriate Ollama download URL based on the operating system."""
    os_name = platform.system().lower()
    
    if os_name == "windows":
        return "https://ollama.ai/download/windows"
    elif os_name == "darwin":  # macOS
        return "https://ollama.ai/download/mac"
    elif os_name == "linux":
        return "https://ollama.ai/download/linux"
    else:
        return "https://ollama.ai/download"

def prompt_ollama_installation():
    """Prompt user to install Ollama and open download page."""
    print_warning("Ollama is required to run TimeLine AI with local models")
    print_info("Opening Ollama download page in your browser...")
    
    download_url = get_ollama_download_url()
    print(f"\n📥 Download URL: {download_url}")
    
    try:
        webbrowser.open(download_url)
        print_success("Browser opened with Ollama download page")
    except Exception as e:
        print_warning(f"Could not open browser: {e}")
        print_info(f"Please manually visit: {download_url}")
    
    print("\n" + "="*60)
    print("Please install Ollama and run this setup script again.")
    print("="*60)
    
    input("\nPress Enter to exit...")
    sys.exit(0)

def check_ollama_model(model_name):
    """Check if a specific Ollama model is available."""
    print_step(2, f"Checking for model: {model_name}")
    
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            models = result.stdout.lower()
            if model_name.lower() in models:
                print_success(f"Model '{model_name}' is already installed")
                return True
            else:
                print_warning(f"Model '{model_name}' is not installed")
                return False
        else:
            print_error("Could not list Ollama models")
            return False
    except Exception as e:
        print_error(f"Error checking models: {e}")
        return False

def download_ollama_model(model_name):
    """Download an Ollama model."""
    print_step(3, f"Downloading model: {model_name}")
    print_info("This may take several minutes depending on your internet connection...")
    
    try:
        print(f"\n🔄 Running: ollama pull {model_name}\n")
        
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print_success(f"\nModel '{model_name}' downloaded successfully!")
            return True
        else:
            print_error(f"\nFailed to download model '{model_name}'")
            return False
    except Exception as e:
        print_error(f"Error downloading model: {e}")
        return False

def install_requirements():
    """Install Python dependencies from requirements.txt."""
    print_step(4, "Installing Python dependencies")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        return False
    
    print_info("Installing packages from requirements.txt...")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success("All dependencies installed successfully!")
            print("\n" + "-"*60)
            print(result.stdout)
            return True
        else:
            print_error("Failed to install some dependencies")
            print("\n" + "-"*60)
            print(result.stderr)
            return False
    except Exception as e:
        print_error(f"Error installing requirements: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist."""
    print_step(5, "Configuring environment variables")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print_info(".env file already exists, skipping...")
        return True
    
    if env_example.exists():
        try:
            with open(env_example, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print_success("Created .env file from .env.example")
            print_info("You can edit .env to configure your API keys and models")
            return True
        except Exception as e:
            print_warning(f"Could not create .env file: {e}")
            return False
    else:
        print_warning(".env.example not found, creating basic .env file...")
        try:
            content = """# Ollama Configuration
OLLAMA_MODEL=qwen2.5vl:3b

# Gemini Configuration
GEMINI_MODEL=gemini-2.5-pro
GEMINI_API_KEY=your_gemini_api_key_here
"""
            with open(env_file, 'w') as f:
                f.write(content)
            
            print_success("Created basic .env file")
            return True
        except Exception as e:
            print_error(f"Could not create .env file: {e}")
            return False

def print_completion_message():
    """Print setup completion message with next steps."""
    message = """
    ╔═══════════════════════════════════════════════╗
    ║                                               ║
    ║          ✅ Setup Complete! ✅                ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝

    🚀 Next Steps:

    1. Review and edit your .env file:
       - Add your GEMINI_API_KEY if you want to use Gemini
       - Adjust model names if needed

    2. Start the application:
       python main.py

    3. Open your browser to:
       http://localhost:5000

    4. Click "Start" to begin capturing and analyzing screenshots!

    📚 For more information, see README.md

    ═══════════════════════════════════════════════
    """
    print(message)

def main():
    """Main setup function."""
    print_banner()
    
    print_info("Starting TimeLine AI setup process...\n")
    
    # Step 1: Check if Ollama is installed
    if not check_ollama_installed():
        prompt_ollama_installation()
        return
    
    # Step 2 & 3: Check and download model
    model_name = "qwen2.5vl:3b"  # Changed from qwen3-vl:2b to the correct model
    
    if not check_ollama_model(model_name):
        print_warning(f"Model '{model_name}' needs to be downloaded")
        
        response = input(f"\n📥 Download {model_name} now? (y/n): ").strip().lower()
        
        if response == 'y':
            if not download_ollama_model(model_name):
                print_error("Model download failed. You can download it later using:")
                print(f"   ollama pull {model_name}")
                response = input("\nContinue setup anyway? (y/n): ").strip().lower()
                if response != 'y':
                    sys.exit(1)
        else:
            print_info("Skipping model download. You can download it later using:")
            print(f"   ollama pull {model_name}")
    
    # Step 4: Install Python dependencies
    if not install_requirements():
        print_warning("Some dependencies failed to install")
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 5: Create .env file
    create_env_file()
    
    # Show completion message
    print_completion_message()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error during setup: {e}")
        sys.exit(1)
