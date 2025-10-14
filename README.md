# Screenshot Analysis Tool

A Python tool for analyzing screenshots using AI vision models (Ollama and Google Gemini).

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   # Copy the example file
   copy .env.example .env
   
   # Edit .env file with your actual API keys and preferences
   ```

3. **Set up your .env file:**
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key
   OLLAMA_MODEL=qwen2.5vl:3b
   GEMINI_MODEL=gemini-1.5-flash
   ```

## Usage

### Analyze with Ollama (default)
```bash
python analyse-screen.py screenshot.png
```

### Analyze with Gemini
```bash
python analyse-screen.py -g screenshot.png
```

### Test Gemini API
```bash
python gemini-test.py
```

## Features

- **Environment Configuration**: Uses `.env` file for secure API key management
- **Token Tracking**: Displays and logs token usage for both models
- **Response Logging**: All responses saved to `responses.json` with timestamps
- **Verbose Output**: Detailed information about processing times and token counts
- **Error Handling**: Comprehensive error logging and recovery

## Output

All analysis results are saved to `responses.json` with the following information:
- Timestamp
- Model used
- Token usage statistics
- Generated title and summary
- Raw model response
- Error information (if applicable)

## Requirements

- Python 3.7+
- Ollama (for local model usage)
- Internet connection (for Gemini API)
- Valid Gemini API key (for Gemini usage)