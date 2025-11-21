# TimeLine AI

An AI-powered activity timeline tracker that automatically captures, analyzes, and logs your computer activities.

## Features

- 📸 **Automatic Screenshot Capture**: Takes screenshots at configurable intervals
- 🧠 **AI-Powered Analysis**: Uses Ollama (local), Google Gemini, or a custom remote server to analyze screen content
- 🔍 **Smart Deduplication**: Uses SSIM algorithm to eliminate similar screenshots (95% threshold)
- 🌐 **Web Interface**: Beautiful Flask-based dashboard to view your activity timeline
- ⚙️ **Configurable**: Adjust capture interval, model selection, and similarity threshold
- 📊 **Token Tracking**: Monitors token usage for each analysis

## Screenshots

The web interface includes:
- **Timeline View**: Chronological display of all captured activities
- **Configuration Panel**: Easy-to-use settings for customization
- **Real-time Status**: Live updates on capture status and statistics

## Installation

### Quick Setup (Recommended)

Run the automated setup script:
```bash
python setup.py
```

This script will:
1. ✅ Check if Ollama is installed (redirects to download if not)
2. ✅ Check if the qwen2.5vl:3b model is available
3. ✅ Download the model if needed
4. ✅ Install all Python dependencies from requirements.txt
5. ✅ Create .env configuration file

### Manual Installation

1. Clone this repository:
```bash
git clone https://github.com/Developerr86/TimeLine-AI.git
cd TimeLine-AI
```

2. Install Ollama (for local AI analysis):
- Download from [ollama.ai](https://ollama.ai)
- Pull a vision model: `ollama pull qwen2.5vl:3b`

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your configuration:
- `GEMINI_API_KEY`: Required if using Google Gemini model
- `OLLAMA_MODEL`: Default Ollama model to use (default: qwen2.5vl:3b)
- `GEMINI_MODEL`: Default Gemini model to use (default: gemini-2.5-pro)

## Usage

### Starting the Application

Run the Flask web application:
```bash
python main.py
```

The web interface will be available at `http://localhost:5000`

### Using the Original CLI Tool

The original command-line analysis tool is still available:

```bash
# Analyze with Ollama (local)
python analyse-screen.py path/to/screenshot.png

# Analyze with Gemini
python analyse-screen.py -g path/to/screenshot.png

# Analyze with Remote Server
python analyse-screen.py -r --url http://localhost:5001/predict path/to/screenshot.png
```

## Configuration

Access the configuration page at `http://localhost:5000/config` to adjust:

- **Screenshot Interval**: How often to capture (1-300 seconds)
- **Similarity Threshold**: SSIM threshold for duplicate detection (0-1)
- **Model Type**: Choose between Ollama (local), Gemini (cloud), or Remote Server
- **Model Names/URL**: Specify which AI model or server URL to use

## How It Works

1. **Capture**: Takes a screenshot every N seconds (configurable)
2. **Compare**: Uses SSIM algorithm to compare with the last screenshot
3. **Filter**: Discards screenshots that are >95% similar (configurable)
4. **Analyze**: Sends unique screenshots to AI model for analysis
5. **Store**: Saves analysis results to `responses.json`
6. **Display**: Shows timeline in the web interface

## API Endpoints

- `GET /` - Main timeline view
- `GET /config` - Configuration page
- `GET /api/status` - Get current status
- `POST /api/start` - Start screenshot capture
- `POST /api/stop` - Stop screenshot capture
- `GET/POST /api/config` - Get or update configuration
- `GET /api/responses` - Get all responses as JSON

## File Structure

```
TimeLine-AI/
├── main.py                 # Flask web application
├── analyse-screen.py       # Original CLI analysis tool
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from .env.example)
├── config.json             # Runtime configuration (auto-generated)
├── responses.json          # Analysis results (auto-generated)
├── screenshots/            # Captured screenshots (auto-generated)
├── templates/              # HTML templates
│   ├── index.html         # Timeline view
│   └── config.html        # Configuration page
└── static/                 # Static assets
    └── style.css          # Stylesheet

```

## Requirements

- Python 3.8+
- Ollama (optional, for local AI analysis)
- Google Gemini API key (optional, for cloud AI analysis)

## Dependencies

See [requirements.txt](requirements.txt) for full list:
- Flask (web framework)
- Pillow (screenshot capture)
- requests (HTTP client)
- opencv-python (image processing)
- scikit-image (SSIM comparison)
- ollama (Ollama API client)
- google-generativeai (Gemini API client)

## Privacy & Security

- All screenshots are stored locally in the `screenshots/` folder
- When using Ollama, all processing happens on your machine
- When using Gemini, screenshots are sent to Google's API
- When using Remote Server, screenshots are sent to your configured URL
- No data is shared with third parties beyond your chosen AI provider

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Ollama connection errors
- Ensure Ollama is running: `ollama serve`
- Check model is installed: `ollama list`
- Pull model if needed: `ollama pull qwen2.5vl:3b`

### Gemini API errors
- Verify `GEMINI_API_KEY` is set in `.env`
- Check API key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)

### Screenshot capture not working
- Ensure no other screen capture tools are blocking access
- Check file permissions for the `screenshots/` directory
