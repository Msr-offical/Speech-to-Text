# Speech-to-Text Recognition Tool

A Flask-based web application that provides speech-to-text transcription using OpenAI's Whisper model, with support for file uploads and microphone recording. The application includes a WER (Word Error Rate) comparison tool to evaluate transcription accuracy.

## Features

- Audio file transcription (supports .wav, .mp3, and .m4a formats)
- Real-time microphone recording and transcription
- Word Error Rate (WER) calculation and visualization
- Word-level difference highlighting
- Detailed statistics for transcription accuracy
- Modern, responsive UI

## Prerequisites

- Python 3.10 (recommended) or Python 3.9
- FFmpeg (required for audio processing)
- Windows 10 or later (for Windows users)

## Installation

### Windows Setup

For detailed Windows setup instructions, see [SETUP.md](SETUP.md).

### Quick Start

1. Create a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

3. Install FFmpeg:
- **Windows**: See [SETUP.md](SETUP.md) for detailed instructions
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the application:
   - Upload audio files using the file input
   - Record audio using the microphone button
   - Enter ground truth text for comparison
   - Click "Compare" to see the WER analysis

## Features in Detail

### Audio Input
- **File Upload**: Supports .wav, .mp3, and .m4a audio files
- **Microphone Recording**: Real-time audio recording with visual feedback

### Transcription
- Uses OpenAI's Whisper model for accurate speech-to-text conversion
- Displays transcribed text in a clean, editable text area

### Comparison
- Word Error Rate (WER) calculation
- Visual highlighting of differences:
  - Deletions (red)
  - Insertions (green)
  - Substitutions (yellow)
- Detailed statistics including:
  - WER percentage
  - Count of substitutions, deletions, and insertions
  - List of substituted words

## Technical Details

- **Backend**: Flask (Python)
- **Speech Recognition**: OpenAI Whisper
- **WER Calculation**: jiwer library
- **Frontend**: HTML5, CSS3, JavaScript
- **Audio Processing**: pydub, soundfile
- **CPU Support**: Optimized for CPU-only operation

## Troubleshooting

If you encounter installation issues:
1. Ensure you're using Python 3.10 or 3.9
2. Check FFmpeg installation and PATH
3. Try the fallback installation method in [SETUP.md](SETUP.md)
4. For Windows-specific issues, refer to [SETUP.md](SETUP.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 