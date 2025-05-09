# Windows Setup Guide for Speech-to-Text Project

## Prerequisites

1. **Python Installation**:
   - Download and install Python 3.10 from [python.org](https://www.python.org/downloads/release/python-3109/)
   - During installation, check "Add Python to PATH"
   - Verify installation: `python --version` should show 3.10.x

2. **FFmpeg Installation**:
   - Download FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
   - Choose "ffmpeg-release-essentials.zip"
   - Extract to `C:\ffmpeg`
   - Add to PATH:
     ```powershell
     $env:Path += ";C:\ffmpeg\bin"
     ```

## Virtual Environment Setup

1. **Create and Activate Virtual Environment**:
   ```powershell
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   # Upgrade pip
   python -m pip install --upgrade pip

   # Install requirements
   pip install -r requirements.txt
   ```

## Troubleshooting

### Common Issues and Solutions

1. **FFmpeg Not Found**:
   - Verify FFmpeg is in PATH: `ffmpeg -version`
   - If not found, add manually:
     ```powershell
     [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ffmpeg\bin", "User")
     ```

2. **Whisper Installation Issues**:
   - If `openai-whisper` fails, try:
     ```powershell
     pip install git+https://github.com/openai/whisper.git
     ```

3. **Torch Installation Issues**:
   - If torch fails, try CPU-only version:
     ```powershell
     pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```

### Fallback Installation Method

If the standard installation fails, use this step-by-step approach:

```powershell
# 1. Install core dependencies first
pip install Flask==2.3.3 Werkzeug==2.3.7

# 2. Install PyTorch CPU version
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install Whisper
pip install git+https://github.com/openai/whisper.git

# 4. Install remaining dependencies
pip install numpy==1.24.3 jiwer==3.0.3 soundfile==0.12.1 librosa==0.10.1 pydub==0.25.1 ffmpeg-python==0.2.0 tqdm==4.66.1 regex==2023.10.3
```

## Running the Application

1. **Start the Flask Server**:
   ```powershell
   python app.py
   ```

2. **Access the Application**:
   - Open browser and navigate to: `http://localhost:5000`

## System Requirements

- Windows 10 or later
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Python 3.10.x
- FFmpeg installed and in PATH

## Notes

- The application uses CPU-only mode by default
- No GPU dependencies are required
- Audio processing may be slower on CPU but will work reliably 