# Complete Video Editor - AI-Powered Video Enhancement Tool 🎬

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-green.svg)](https://ffmpeg.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A revolutionary single-file video processing pipeline that combines AI-powered audio enhancement, vocal separation, noise reduction, and multi-language subtitle generation. Features silence-first optimization for 25-45% faster processing while maintaining zero content loss.

## 🚀 Key Features

- **🎤 AI Vocal Separation**: Advanced Demucs model for professional vocal isolation
- **🔇 Intelligent Noise Reduction**: Smart noise profiling with targeted denoising
- **⚡ Silence-First Optimization**: Process only speech content for maximum efficiency
- **📝 AI Transcription**: Whisper-powered transcription with word-level timestamps
- **🌐 Multi-Language Subtitles**: Automatic English and Hindi subtitle generation
- **🎬 Zero-Loss Video Assembly**: Perfect audio-video synchronization with content preservation
- **💾 Smart Model Caching**: Persistent model storage for faster subsequent runs
- **🧠 Professional Audio Enhancement**: Surgical precision audio editing

## 🎯 Processing Pipeline

```
Input Video → Audio Extraction → Vocal Separation → Noise Profiling → 
Smart Silence Removal → Targeted Denoising → Transcription → 
Subtitle Generation → Intelligent Video Assembly → Enhanced Output
```

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for large videos)
- **Storage**: 2-5GB free space (for models and temporary files)

### Required Software
- **FFmpeg**: Must be installed and accessible from command line
- **Python 3.8+**: With pip package manager

## 🛠️ Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Veereshamaragatti/Capstone_Veer.git
cd Capstone_Veer
```

### Step 2: Install FFmpeg

#### Windows:
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your PATH environment variable
4. Verify installation:
```cmd
ffmpeg -version
```

#### macOS (using Homebrew):
```bash
brew install ffmpeg
```

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

### Step 3: Create Virtual Environment

#### Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv video_editor_env

# Activate virtual environment
.\video_editor_env\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Windows (Command Prompt):
```cmd
# Create virtual environment
python -m venv video_editor_env

# Activate virtual environment
video_editor_env\Scripts\activate.bat
```

#### macOS/Linux:
```bash
# Create virtual environment
python3 -m venv video_editor_env

# Activate virtual environment
source video_editor_env/bin/activate
```

### Step 4: Install Dependencies

The script automatically installs required dependencies on first run, but you can manually install them:

```bash
pip install torch torchaudio demucs noisereduce pydub openai-whisper librosa soundfile googletrans==4.0.0-rc1 numpy
```

## 🎮 Usage

### Basic Usage
```bash
# Make sure your virtual environment is activated
python complete_video_editor_single_file.py
```

### Advanced Usage
The script is configured to process `Unit-2 TA Session 1.mp4` by default. To process a different video:

1. **Edit the script**: Open `complete_video_editor_single_file.py` and modify the `input_video` variable in the `main()` function:
```python
def main():
    # Change this to your video file
    input_video = "your_video_file.mp4"
```

2. **Place your video**: Put your video file in the project directory

3. **Run the script**:
```bash
python complete_video_editor_single_file.py
```

## 📁 Project Structure

```
Capstone_Veer/
├── complete_video_editor_single_file.py    # Main processing script
├── video_editor_env/                       # Virtual environment
├── model_cache/                            # AI model cache (auto-created)
│   ├── demucs_htdemucs.pkl                 # Cached Demucs model
│   └── tiny.pt                             # Cached Whisper model
├── output/                                 # Generated output files
│   ├── enhanced_[video_name].wav           # Enhanced audio
│   ├── FINAL_Enhanced_[video_name].mp4     # Final enhanced video
│   ├── [video_name]_english.srt            # English subtitles
│   ├── [video_name]_hindi.srt              # Hindi subtitles
│   └── transcript_[video_name].json        # Full transcript
├── [input_videos].mp4                      # Your input video files
└── README.md                               # This file
```

## 📊 Performance & Time Estimates

### Processing Time (Silence-First Optimized):
- **5-minute video**: ~3-4 minutes processing
- **15-minute video**: ~8-12 minutes processing
- **30-minute video**: ~15-20 minutes processing

### First Run vs Subsequent Runs:
- **First run**: +2-3 minutes (model downloads)
- **Subsequent runs**: Optimized with cached models

### Time Breakdown:
1. **Audio Extraction**: 10-30 seconds
2. **Vocal Separation**: 1-3 minutes (CPU optimized)
3. **Noise Profiling**: 5-15 seconds
4. **Smart Silence Removal**: 15-45 seconds
5. **Targeted Denoising**: 30-90 seconds (only speech content!)
6. **Transcription**: 15-60 seconds
7. **Subtitle Generation**: 10-30 seconds
8. **Video Assembly**: 30-120 seconds

## 🎯 Output Files

After processing, you'll find these files in the `output/` directory:

### Video Files:
- `FINAL_Enhanced_[video_name].mp4` - Final enhanced video with improved audio

### Audio Files:
- `enhanced_[video_name].wav` - Standalone enhanced audio

### Subtitle Files:
- `[video_name]_english.srt` - English subtitles
- `[video_name]_hindi.srt` - Hindi subtitles (auto-translated)

### Transcript Files:
- `transcript_[video_name].json` - Complete transcript with timestamps

## 📱 How to Use Subtitles

### VLC Media Player:
1. Open the enhanced video in VLC
2. Go to **Subtitle** → **Add Subtitle File**
3. Select either English or Hindi SRT file
4. Enjoy your enhanced video with subtitles!

### Other Media Players:
Most modern media players support SRT files. Simply:
1. Ensure the SRT file has the same name as your video
2. Place it in the same directory
3. Open the video - subtitles should load automatically

## 🧠 Advanced Features

### Silence-First Optimization
- Removes silence **before** denoising to save 25-45% processing time
- Maintains perfect content synchronization
- Zero loss of actual speech content

### Intelligent Noise Profiling
- Analyzes silence segments to understand noise characteristics
- Applies targeted noise reduction only where needed
- Preserves natural speech dynamics

### Professional Audio Enhancement
- Separates vocals from background music/noise
- Applies spectral gating for clean audio
- Maintains audio quality while removing unwanted noise

### Zero-Loss Video Assembly
- Creates timestamp mapping for perfect synchronization
- Removes only silence, preserves all speech content
- Maintains original video quality

## 🔧 Troubleshooting

### Common Issues:

#### 1. FFmpeg Not Found
```
❌ Error: FFmpeg not found
```
**Solution**: Install FFmpeg and ensure it's in your system PATH

#### 2. Virtual Environment Issues
```
❌ Error: Cannot activate virtual environment
```
**Solution (Windows)**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3. Memory Issues
```
❌ Error: Out of memory
```
**Solution**: 
- Close other applications
- Use a smaller video file for testing
- Ensure at least 8GB RAM available

#### 4. Model Download Issues
```
❌ Error: Failed to download models
```
**Solution**:
- Check internet connection
- Clear model cache: Delete `model_cache/` folder
- Run script again

### Performance Optimization:

#### For Faster Processing:
- Use SSD storage for temporary files
- Close unnecessary applications
- Ensure good internet connection for first run (model downloads)

#### For Large Videos:
- Consider splitting very long videos (>2 hours)
- Ensure sufficient disk space (3x video size recommended)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Demucs**: For state-of-the-art audio source separation
- **OpenAI Whisper**: For accurate speech transcription
- **FFmpeg**: For robust multimedia processing
- **PyTorch**: For deep learning capabilities
- **Google Translate**: For multi-language subtitle support

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#🔧-troubleshooting) section
2. Create an issue in the GitHub repository
3. Include your error message and system information

## 🌟 Star History

If this project helped you, please consider giving it a star! ⭐

---

**Made with ❤️ for content creators and video enthusiasts**

*Transform your videos with AI-powered enhancement - No content loss, Maximum quality!*
