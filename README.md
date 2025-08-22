# Advanced Video Enhancer

A powerful, user-friendly GUI application for video enhancement, upscaling, and format conversion using OpenCV and FFmpeg.

![Video Enhancer GUI](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

<p align="center">
  <img src="https://github.com/leonelpedroza/videoenhancer/blob/main/screenshoot.png">
</p>


### 🎨 Video Enhancement
- **Multiple Enhancement Methods:**
  - Unsharp Mask (fast sharpening)
  - CLAHE (contrast enhancement)
  - Bilateral Filter (noise reduction)
  - Combined (balanced enhancement)
  - Advanced AI-like processing

### 📏 Video Upscaling
- **Scaling Options:** 0.1x to 10x resolution scaling
- **Interpolation Methods:**
  - Bicubic (recommended)
  - Lanczos (highest quality)
  - Linear (fast)
  - Nearest (fastest)

### 🔄 Format Conversion
- **Supported Formats:** AVI, MP4, MOV, MKV, WMV
- **Fast conversion** using FFmpeg
- **Audio preservation** during processing

### 🎵 Audio Features
- **Preserve original audio** during enhancement/upscaling
- **Automatic audio merging** using FFmpeg
- **Format-specific audio codecs**

### ⚙️ Processing Options
- **Operation Modes:**
  - Enhance only (improve quality)
  - Upscale only (increase resolution)
  - Both (enhance + upscale)
  - Convert (format conversion only)

### 🎯 Range Selection
- **Frame-based range:** Process specific frame ranges
- **Time-based range:** Process specific time segments
- **Real-time synchronization** between frame and time values

### 🔧 Advanced Controls
- **Pause/Resume/Stop** processing
- **Real-time progress monitoring**
- **Comprehensive logging** with file export
- **Format-specific codec optimization**

## 📋 Requirements

### System Requirements
- **Python:** 3.8 or higher
- **Operating System:** Windows, macOS, or Linux
- **Memory:** 4GB RAM minimum (8GB recommended)
- **Storage:** Space for input and output videos

### Dependencies
- **OpenCV** (opencv-python)
- **PySide6** (Qt6 GUI framework)
- **NumPy** (numerical operations)
- **FFmpeg** (optional, for audio preservation and format conversion)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/advanced-video-enhancer.git
cd advanced-video-enhancer
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg (Optional but Recommended)

#### Windows:
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract and add to PATH, or use:
```bash
# Using chocolatey
choco install ffmpeg

# Using winget
winget install ffmpeg
```

#### macOS:
```bash
# Using Homebrew
brew install ffmpeg
```

#### Linux:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# CentOS/RHEL/Fedora
sudo dnf install ffmpeg
```

## 🎮 Usage

### Starting the Application
```bash
python video_enhancer_gui.py
```

### Basic Workflow
1. **Select Input Video:** Click "Browse" to choose your video file
2. **Choose Output Format:** Select desired output format (MP4, AVI, etc.)
3. **Select Operation:** Choose enhancement, upscaling, both, or conversion
4. **Configure Settings:** Adjust enhancement method and scaling options
5. **Set Range (Optional):** Process entire video or specific segments
6. **Enable Audio Preservation:** Check "Preserve Audio" if desired
7. **Start Processing:** Click "Start" and monitor progress

### Operation Modes

#### 🎨 Enhance Only
- Improves video quality without changing resolution
- Faster processing, maintains file size
- Good for sharpening, noise reduction, contrast improvement

#### 📏 Upscale Only
- Increases resolution using interpolation
- Creates larger, potentially clearer images
- May introduce artifacts with poor source quality

#### 🚀 Both (Recommended)
- Enhances quality first, then upscales
- Best quality results but slower processing
- Maximum improvement for most videos

#### 🔄 Convert
- Format conversion only using FFmpeg
- Fastest option, preserves original quality
- Perfect for format compatibility

### Enhancement Methods

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| **Unsharp Mask** | ⚡⚡⚡ | ⭐⭐ | Slightly blurry videos |
| **CLAHE** | ⚡⚡ | ⭐⭐⭐ | Dark/bright area enhancement |
| **Bilateral** | ⚡ | ⭐⭐⭐ | Noise reduction |
| **Combined** | ⚡⚡ | ⭐⭐⭐⭐ | General purpose (recommended) |
| **Advanced** | ⚡ | ⭐⭐⭐⭐⭐ | Maximum quality |

## 📊 Examples

### Basic Enhancement
```
Input: 720x480 AVI video
Operation: Enhance (Combined method)
Output: 720x480 MP4 with improved quality
```

### Upscaling
```
Input: 480p video
Operation: Upscale 2x (Bicubic)
Output: 960p video with doubled resolution
```

### Complete Processing
```
Input: Old 320x240 AVI
Operation: Both (Combined + 3x Lanczos)
Output: 960x720 MP4 with enhanced quality and audio
```

## 🔧 Configuration

### Logging
- **Enable/Disable:** Use checkbox in GUI
- **Auto-save:** Logs saved to same directory as input video
- **Manual save:** Export logs using "Save Log" button

### Audio Preservation
- **Requires FFmpeg:** Install FFmpeg for audio features
- **Automatic detection:** GUI shows FFmpeg availability
- **Format support:** Works with all supported video formats

### Performance Tips
- **Use "Combined" enhancement** for best quality/speed balance
- **Choose appropriate scaling factors** (2x-4x recommended)
- **Process shorter segments** for testing settings
- **Enable logging** for troubleshooting

## 🐛 Troubleshooting

### Common Issues

#### No Audio in Output
- **Solution:** Install FFmpeg and enable "Preserve Audio"
- **Check:** FFmpeg status indicator in GUI

#### Processing Fails
- **Check:** Log display for detailed error messages
- **Try:** Different output format (MP4 recommended)
- **Verify:** Input video file is not corrupted

#### Slow Processing
- **Use:** Lower enhancement methods for speed
- **Reduce:** Scaling factor if not needed
- **Process:** Smaller time ranges for testing

#### FFmpeg Warnings
- Application handles codec compatibility automatically
- Warnings don't affect output quality
- Check logs for detailed codec information

### Getting Help
1. **Check logs** for detailed error information
2. **Try different formats** (MP4 is most compatible)
3. **Test with shorter videos** first
4. **Verify FFmpeg installation** for audio features

## 🤝 Contributing

We welcome contributions! Please feel free to submit:

- **Bug reports** with detailed logs
- **Feature requests** for new functionality  
- **Pull requests** with improvements
- **Documentation** updates

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** for powerful computer vision capabilities
- **PySide6/Qt** for the modern GUI framework
- **FFmpeg** for audio processing and format conversion
- **NumPy** for efficient numerical operations

## 📞 Support

If you find this project helpful, please:
- ⭐ Star the repository
- 🐛 Report bugs in Issues
- 💡 Suggest features
- 🔄 Share with others

