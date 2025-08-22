import sys
import os
import cv2
import numpy as np
import threading
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QGridLayout, QWidget, QPushButton, QLabel, QLineEdit, 
                               QComboBox, QProgressBar, QTextEdit, QFileDialog, 
                               QSpinBox, QDoubleSpinBox, QGroupBox, QFrame, QCheckBox,
                               QMessageBox, QSplitter, QRadioButton, QButtonGroup)
from PySide6.QtCore import QThread, Signal, QTimer, Qt
from PySide6.QtGui import QFont, QPixmap, QIcon

class VideoProcessor(QThread):
    progress_update = Signal(int, int, int)  # current_frame, total_frames, percentage
    status_update = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)
    log_signal = Signal(str)  # New signal for logging
    
    def __init__(self):
        super().__init__()
        self.input_path = ""
        self.output_path = ""
        self.operation = "enhance"
        self.enhancement_method = "combined"
        self.scale_factor = 2.0
        self.upscale_method = "bicubic"
        self.start_frame = 0
        self.end_frame = -1
        self.start_time = 0.0
        self.end_time = -1.0
        self.use_time_range = False
        self.output_format = "avi"
        self.preserve_audio = False
        self.paused = False
        self.stopped = False
        self.logger = None
        self.log_file = None
        
    def setup_logging(self, log_file_path):
        """Setup logging to file"""
        self.log_file = log_file_path
        self.logger = logging.getLogger('VideoProcessor')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Log initial info
        self.log("="*60)
        self.log("Video Enhancement Session Started")
        self.log("="*60)
    
    def log(self, message, level="INFO"):
        """Log message to file and emit signal"""
        if self.logger:
            if level == "ERROR":
                self.logger.error(message)
            elif level == "WARNING":
                self.logger.warning(message)
            elif level == "DEBUG":
                self.logger.debug(message)
            else:
                self.logger.info(message)
        
        # Emit signal for GUI display
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_signal.emit(log_entry)
        
        # Also update status for important messages
        if level in ["ERROR", "WARNING"] or "frame" not in message.lower():
            self.status_update.emit(message)
        
    def set_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)
    
    def pause(self):
        self.paused = True
        self.log("Processing paused by user")
    
    def resume(self):
        self.paused = False
        self.log("Processing resumed by user")
    
    def stop(self):
        self.stopped = True
        self.log("Processing stopped by user")
    
    def get_codec_fourcc(self, format_name):
        """Get format-appropriate codec for different video formats"""
        # Use format-specific codecs for better compatibility
        codecs = {
            'avi': cv2.VideoWriter_fourcc(*'MJPG'),  # MJPG works great with AVI
            'mp4': cv2.VideoWriter_fourcc(*'mp4v'),  # MP4V is standard for MP4
            'mov': cv2.VideoWriter_fourcc(*'mp4v'),  # MP4V works with MOV
            'mkv': cv2.VideoWriter_fourcc(*'XVID'),  # XVID for MKV
            'wmv': cv2.VideoWriter_fourcc(*'WMV2'),  # WMV2 for WMV files
            'flv': cv2.VideoWriter_fourcc(*'FLV1')   # FLV1 for FLV files
        }
        return codecs.get(format_name.lower(), cv2.VideoWriter_fourcc(*'XVID'))
    
    def enhance_frame_traditional(self, frame, method='unsharp_mask'):
        """Traditional enhancement methods"""
        if method == 'unsharp_mask':
            gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
            enhanced = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
        elif method == 'clahe':
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        elif method == 'bilateral':
            enhanced = cv2.bilateralFilter(frame, 9, 75, 75)
        elif method == 'combined':
            denoised = cv2.bilateralFilter(frame, 9, 75, 75)
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.5)
            sharpened = cv2.addWeighted(denoised, 1.8, gaussian, -0.8, 0)
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            enhanced = frame
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def enhance_frame_advanced(self, frame):
        """Advanced enhancement methods"""
        edge_preserved = cv2.edgePreservingFilter(frame, flags=1, sigma_s=60, sigma_r=0.4)
        detail_enhanced = cv2.detailEnhance(edge_preserved, sigma_s=10, sigma_r=0.15)
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(detail_enhanced, -1, kernel)
        enhanced = cv2.addWeighted(detail_enhanced, 0.7, sharpened, 0.3, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def check_ffmpeg_available(self):
        """Check if FFmpeg is available for audio processing"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.log("FFmpeg is available for audio processing")
                return True
            else:
                self.log("FFmpeg not found - audio preservation will be disabled", "WARNING")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            self.log(f"FFmpeg check failed: {e} - audio preservation will be disabled", "WARNING")
            return False
    
    def add_audio_to_video(self, video_path, audio_source_path, output_path):
        """Add audio from source video to processed video using FFmpeg"""
        try:
            self.log("Adding audio to processed video using FFmpeg...")
            
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', video_path,  # Video input (processed video without audio)
                '-i', audio_source_path,  # Audio source (original video)
                '-c:v', 'copy',  # Copy video stream as-is
                '-c:a', 'aac',   # Encode audio as AAC
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input
                '-shortest',      # End when shortest stream ends
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log("Audio successfully added to video")
                return True
            else:
                self.log(f"FFmpeg error: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("FFmpeg timeout - audio processing took too long", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error adding audio: {e}", "ERROR")
            return False
    
    def convert_format_only(self):
        """Convert video format only using FFmpeg"""
        try:
            self.log("Starting format conversion using FFmpeg...")
            self.log(f"Converting from {self.input_path} to {self.output_path}")
            
            # Calculate time range for FFmpeg
            start_time_str = ""
            duration_str = ""
            
            if self.use_time_range and (self.start_time > 0 or self.end_time > 0):
                start_time_str = f"-ss {self.start_time}"
                if self.end_time > 0:
                    duration = self.end_time - self.start_time
                    duration_str = f"-t {duration}"
            elif not self.use_time_range and (self.start_frame > 0 or self.end_frame > 0):
                # Convert frames to time for FFmpeg
                cap = cv2.VideoCapture(self.input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                start_time = self.start_frame / fps if fps > 0 else 0
                if self.end_frame > 0:
                    end_time = self.end_frame / fps
                    duration = end_time - start_time
                    duration_str = f"-t {duration}"
                start_time_str = f"-ss {start_time}"
            
            # Build FFmpeg command
            cmd = ['ffmpeg', '-y']  # -y to overwrite
            
            if start_time_str:
                cmd.extend(start_time_str.split())
            
            cmd.extend(['-i', self.input_path])
            
            if duration_str:
                cmd.extend(duration_str.split())
            
            # Set codec based on output format
            format_codecs = {
                'mp4': ['-c:v', 'libx264', '-c:a', 'aac'],
                'avi': ['-c:v', 'libx264', '-c:a', 'mp3'],
                'mov': ['-c:v', 'libx264', '-c:a', 'aac'],
                'mkv': ['-c:v', 'libx264', '-c:a', 'aac'],
                'wmv': ['-c:v', 'wmv2', '-c:a', 'wmav2'],
            }
            
            codec_args = format_codecs.get(self.output_format, ['-c:v', 'libx264', '-c:a', 'aac'])
            cmd.extend(codec_args)
            cmd.append(self.output_path)
            
            self.log(f"FFmpeg command: {' '.join(cmd)}")
            
            # Run FFmpeg with progress monitoring
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     text=True, universal_newlines=True)
            
            # Monitor progress
            while True:
                if self.stopped:
                    process.terminate()
                    self.log("Format conversion stopped by user")
                    return False
                
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output and 'time=' in output:
                    # Extract time progress from FFmpeg output
                    try:
                        time_part = [part for part in output.split() if part.startswith('time=')]
                        if time_part:
                            time_str = time_part[0].split('=')[1]
                            # Convert time to seconds and update progress
                            time_parts = time_str.split(':')
                            if len(time_parts) == 3:
                                current_seconds = float(time_parts[0]) * 3600 + float(time_parts[1]) * 60 + float(time_parts[2])
                                # Estimate total duration for progress calculation
                                cap = cv2.VideoCapture(self.input_path)
                                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                total_duration = total_frames / fps if fps > 0 else 1
                                cap.release()
                                
                                percentage = min(100, int((current_seconds / total_duration) * 100))
                                self.progress_update.emit(int(current_seconds), int(total_duration), percentage)
                    except:
                        pass
            
            return_code = process.poll()
            if return_code == 0:
                self.log("Format conversion completed successfully")
                return True
            else:
                stderr_output = process.stderr.read()
                self.log(f"FFmpeg conversion failed: {stderr_output}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error during format conversion: {e}", "ERROR")
            return False
    
    def run(self):
        try:
            self.log("Starting video processing")
            self.log(f"Input file: {self.input_path}")
            self.log(f"Output file: {self.output_path}")
            self.log(f"Operation: {self.operation}")
            
            # Handle format conversion only
            if self.operation == "convert":
                if self.convert_format_only():
                    file_size = os.path.getsize(self.output_path) / (1024 * 1024)
                    self.finished_signal.emit(f"Format conversion completed!\nOutput file: {file_size:.1f} MB")
                else:
                    self.error_signal.emit("Format conversion failed. Check log for details.")
                return
            
            # Regular video processing continues here...
            self.log(f"Enhancement method: {self.enhancement_method}")
            self.log(f"Scale factor: {self.scale_factor}")
            self.log(f"Output format: {self.output_format}")
            self.log(f"Preserve audio: {self.preserve_audio}")
            
            # Check FFmpeg availability if audio preservation is requested
            ffmpeg_available = False
            if self.preserve_audio:
                ffmpeg_available = self.check_ffmpeg_available()
                if not ffmpeg_available:
                    self.log("Audio preservation requested but FFmpeg not available", "WARNING")
            
            # Check input file
            if not os.path.exists(self.input_path):
                error_msg = f"Input file does not exist: {self.input_path}"
                self.log(error_msg, "ERROR")
                self.error_signal.emit(error_msg)
                return
            
            file_size = os.path.getsize(self.input_path) / (1024 * 1024)
            self.log(f"Input file size: {file_size:.2f} MB")
            
            # For audio preservation, create temporary video file
            temp_video_path = None
            final_output_path = self.output_path
            
            if self.preserve_audio and ffmpeg_available:
                temp_video_path = str(Path(self.output_path).parent / f"temp_video_{int(time.time())}.avi")
                self.output_path = temp_video_path  # Process to temp file first
                self.log(f"Audio preservation enabled - using temporary file: {temp_video_path}")
            
            # Open input video
            self.log("Opening video file...")
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                error_msg = f"Cannot open video file: {self.input_path}"
                self.log(error_msg, "ERROR")
                self.error_signal.emit(error_msg)
                return
            
            # Get video properties
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.log(f"Video properties:")
            self.log(f"  Resolution: {original_width}x{original_height}")
            self.log(f"  FPS: {fps}")
            self.log(f"  Total frames: {total_frames}")
            self.log(f"  Duration: {total_frames/fps:.2f} seconds")
            
            # Validate video properties
            if original_width <= 0 or original_height <= 0:
                error_msg = f"Invalid video dimensions: {original_width}x{original_height}"
                self.log(error_msg, "ERROR")
                self.error_signal.emit(error_msg)
                return
            
            if fps <= 0:
                error_msg = f"Invalid FPS: {fps}"
                self.log(error_msg, "ERROR")
                self.error_signal.emit(error_msg)
                return
            
            # Calculate frame range
            if self.use_time_range:
                start_frame = int(self.start_time * fps) if self.start_time > 0 else 0
                end_frame = int(self.end_time * fps) if self.end_time > 0 else total_frames
                self.log(f"Using time range: {self.start_time}s to {self.end_time}s")
            else:
                start_frame = max(0, self.start_frame)
                end_frame = min(total_frames, self.end_frame) if self.end_frame > 0 else total_frames
                self.log(f"Using frame range: {self.start_frame} to {self.end_frame}")
            
            # Ensure valid range
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            
            self.log(f"Final frame range: {start_frame} to {end_frame} ({end_frame - start_frame} frames)")
            
            if end_frame <= start_frame:
                error_msg = f"Invalid frame range: start={start_frame}, end={end_frame}"
                self.log(error_msg, "ERROR")
                self.error_signal.emit(error_msg)
                return
            
            # Calculate output dimensions
            if self.operation in ['upscale', 'both']:
                new_width = int(original_width * self.scale_factor)
                new_height = int(original_height * self.scale_factor)
            else:
                new_width = original_width
                new_height = original_height
            
            # Ensure dimensions are even (required by some codecs)
            if new_width % 2 != 0:
                new_width += 1
            if new_height % 2 != 0:
                new_height += 1
            
            self.log(f"Output dimensions: {new_width}x{new_height}")
            
            # Test frame reading at start position
            self.log(f"Testing frame reading at position {start_frame}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            actual_start = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.log(f"Requested start frame: {start_frame}, actual position: {actual_start}")
            
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                error_msg = f"Cannot read test frame at position {start_frame}"
                self.log(error_msg, "ERROR")
                self.error_signal.emit(error_msg)
                return
            
            self.log(f"Test frame read successfully: shape={test_frame.shape}, dtype={test_frame.dtype}")
            
            # Reset to start position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Set up video writer with format-specific codec testing
            self.log("Setting up video writer...")
            success_writer = False
            
            # Get format-appropriate codecs based on output file extension
            output_format = self.output_format.lower()
            
            if output_format == 'mp4':
                # MP4-specific codecs in order of preference
                fourcc_options = [
                    ('MP4V', cv2.VideoWriter_fourcc(*'mp4v')),      # Standard MP4 codec
                    ('H264', cv2.VideoWriter_fourcc(*'H264')),      # H.264 for MP4
                    ('XVID', cv2.VideoWriter_fourcc(*'XVID')),      # XVID fallback
                    ('Auto', -1)                                     # Let OpenCV choose
                ]
            elif output_format == 'avi':
                # AVI-specific codecs
                fourcc_options = [
                    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),      # MJPG works great with AVI
                    ('XVID', cv2.VideoWriter_fourcc(*'XVID')),      # XVID for AVI
                    ('DIVX', cv2.VideoWriter_fourcc(*'DIVX')),      # DIVX fallback
                    ('Auto', -1)
                ]
            elif output_format == 'mov':
                # MOV-specific codecs
                fourcc_options = [
                    ('MP4V', cv2.VideoWriter_fourcc(*'mp4v')),      # MP4V for MOV
                    ('H264', cv2.VideoWriter_fourcc(*'H264')),      # H.264 for MOV
                    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),      # MJPG fallback
                    ('Auto', -1)
                ]
            else:
                # Generic fallback for other formats
                fourcc_options = [
                    ('XVID', cv2.VideoWriter_fourcc(*'XVID')),      # XVID is widely supported
                    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),      # MJPG fallback
                    ('Auto', -1)
                ]
            
            for codec_name, fourcc in fourcc_options:
                self.log(f"Trying {output_format.upper()} with codec: {codec_name} ({fourcc})")
                
                try:
                    # Create writer with specific parameters for better compatibility
                    out = cv2.VideoWriter(
                        self.output_path, 
                        fourcc, 
                        fps, 
                        (new_width, new_height),
                        True  # isColor parameter - explicitly set to True
                    )
                    
                    if out.isOpened():
                        # Test if we can actually write to it
                        self.log(f"Writer opened successfully, testing write capability...")
                        
                        # Try to write the test frame
                        write_result = out.write(test_frame)
                        
                        # Force flush and check file
                        out.release()
                        
                        # Check if file was created and has content
                        if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0:
                            self.log(f"Write test successful with {codec_name} codec")
                            success_writer = True
                            working_fourcc = fourcc
                            working_codec_name = codec_name
                            break
                        else:
                            self.log(f"Write test failed with {codec_name} - no output file created", "WARNING")
                    else:
                        self.log(f"Failed to open writer with codec: {codec_name}", "WARNING")
                        if out:
                            out.release()
                            
                except Exception as codec_error:
                    self.log(f"Exception testing codec {codec_name}: {codec_error}", "WARNING")
                    if 'out' in locals() and out:
                        out.release()
            
            if not success_writer:
                error_msg = f"Cannot create working {output_format.upper()} file with any compatible codec"
                self.log(error_msg, "ERROR")
                self.error_signal.emit(error_msg)
                return
            
            self.log(f"Successfully configured video writer: {working_codec_name} codec for {output_format.upper()}")
            
            # Reopen the writer for actual processing with the working codec
            out = cv2.VideoWriter(
                self.output_path, 
                working_fourcc, 
                fps, 
                (new_width, new_height),
                True
            )
            
            if not out.isOpened():
                error_msg = "Cannot reopen video writer after successful test"
                self.log(error_msg, "ERROR")
                self.error_signal.emit(error_msg)
                return
            
            # Interpolation methods
            interpolation_methods = {
                'nearest': cv2.INTER_NEAREST,
                'linear': cv2.INTER_LINEAR,
                'bicubic': cv2.INTER_CUBIC,
                'lanczos': cv2.INTER_LANCZOS4
            }
            
            # Process frames
            processed_frames = 0
            failed_frames = 0
            last_log_frame = 0
            
            self.log("Starting frame processing...")
            
            for frame_num in range(start_frame, end_frame):
                if self.stopped:
                    self.log("Processing stopped by user")
                    break
                
                # Handle pause
                while self.paused and not self.stopped:
                    time.sleep(0.1)
                
                ret, frame = cap.read()
                if not ret:
                    failed_frames += 1
                    if frame_num - last_log_frame >= 10:  # Log every 10 frames
                        self.log(f"Failed to read frame {frame_num} (total failed: {failed_frames})", "WARNING")
                        last_log_frame = frame_num
                    if failed_frames > 10:  # Stop if too many failures
                        self.log("Too many failed frame reads, stopping...", "ERROR")
                        break
                    continue
                
                if frame is None or frame.size == 0:
                    failed_frames += 1
                    self.log(f"Empty frame at {frame_num}", "WARNING")
                    continue
                
                try:
                    # Log frame details occasionally
                    if processed_frames % 50 == 0:
                        self.log(f"Processing frame {frame_num}: shape={frame.shape}, dtype={frame.dtype}")
                    
                    # Apply enhancement
                    if self.operation in ['enhance', 'both']:
                        if self.enhancement_method == 'advanced':
                            enhanced_frame = self.enhance_frame_advanced(frame)
                        else:
                            enhanced_frame = self.enhance_frame_traditional(frame, self.enhancement_method)
                    else:
                        enhanced_frame = frame.copy()
                    
                    # Apply scaling
                    if self.operation in ['upscale', 'both'] and self.scale_factor != 1.0:
                        enhanced_frame = cv2.resize(
                            enhanced_frame,
                            (new_width, new_height),
                            interpolation=interpolation_methods[self.upscale_method]
                        )
                    elif enhanced_frame.shape[:2] != (new_height, new_width):
                        # Ensure frame matches output dimensions
                        enhanced_frame = cv2.resize(enhanced_frame, (new_width, new_height))
                    
                    # Validate enhanced frame
                    if enhanced_frame is None or enhanced_frame.size == 0:
                        self.log(f"Enhanced frame is empty at {frame_num}", "WARNING")
                        failed_frames += 1
                        continue
                    
                    if len(enhanced_frame.shape) != 3 or enhanced_frame.shape[2] != 3:
                        self.log(f"Invalid enhanced frame format at {frame_num}: {enhanced_frame.shape}", "WARNING")
                        failed_frames += 1
                        continue
                    
                    # Ensure correct data type
                    if enhanced_frame.dtype != np.uint8:
                        enhanced_frame = enhanced_frame.astype(np.uint8)
                    
                    # Write frame with better error handling
                    try:
                        # Ensure frame is in correct format before writing
                        if enhanced_frame.dtype != np.uint8:
                            enhanced_frame = enhanced_frame.astype(np.uint8)
                        
                        # Ensure frame has correct dimensions
                        if enhanced_frame.shape[:2] != (new_height, new_width):
                            enhanced_frame = cv2.resize(enhanced_frame, (new_width, new_height))
                        
                        # Ensure frame is BGR format (3 channels)
                        if len(enhanced_frame.shape) == 3 and enhanced_frame.shape[2] == 3:
                            # Use different write approach to avoid FFmpeg warnings
                            write_success = out.write(enhanced_frame)
                            
                            # Check if write was successful
                            if write_success is not False:  # Accept None or True as success
                                processed_frames += 1
                                
                                # Verify file is growing every 50 frames
                                if processed_frames % 50 == 0:
                                    try:
                                        current_size = os.path.getsize(self.output_path)
                                        if current_size > 0:
                                            self.log(f"Output file size at frame {processed_frames}: {current_size} bytes")
                                        else:
                                            self.log(f"Warning: Output file size is 0 at frame {processed_frames}", "WARNING")
                                    except Exception as size_error:
                                        self.log(f"Could not check file size: {size_error}", "WARNING")
                            else:
                                self.log(f"Write returned False for frame {frame_num}", "WARNING")
                                failed_frames += 1
                        else:
                            self.log(f"Invalid frame format at {frame_num}: shape={enhanced_frame.shape}", "WARNING")
                            failed_frames += 1
                            
                    except Exception as write_error:
                        self.log(f"Exception writing frame {frame_num}: {write_error}", "WARNING")
                        failed_frames += 1
                        
                        # If we get too many write errors, try to continue but warn user
                        if failed_frames > 50:
                            self.log("Many write failures detected - video file may be corrupted", "ERROR")
                    
                except Exception as frame_error:
                    self.log(f"Error processing frame {frame_num}: {str(frame_error)}", "ERROR")
                    failed_frames += 1
                    continue
                
                # Update progress
                current_progress = frame_num - start_frame + 1
                total_processing_frames = end_frame - start_frame
                percentage = int((current_progress / total_processing_frames) * 100)
                self.progress_update.emit(current_progress, total_processing_frames, percentage)
                
                # Log progress every 100 frames
                if processed_frames % 100 == 0 and processed_frames > 0:
                    self.log(f"Progress: {processed_frames} frames processed, {failed_frames} failed ({percentage}%)")
            
            # Clean up
            cap.release()
            out.release()
            
            self.log(f"Video processing completed: {processed_frames} frames processed, {failed_frames} failed")
            
            if self.stopped:
                self.log("Processing was stopped by user")
                try:
                    os.remove(self.output_path)
                    self.log("Removed incomplete output file")
                except Exception as e:
                    self.log(f"Could not remove incomplete file: {e}", "WARNING")
            elif processed_frames > 0:
                # Handle audio preservation
                audio_success = True
                if temp_video_path and self.preserve_audio and ffmpeg_available:
                    self.log("Adding audio from original video...")
                    audio_success = self.add_audio_to_video(temp_video_path, self.input_path, final_output_path)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_video_path)
                        self.log("Removed temporary video file")
                    except Exception as e:
                        self.log(f"Could not remove temp file: {e}", "WARNING")
                
                try:
                    output_file = final_output_path if temp_video_path else self.output_path
                    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                    self.log(f"Output file created successfully: {file_size:.1f} MB")
                    
                    audio_status = ""
                    if self.preserve_audio:
                        audio_status = " (with audio)" if audio_success else " (audio failed)"
                    
                    success_msg = f"Video processing completed!\nProcessed {processed_frames} frames\nFailed: {failed_frames} frames\nOutput file: {file_size:.1f} MB{audio_status}"
                    self.finished_signal.emit(success_msg)
                except Exception as e:
                    self.log(f"Error checking output file: {e}", "ERROR")
                    self.error_signal.emit(f"Processing completed but output file may be corrupted: {e}")
            else:
                error_msg = f"No frames were processed successfully.\nFailed frames: {failed_frames}\nCheck the log file for details."
                self.log(error_msg, "ERROR")
                self.error_signal.emit(error_msg)
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log(f"Critical error during processing: {str(e)}", "ERROR")
            self.log(f"Stack trace: {error_details}", "ERROR")
            self.error_signal.emit(f"Critical error: {str(e)}\n\nCheck log file for details.")

class VideoEnhancerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = None
        self.video_info = {}
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Advanced Video Enhancer")
        self.setGeometry(100, 100, 1200, 650)  # Wider but shorter
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # HORIZONTAL layout
        
        # LEFT PANEL - Controls (limited width)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(700)
        
        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout(file_group)
        
        file_layout.addWidget(QLabel("Input:"), 0, 0)
        self.input_edit = QLineEdit()
        file_layout.addWidget(self.input_edit, 0, 1)
        self.browse_input_btn = QPushButton("Browse")
        self.browse_input_btn.clicked.connect(self.browse_input_file)
        file_layout.addWidget(self.browse_input_btn, 0, 2)
        
        file_layout.addWidget(QLabel("Output:"), 1, 0)
        self.output_edit = QLineEdit()
        file_layout.addWidget(self.output_edit, 1, 1)
        self.browse_output_btn = QPushButton("Browse")
        self.browse_output_btn.clicked.connect(self.browse_output_file)
        file_layout.addWidget(self.browse_output_btn, 1, 2)
        
        self.info_label = QLabel("Select a video file to see information")
        self.info_label.setStyleSheet("color: #666; font-style: italic;")
        file_layout.addWidget(self.info_label, 2, 0, 1, 3)
        
        left_layout.addWidget(file_group)
        
        # Format and Operation in one row
        format_operation_layout = QHBoxLayout()
        
        # Output format (compact)
        format_group = QGroupBox("Format")
        format_layout = QHBoxLayout(format_group)
        self.format_group = QButtonGroup()
        formats = [("MP4", "mp4"), ("AVI", "avi"), ("MOV", "mov"), ("MKV", "mkv")]
        
        for i, (name, ext) in enumerate(formats):
            radio = QRadioButton(name)
            radio.setProperty("format", ext)
            radio.toggled.connect(self.on_format_changed)
            self.format_group.addButton(radio, i)
            format_layout.addWidget(radio)
            if ext == "mp4":
                radio.setChecked(True)
        
        format_operation_layout.addWidget(format_group)
        
        # Operation
        operation_group = QGroupBox("Operation")
        operation_layout = QVBoxLayout(operation_group)
        self.operation_combo = QComboBox()
        self.operation_combo.addItems(["enhance", "upscale", "both", "convert"])
        self.operation_combo.currentTextChanged.connect(self.on_operation_changed)
        operation_layout.addWidget(self.operation_combo)
        format_operation_layout.addWidget(operation_group)
        
        left_layout.addLayout(format_operation_layout)
        
        # Enhancement and Scaling settings
        settings_layout = QHBoxLayout()
        
        # Enhancement
        enhance_group = QGroupBox("Enhancement")
        enhance_layout = QVBoxLayout(enhance_group)
        self.enhancement_combo = QComboBox()
        self.enhancement_combo.addItems(["combined", "unsharp_mask", "clahe", "bilateral", "advanced"])
        self.enhancement_combo.currentTextChanged.connect(self.update_explanation)
        enhance_layout.addWidget(self.enhancement_combo)
        settings_layout.addWidget(enhance_group)
        
        # Scaling
        scale_group = QGroupBox("Scaling")
        scale_layout = QGridLayout(scale_group)
        scale_layout.addWidget(QLabel("Factor:"), 0, 0)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 10.0)
        self.scale_spin.setValue(2.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.valueChanged.connect(self.update_explanation)
        scale_layout.addWidget(self.scale_spin, 0, 1)
        
        scale_layout.addWidget(QLabel("Method:"), 1, 0)
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["bicubic", "lanczos", "linear", "nearest"])
        self.upscale_combo.currentTextChanged.connect(self.update_explanation)
        scale_layout.addWidget(self.upscale_combo, 1, 1)
        settings_layout.addWidget(scale_group)
        
        left_layout.addLayout(settings_layout)
        
        # Range and Audio options
        range_audio_layout = QHBoxLayout()
        
        # Range (compact)
        range_group = QGroupBox("Range")
        range_layout = QGridLayout(range_group)
        
        self.range_button_group = QButtonGroup()
        self.frame_radio = QRadioButton("Frames")
        self.time_radio = QRadioButton("Time(s)")
        self.frame_radio.setChecked(True)
        self.frame_radio.toggled.connect(self.on_range_type_changed)
        self.time_radio.toggled.connect(self.on_range_type_changed)
        self.range_button_group.addButton(self.frame_radio)
        self.range_button_group.addButton(self.time_radio)
        
        range_layout.addWidget(self.frame_radio, 0, 0)
        range_layout.addWidget(self.time_radio, 0, 1)
        
        range_layout.addWidget(QLabel("Start:"), 1, 0)
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 999999)
        self.start_frame_spin.valueChanged.connect(self.update_time_from_frame)
        range_layout.addWidget(self.start_frame_spin, 1, 1)
        
        range_layout.addWidget(QLabel("End:"), 2, 0)
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(-1, 999999)
        self.end_frame_spin.setValue(-1)
        self.end_frame_spin.valueChanged.connect(self.update_time_from_frame)
        range_layout.addWidget(self.end_frame_spin, 2, 1)
        
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0.0, 999999.0)
        self.start_time_spin.setDecimals(1)
        self.start_time_spin.valueChanged.connect(self.update_frame_from_time)
        range_layout.addWidget(self.start_time_spin, 3, 1)
        
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(-1.0, 999999.0)
        self.end_time_spin.setValue(-1.0)
        self.end_time_spin.setDecimals(1)
        self.end_time_spin.valueChanged.connect(self.update_frame_from_time)
        range_layout.addWidget(self.end_time_spin, 4, 1)
        
        range_audio_layout.addWidget(range_group)
        
        # Audio options
        audio_group = QGroupBox("Audio")
        audio_layout = QVBoxLayout(audio_group)
        self.preserve_audio_check = QCheckBox("Preserve Audio")
        self.preserve_audio_check.setToolTip("Requires FFmpeg")
        audio_layout.addWidget(self.preserve_audio_check)
        
        self.ffmpeg_status_label = QLabel("FFmpeg: Checking...")
        self.ffmpeg_status_label.setStyleSheet("color: #666; font-size: 10px;")
        audio_layout.addWidget(self.ffmpeg_status_label)
        range_audio_layout.addWidget(audio_group)
        
        left_layout.addLayout(range_audio_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_processing)
        self.pause_btn.setEnabled(False)
        
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(self.resume_processing)
        self.resume_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.resume_btn)
        control_layout.addWidget(self.stop_btn)
        left_layout.addLayout(control_layout)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready to process")
        progress_layout.addWidget(self.progress_label)
        left_layout.addWidget(progress_group)
        
        # RIGHT PANEL - Information
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setMinimumWidth(450)
        
        # Explanation
        explanation_group = QGroupBox("Option Explanation")
        explanation_layout = QVBoxLayout(explanation_group)
        self.explanation_text = QTextEdit()
        self.explanation_text.setMaximumHeight(180)
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
                color: #212529;
            }
        """)
        explanation_layout.addWidget(self.explanation_text)
        right_layout.addWidget(explanation_group)
        
        # Log
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        log_controls = QHBoxLayout()
        self.enable_logging_check = QCheckBox("Logging")
        self.enable_logging_check.setChecked(True)
        log_controls.addWidget(self.enable_logging_check)
        
        self.save_log_btn = QPushButton("Save")
        self.save_log_btn.clicked.connect(self.save_log_file)
        log_controls.addWidget(self.save_log_btn)
        
        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_controls.addWidget(self.clear_log_btn)
        log_controls.addStretch()
        log_layout.addLayout(log_controls)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        log_layout.addWidget(self.log_display)
        right_layout.addWidget(log_group)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        main_layout.setStretch(0, 3)  # Left panel takes more space
        main_layout.setStretch(1, 2)  # Right panel smaller
        
        # Store references for enabling/disabling
        self.enhancement_label = QLabel()  # Dummy for compatibility
        self.scale_label = QLabel()        # Dummy for compatibility
        self.upscale_label = QLabel()      # Dummy for compatibility
        
        # Initialize
        self.on_operation_changed()
        self.on_range_type_changed()
        self.update_explanation()
        self.check_ffmpeg_status()
    
    def check_ffmpeg_status(self):
        """Check if FFmpeg is available and update status"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.ffmpeg_status_label.setText("FFmpeg: Available ✓")
                self.ffmpeg_status_label.setStyleSheet("color: #28a745; font-size: 10px;")
                self.preserve_audio_check.setEnabled(True)
            else:
                self.ffmpeg_status_label.setText("FFmpeg: Not found")
                self.ffmpeg_status_label.setStyleSheet("color: #dc3545; font-size: 10px;")
                self.preserve_audio_check.setEnabled(False)
        except:
            self.ffmpeg_status_label.setText("FFmpeg: Not found")
            self.ffmpeg_status_label.setStyleSheet("color: #dc3545; font-size: 10px;")
            self.preserve_audio_check.setEnabled(False)
    
    def clear_log(self):
        """Clear the log display"""
        self.log_display.clear()
    
    def save_log_file(self):
        """Save the current log to a file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Log File", 
            f"video_enhancer_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_display.toPlainText())
                QMessageBox.information(self, "Success", f"Log saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save log file: {e}")
    
    def add_log_entry(self, message):
        """Add entry to log display"""
        self.log_display.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Input Video", 
            "", 
            "Video Files (*.avi *.mp4 *.mov *.mkv *.wmv *.flv *.m4v);;All Files (*)"
        )
        if file_path:
            self.input_edit.setText(file_path)
            self.load_video_info(file_path)
            self.update_output_filename()
    
    def browse_output_file(self):
        selected_format = self.get_selected_format()
        filter_text = f"{selected_format.upper()} Files (*.{selected_format});;All Files (*)"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Enhanced Video", 
            "", 
            filter_text
        )
        if file_path:
            self.output_edit.setText(file_path)
    
    def get_selected_format(self):
        checked_button = self.format_group.checkedButton()
        if checked_button:
            return checked_button.property("format")
        return "mp4"
    
    def on_format_changed(self):
        self.update_output_filename()
    
    def update_output_filename(self):
        if self.input_edit.text():
            input_path = Path(self.input_edit.text())
            selected_format = self.get_selected_format()
            operation = self.operation_combo.currentText()
            output_path = input_path.parent / f"{input_path.stem}_{operation}.{selected_format}"
            self.output_edit.setText(str(output_path))
    
    def load_video_info(self, file_path):
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                self.video_info = {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': duration
                }
                
                info_text = f"Resolution: {width}x{height} | FPS: {fps:.1f} | Duration: {duration:.1f}s | Frames: {frame_count}"
                self.info_label.setText(info_text)
                self.info_label.setStyleSheet("color: #333; font-weight: bold;")
                
                # Update range controls
                self.end_frame_spin.setMaximum(frame_count)
                self.end_time_spin.setMaximum(duration)
                
                cap.release()
            else:
                self.info_label.setText("Cannot read video file")
                self.info_label.setStyleSheet("color: red;")
        except Exception as e:
            self.info_label.setText(f"Error loading video: {str(e)}")
            self.info_label.setStyleSheet("color: red;")
    
    def on_operation_changed(self):
        operation = self.operation_combo.currentText()
        
        # Enable/disable enhancement controls
        enhance_enabled = operation in ['enhance', 'both']
        self.enhancement_combo.setEnabled(enhance_enabled)
        
        # Enable/disable upscale controls
        upscale_enabled = operation in ['upscale', 'both']
        self.scale_spin.setEnabled(upscale_enabled)
        self.upscale_combo.setEnabled(upscale_enabled)
        
        # Enable/disable audio preservation for convert operation
        audio_enabled = operation != "convert"  # Disable for convert since FFmpeg handles it
        self.preserve_audio_check.setEnabled(audio_enabled and self.ffmpeg_status_label.text().endswith("✓"))
        
        self.update_output_filename()
        self.update_explanation()
    
    def on_range_type_changed(self):
        use_time = self.time_radio.isChecked()
        
        # Enable appropriate controls
        self.start_frame_spin.setEnabled(not use_time)
        self.end_frame_spin.setEnabled(not use_time)
        self.start_time_spin.setEnabled(use_time)
        self.end_time_spin.setEnabled(use_time)
    
    def update_time_from_frame(self):
        if self.frame_radio.isChecked() and 'fps' in self.video_info:
            fps = self.video_info['fps']
            if fps > 0:
                start_time = self.start_frame_spin.value() / fps
                end_time = self.end_frame_spin.value() / fps if self.end_frame_spin.value() > 0 else -1
                
                self.start_time_spin.blockSignals(True)
                self.end_time_spin.blockSignals(True)
                self.start_time_spin.setValue(start_time)
                if end_time > 0:
                    self.end_time_spin.setValue(end_time)
                self.start_time_spin.blockSignals(False)
                self.end_time_spin.blockSignals(False)
    
    def update_frame_from_time(self):
        if self.time_radio.isChecked() and 'fps' in self.video_info:
            fps = self.video_info['fps']
            if fps > 0:
                start_frame = int(self.start_time_spin.value() * fps)
                end_frame = int(self.end_time_spin.value() * fps) if self.end_time_spin.value() > 0 else -1
                
                self.start_frame_spin.blockSignals(True)
                self.end_frame_spin.blockSignals(True)
                self.start_frame_spin.setValue(start_frame)
                if end_frame > 0:
                    self.end_frame_spin.setValue(end_frame)
                self.start_frame_spin.blockSignals(False)
                self.end_frame_spin.blockSignals(False)
    
    def update_explanation(self):
        explanations = {
            'operations': {
                'enhance': '<b style="color: #28a745;">✨ Enhance Only:</b> Improves image quality without changing resolution. Faster processing, same file size. Good for sharpening, noise reduction, and contrast improvement.',
                'upscale': '<b style="color: #007bff;">📏 Upscale Only:</b> Increases resolution using interpolation. Creates larger, potentially clearer images but may introduce artifacts if source quality is poor.',
                'both': '<b style="color: #dc3545;">🚀 Enhance + Upscale:</b> Best quality results. Enhances image quality first, then upscales. Slower processing but maximum improvement. Recommended for most videos.',
                'convert': '<b style="color: #6f42c1;">🔄 Format Convert:</b> Changes video format only using FFmpeg. Fastest option, preserves original quality and audio. Perfect for format compatibility.'
            },
            'enhancement_methods': {
                'unsharp_mask': '<b>🔍 Unsharp Mask:</b> Fast sharpening technique. <span style="color: #28a745;">Pros:</span> Quick, good for slightly blurry videos. <span style="color: #dc3545;">Cons:</span> Can enhance noise, may create halos around edges.',
                'clahe': '<b>🌟 CLAHE:</b> Contrast enhancement. <span style="color: #28a745;">Pros:</span> Brings out details in dark/bright areas, natural look. <span style="color: #dc3545;">Cons:</span> Slower processing, may not sharpen much.',
                'bilateral': '<b>🎯 Bilateral Filter:</b> Noise reduction while preserving edges. <span style="color: #28a745;">Pros:</span> Smooth results, reduces grain. <span style="color: #dc3545;">Cons:</span> Slow processing, may soften some details.',
                'combined': '<b>🏆 Combined (Recommended):</b> Noise reduction + sharpening + contrast. <span style="color: #28a745;">Pros:</span> Balanced results, handles most videos well. <span style="color: #dc3545;">Cons:</span> Slower than single methods.',
                'advanced': '<b>🤖 Advanced AI-like:</b> Multiple AI techniques. <span style="color: #28a745;">Pros:</span> Best quality, professional results. <span style="color: #dc3545;">Cons:</span> Slowest processing, highest resource usage.'
            },
            'upscale_methods': {
                'nearest': '<b>⚡ Nearest:</b> Fastest upscaling. <span style="color: #28a745;">Pros:</span> Very fast. <span style="color: #dc3545;">Cons:</span> Pixelated results, poor quality.',
                'linear': '<b>📈 Linear:</b> Basic smooth upscaling. <span style="color: #28a745;">Pros:</span> Fast, smoother than nearest. <span style="color: #dc3545;">Cons:</span> Still somewhat blurry.',
                'bicubic': '<b>🎨 Bicubic (Recommended):</b> Good quality balance. <span style="color: #28a745;">Pros:</span> Good detail preservation, reasonable speed. <span style="color: #dc3545;">Cons:</span> May introduce slight artifacts.',
                'lanczos': '<b>💎 Lanczos:</b> Highest quality upscaling. <span style="color: #28a745;">Pros:</span> Best detail preservation, sharp results. <span style="color: #dc3545;">Cons:</span> Slowest processing, may enhance noise.'
            }
        }
        
        operation = self.operation_combo.currentText()
        enhancement = self.enhancement_combo.currentText()
        upscale_method = self.upscale_combo.currentText()
        scale_factor = self.scale_spin.value()
        
        explanation_parts = [explanations['operations'][operation]]
        
        if operation in ['enhance', 'both']:
            explanation_parts.append(explanations['enhancement_methods'][enhancement])
        
        if operation in ['upscale', 'both']:
            explanation_parts.append(explanations['upscale_methods'][upscale_method])
            scale_text = f'<b>Scale Factor:</b> {scale_factor}x - {"No size change" if scale_factor == 1.0 else f"Output will be {scale_factor}x larger"}'
            explanation_parts.append(scale_text)
        
        if operation == 'convert':
            explanation_parts.append('<b>📝 Note:</b> Format conversion preserves original video and audio quality. Uses FFmpeg for best compatibility and speed.')
        
        explanation = '<div style="line-height: 1.5;">' + '<br><br>'.join(explanation_parts) + '</div>'
        self.explanation_text.setHtml(explanation)
    
    def start_processing(self):
        if not self.input_edit.text() or not self.output_edit.text():
            QMessageBox.warning(self, "Warning", "Please select input and output files")
            return
        
        if not os.path.exists(self.input_edit.text()):
            QMessageBox.warning(self, "Warning", "Input file does not exist")
            return
        
        # Setup logging if enabled
        log_file_path = None
        if self.enable_logging_check.isChecked():
            # Create log file in same directory as input file
            input_path = Path(self.input_edit.text())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = input_path.parent / f"video_enhancer_log_{timestamp}.txt"
            self.add_log_entry(f"Logging enabled: {log_file_path}")
        
        # Prepare parameters
        params = {
            'input_path': self.input_edit.text(),
            'output_path': self.output_edit.text(),
            'operation': self.operation_combo.currentText(),
            'enhancement_method': self.enhancement_combo.currentText(),
            'scale_factor': self.scale_spin.value(),
            'upscale_method': self.upscale_combo.currentText(),
            'use_time_range': self.time_radio.isChecked(),
            'start_frame': self.start_frame_spin.value(),
            'end_frame': self.end_frame_spin.value(),
            'start_time': self.start_time_spin.value(),
            'end_time': self.end_time_spin.value(),
            'output_format': self.get_selected_format(),
            'preserve_audio': self.preserve_audio_check.isChecked()
        }
        
        # Create and start processor thread
        self.processor = VideoProcessor()
        
        # Setup logging
        if log_file_path:
            self.processor.setup_logging(str(log_file_path))
        
        self.processor.set_parameters(params)
        self.processor.progress_update.connect(self.update_progress)
        self.processor.status_update.connect(self.update_status)
        self.processor.finished_signal.connect(self.processing_finished)
        self.processor.error_signal.connect(self.processing_error)
        self.processor.log_signal.connect(self.add_log_entry)
        
        self.processor.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
    
    def pause_processing(self):
        if self.processor:
            self.processor.pause()
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)
    
    def resume_processing(self):
        if self.processor:
            self.processor.resume()
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)
    
    def stop_processing(self):
        if self.processor:
            self.processor.stop()
            self.reset_ui()
    
    def update_progress(self, current_frame, total_frames, percentage):
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"Processing frame {current_frame}/{total_frames} ({percentage}%)")
    
    def update_status(self, status):
        self.progress_label.setText(status)
    
    def processing_finished(self, message):
        QMessageBox.information(self, "Success", message)
        self.reset_ui()
    
    def processing_error(self, error):
        QMessageBox.critical(self, "Error", error)
        self.reset_ui()
    
    def reset_ui(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready to process")
        if self.processor:
            self.processor.quit()
            self.processor.wait()
            self.processor = None

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = VideoEnhancerGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()