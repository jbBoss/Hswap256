import os
import sys
from flask import Flask, render_template, request, send_file, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
import uuid
import warnings
import queue
import threading
import zipfile
from io import BytesIO
import subprocess

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub.file_download')
os.environ['OMP_NUM_THREADS'] = '1'

# Add project root to path to import facefusion modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy
import onnxruntime
from facefusion import state_manager, face_detector, face_landmarker, ffmpeg
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.face_analyser import get_one_face, get_many_faces
from facefusion.face_helper import warp_face_by_face_landmark_5, paste_back
from facefusion.face_masker import create_box_mask
from facefusion.filesystem import resolve_relative_path
from facefusion.vision import read_image, write_image
from face_restoration import FaceRestorer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size for videos
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm', 'mkv'}

# Global variables for model caching
face_swapper_model = None
face_restorer = None
models_initialized = False

# Progress tracking
progress_queues = {}  # Dictionary to store progress queues for each task


def get_execution_providers():
    """Get available execution providers, GPU ONLY - no CPU fallback."""
    available_providers = onnxruntime.get_available_providers()
    print(f"Available ONNX providers: {available_providers}")
    
    # Force GPU only - CUDA or DirectML, NO CPU fallback
    if 'CUDAExecutionProvider' in available_providers:
        print("🚀 Using CUDA (GPU) for acceleration!")
        return ['CUDAExecutionProvider']
    elif 'DmlExecutionProvider' in available_providers:
        print("🚀 Using DirectML (GPU) for acceleration!")
        return ['DmlExecutionProvider']
    else:
        error_msg = "❌ ERROR: No GPU detected! This application requires GPU (CUDA or DirectML)."
        print(error_msg)
        raise RuntimeError(error_msg)


def is_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def is_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def initialize_models():
    """Initialize face swapper model and face analysis models (once)."""
    global face_swapper_model, face_restorer, models_initialized
    
    if models_initialized:
        return True
    
    try:
        # Get optimal execution providers
        execution_providers = get_execution_providers()
        
        # Download and load face swapper model
        model_path = resolve_relative_path('../.assets/models/hyperswap_1a_256.onnx')
        model_url = 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx'
        hash_url = 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.hash'
        hash_path = resolve_relative_path('../.assets/models/hyperswap_1a_256.hash')

        model_set = {'face_swapper': {'url': model_url, 'path': model_path}}
        hash_set = {'face_swapper': {'url': hash_url, 'path': hash_path}}

        print("Checking for model files...")
        if not conditional_download_hashes(hash_set) or not conditional_download_sources(model_set):
            print("Model download failed.")
            return False

        print("Loading face swapper model...")
        face_swapper_model = onnxruntime.InferenceSession(model_path, providers=execution_providers)
        print(f"Face swapper model loaded with: {face_swapper_model.get_providers()}")

        # Initialize state for face analysis
        print("Initializing face analyser...")
        state_manager.init_item('face_detector_model', 'retinaface')
        state_manager.init_item('face_detector_size', '640x640')
        state_manager.init_item('face_detector_score', 0.5)
        state_manager.init_item('face_detector_angles', [0])
        state_manager.init_item('face_landmarker_model', '2dfan4')
        state_manager.init_item('face_landmarker_score', 0.5)
        state_manager.init_item('execution_providers', execution_providers)
        state_manager.init_item('execution_device_ids', ['0'])
        state_manager.init_item('download_providers', ['github'])
        state_manager.init_item('face_recognizer_model', 'arcface_inswapper')

        # Pre-check and load face detector models
        print("Loading face detector models...")
        if not face_detector.pre_check():
            print("Failed to load face detector models.")
            return False

        # Pre-check and load face landmarker models
        print("Loading face landmarker models...")
        if not face_landmarker.pre_check():
            print("Failed to load face landmarker models.")
            return False

        # Initialize face restoration model (CodeFormer)
        print("Loading face restoration model (CodeFormer)...")
        face_restorer = FaceRestorer(model_name='codeformer', execution_providers=execution_providers)
        print("Face restoration model loaded successfully!")

        models_initialized = True
        print("All models initialized successfully!")
        return True
    
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False


def extract_source_face(source_path):
    """
    Extract and analyze source face once, return embedding.
    Returns (source_embedding, error_message) - embedding is None if failed.
    """
    try:
        source_frame = read_image(source_path)
        if source_frame is None:
            return None, "Could not read source image."

        source_face = get_one_face(get_many_faces([source_frame]))
        if not source_face:
            return None, "Could not detect a face in the source image."
        
        source_height, source_width, _ = source_frame.shape
        face_width = int(source_face.bounding_box[2] - source_face.bounding_box[0])
        face_height = int(source_face.bounding_box[3] - source_face.bounding_box[1])
        print(f"✓ Source image resolution: {source_width}x{source_height}")
        print(f"✓ Detected source face resolution: {face_width}x{face_height}")

        source_embedding = source_face.embedding_norm.reshape((1, -1))
        return source_embedding, None
    except Exception as e:
        return None, f"Error extracting source face: {str(e)}"


def process_image(source_embedding, target_path, output_path, enhance_face=True, enhancement_weight=0.5, enhancement_blend=80):
    """
    Performs face swap from source embedding to target image with optional face enhancement.
    Returns (success, message).
    
    Args:
        source_embedding: Pre-extracted source face embedding (numpy array)
        target_path: Path to target image
        output_path: Path to save output
        enhance_face: Whether to enhance faces after swapping
        enhancement_weight: Enhancement strength (0.0-1.0)
        enhancement_blend: Blend amount (0-100)
    """
    try:
        model_template = 'arcface_128'
        model_size = (256, 256)

        # Read target image
        target_frame = read_image(target_path)
        if target_frame is None:
            return False, "Could not read target image."

        target_face = get_one_face(get_many_faces([target_frame]))
        if not target_face:
            return False, "Could not detect a face in the target image."
        
        target_height, target_width, _ = target_frame.shape
        print(f"Target image resolution: {target_width}x{target_height}")

        # Warp and crop target face
        crop_frame, affine_matrix = warp_face_by_face_landmark_5(
            target_frame, 
            target_face.landmark_set.get('5/68'), 
            model_template, 
            model_size
        )

        # Prepare crop frame for inference
        crop_frame_for_onnx = crop_frame[:, :, ::-1] / 255.0
        crop_frame_for_onnx = (crop_frame_for_onnx - 0.5) / 0.5  # Normalize to [-1,1]
        crop_frame_for_onnx = crop_frame_for_onnx.transpose(2, 0, 1)
        crop_frame_for_onnx = numpy.expand_dims(crop_frame_for_onnx, axis=0).astype(numpy.float32)

        # Run inference
        swapped_face = face_swapper_model.run(None, {
            'source': source_embedding,
            'target': crop_frame_for_onnx
        })[0][0]

        # Post-process swapped face
        swapped_face = swapped_face.transpose(1, 2, 0)
        swapped_face = (swapped_face * 0.5 + 0.5) * 255  # Denormalize to [0,255]
        swapped_face = swapped_face[:, :, ::-1].clip(0, 255).astype(numpy.uint8)

        # Create mask and paste back
        box_mask = create_box_mask(swapped_face, face_mask_blur=0.1, face_mask_padding=(0, 0, 0, 0))
        result_frame = paste_back(target_frame, swapped_face, box_mask, affine_matrix)

        # Apply face enhancement/restoration if enabled
        if enhance_face and face_restorer:
            print("Applying face restoration...")
            result_frame = face_restorer.enhance_frame(result_frame, weight=enhancement_weight, blend=enhancement_blend)

        # Save output
        write_image(output_path, result_frame)
        
        message = "Face swap completed successfully!"
        if enhance_face:
            message += " (with face restoration)"
        return True, message

    except Exception as e:
        return False, f"Error during face swap: {str(e)}"


def process_video(source_embedding, target_path, output_path, task_id=None, enhance_face=True, enhancement_weight=0.5, enhancement_blend=80):
    """
    Performs face swap on a video target using OpenCV with optional face enhancement.
    Returns (success, message).
    
    Args:
        source_embedding: Pre-extracted source face embedding (numpy array)
        target_path: Path to target video
        output_path: Path to save output video
        task_id: Task ID for progress tracking
        enhance_face: Whether to enhance faces after swapping
        enhancement_weight: Enhancement strength (0.0-1.0)
        enhancement_blend: Blend amount (0-100)
    """
    try:
        model_template = 'arcface_128'
        model_size = (256, 256)

        # Get progress queue if available
        progress_queue = progress_queues.get(task_id) if task_id else None

        def update_progress(current, total, message):
            """Helper to update progress"""
            if progress_queue:
                try:
                    progress_queue.put({
                        'current': current,
                        'total': total,
                        'message': message,
                        'percentage': int((current / total) * 100) if total > 0 else 0
                    })
                except:
                    pass

        # Open video
        update_progress(0, 100, "Opening video...")
        print("Opening video...")
        video_capture = cv2.VideoCapture(target_path)
        if not video_capture.isOpened():
            return False, "Could not open video file."

        # Get video properties
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        enhancement_msg = " with face restoration" if enhance_face else ""
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames{enhancement_msg}")
        update_progress(10, 100, f"Processing {total_frames} frames{enhancement_msg}...")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        processed_count = 0

        print("Processing video frames...")
        while True:
            ret, target_frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            
            # Update progress
            progress_percentage = 10 + int((frame_count / total_frames) * 85)  # 10-95%
            update_progress(progress_percentage, 100, f"Processing frame {frame_count}/{total_frames}{enhancement_msg}...")
            
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames}...")

            try:
                # Detect face in current frame
                target_face = get_one_face(get_many_faces([target_frame]))

                if target_face:
                    # Warp and crop target face
                    crop_frame, affine_matrix = warp_face_by_face_landmark_5(
                        target_frame,
                        target_face.landmark_set.get('5/68'),
                        model_template,
                        model_size
                    )

                    # Prepare crop frame for inference
                    crop_frame_for_onnx = crop_frame[:, :, ::-1] / 255.0
                    crop_frame_for_onnx = (crop_frame_for_onnx - 0.5) / 0.5
                    crop_frame_for_onnx = crop_frame_for_onnx.transpose(2, 0, 1)
                    crop_frame_for_onnx = numpy.expand_dims(crop_frame_for_onnx, axis=0).astype(numpy.float32)

                    # Run inference
                    swapped_face = face_swapper_model.run(None, {
                        'source': source_embedding,
                        'target': crop_frame_for_onnx
                    })[0][0]

                    # Post-process swapped face
                    swapped_face = swapped_face.transpose(1, 2, 0)
                    swapped_face = (swapped_face * 0.5 + 0.5) * 255
                    swapped_face = swapped_face[:, :, ::-1].clip(0, 255).astype(numpy.uint8)

                    # Create mask and paste back
                    box_mask = create_box_mask(swapped_face, face_mask_blur=0.1, face_mask_padding=(0, 0, 0, 0))
                    target_frame = paste_back(target_frame, swapped_face, box_mask, affine_matrix)
                    
                    # Apply face enhancement/restoration if enabled
                    if enhance_face and face_restorer:
                        target_frame = face_restorer.enhance_frame(target_frame, weight=enhancement_weight, blend=enhancement_blend)
                    
                    processed_count += 1

            except Exception as e:
                # If face swap fails for a frame, just use the original frame
                print(f"Warning: Frame {frame_count} face swap failed: {e}")

            # Write frame to output video
            video_writer.write(target_frame)

        video_capture.release()
        video_writer.release()

        update_progress(95, 100, "Merging audio from original video...")
        print("Merging audio from original video...")
        
        # Use ffmpeg to merge audio from original video with processed video
        temp_output_path = output_path.replace('.mp4', '_no_audio.mp4')
        os.rename(output_path, temp_output_path)
        
        try:
            # Extract audio from original and merge with processed video
            import subprocess
            ffmpeg_command = [
                'ffmpeg', '-y',
                '-i', temp_output_path,  # Processed video (no audio)
                '-i', target_path,  # Original video (with audio)
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',  # Encode audio to AAC
                '-map', '0:v:0',  # Video from first input
                '-map', '1:a:0?',  # Audio from second input (? makes it optional)
                '-shortest',  # Match shortest stream duration
                output_path
            ]
            
            result = subprocess.run(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            if result.returncode == 0:
                print("✓ Audio merged successfully!")
            else:
                print("⚠ Warning: Could not merge audio, video saved without audio")
                # If merge fails, keep the no-audio version
                if not os.path.exists(output_path) and os.path.exists(temp_output_path):
                    os.rename(temp_output_path, output_path)
        
        except Exception as audio_error:
            print(f"⚠ Warning: Audio merge failed: {audio_error}")
            # If merge fails, keep the no-audio version
            if not os.path.exists(output_path) and os.path.exists(temp_output_path):
                os.rename(temp_output_path, output_path)

        update_progress(100, 100, f"Complete! Processed {processed_count}/{frame_count} frames with faces{enhancement_msg}.")
        print(f"Video processing complete! Processed {processed_count}/{frame_count} frames with faces{enhancement_msg}.")
        message = f"Video face swap completed! Processed {processed_count} frames with faces."
        if enhance_face:
            message += " (with face restoration)"
        return True, message

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error during video face swap: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/swap', methods=['POST'])
def swap():
    # Check if files are present
    if 'source' not in request.files:
        return jsonify({'success': False, 'message': 'Source image is required.'}), 400
    
    source_file = request.files['source']
    
    # Get all target files (support multiple targets)
    target_files = request.files.getlist('target')
    
    # Check if files are selected
    if source_file.filename == '':
        return jsonify({'success': False, 'message': 'Please select source image.'}), 400
    
    if not target_files or all(f.filename == '' for f in target_files):
        return jsonify({'success': False, 'message': 'Please select at least one target file.'}), 400
    
    # Get enhancement parameters (optional)
    enhance_face = request.form.get('enhance_face', 'true').lower() == 'true'
    enhancement_weight = float(request.form.get('enhancement_weight', '0.5'))
    enhancement_blend = int(request.form.get('enhancement_blend', '80'))
    
    # Validate enhancement parameters
    enhancement_weight = max(0.0, min(1.0, enhancement_weight))
    enhancement_blend = max(0, min(100, enhancement_blend))
    
    # Validate source file type
    if not is_image_file(source_file.filename):
        return jsonify({'success': False, 'message': 'Source must be an image.'}), 400
    
    # Models should already be initialized at startup
    if not models_initialized:
        return jsonify({'success': False, 'message': 'Models not initialized. Please restart the server.'}), 500
    
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Save source file
        source_filename = secure_filename(f"{task_id}_source_{source_file.filename}")
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
        source_file.save(source_path)
        
        # Extract source face ONCE (optimization for batch processing)
        print(f"\n{'='*60}")
        print(f"🎯 Processing source face (one-time operation)...")
        print(f"{'='*60}")
        source_embedding, error_msg = extract_source_face(source_path)
        
        if source_embedding is None:
            # Clean up source file
            try:
                os.remove(source_path)
            except:
                pass
            return jsonify({'success': False, 'message': error_msg}), 400
        
        print(f"✅ Source face extracted successfully! Ready for batch processing.\n")
        
        # Process all target files
        results = []
        output_filenames = []
        
        # Create enhancement suffix for filename
        if enhance_face:
            weight_str = str(int(enhancement_weight * 10)).zfill(2)  # 0.5 -> "05", 1.0 -> "10"
            blend_str = str(enhancement_blend).zfill(2)  # 85 -> "85", 5 -> "05"
            enhancement_suffix = f"E{weight_str}P{blend_str}"
        else:
            enhancement_suffix = "E00P00"  # No enhancement
        
        for idx, target_file in enumerate(target_files):
            if target_file.filename == '':
                continue
            
            # Validate file type
            is_target_image = is_image_file(target_file.filename)
            is_target_video = is_video_file(target_file.filename)
            
            if not (is_target_image or is_target_video):
                results.append({
                    'filename': target_file.filename,
                    'success': False,
                    'message': 'Invalid file type. Must be image or video.'
                })
                continue
            
            # Generate filenames with enhancement parameters
            target_filename = secure_filename(f"{task_id}_target_{idx}_{target_file.filename}")
            unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
            
            if is_target_image:
                output_filename = f"{enhancement_suffix}_{unique_id}.jpg"
            else:
                output_filename = f"{enhancement_suffix}_{unique_id}.mp4"
            
            # Save target file
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            target_file.save(target_path)
            
            # Process based on type (using cached source embedding)
            print(f"\n📂 Processing target {idx+1}/{len(target_files)}: {target_file.filename}")
            
            if is_target_image:
                success, message = process_image(source_embedding, target_path, output_path,
                                               enhance_face=enhance_face,
                                               enhancement_weight=enhancement_weight,
                                               enhancement_blend=enhancement_blend)
            else:  # is_target_video
                # Create progress queue for this task
                video_task_id = f"{task_id}_{idx}"
                progress_queues[video_task_id] = queue.Queue()
                
                # Process video with progress tracking (using cached source embedding)
                success, message = process_video(source_embedding, target_path, output_path, video_task_id,
                                               enhance_face=enhance_face,
                                               enhancement_weight=enhancement_weight,
                                               enhancement_blend=enhancement_blend)
                
                # Clean up progress queue
                if video_task_id in progress_queues:
                    del progress_queues[video_task_id]
            
            # Clean up target file
            try:
                os.remove(target_path)
            except:
                pass
            
            # Store result
            results.append({
                'filename': target_file.filename,
                'success': success,
                'message': message,
                'output_filename': output_filename if success else None
            })
            
            if success:
                output_filenames.append(output_filename)
        
        # Clean up source file
        try:
            os.remove(source_path)
        except:
            pass
        
        # Check if any succeeded
        any_success = any(r['success'] for r in results)
        
        if any_success:
            # Create zip file if multiple successful outputs
            zip_filename = None
            if len(output_filenames) > 1:
                zip_filename = f"{task_id}_batch_results.zip"
                print(f"\n📦 Creating zip file with {len(output_filenames)} results...")
            
            return jsonify({
                'success': True,
                'message': f"Processed {len(output_filenames)} of {len(target_files)} files successfully.",
                'results': results,
                'output_filenames': output_filenames,
                'task_id': task_id,
                'zip_filename': zip_filename
            })
        else:
            return jsonify({
                'success': False,
                'message': 'All files failed to process.',
                'results': results
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/progress/<task_id>')
def progress(task_id):
    """Stream progress updates for a specific task using Server-Sent Events."""
    def generate():
        if task_id not in progress_queues:
            yield f"data: {jsonify({'error': 'Task not found'}).get_data(as_text=True)}\n\n"
            return
        
        progress_queue = progress_queues[task_id]
        while True:
            try:
                # Get progress update with timeout
                progress_data = progress_queue.get(timeout=30)
                yield f"data: {jsonify(progress_data).get_data(as_text=True)}\n\n"
                
                # If processing is complete, end the stream
                if progress_data.get('percentage') == 100:
                    break
            except queue.Empty:
                # Send keepalive
                yield f"data: {jsonify({'keepalive': True}).get_data(as_text=True)}\n\n"
            except Exception as e:
                yield f"data: {jsonify({'error': str(e)}).get_data(as_text=True)}\n\n"
                break
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/download/<filename>')
def download(filename):
    try:
        # Check if it's a zip file request for batch download
        if filename.endswith('_batch_results.zip'):
            task_id = filename.replace('_batch_results.zip', '')
            # Find all output files for this task (now using enhancement-based naming)
            output_files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) 
                          if f.startswith('E') and f.endswith(('.jpg', '.mp4'))]
            
            # Filter by creation time to get recent batch (within last 5 minutes)
            import time
            current_time = time.time()
            recent_files = []
            for f in output_files:
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], f)
                if os.path.exists(file_path):
                    file_time = os.path.getmtime(file_path)
                    if current_time - file_time < 300:  # 5 minutes
                        recent_files.append(f)
            
            if not recent_files:
                return jsonify({'success': False, 'message': 'No files found for batch download.'}), 404
            
            # Create zip file in memory
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                for output_file in recent_files:
                    file_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
                    # Keep the enhancement-encoded filename
                    zf.write(file_path, output_file)
            
            memory_file.seek(0)
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name='face_swap_results.zip'
            )
        
        # Single file download
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
        if os.path.exists(output_path):
            # Keep original filename with enhancement encoding
            return send_file(output_path, as_attachment=True, download_name=filename)
        else:
            return jsonify({'success': False, 'message': 'File not found.'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/view/<filename>')
def view(filename):
    try:
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
        if os.path.exists(output_path):
            if filename.endswith('.jpg'):
                return send_file(output_path, mimetype='image/jpeg')
            else:
                return send_file(output_path, mimetype='video/mp4')
        else:
            return jsonify({'success': False, 'message': 'File not found.'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Starting Flask server...")
    print("=" * 60)
    
    # Initialize models at startup (before accepting requests)
    print("\n🔄 Initializing models at startup...")
    try:
        if initialize_models():
            print("✅ All models loaded successfully!")
            print("🚀 Server ready to accept requests!")
        else:
            print("❌ Failed to initialize models!")
            print("⚠️  Server will start but requests will fail.")
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        print("⚠️  Server will start but requests will fail.")
    
    print("=" * 60)
    print(f"🌐 Server running at: http://localhost:5050")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5050)
