"""
Batch Face Restoration Script
Applies CodeFormer face restoration to all images and videos in a directory.
"""

import os
import sys
import argparse
import cv2
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from facefusion import state_manager, face_detector, face_landmarker
from facefusion.vision import read_image, write_image
from face_restoration import FaceRestorer


def get_execution_providers():
    """Get available execution providers, prioritizing CUDA."""
    import onnxruntime
    available_providers = onnxruntime.get_available_providers()
    print(f"Available ONNX providers: {available_providers}")
    
    if 'CUDAExecutionProvider' in available_providers:
        print("🚀 Using CUDA (GPU) for acceleration!")
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif 'DmlExecutionProvider' in available_providers:
        print("🚀 Using DirectML (GPU) for acceleration!")
        return ['DmlExecutionProvider', 'CPUExecutionProvider']
    else:
        print("⚠️ No GPU detected, using CPU")
        return ['CPUExecutionProvider']


def initialize_facefusion():
    """Initialize FaceFusion face detection components."""
    execution_providers = get_execution_providers()
    
    print("Initializing face analysis components...")
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

    if not face_detector.pre_check():
        raise RuntimeError("Failed to load face detector models.")
    
    if not face_landmarker.pre_check():
        raise RuntimeError("Failed to load face landmarker models.")
    
    print("Face analysis initialized!")


def process_image(input_path, output_path, face_restorer, weight=0.5, blend=80):
    """Process a single image."""
    print(f"Processing image: {input_path}")
    
    # Read image
    frame = read_image(input_path)
    if frame is None:
        print(f"  ❌ Failed to read image")
        return False
    
    # Apply restoration
    restored_frame = face_restorer.enhance_frame(frame, weight=weight, blend=blend)
    
    # Save result
    write_image(output_path, restored_frame)
    print(f"  ✅ Saved to: {output_path}")
    return True


def process_video(input_path, output_path, face_restorer, weight=0.5, blend=80):
    """Process a single video."""
    print(f"Processing video: {input_path}")
    
    # Open video
    video_capture = cv2.VideoCapture(input_path)
    if not video_capture.isOpened():
        print(f"  ❌ Failed to open video")
        return False
    
    # Get video properties
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Processing frame {frame_count}/{total_frames}...")
        
        # Apply restoration
        restored_frame = face_restorer.enhance_frame(frame, weight=weight, blend=blend)
        
        # Write frame
        video_writer.write(restored_frame)
    
    video_capture.release()
    video_writer.release()
    
    print(f"  ✅ Processed {frame_count} frames, saved to: {output_path}")
    return True


def process_directory(input_dir, output_dir, model_name='codeformer', weight=0.5, blend=80):
    """Process all images and videos in a directory."""
    
    # Supported formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize FaceFusion
    initialize_facefusion()
    
    # Initialize face restorer
    execution_providers = get_execution_providers()
    print(f"\nLoading {model_name} model...")
    face_restorer = FaceRestorer(model_name=model_name, execution_providers=execution_providers)
    print(f"{model_name} loaded successfully!\n")
    
    # Process all files
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    processed_count = 0
    failed_count = 0
    
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        # Create output filename
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}_restored{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            if ext in image_extensions:
                if process_image(input_path, output_path, face_restorer, weight, blend):
                    processed_count += 1
                else:
                    failed_count += 1
            elif ext in video_extensions:
                if process_video(input_path, output_path, face_restorer, weight, blend):
                    processed_count += 1
                else:
                    failed_count += 1
            else:
                print(f"Skipping unsupported file: {filename}")
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Batch face restoration using CodeFormer')
    parser.add_argument('input_dir', help='Input directory containing images/videos')
    parser.add_argument('output_dir', help='Output directory for restored files')
    parser.add_argument('--model', default='codeformer', 
                       choices=['codeformer', 'gfpgan_1.4', 'gpen_bfr_512'],
                       help='Face restoration model to use')
    parser.add_argument('--weight', type=float, default=0.5,
                       help='Enhancement strength (0.0-1.0, default: 0.5)')
    parser.add_argument('--blend', type=int, default=80,
                       help='Original face preservation (0-100, default: 80)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    # Validate parameters
    weight = max(0.0, min(1.0, args.weight))
    blend = max(0, min(100, args.blend))
    
    print("="*60)
    print("Batch Face Restoration")
    print("="*60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model:            {args.model}")
    print(f"Enhancement strength: {weight}")
    print(f"Face preservation:    {blend}%")
    print("="*60)
    print()
    
    process_directory(args.input_dir, args.output_dir, args.model, weight, blend)


if __name__ == '__main__':
    main()
