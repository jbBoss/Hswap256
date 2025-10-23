"""
Face Restoration Module using CodeFormer
This module enhances and restores facial features after face swapping.
"""

import os
import sys
import numpy
import onnxruntime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.filesystem import resolve_relative_path
from facefusion.face_helper import warp_face_by_face_landmark_5, paste_back
from facefusion.face_masker import create_box_mask
from facefusion.face_analyser import get_many_faces
from facefusion.types import Face, VisionFrame


class FaceRestorer:
    """Face restoration using CodeFormer or other enhancement models"""
    
    def __init__(self, model_name='codeformer', execution_providers=None):
        """
        Initialize face restorer.
        
        Args:
            model_name: Name of the enhancement model ('codeformer', 'gfpgan_1.4', etc.)
            execution_providers: ONNX execution providers (CUDA, CPU, etc.)
        """
        self.model_name = model_name
        self.execution_providers = execution_providers or ['CPUExecutionProvider']
        self.face_enhancer = None
        self.model_template = None
        self.model_size = None
        self._load_model()
    
    def _load_model(self):
        """Load the face enhancement model."""
        model_configs = {
            'codeformer': {
                'model_path': resolve_relative_path('../.assets/models/codeformer.onnx'),
                'model_url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx',
                'hash_url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.hash',
                'hash_path': resolve_relative_path('../.assets/models/codeformer.hash'),
                'template': 'ffhq_512',
                'size': (512, 512)
            },
            'gfpgan_1.4': {
                'model_path': resolve_relative_path('../.assets/models/gfpgan_1.4.onnx'),
                'model_url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx',
                'hash_url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.hash',
                'hash_path': resolve_relative_path('../.assets/models/gfpgan_1.4.hash'),
                'template': 'ffhq_512',
                'size': (512, 512)
            },
            'gpen_bfr_512': {
                'model_path': resolve_relative_path('../.assets/models/gpen_bfr_512.onnx'),
                'model_url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_512.onnx',
                'hash_url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_512.hash',
                'hash_path': resolve_relative_path('../.assets/models/gpen_bfr_512.hash'),
                'template': 'ffhq_512',
                'size': (512, 512)
            }
        }
        
        if self.model_name not in model_configs:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        config = model_configs[self.model_name]
        self.model_template = config['template']
        self.model_size = config['size']
        
        # Download model if needed
        model_set = {
            'face_enhancer': {
                'url': config['model_url'],
                'path': config['model_path']
            }
        }
        hash_set = {
            'face_enhancer': {
                'url': config['hash_url'],
                'path': config['hash_path']
            }
        }
        
        print(f"Checking for {self.model_name} model files...")
        if not conditional_download_hashes(hash_set) or not conditional_download_sources(model_set):
            raise RuntimeError(f"Failed to download {self.model_name} model.")
        
        print(f"Loading {self.model_name} model...")
        self.face_enhancer = onnxruntime.InferenceSession(
            config['model_path'],
            providers=self.execution_providers
        )
        print(f"{self.model_name} model loaded with: {self.face_enhancer.get_providers()}")
    
    def enhance_face(self, target_face: Face, temp_vision_frame: VisionFrame, 
                     weight: float = 0.5, blend: int = 80) -> VisionFrame:
        """
        Enhance a single face in the frame.
        
        Args:
            target_face: Face object with landmarks
            temp_vision_frame: Frame containing the face
            weight: Enhancement weight (0.0-1.0), higher = more enhancement
            blend: Blend percentage (0-100), higher = more original preserved
            
        Returns:
            Enhanced vision frame
        """
        # Warp and crop the face
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
            temp_vision_frame,
            target_face.landmark_set.get('5/68'),
            self.model_template,
            self.model_size
        )
        
        # Create mask
        box_mask = create_box_mask(crop_vision_frame, face_mask_blur=0.1, face_mask_padding=(0, 0, 0, 0))
        
        # Prepare frame for model
        crop_vision_frame_prepared = self._prepare_crop_frame(crop_vision_frame)
        
        # Run enhancement
        crop_vision_frame_enhanced = self._forward(crop_vision_frame_prepared, weight)
        
        # Normalize output
        crop_vision_frame_enhanced = self._normalize_crop_frame(crop_vision_frame_enhanced)
        
        # Paste back
        paste_vision_frame = paste_back(
            temp_vision_frame,
            crop_vision_frame_enhanced,
            box_mask,
            affine_matrix
        )
        
        # Blend with original
        blend_factor = 1 - (blend / 100)
        temp_vision_frame = self._blend_frame(temp_vision_frame, paste_vision_frame, blend_factor)
        
        return temp_vision_frame
    
    def enhance_frame(self, vision_frame: VisionFrame, weight: float = 0.5, 
                     blend: int = 80) -> VisionFrame:
        """
        Enhance all faces in a frame.
        
        Args:
            vision_frame: Input frame
            weight: Enhancement weight (0.0-1.0)
            blend: Blend percentage (0-100)
            
        Returns:
            Enhanced vision frame
        """
        # Detect all faces in the frame
        target_faces = get_many_faces([vision_frame])
        
        if not target_faces:
            return vision_frame
        
        temp_vision_frame = vision_frame.copy()
        
        # Enhance each face
        for target_face in target_faces:
            temp_vision_frame = self.enhance_face(target_face, temp_vision_frame, weight, blend)
        
        return temp_vision_frame
    
    def _prepare_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        """Prepare frame for model inference."""
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0  # BGR to RGB and normalize
        crop_vision_frame = (crop_vision_frame - 0.5) / 0.5  # Normalize to [-1, 1]
        crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis=0).astype(numpy.float32)
        return crop_vision_frame
    
    def _normalize_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        """Normalize model output back to image format."""
        crop_vision_frame = numpy.clip(crop_vision_frame, -1, 1)
        crop_vision_frame = (crop_vision_frame + 1) / 2  # [-1, 1] to [0, 1]
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        crop_vision_frame = (crop_vision_frame * 255.0).round()
        crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]  # RGB to BGR
        return crop_vision_frame
    
    def _forward(self, crop_vision_frame: VisionFrame, weight: float) -> VisionFrame:
        """Run model inference."""
        face_enhancer_inputs = {}
        
        # Check model inputs
        for face_enhancer_input in self.face_enhancer.get_inputs():
            if face_enhancer_input.name == 'input':
                face_enhancer_inputs[face_enhancer_input.name] = crop_vision_frame
            if face_enhancer_input.name == 'weight':
                face_enhancer_inputs[face_enhancer_input.name] = numpy.array([weight]).astype(numpy.double)
        
        crop_vision_frame = self.face_enhancer.run(None, face_enhancer_inputs)[0][0]
        return crop_vision_frame
    
    def _blend_frame(self, base_frame: VisionFrame, overlay_frame: VisionFrame, 
                     blend_factor: float) -> VisionFrame:
        """Blend two frames together."""
        base_frame = base_frame.astype(numpy.float32)
        overlay_frame = overlay_frame.astype(numpy.float32)
        blended = (1 - blend_factor) * base_frame + blend_factor * overlay_frame
        return blended.clip(0, 255).astype(numpy.uint8)
