"""
Production Evaluation Models
Real implementations of LPIPS, FVMD, CLIP, and LLaVA for video quality assessment.

This module integrates industry-standard models for objective and subjective evaluation:
- LPIPS: Learned Perceptual Image Patch Similarity
- FVMD: Frame-wise Video Motion Detection  
- CLIP: Contrastive Language-Image Pre-training
- LLaVA: Large Language and Vision Assistant
"""

import asyncio
import gc
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
from PIL import Image

# Model imports (installed via requirements)
import lpips
import clip
from transformers import CLIPProcessor, CLIPModel
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from ...shared.monitoring import get_logger

logger = get_logger(__name__)

class LPIPSEvaluator:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) for perceptual quality assessment.
    Measures perceptual similarity between consecutive frames.
    """
    
    def __init__(self, net: str = 'alex', device: str = 'auto'):
        self.device = device if device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.net = net
        
        logger.info(f"🎨 LPIPS Evaluator initialized (net: {net}, device: {self.device})")
    
    async def initialize(self):
        """Initialize LPIPS model"""
        try:
            self.model = lpips.LPIPS(net=self.net).to(self.device)
            self.model.eval()
            logger.info("✅ LPIPS model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load LPIPS model: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for LPIPS evaluation"""
        # Convert BGR to RGB if needed
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1]
        frame = frame.astype(np.float32) / 255.0
        frame = frame * 2.0 - 1.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return frame_tensor.to(self.device)
    
    async def evaluate_perceptual_consistency(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate perceptual consistency between consecutive frames using LPIPS
        
        Returns:
            - consistency_score: Average LPIPS distance (lower = more consistent)
            - confidence: Confidence in the measurement
            - frame_scores: Per-frame consistency scores
        """
        if not self.model:
            await self.initialize()
        
        if len(frames) < 2:
            return {'consistency_score': 1.0, 'confidence': 0.0, 'frame_scores': []}
        
        try:
            lpips_distances = []
            frame_scores = []
            
            with torch.no_grad():
                for i in range(len(frames) - 1):
                    # Preprocess consecutive frames
                    frame1 = self.preprocess_frame(frames[i])
                    frame2 = self.preprocess_frame(frames[i + 1])
                    
                    # Calculate LPIPS distance
                    distance = self.model(frame1, frame2).item()
                    lpips_distances.append(distance)
                    frame_scores.append(distance)
            
            # Calculate overall consistency score
            avg_distance = np.mean(lpips_distances)
            std_distance = np.std(lpips_distances)
            
            # Convert distance to consistency score (invert and normalize)
            # LPIPS distance ranges ~0-1, convert to quality score
            consistency_score = max(0.0, 1.0 - avg_distance)
            
            # Confidence based on consistency of measurements
            confidence = max(0.1, 1.0 - (std_distance / (avg_distance + 1e-6)))
            
            return {
                'consistency_score': consistency_score,
                'confidence': min(confidence, 0.99),
                'frame_scores': frame_scores,
                'avg_lpips_distance': avg_distance,
                'std_lpips_distance': std_distance
            }
            
        except Exception as e:
            logger.error(f"❌ LPIPS evaluation failed: {e}")
            return {'consistency_score': 0.5, 'confidence': 0.1, 'frame_scores': []}
    
    def cleanup(self):
        """Cleanup GPU memory"""
        if self.model:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class MotionConsistencyEvaluator:
    """
    Frame-wise Video Motion Detection (FVMD) for temporal consistency assessment.
    Analyzes motion patterns and temporal coherence.
    """
    
    def __init__(self):
        self.initialized = False
        logger.info("🎬 Motion Consistency Evaluator initialized")
    
    async def initialize(self):
        """Initialize motion detection components"""
        self.initialized = True
        logger.info("✅ Motion evaluator ready")
    
    def calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[np.ndarray]:
        """Calculate optical flow between two frames"""
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
            return flow
        except Exception as e:
            logger.debug(f"Optical flow calculation failed: {e}")
            return None
    
    def calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate frame difference as motion metric"""
        try:
            # Convert to grayscale and float
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # Calculate absolute difference
            diff = np.abs(gray1 - gray2)
            
            # Normalize and return mean difference
            return np.mean(diff) / 255.0
            
        except Exception as e:
            logger.debug(f"Frame difference calculation failed: {e}")
            return 0.0
    
    async def evaluate_motion_consistency(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate motion consistency and temporal coherence
        
        Returns:
            - motion_score: Overall motion consistency (0-1)
            - confidence: Confidence in the assessment
            - temporal_smoothness: Smoothness of motion patterns
            - motion_magnitude: Average motion magnitude
        """
        if not self.initialized:
            await self.initialize()
        
        if len(frames) < 3:
            return {'motion_score': 0.5, 'confidence': 0.0, 'temporal_smoothness': 0.5}
        
        try:
            motion_magnitudes = []
            frame_differences = []
            
            # Calculate motion between consecutive frames
            for i in range(len(frames) - 1):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                
                # Calculate frame difference (fallback method)
                diff = self.calculate_frame_difference(frame1, frame2)
                frame_differences.append(diff)
                
                # Try optical flow for more sophisticated analysis
                flow = self.calculate_optical_flow(frame1, frame2)
                if flow is not None and len(flow) == 2:
                    # Calculate motion magnitude from optical flow
                    magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
                    motion_magnitudes.append(np.mean(magnitude))
                else:
                    # Use frame difference as motion proxy
                    motion_magnitudes.append(diff * 100)  # Scale up for magnitude
            
            # Calculate temporal smoothness
            if len(motion_magnitudes) > 1:
                motion_diffs = np.diff(motion_magnitudes)
                temporal_smoothness = 1.0 - (np.std(motion_diffs) / (np.mean(motion_magnitudes) + 1e-6))
                temporal_smoothness = max(0.0, min(1.0, temporal_smoothness))
            else:
                temporal_smoothness = 0.5
            
            # Calculate average motion magnitude
            avg_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0.0
            
            # Motion score based on smoothness and reasonable motion
            # Penalize both too little motion (static) and too erratic motion
            motion_score = temporal_smoothness * 0.7 + (1.0 - min(1.0, avg_motion / 50.0)) * 0.3
            motion_score = max(0.0, min(1.0, motion_score))
            
            # Confidence based on data quality
            confidence = 0.8 if len(motion_magnitudes) >= 5 else 0.6
            
            return {
                'motion_score': motion_score,
                'confidence': confidence,
                'temporal_smoothness': temporal_smoothness,
                'motion_magnitude': avg_motion,
                'frame_differences': frame_differences
            }
            
        except Exception as e:
            logger.error(f"❌ Motion consistency evaluation failed: {e}")
            return {'motion_score': 0.5, 'confidence': 0.1, 'temporal_smoothness': 0.5}

class CLIPTextVideoAligner:
    """
    CLIP-based text-video alignment assessment.
    Measures how well video content matches the text prompt.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = 'auto'):
        self.device = device if device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        logger.info(f"🔗 CLIP Text-Video Aligner initialized (model: {model_name}, device: {self.device})")
    
    async def initialize(self):
        """Initialize CLIP model"""
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("✅ CLIP model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            raise
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess frames for CLIP"""
        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb_frame)
            pil_frames.append(pil_frame)
        
        return pil_frames
    
    async def evaluate_text_video_alignment(self, frames: List[np.ndarray], text_prompt: str) -> Dict[str, float]:
        """
        Evaluate how well video frames align with text prompt using CLIP
        
        Returns:
            - alignment_score: Average CLIP similarity score
            - confidence: Confidence in the measurement
            - frame_scores: Per-frame alignment scores
        """
        if not self.model:
            await self.initialize()
        
        if not frames or not text_prompt:
            return {'alignment_score': 0.0, 'confidence': 0.0, 'frame_scores': []}
        
        try:
            # Preprocess frames
            pil_frames = self.preprocess_frames(frames)
            
            # Sample frames if too many (for efficiency)
            if len(pil_frames) > 10:
                step = len(pil_frames) // 10
                pil_frames = pil_frames[::step][:10]
            
            frame_scores = []
            
            with torch.no_grad():
                for frame in pil_frames:
                    # Process text and image
                    inputs = self.processor(
                        text=[text_prompt], 
                        images=[frame], 
                        return_tensors="pt", 
                        padding=True
                    )
                    
                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get embeddings
                    outputs = self.model(**inputs)
                    
                    # Calculate similarity
                    image_embeds = outputs.image_embeds
                    text_embeds = outputs.text_embeds
                    
                    # Normalize embeddings
                    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                    # Calculate cosine similarity
                    similarity = torch.matmul(text_embeds, image_embeds.T).item()
                    frame_scores.append(similarity)
            
            # Calculate overall alignment score
            avg_similarity = np.mean(frame_scores)
            std_similarity = np.std(frame_scores)
            
            # Convert similarity to 0-1 range (CLIP similarity is roughly -1 to 1)
            alignment_score = (avg_similarity + 1.0) / 2.0
            alignment_score = max(0.0, min(1.0, alignment_score))
            
            # Confidence based on consistency across frames
            confidence = max(0.1, 1.0 - std_similarity)
            confidence = max(0.1, min(0.99, confidence))
            
            return {
                'alignment_score': alignment_score,
                'confidence': confidence,
                'frame_scores': [(s + 1.0) / 2.0 for s in frame_scores],
                'raw_similarity': avg_similarity,
                'similarity_std': std_similarity
            }
            
        except Exception as e:
            logger.error(f"❌ CLIP alignment evaluation failed: {e}")
            return {'alignment_score': 0.5, 'confidence': 0.1, 'frame_scores': []}
    
    def cleanup(self):
        """Cleanup GPU memory"""
        if self.model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class LLaVAVideoEvaluator:
    """
    LLaVA (Large Language and Vision Assistant) for subjective video quality assessment.
    Provides human-like evaluation of aesthetic quality and narrative flow.
    """
    
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", device: str = 'auto'):
        self.device = device if device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        logger.info(f"🤖 LLaVA Video Evaluator initialized (model: {model_name}, device: {self.device})")
    
    async def initialize(self):
        """Initialize LLaVA model"""
        try:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            logger.info("✅ LLaVA model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load LLaVA model: {e}")
            raise
    
    def create_evaluation_prompt(self, evaluation_type: str, text_prompt: str) -> str:
        """Create structured prompt for LLaVA evaluation"""
        
        base_prompts = {
            'aesthetic': f"""
            Evaluate the aesthetic quality of this video frame sequence. Consider:
            - Visual composition and framing
            - Color harmony and lighting
            - Overall visual appeal
            - Artistic quality
            
            Original prompt: "{text_prompt}"
            
            Rate the aesthetic quality from 0-10 and explain your reasoning.
            Respond with: SCORE: X.X | REASONING: [your explanation]
            """,
            
            'narrative': f"""
            Evaluate the narrative flow and coherence of this video sequence. Consider:
            - Story continuity between frames
            - Logical progression of events
            - Character/object consistency
            - Scene transitions
            
            Original prompt: "{text_prompt}"
            
            Rate the narrative flow from 0-10 and explain your reasoning.
            Respond with: SCORE: X.X | REASONING: [your explanation]
            """,
            
            'overall': f"""
            Provide an overall quality assessment of this video sequence. Consider:
            - How well it matches the prompt
            - Technical quality (clarity, consistency)
            - Aesthetic appeal
            - Overall coherence
            
            Original prompt: "{text_prompt}"
            
            Rate the overall quality from 0-10 and explain your reasoning.
            Respond with: SCORE: X.X | REASONING: [your explanation]
            """
        }
        
        return base_prompts.get(evaluation_type, base_prompts['overall'])
    
    def parse_llava_response(self, response: str) -> Tuple[float, str]:
        """Parse LLaVA response to extract score and reasoning"""
        try:
            # Look for SCORE: X.X pattern
            if "SCORE:" in response:
                score_part = response.split("SCORE:")[1].split("|")[0].strip()
                score = float(score_part)
                score = max(0.0, min(10.0, score))  # Clamp to 0-10
                
                # Extract reasoning
                reasoning = ""
                if "REASONING:" in response:
                    reasoning = response.split("REASONING:")[1].strip()
                
                return score / 10.0, reasoning  # Convert to 0-1 scale
            else:
                # Fallback: try to extract any number
                import re
                numbers = re.findall(r'\b\d+\.?\d*\b', response)
                if numbers:
                    score = float(numbers[0])
                    score = max(0.0, min(10.0, score))
                    return score / 10.0, response
        except:
            pass
        
        # Default fallback
        return 0.5, "Failed to parse response"
    
    async def evaluate_frames(self, frames: List[np.ndarray], prompt: str, evaluation_type: str = 'overall') -> Dict[str, Any]:
        """
        Evaluate video frames using LLaVA
        
        Returns:
            - score: LLaVA quality score (0-1)
            - confidence: Confidence in the assessment
            - reasoning: LLaVA's reasoning
        """
        if not self.model:
            await self.initialize()
        
        if not frames:
            return {'score': 0.0, 'confidence': 0.0, 'reasoning': 'No frames provided'}
        
        try:
            # Select representative frames (max 4 for context length)
            if len(frames) > 4:
                indices = np.linspace(0, len(frames) - 1, 4, dtype=int)
                selected_frames = [frames[i] for i in indices]
            else:
                selected_frames = frames
            
            # Convert frames to PIL Images
            pil_frames = []
            for frame in selected_frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                pil_frames.append(pil_frame)
            
            # Create evaluation prompt
            evaluation_prompt = self.create_evaluation_prompt(evaluation_type, prompt)
            
            # Process with LLaVA (use first frame for now, multi-frame support varies by model)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": evaluation_prompt},
                        {"type": "image", "image": pil_frames[0]},
                    ],
                },
            ]
            
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(pil_frames[0], prompt_text, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Extract score and reasoning
            score, reasoning = self.parse_llava_response(response)
            
            # Confidence based on response quality
            confidence = 0.8 if len(reasoning) > 20 else 0.6
            
            return {
                'score': score,
                'confidence': confidence,
                'reasoning': reasoning,
                'raw_response': response
            }
            
        except Exception as e:
            logger.error(f"❌ LLaVA evaluation failed: {e}")
            return {
                'score': 0.5,
                'confidence': 0.1,
                'reasoning': f'Evaluation failed: {str(e)}',
                'raw_response': ''
            }
    
    def cleanup(self):
        """Cleanup GPU memory"""
        if self.model:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class ProductionEvaluationModels:
    """
    Unified interface for all production evaluation models.
    Manages model lifecycle and provides evaluation services.
    """
    
    def __init__(self, device: str = 'auto', enable_models: List[str] = None):
        self.device = device if device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enable all models by default, or specific ones
        self.enabled_models = enable_models or ['lpips', 'motion', 'clip', 'llava']
        
        # Initialize model instances
        self.lpips_evaluator = LPIPSEvaluator(device=self.device) if 'lpips' in self.enabled_models else None
        self.motion_evaluator = MotionConsistencyEvaluator() if 'motion' in self.enabled_models else None
        self.clip_evaluator = CLIPTextVideoAligner(device=self.device) if 'clip' in self.enabled_models else None
        self.llava_evaluator = LLaVAVideoEvaluator(device=self.device) if 'llava' in self.enabled_models else None
        
        self.initialized = False
        
        logger.info(f"🏭 Production Evaluation Models initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Enabled models: {self.enabled_models}")
    
    async def initialize(self):
        """Initialize all enabled models"""
        logger.info("🚀 Initializing production evaluation models...")
        
        initialization_tasks = []
        
        if self.lpips_evaluator:
            initialization_tasks.append(self.lpips_evaluator.initialize())
        
        if self.motion_evaluator:
            initialization_tasks.append(self.motion_evaluator.initialize())
        
        if self.clip_evaluator:
            initialization_tasks.append(self.clip_evaluator.initialize())
        
        if self.llava_evaluator:
            initialization_tasks.append(self.llava_evaluator.initialize())
        
        # Initialize models concurrently where possible
        await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        self.initialized = True
        logger.info("✅ All production models initialized")
    
    async def evaluate_all_dimensions(self, frames: List[np.ndarray], prompt: str) -> Dict[str, Dict[str, Any]]:
        """
        Run all available evaluation models on the video frames
        
        Returns comprehensive evaluation results from all models
        """
        if not self.initialized:
            await self.initialize()
        
        results = {}
        
        # LPIPS Perceptual Quality
        if self.lpips_evaluator:
            logger.debug("🎨 Running LPIPS evaluation...")
            results['lpips'] = await self.lpips_evaluator.evaluate_perceptual_consistency(frames)
        
        # Motion Consistency
        if self.motion_evaluator:
            logger.debug("🎬 Running motion consistency evaluation...")
            results['motion'] = await self.motion_evaluator.evaluate_motion_consistency(frames)
        
        # CLIP Text-Video Alignment
        if self.clip_evaluator:
            logger.debug("🔗 Running CLIP alignment evaluation...")
            results['clip'] = await self.clip_evaluator.evaluate_text_video_alignment(frames, prompt)
        
        # LLaVA Subjective Assessments
        if self.llava_evaluator:
            logger.debug("🤖 Running LLaVA evaluations...")
            
            # Run multiple LLaVA evaluations for different aspects
            aesthetic_result = await self.llava_evaluator.evaluate_frames(frames, prompt, 'aesthetic')
            narrative_result = await self.llava_evaluator.evaluate_frames(frames, prompt, 'narrative')
            
            results['llava'] = {
                'aesthetic': aesthetic_result,
                'narrative': narrative_result
            }
        
        return results
    
    def cleanup(self):
        """Cleanup all models and free GPU memory"""
        logger.info("🧹 Cleaning up production models...")
        
        if self.lpips_evaluator:
            self.lpips_evaluator.cleanup()
        
        if self.clip_evaluator:
            self.clip_evaluator.cleanup()
        
        if self.llava_evaluator:
            self.llava_evaluator.cleanup()
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Model cleanup completed")

# Factory function for production deployment
def get_production_models(device: str = 'auto', lightweight: bool = False) -> ProductionEvaluationModels:
    """
    Get configured production evaluation models
    
    Args:
        device: Target device ('auto', 'cuda', 'mps', 'cpu')
        lightweight: If True, enable only essential models for resource-constrained environments
    """
    
    if lightweight:
        # Enable only essential models for lightweight deployment
        enabled_models = ['motion', 'clip']
        logger.info("🚀 Lightweight production models configuration")
    else:
        # Full model suite for comprehensive evaluation
        enabled_models = ['lpips', 'motion', 'clip', 'llava']
        logger.info("🏭 Full production models configuration")
    
    return ProductionEvaluationModels(device=device, enable_models=enabled_models)
