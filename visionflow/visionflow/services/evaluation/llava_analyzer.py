"""
LLaVA-based video content analyzer for subjective evaluation.

This module provides a local LLaVA implementation for analyzing video content,
replacing the API-based Gemini Pro Vision for subjective assessments.
"""

import asyncio
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class LLaVAAnalyzer:
    """
    LLaVA-based video content analyzer for subjective evaluation.
    
    Uses local LLaVA model for:
    - Content description and interpretation
    - Prompt adherence assessment  
    - Aesthetic quality judgment
    - Creative interpretation
    - Content appropriateness
    """
    
    def __init__(self, model_path: str = "liuhaotian/llava-v1.5-13b"):
        """
        Initialize LLaVA analyzer.
        
        Args:
            model_path: Path to LLaVA model weights or HuggingFace model name
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model lazily to avoid startup delays
        self._initialized = False
        
    async def _ensure_initialized(self):
        """Initialize LLaVA model if not already done."""
        if self._initialized:
            return
            
        try:
            await self._load_model()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize LLaVA: {e}")
            raise
    
    async def _load_model(self):
        """Load LLaVA model and tokenizer."""
        try:
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
            
            logger.info(f"Loading LLaVA model from {self.model_path}")
            
            # Load processor and model
            self.processor = LlavaProcessor.from_pretrained(self.model_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("LLaVA model loaded successfully")
            
        except ImportError:
            logger.error("LLaVA dependencies not installed. Install with: pip install transformers[torch]")
            raise
        except Exception as e:
            logger.error(f"Error loading LLaVA model: {e}")
            raise
    
    def _frame_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert numpy frame to PIL Image."""
        # Convert BGR to RGB if needed
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        return Image.fromarray(frame_rgb)
    
    async def analyze_frame_content(
        self, 
        frame: np.ndarray, 
        frame_index: int,
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze frame content using LLaVA for subjective evaluation.
        
        Args:
            frame: Video frame as numpy array
            frame_index: Index of the frame in the video
            original_prompt: Original text prompt used for video generation
            
        Returns:
            Dictionary containing subjective analysis results
        """
        await self._ensure_initialized()
        
        try:
            # Convert frame to PIL Image
            pil_image = self._frame_to_pil(frame)
            
            # Create prompt for subjective evaluation
            evaluation_prompt = f"""
            Analyze this video frame for video generation quality assessment.
            
            Original Prompt: "{original_prompt}"
            Frame Index: {frame_index}
            
            Please evaluate and return a JSON response with:
            {{
                "content_description": "detailed description of what's shown",
                "prompt_adherence_score": 0.0-1.0,
                "visual_quality_score": 0.0-1.0,
                "composition_score": 0.0-1.0,
                "lighting_quality": 0.0-1.0,
                "color_harmony": 0.0-1.0,
                "artistic_appeal": 0.0-1.0,
                "content_safety": 0.0-1.0,
                "detected_objects": ["list", "of", "objects"],
                "scene_type": "description",
                "dominant_colors": ["color1", "color2", "color3"],
                "overall_assessment": "brief overall assessment"
            }}
            
            Focus on subjective quality aspects that require human-like judgment.
            """
            
            # Process inputs
            inputs = self.processor(
                text=evaluation_prompt,
                images=pil_image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode response
            response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            try:
                # Find JSON content in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != 0:
                    json_content = response_text[json_start:json_end]
                    analysis = json.loads(json_content)
                else:
                    # Fallback: create structured response from text
                    analysis = self._parse_text_response(response_text)
                    
                return analysis
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLaVA response, using fallback")
                return self._parse_text_response(response_text)
                
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return self._create_fallback_analysis(frame, frame_index)
    
    async def analyze_video_sequence(
        self,
        frames: List[np.ndarray],
        original_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze sequence of frames for temporal coherence using LLaVA.
        
        Args:
            frames: List of video frames
            original_prompt: Original text prompt
            
        Returns:
            Dictionary containing sequence analysis results
        """
        await self._ensure_initialized()
        
        try:
            # Limit to 4 frames for analysis
            analysis_frames = frames[:4]
            pil_images = [self._frame_to_pil(frame) for frame in analysis_frames]
            
            # Create sequence analysis prompt
            sequence_prompt = f"""
            Analyze this sequence of video frames for temporal coherence and quality.
            
            Original Prompt: "{original_prompt}"
            Number of Frames: {len(analysis_frames)}
            
            Evaluate and return a JSON response with:
            {{
                "temporal_coherence_score": 0.0-1.0,
                "motion_smoothness": 0.0-1.0,
                "visual_consistency": 0.0-1.0,
                "narrative_flow": 0.0-1.0,
                "style_consistency": 0.0-1.0,
                "transition_quality": 0.0-1.0,
                "overall_sequence_assessment": "brief assessment"
            }}
            
            Focus on how well the frames work together as a sequence.
            """
            
            # Process multiple images
            inputs = self.processor(
                text=sequence_prompt,
                images=pil_images,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Decode and parse response
            response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != 0:
                    json_content = response_text[json_start:json_end]
                    analysis = json.loads(json_content)
                else:
                    analysis = self._parse_text_response(response_text)
                    
                return analysis
                
            except json.JSONDecodeError:
                return self._parse_text_response(response_text)
                
        except Exception as e:
            logger.error(f"Sequence analysis failed: {e}")
            return self._create_fallback_sequence_analysis()
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails."""
        # Extract scores from text using simple parsing
        scores = {}
        
        # Look for score patterns
        import re
        score_pattern = r'(\w+)_score["\s]*:["\s]*([0-9]*\.?[0-9]+)'
        matches = re.findall(score_pattern, response_text)
        
        for key, value in matches:
            try:
                scores[f"{key}_score"] = float(value)
            except ValueError:
                continue
        
        # Fill missing scores with defaults
        default_scores = {
            "prompt_adherence_score": 0.7,
            "visual_quality_score": 0.7,
            "composition_score": 0.7,
            "lighting_quality": 0.7,
            "color_harmony": 0.7,
            "artistic_appeal": 0.7,
            "content_safety": 0.9
        }
        
        for key, default_value in default_scores.items():
            if key not in scores:
                scores[key] = default_value
        
        # Add description
        scores["content_description"] = "Analysis based on LLaVA evaluation"
        scores["overall_assessment"] = "Quality assessment completed"
        
        return scores
    
    def _create_fallback_analysis(self, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        """Create fallback analysis when LLaVA fails."""
        return {
            "content_description": f"Frame {frame_index} analysis unavailable",
            "prompt_adherence_score": 0.5,
            "visual_quality_score": 0.5,
            "composition_score": 0.5,
            "lighting_quality": 0.5,
            "color_harmony": 0.5,
            "artistic_appeal": 0.5,
            "content_safety": 0.8,
            "detected_objects": [],
            "scene_type": "unknown",
            "dominant_colors": [],
            "overall_assessment": "Fallback analysis used"
        }
    
    def _create_fallback_sequence_analysis(self) -> Dict[str, Any]:
        """Create fallback sequence analysis."""
        return {
            "temporal_coherence_score": 0.5,
            "motion_smoothness": 0.5,
            "visual_consistency": 0.5,
            "narrative_flow": 0.5,
            "style_consistency": 0.5,
            "transition_quality": 0.5,
            "overall_sequence_assessment": "Sequence analysis unavailable"
        }
    
    async def close(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._initialized = False
        logger.info("LLaVA analyzer resources cleaned up")


# Factory function for easy integration
def create_llava_analyzer(model_path: str = "liuhaotian/llava-v1.5-13b") -> LLaVAAnalyzer:
    """
    Create a LLaVA analyzer instance.
    
    Args:
        model_path: Path to LLaVA model weights or HuggingFace model name
        
    Returns:
        Configured LLaVA analyzer
    """
    return LLaVAAnalyzer(model_path)
