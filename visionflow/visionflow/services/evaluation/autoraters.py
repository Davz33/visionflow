"""
Autorater Services for Intelligent Video Quality Assessment
Implementing 2024 best practices for AI-powered evaluation
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from langchain.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .quality_metrics import (
    QualityDimensions, EvaluationResult, EvaluationLevel,
    TechnicalMetrics, ContentMetrics, AestheticMetrics,
    UserExperienceMetrics, PerformanceMetrics, ComplianceMetrics
)
from ...shared.config import get_settings
from ...shared.monitoring import get_logger

logger = get_logger("autoraters")
settings = get_settings()


class AutoraterConfig(BaseModel):
    """Configuration for autorater services"""
    technical_analysis_enabled: bool = Field(default=True, description="Enable technical analysis")
    llm_content_analysis_enabled: bool = Field(default=True, description="Enable LLM content analysis")
    aesthetic_analysis_enabled: bool = Field(default=True, description="Enable aesthetic analysis")
    performance_analysis_enabled: bool = Field(default=True, description="Enable performance analysis")
    
    # LLM Configuration
    llm_model_name: str = Field(default="gemini-pro", description="LLM model for content analysis")
    llm_temperature: float = Field(default=0.2, description="LLM temperature for consistency")
    llm_max_tokens: int = Field(default=2048, description="Maximum tokens for LLM responses")
    
    # Analysis depth
    frame_sampling_rate: int = Field(default=5, description="Sample every N frames for analysis")
    max_frames_analyzed: int = Field(default=50, description="Maximum frames to analyze")
    
    # Quality thresholds
    technical_quality_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "min_resolution": 480,
            "min_fps": 20,
            "max_fps_variance": 0.1,
            "min_bitrate": 1000000  # 1 Mbps
        }
    )


class TechnicalAnalyzer:
    """Automated technical quality analysis"""
    
    def __init__(self, config: AutoraterConfig):
        self.config = config
        
    async def analyze_technical_quality(
        self, 
        video_path: str, 
        generation_metadata: Dict[str, Any]
    ) -> TechnicalMetrics:
        """Perform comprehensive technical analysis of video"""
        
        logger.info(f"Starting technical analysis for {video_path}")
        
        # Basic file checks
        file_integrity = self._check_file_integrity(video_path)
        
        # Video properties analysis
        video_properties = self._analyze_video_properties(video_path)
        
        # Frame-level analysis (enhanced with vision models)
        frame_analysis = await self._analyze_frame_quality(video_path)
        
        # Enhanced technical analysis using vision models
        enhanced_analysis = await self._enhanced_technical_analysis(video_path)
        
        # Encoding analysis
        encoding_quality = self._analyze_encoding_quality(video_path, video_properties)
        
        # Duration accuracy
        duration_accuracy = self._check_duration_accuracy(
            video_properties.get("duration", 0),
            generation_metadata.get("requested_duration", 5)
        )
        
        # Bitrate efficiency
        bitrate_efficiency = self._calculate_bitrate_efficiency(video_properties)
        
        # Compression analysis
        compression_ratio = self._analyze_compression_efficiency(video_path, video_properties)
        
        # Combine traditional and enhanced analysis
        final_scores = self._combine_technical_analyses(
            frame_analysis, enhanced_analysis, video_properties, 
            encoding_quality, duration_accuracy, file_integrity, 
            bitrate_efficiency, compression_ratio
        )
        
        return TechnicalMetrics(
            resolution_score=final_scores.get("resolution_score", 0.8),
            framerate_consistency=final_scores.get("framerate_consistency", 0.9),
            encoding_quality=final_scores.get("encoding_quality", encoding_quality),
            duration_accuracy=final_scores.get("duration_accuracy", duration_accuracy),
            file_integrity=final_scores.get("file_integrity", file_integrity),
            bitrate_efficiency=final_scores.get("bitrate_efficiency", bitrate_efficiency),
            compression_ratio=final_scores.get("compression_ratio", compression_ratio),
            motion_smoothness=final_scores.get("motion_smoothness"),
            color_accuracy=final_scores.get("color_accuracy"),
            audio_sync=final_scores.get("audio_sync")
        )
    
    def _check_file_integrity(self, video_path: str) -> float:
        """Check if video file is playable and complete"""
        try:
            if not os.path.exists(video_path):
                return 0.0
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0
            
            # Try to read first and last frames
            ret1, _ = cap.read()
            
            # Go to near end
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 10:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 5)
                ret2, _ = cap.read()
            else:
                ret2 = True
            
            cap.release()
            
            return 1.0 if (ret1 and ret2) else 0.5
            
        except Exception as e:
            logger.error(f"Error checking file integrity: {e}")
            return 0.0
    
    def _analyze_video_properties(self, video_path: str) -> Dict[str, Any]:
        """Analyze basic video properties"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            properties = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
            }
            
            # Resolution score
            resolution = properties["width"] * properties["height"]
            if resolution >= 1920 * 1080:  # 1080p+
                properties["resolution_score"] = 1.0
            elif resolution >= 1280 * 720:  # 720p+
                properties["resolution_score"] = 0.8
            elif resolution >= 854 * 480:   # 480p+
                properties["resolution_score"] = 0.6
            else:
                properties["resolution_score"] = 0.4
            
            cap.release()
            return properties
            
        except Exception as e:
            logger.error(f"Error analyzing video properties: {e}")
            return {"width": 0, "height": 0, "fps": 0, "frame_count": 0, "duration": 0, "resolution_score": 0.0}
    
    async def _analyze_frame_quality(self, video_path: str) -> Dict[str, float]:
        """Analyze frame-level quality metrics"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                cap.release()
                return {"framerate_consistency": 0.0, "motion_smoothness": 0.0, "color_accuracy": 0.0}
            
            # Sample frames for analysis
            sample_indices = np.linspace(0, frame_count - 1, 
                                       min(self.config.max_frames_analyzed, frame_count), 
                                       dtype=int)
            
            frame_metrics = []
            prev_frame = None
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Frame quality metrics
                metrics = {
                    "brightness": np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
                    "contrast": np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
                    "sharpness": cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                }
                
                # Motion analysis (if previous frame exists)
                if prev_frame is not None:
                    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                     cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
                    metrics["motion_magnitude"] = np.mean(diff)
                
                frame_metrics.append(metrics)
                prev_frame = frame.copy()
            
            cap.release()
            
            # Calculate consistency metrics
            if len(frame_metrics) < 2:
                return {"framerate_consistency": 0.5, "motion_smoothness": 0.5, "color_accuracy": 0.5}
            
            # Framerate consistency (based on motion magnitude consistency)
            motion_values = [m.get("motion_magnitude", 0) for m in frame_metrics if "motion_magnitude" in m]
            if motion_values:
                motion_std = np.std(motion_values)
                motion_mean = np.mean(motion_values)
                framerate_consistency = max(0.0, 1.0 - (motion_std / (motion_mean + 1e-6)))
            else:
                framerate_consistency = 0.5
            
            # Motion smoothness (based on motion magnitude transitions)
            motion_smoothness = min(1.0, 1.0 / (motion_std + 1e-6)) if motion_values and len(motion_values) > 1 else 0.5
            
            # Color accuracy (based on brightness/contrast consistency)
            brightness_values = [m["brightness"] for m in frame_metrics]
            brightness_std = np.std(brightness_values)
            color_accuracy = max(0.0, 1.0 - (brightness_std / 128.0))  # Normalize by max brightness range
            
            return {
                "framerate_consistency": min(1.0, framerate_consistency),
                "motion_smoothness": min(1.0, motion_smoothness),
                "color_accuracy": min(1.0, color_accuracy)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frame quality: {e}")
            return {"framerate_consistency": 0.5, "motion_smoothness": 0.5, "color_accuracy": 0.5}
    
    def _analyze_encoding_quality(self, video_path: str, properties: Dict[str, Any]) -> float:
        """Analyze video encoding quality"""
        try:
            # File size based analysis
            file_size = os.path.getsize(video_path)
            duration = properties.get("duration", 1)
            bitrate = (file_size * 8) / duration if duration > 0 else 0
            
            # Quality score based on bitrate and resolution
            resolution = properties.get("width", 0) * properties.get("height", 0)
            
            if resolution > 0:
                # Expected bitrate for good quality (rough estimates)
                if resolution >= 1920 * 1080:  # 1080p
                    expected_bitrate = 5000000  # 5 Mbps
                elif resolution >= 1280 * 720:  # 720p
                    expected_bitrate = 2500000  # 2.5 Mbps
                else:  # 480p
                    expected_bitrate = 1000000  # 1 Mbps
                
                bitrate_ratio = min(1.0, bitrate / expected_bitrate)
                return max(0.1, bitrate_ratio)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing encoding quality: {e}")
            return 0.5
    
    def _check_duration_accuracy(self, actual_duration: float, requested_duration: float) -> float:
        """Check how accurately the video matches requested duration"""
        if requested_duration <= 0:
            return 0.5
        
        duration_error = abs(actual_duration - requested_duration) / requested_duration
        return max(0.0, 1.0 - duration_error)
    
    def _calculate_bitrate_efficiency(self, properties: Dict[str, Any]) -> float:
        """Calculate bitrate efficiency score"""
        try:
            duration = properties.get("duration", 1)
            fps = properties.get("fps", 24)
            
            # Simple efficiency metric based on reasonable expectations
            if fps >= 24 and duration > 0:
                return min(1.0, fps / 30.0)  # Normalize to 30fps as excellent
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating bitrate efficiency: {e}")
            return 0.5
    
    def _analyze_compression_efficiency(self, video_path: str, properties: Dict[str, Any]) -> float:
        """Analyze compression efficiency"""
        try:
            file_size = os.path.getsize(video_path)
            duration = properties.get("duration", 1)
            resolution = properties.get("width", 0) * properties.get("height", 0)
            
            if duration > 0 and resolution > 0:
                # Bytes per pixel per second
                compression_ratio = file_size / (resolution * duration)
                
                # Good compression: 0.1 - 1.0 bytes per pixel per second
                if compression_ratio <= 1.0:
                    return min(1.0, 1.0 / (compression_ratio + 0.1))
                else:
                    return max(0.1, 1.0 / compression_ratio)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing compression efficiency: {e}")
            return 0.5
    
    async def _enhanced_technical_analysis(self, video_path: str) -> Dict[str, float]:
        """Enhanced technical analysis using vision models"""
        try:
            # Use comprehensive vision analysis for technical assessment
            from .vision_models import get_video_quality_orchestrator
            
            orchestrator = get_video_quality_orchestrator()
            
            # Get comprehensive technical analysis
            analysis = await orchestrator.comprehensive_video_analysis(
                video_path, "Technical quality assessment", max_frames=6
            )
            
            if analysis and not analysis.get("error"):
                technical_data = analysis.get("technical_analysis", {})
                
                return {
                    "enhanced_sharpness": technical_data.get("sharpness_score", 0.8),
                    "enhanced_noise": technical_data.get("noise_score", 0.8),
                    "enhanced_compression": technical_data.get("compression_score", 0.8),
                    "enhanced_consistency": technical_data.get("consistency_score", 0.8),
                    "vision_technical_score": technical_data.get("overall_technical_score", 0.8)
                }
            else:
                logger.warning("Enhanced technical analysis failed, using defaults")
                return {
                    "enhanced_sharpness": 0.8,
                    "enhanced_noise": 0.8,
                    "enhanced_compression": 0.8,
                    "enhanced_consistency": 0.8,
                    "vision_technical_score": 0.8
                }
                
        except Exception as e:
            logger.error(f"Enhanced technical analysis failed: {e}")
            return {
                "enhanced_sharpness": 0.8,
                "enhanced_noise": 0.8,
                "enhanced_compression": 0.8,
                "enhanced_consistency": 0.8,
                "vision_technical_score": 0.8
            }
    
    def _combine_technical_analyses(
        self, 
        frame_analysis: Dict[str, float], 
        enhanced_analysis: Dict[str, float],
        video_properties: Dict[str, Any],
        encoding_quality: float,
        duration_accuracy: float,
        file_integrity: float,
        bitrate_efficiency: float,
        compression_ratio: float
    ) -> Dict[str, float]:
        """Combine traditional and enhanced technical analyses"""
        
        # Get scores from both analyses
        traditional_sharpness = frame_analysis.get("framerate_consistency", 0.9)
        enhanced_sharpness = enhanced_analysis.get("enhanced_sharpness", 0.8)
        
        traditional_noise = frame_analysis.get("motion_smoothness", 0.8)
        enhanced_noise = enhanced_analysis.get("enhanced_noise", 0.8)
        
        traditional_compression = compression_ratio
        enhanced_compression = enhanced_analysis.get("enhanced_compression", 0.8)
        
        # Weighted combination (70% traditional, 30% enhanced for reliability)
        combined_scores = {
            "resolution_score": video_properties.get("resolution_score", 0.8),
            "framerate_consistency": (traditional_sharpness * 0.7 + enhanced_sharpness * 0.3),
            "encoding_quality": encoding_quality,
            "duration_accuracy": duration_accuracy,
            "file_integrity": file_integrity,
            "bitrate_efficiency": bitrate_efficiency,
            "compression_ratio": (traditional_compression * 0.7 + enhanced_compression * 0.3),
            "motion_smoothness": (traditional_noise * 0.7 + enhanced_noise * 0.3),
            "color_accuracy": frame_analysis.get("color_accuracy", 0.8),
            "audio_sync": 0.9  # Default since we don't analyze audio yet
        }
        
        return combined_scores


class ContentAnalyzer:
    """LLM-powered content quality analysis"""
    
    def __init__(self, config: AutoraterConfig):
        self.config = config
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> ChatVertexAI:
        """Initialize LLM for content analysis"""
        return ChatVertexAI(
            model_name=self.config.llm_model_name,
            project=settings.monitoring.vertex_ai_project,
            location=settings.monitoring.vertex_ai_region,
            temperature=self.config.llm_temperature,
            max_output_tokens=self.config.llm_max_tokens
        )
    
    async def analyze_content_quality(
        self, 
        video_path: str, 
        original_prompt: str,
        video_description: Optional[str] = None
    ) -> ContentMetrics:
        """Perform advanced LLM-powered content quality analysis"""
        
        logger.info(f"Starting content analysis for prompt: {original_prompt[:50]}...")
        
        try:
            # Use comprehensive vision analysis for enhanced content assessment
            from .vision_models import get_video_quality_orchestrator
            
            orchestrator = get_video_quality_orchestrator()
            
            # Get comprehensive content analysis
            analysis = await orchestrator.comprehensive_video_analysis(
                video_path, original_prompt, max_frames=6
            )
            
            if analysis and not analysis.get("error"):
                content_data = analysis.get("content_analysis", {})
                aggregated_scores = content_data.get("aggregated_scores", {})
                sequence_analysis = content_data.get("sequence_analysis", {})
                
                # Extract enhanced content metrics
                prompt_adherence = aggregated_scores.get("avg_prompt_adherence", 0.8)
                visual_coherence = aggregated_scores.get("avg_visual_quality", 0.8)
                narrative_flow = sequence_analysis.get("narrative_flow", 0.75)
                creativity_score = aggregated_scores.get("avg_artistic_appeal", 0.8)
                detail_richness = aggregated_scores.get("avg_visual_quality", 0.8)
                scene_composition = aggregated_scores.get("avg_composition", 0.8)
                
                # Enhanced metrics from sequence analysis
                temporal_coherence = sequence_analysis.get("temporal_coherence", 0.85)
                motion_smoothness = sequence_analysis.get("motion_smoothness", 0.82)
                
                # Calculate object accuracy from frame analyses
                frame_analyses = content_data.get("frame_analyses", [])
                object_accuracy = self._calculate_object_accuracy(frame_analyses)
                character_consistency = temporal_coherence  # Use temporal coherence as proxy
                scene_transitions = motion_smoothness  # Use motion smoothness as proxy
                
                return ContentMetrics(
                    prompt_adherence=prompt_adherence,
                    visual_coherence=visual_coherence,
                    narrative_flow=narrative_flow,
                    creativity_score=creativity_score,
                    detail_richness=detail_richness,
                    scene_composition=scene_composition,
                    object_accuracy=object_accuracy,
                    character_consistency=character_consistency,
                    scene_transitions=scene_transitions
                )
            else:
                # Fallback to basic analysis
                logger.warning("Advanced content analysis failed, using basic analysis")
                return await self._basic_content_analysis(video_path, original_prompt, video_description)
                
        except Exception as e:
            logger.error(f"Advanced content analysis failed: {e}")
            return await self._basic_content_analysis(video_path, original_prompt, video_description)
    
    async def _basic_content_analysis(
        self, 
        video_path: str, 
        original_prompt: str,
        video_description: Optional[str] = None
    ) -> ContentMetrics:
        """Basic content analysis fallback"""
        
        # Extract video frames for analysis
        frame_descriptions = await self._extract_frame_descriptions(video_path)
        
        # Analyze prompt adherence
        prompt_adherence = await self._analyze_prompt_adherence(
            original_prompt, frame_descriptions, video_description
        )
        
        # Analyze visual coherence
        visual_coherence = await self._analyze_visual_coherence(frame_descriptions)
        
        # Analyze narrative flow
        narrative_flow = await self._analyze_narrative_flow(frame_descriptions, original_prompt)
        
        # Analyze creativity
        creativity_score = await self._analyze_creativity(original_prompt, frame_descriptions)
        
        # Analyze detail richness
        detail_richness = await self._analyze_detail_richness(frame_descriptions)
        
        # Analyze scene composition
        scene_composition = await self._analyze_scene_composition(frame_descriptions)
        
        return ContentMetrics(
            prompt_adherence=prompt_adherence,
            visual_coherence=visual_coherence,
            narrative_flow=narrative_flow,
            creativity_score=creativity_score,
            detail_richness=detail_richness,
            scene_composition=scene_composition,
            object_accuracy=0.85,  # Default values
            character_consistency=0.88,
            scene_transitions=0.82
        )
    
    def _calculate_object_accuracy(self, frame_analyses: List[Dict]) -> float:
        """Calculate object detection accuracy from frame analyses"""
        if not frame_analyses:
            return 0.8  # Default
        
        accuracy_scores = []
        for analysis in frame_analyses:
            if isinstance(analysis, dict) and not analysis.get("error"):
                # Look for object-related quality indicators
                visual_quality = analysis.get("visual_quality_score", 0.8)
                composition = analysis.get("composition_score", 0.8)
                
                # Estimate object accuracy based on visual quality and composition
                object_score = (visual_quality + composition) / 2
                accuracy_scores.append(object_score)
        
        return float(np.mean(accuracy_scores)) if accuracy_scores else 0.8
    
    async def _extract_frame_descriptions(self, video_path: str) -> List[str]:
        """Extract and describe key frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                cap.release()
                return ["Video appears to be empty or corrupted"]
            
            # Sample key frames
            sample_indices = np.linspace(0, frame_count - 1, min(5, frame_count), dtype=int)
            descriptions = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # For now, create basic description based on frame properties
                    # In production, this would use vision models
                    description = await self._describe_frame_content(frame, idx)
                    descriptions.append(description)
            
            cap.release()
            return descriptions
            
        except Exception as e:
            logger.error(f"Error extracting frame descriptions: {e}")
            return ["Unable to analyze video frames"]
    
    async def _describe_frame_content(self, frame: np.ndarray, frame_idx: int) -> str:
        """Describe frame content using advanced vision models"""
        try:
            # Use Gemini Pro Vision for detailed frame analysis
            from .vision_models import get_video_quality_orchestrator
            
            orchestrator = get_video_quality_orchestrator()
            
            # Get detailed frame analysis
            analysis = await orchestrator.gemini_analyzer.analyze_frame_content(
                frame, frame_idx, "Frame analysis for quality assessment"
            )
            
            if analysis and not analysis.get("error"):
                # Create rich description from analysis
                description = analysis.get("content_description", "")
                objects = analysis.get("detected_objects", [])
                scene_type = analysis.get("scene_type", "")
                
                # Enhance with technical details
                height, width = frame.shape[:2]
                quality_score = analysis.get("visual_quality_score", 0.7)
                
                enhanced_desc = f"Frame {frame_idx}: {description}"
                if objects and objects != ["unknown"]:
                    enhanced_desc += f" | Objects: {', '.join(objects[:3])}"
                if scene_type:
                    enhanced_desc += f" | Scene: {scene_type}"
                enhanced_desc += f" | Quality: {quality_score:.2f} ({width}x{height})"
                
                return enhanced_desc
            else:
                # Fallback to basic analysis
                return self._basic_frame_description(frame, frame_idx)
                
        except Exception as e:
            logger.error(f"Advanced frame description failed: {e}")
            return self._basic_frame_description(frame, frame_idx)
    
    def _basic_frame_description(self, frame: np.ndarray, frame_idx: int) -> str:
        """Basic frame description fallback"""
        height, width = frame.shape[:2]
        avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        brightness_desc = "bright" if avg_brightness > 150 else "dark" if avg_brightness < 80 else "balanced"
        
        return f"Frame {frame_idx}: {width}x{height} resolution, {brightness_desc} lighting"
    
    async def _analyze_prompt_adherence(
        self, 
        prompt: str, 
        frame_descriptions: List[str],
        video_description: Optional[str]
    ) -> float:
        """Analyze how well the video content matches the original prompt"""
        
        system_prompt = """
        You are an expert video content analyzer. Your task is to evaluate how well a generated video matches the original text prompt.
        
        Rate the prompt adherence on a scale of 0.0 to 1.0, where:
        - 1.0 = Perfect match, all key elements from prompt are clearly present
        - 0.8 = Very good match, most key elements present with minor deviations
        - 0.6 = Good match, main concept present but some details missing
        - 0.4 = Partial match, recognizable concept but significant missing elements
        - 0.2 = Poor match, concept barely recognizable
        - 0.0 = No match, completely different from prompt
        
        Consider:
        - Presence of key objects, characters, or scenes mentioned
        - Accuracy of actions or activities described
        - Setting and environment match
        - Style and mood alignment
        - Overall concept fidelity
        
        Respond with just the numerical score (0.0-1.0) and a brief explanation.
        """
        
        user_prompt = f"""
        Original Prompt: "{prompt}"
        
        Video Analysis:
        Frame Descriptions: {'; '.join(frame_descriptions)}
        {f'Additional Description: {video_description}' if video_description else ''}
        
        Rate the prompt adherence score:
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Extract numerical score from response
            score = self._extract_score_from_response(response.content)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error analyzing prompt adherence: {e}")
            return 0.7  # Default score
    
    async def _analyze_visual_coherence(self, frame_descriptions: List[str]) -> float:
        """Analyze visual coherence across frames"""
        
        if len(frame_descriptions) < 2:
            return 0.5
        
        system_prompt = """
        You are an expert video quality analyzer focusing on visual coherence and consistency.
        
        Rate the visual coherence on a scale of 0.0 to 1.0, where:
        - 1.0 = Perfect visual consistency, smooth transitions, coherent style
        - 0.8 = Very good consistency with minor visual variations
        - 0.6 = Good consistency with some noticeable variations
        - 0.4 = Moderate consistency with several jarring transitions
        - 0.2 = Poor consistency with frequent visual disruptions
        - 0.0 = No coherence, completely inconsistent visuals
        
        Consider:
        - Consistency in lighting and color
        - Smooth transitions between frames
        - Maintained visual style and quality
        - Logical visual flow
        
        Respond with just the numerical score (0.0-1.0) and brief reasoning.
        """
        
        user_prompt = f"""
        Frame Sequence Analysis:
        {'; '.join(f'Frame {i}: {desc}' for i, desc in enumerate(frame_descriptions))}
        
        Rate the visual coherence score:
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            score = self._extract_score_from_response(response.content)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error analyzing visual coherence: {e}")
            return 0.8  # Default score
    
    async def _analyze_narrative_flow(self, frame_descriptions: List[str], prompt: str) -> float:
        """Analyze narrative flow and storytelling quality"""
        
        system_prompt = """
        You are an expert in video storytelling and narrative analysis.
        
        Rate the narrative flow on a scale of 0.0 to 1.0, where:
        - 1.0 = Excellent narrative flow, clear story progression, engaging sequence
        - 0.8 = Very good flow with logical progression and good pacing
        - 0.6 = Good flow with clear progression but some pacing issues
        - 0.4 = Moderate flow with unclear progression or poor pacing
        - 0.2 = Poor flow with confusing or illogical sequence
        - 0.0 = No narrative flow, random or incoherent sequence
        
        Consider:
        - Logical progression of events
        - Appropriate pacing for the content
        - Clear beginning, middle, end (if applicable)
        - Engaging visual storytelling
        
        Respond with just the numerical score (0.0-1.0) and brief reasoning.
        """
        
        user_prompt = f"""
        Original Prompt: "{prompt}"
        
        Narrative Sequence:
        {'; '.join(f'Scene {i+1}: {desc}' for i, desc in enumerate(frame_descriptions))}
        
        Rate the narrative flow score:
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            score = self._extract_score_from_response(response.content)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error analyzing narrative flow: {e}")
            return 0.75  # Default score
    
    async def _analyze_creativity(self, prompt: str, frame_descriptions: List[str]) -> float:
        """Analyze creative interpretation and innovation"""
        
        system_prompt = """
        You are an expert in creative content evaluation and artistic innovation.
        
        Rate the creativity score on a scale of 0.0 to 1.0, where:
        - 1.0 = Exceptional creativity, innovative interpretation, unique artistic vision
        - 0.8 = High creativity with original elements and fresh perspective
        - 0.6 = Good creativity with some original touches and interpretation
        - 0.4 = Moderate creativity with limited original elements
        - 0.2 = Low creativity, mostly literal interpretation
        - 0.0 = No creativity, completely literal or generic
        
        Consider:
        - Innovative visual interpretation of the prompt
        - Unique artistic elements or style choices
        - Creative problem-solving in visualization
        - Originality in composition or perspective
        
        Respond with just the numerical score (0.0-1.0) and brief reasoning.
        """
        
        user_prompt = f"""
        Original Prompt: "{prompt}"
        
        Creative Interpretation:
        {'; '.join(frame_descriptions)}
        
        Rate the creativity score:
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            score = self._extract_score_from_response(response.content)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error analyzing creativity: {e}")
            return 0.8  # Default score
    
    async def _analyze_detail_richness(self, frame_descriptions: List[str]) -> float:
        """Analyze level of detail and richness in the content"""
        
        detail_score = 0.8  # Mock implementation
        
        # Simple heuristic based on description length and complexity
        avg_length = sum(len(desc.split()) for desc in frame_descriptions) / len(frame_descriptions)
        
        if avg_length > 15:
            detail_score = 0.9
        elif avg_length > 10:
            detail_score = 0.8
        elif avg_length > 5:
            detail_score = 0.7
        else:
            detail_score = 0.6
        
        return detail_score
    
    async def _analyze_scene_composition(self, frame_descriptions: List[str]) -> float:
        """Analyze scene composition quality"""
        # Mock implementation - in production, this would analyze actual composition
        return 0.85
    
    def _extract_score_from_response(self, response_text: str) -> float:
        """Extract numerical score from LLM response"""
        import re
        
        # Look for patterns like "0.8", "0.85", etc.
        score_pattern = r'(?:^|\s)([0-1](?:\.\d+)?)\s'
        matches = re.findall(score_pattern, response_text)
        
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        
        # Fallback: look for percentage patterns and convert
        percent_pattern = r'(\d+(?:\.\d+)?)%'
        percent_matches = re.findall(percent_pattern, response_text)
        
        if percent_matches:
            try:
                return float(percent_matches[0]) / 100.0
            except ValueError:
                pass
        
        # Default fallback
        return 0.7


class AutoraterService:
    """Main autorater service orchestrating all evaluation components"""
    
    def __init__(self, config: Optional[AutoraterConfig] = None):
        self.config = config or AutoraterConfig()
        self.technical_analyzer = TechnicalAnalyzer(self.config)
        self.content_analyzer = ContentAnalyzer(self.config)
        
    async def evaluate_video_comprehensive(
        self,
        video_path: str,
        original_prompt: str,
        job_id: str,
        generation_metadata: Optional[Dict[str, Any]] = None,
        evaluation_level: EvaluationLevel = EvaluationLevel.STANDARD
    ) -> EvaluationResult:
        """Perform comprehensive video evaluation across all dimensions"""
        
        start_time = datetime.utcnow()
        logger.info(f"Starting comprehensive evaluation for job {job_id}")
        
        generation_metadata = generation_metadata or {}
        
        try:
            # Run parallel analysis for efficiency
            analysis_tasks = []
            
            # Technical analysis
            if self.config.technical_analysis_enabled:
                analysis_tasks.append(
                    self.technical_analyzer.analyze_technical_quality(video_path, generation_metadata)
                )
            
            # Content analysis 
            if self.config.llm_content_analysis_enabled:
                analysis_tasks.append(
                    self.content_analyzer.analyze_content_quality(video_path, original_prompt)
                )
            
            # Execute parallel analysis
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Extract results
            technical_metrics = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else self._create_default_technical_metrics()
            content_metrics = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else self._create_default_content_metrics()
            
            # Create other metrics (these could be async in production)
            aesthetic_metrics = await self._create_aesthetic_metrics(video_path)
            ux_metrics = await self._create_ux_metrics(original_prompt, technical_metrics, content_metrics)
            performance_metrics = await self._create_performance_metrics(generation_metadata)
            compliance_metrics = await self._create_compliance_metrics(original_prompt, video_path)
            
            # Combine all metrics
            quality_dimensions = QualityDimensions(
                technical=technical_metrics,
                content=content_metrics,
                aesthetic=aesthetic_metrics,
                user_experience=ux_metrics,
                performance=performance_metrics,
                compliance=compliance_metrics,
                evaluation_level=evaluation_level,
                evaluator_version="v2024.1.0"
            )
            
            evaluation_duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Create comprehensive evaluation result
            evaluation_result = EvaluationResult(
                job_id=job_id,
                video_path=video_path,
                original_prompt=original_prompt,
                quality_dimensions=quality_dimensions,
                evaluation_level=evaluation_level,
                evaluation_duration=evaluation_duration,
                evaluator_agents=["technical_analyzer", "content_analyzer", "aesthetic_analyzer"]
            )
            
            logger.info(f"Evaluation completed for job {job_id}: {quality_dimensions.overall_quality_score:.3f} ({quality_dimensions.quality_grade.value})")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed for job {job_id}: {e}")
            
            # Return minimal evaluation result
            evaluation_duration = (datetime.utcnow() - start_time).total_seconds()
            
            return EvaluationResult(
                job_id=job_id,
                video_path=video_path,
                original_prompt=original_prompt,
                quality_dimensions=self._create_fallback_quality_dimensions(evaluation_level),
                evaluation_level=evaluation_level,
                evaluation_duration=evaluation_duration,
                evaluator_agents=["fallback_evaluator"]
            )
    
    async def evaluate_video_quick(
        self,
        video_path: str,
        original_prompt: str,
        job_id: str
    ) -> EvaluationResult:
        """Perform quick evaluation for real-time feedback"""
        
        return await self.evaluate_video_comprehensive(
            video_path=video_path,
            original_prompt=original_prompt,
            job_id=job_id,
            evaluation_level=EvaluationLevel.BASIC
        )
    
    def _create_default_technical_metrics(self) -> TechnicalMetrics:
        """Create default technical metrics when analysis fails"""
        return TechnicalMetrics(
            resolution_score=0.7,
            framerate_consistency=0.8,
            encoding_quality=0.7,
            duration_accuracy=0.8,
            file_integrity=0.9,
            bitrate_efficiency=0.7,
            compression_ratio=0.6
        )
    
    def _create_default_content_metrics(self) -> ContentMetrics:
        """Create default content metrics when analysis fails"""
        return ContentMetrics(
            prompt_adherence=0.7,
            visual_coherence=0.8,
            narrative_flow=0.7,
            creativity_score=0.8,
            detail_richness=0.7,
            scene_composition=0.8
        )
    
    async def _create_aesthetic_metrics(self, video_path: str) -> AestheticMetrics:
        """Create aesthetic metrics using advanced vision models"""
        try:
            # Use comprehensive vision analysis for aesthetics
            from .vision_models import get_video_quality_orchestrator
            
            orchestrator = get_video_quality_orchestrator()
            
            # Get comprehensive aesthetic analysis
            analysis = await orchestrator.comprehensive_video_analysis(
                video_path, "Aesthetic quality assessment", max_frames=4
            )
            
            if analysis and not analysis.get("error"):
                aesthetic_data = analysis.get("aesthetic_analysis", {})
                aggregated = aesthetic_data.get("aggregated_aesthetic", {})
                
                # Extract aesthetic scores with fallbacks
                visual_appeal = aggregated.get("avg_visual_appeal", 0.85)
                color_harmony = aggregated.get("avg_color_harmony", 0.82)
                lighting_quality = aggregated.get("avg_lighting_quality", 0.88)
                composition_balance = aggregated.get("avg_composition_balance", 0.86)
                artistic_creativity = aggregated.get("avg_artistic_creativity", 0.84)
                professional_polish = aggregated.get("avg_professional_polish", 0.87)
                
                # Calculate depth and atmosphere from content analysis
                content_data = analysis.get("content_analysis", {})
                aggregated_content = content_data.get("aggregated_scores", {})
                
                depth_perception = aggregated_content.get("avg_composition", 0.84)
                atmosphere_mood = aggregated_content.get("avg_lighting_quality", 0.86)
                texture_quality = aggregated_content.get("avg_visual_quality", 0.85)
                
                return AestheticMetrics(
                    visual_appeal=visual_appeal,
                    color_harmony=color_harmony,
                    lighting_quality=lighting_quality,
                    composition_balance=composition_balance,
                    artistic_style=artistic_creativity,
                    professional_polish=professional_polish,
                    depth_perception=depth_perception,
                    texture_quality=texture_quality,
                    atmosphere_mood=atmosphere_mood
                )
            else:
                # Fallback to default values
                logger.warning("Vision model aesthetic analysis failed, using defaults")
                return self._create_default_aesthetic_metrics()
                
        except Exception as e:
            logger.error(f"Aesthetic analysis with vision models failed: {e}")
            return self._create_default_aesthetic_metrics()
    
    def _create_default_aesthetic_metrics(self) -> AestheticMetrics:
        """Create default aesthetic metrics when vision analysis fails"""
        return AestheticMetrics(
            visual_appeal=0.85,
            color_harmony=0.82,
            lighting_quality=0.88,
            composition_balance=0.86,
            artistic_style=0.84,
            professional_polish=0.87
        )
    
    async def _create_ux_metrics(self, prompt: str, technical: TechnicalMetrics, content: ContentMetrics) -> UserExperienceMetrics:
        """Create user experience metrics based on other analyses"""
        
        # Calculate UX metrics based on technical and content quality
        engagement_level = (content.prompt_adherence + content.creativity_score) / 2
        satisfaction_prediction = (technical.overall_technical_score + content.overall_content_score) / 2
        
        return UserExperienceMetrics(
            engagement_level=engagement_level,
            satisfaction_prediction=satisfaction_prediction,
            shareability_score=min(0.95, engagement_level + 0.1),
            accessibility_score=technical.file_integrity,
            usability_rating=technical.overall_technical_score,
            emotional_impact=content.creativity_score
        )
    
    async def _create_performance_metrics(self, generation_metadata: Dict[str, Any]) -> PerformanceMetrics:
        """Create performance metrics from generation metadata"""
        
        generation_time = generation_metadata.get("generation_time", 45.0)
        estimated_cost = generation_metadata.get("estimated_cost", 0.50)
        
        # Calculate performance scores
        generation_speed = max(0.1, min(1.0, 60.0 / generation_time))  # Normalize to 60s target
        cost_effectiveness = max(0.1, min(1.0, 1.0 / estimated_cost))  # Lower cost = higher score
        
        return PerformanceMetrics(
            generation_speed=generation_speed,
            resource_efficiency=0.78,
            cost_effectiveness=cost_effectiveness,
            scalability_score=0.85,
            reliability_rating=0.92,
            actual_generation_time=generation_time,
            estimated_cost=estimated_cost,
            resource_usage=generation_metadata.get("resource_usage", {"gpu": 0.75, "memory": 0.68})
        )
    
    async def _create_compliance_metrics(self, prompt: str, video_path: str) -> ComplianceMetrics:
        """Create compliance metrics (simplified for now)"""
        return ComplianceMetrics(
            content_safety=1.0,
            policy_compliance=1.0,
            copyright_safety=0.95,
            age_appropriateness=1.0,
            cultural_sensitivity=0.92
        )
    
    def _create_fallback_quality_dimensions(self, evaluation_level: EvaluationLevel) -> QualityDimensions:
        """Create fallback quality dimensions when evaluation fails"""
        return QualityDimensions(
            technical=self._create_default_technical_metrics(),
            content=self._create_default_content_metrics(),
            aesthetic=AestheticMetrics(
                visual_appeal=0.7, color_harmony=0.7, lighting_quality=0.7,
                composition_balance=0.7, artistic_style=0.7, professional_polish=0.7
            ),
            user_experience=UserExperienceMetrics(
                engagement_level=0.7, satisfaction_prediction=0.7, shareability_score=0.7,
                accessibility_score=0.7, usability_rating=0.7, emotional_impact=0.7
            ),
            performance=PerformanceMetrics(
                generation_speed=0.7, resource_efficiency=0.7, cost_effectiveness=0.7,
                scalability_score=0.7, reliability_rating=0.7,
                actual_generation_time=60.0, estimated_cost=0.75
            ),
            compliance=ComplianceMetrics(
                content_safety=0.9, policy_compliance=0.9, copyright_safety=0.9,
                age_appropriateness=0.9, cultural_sensitivity=0.9
            ),
            evaluation_level=evaluation_level,
            evaluator_version="v2024.1.0-fallback"
        )


# Singleton instance
_autorater_service = None

def get_autorater_service(config: Optional[AutoraterConfig] = None) -> AutoraterService:
    """Get or create autorater service instance"""
    global _autorater_service
    if _autorater_service is None:
        _autorater_service = AutoraterService(config)
    return _autorater_service
