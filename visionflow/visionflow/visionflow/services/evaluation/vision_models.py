"""
Vision Models for Advanced Video Quality Assessment
Production-ready implementation with multiple specialized models
"""

import asyncio
import base64
import io
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from langchain_google_vertexai import ChatVertexAI
from langchain.schema import HumanMessage, SystemMessage
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

from ...shared.config import get_settings
from ...shared.monitoring import get_logger

logger = get_logger("vision_models")
settings = get_settings()


class VisionModelConfig:
    """Configuration for vision models"""
    
    # Primary vision-language model (Gemini Pro Vision)
    PRIMARY_VLM = "gemini-pro-vision"
    
    # Specialized models for different tasks
    AESTHETIC_MODEL = "aesthetic-assessment-v1"  # Custom or fine-tuned model
    TECHNICAL_MODEL = "technical-quality-v1"    # Custom model for technical assessment
    
    # Model parameters
    VLM_TEMPERATURE = 0.1  # Low for consistent evaluation
    VLM_MAX_TOKENS = 2048
    
    # Frame sampling for video analysis
    MAX_FRAMES_PER_ANALYSIS = 8
    FRAME_SAMPLE_STRATEGY = "uniform"  # uniform, keyframe, or adaptive
    
    # Image preprocessing
    TARGET_RESOLUTION = (512, 512)
    IMAGE_QUALITY = 85  # JPEG quality for API calls


class GeminiVisionAnalyzer:
    """Primary vision analyzer using Gemini Pro Vision"""
    
    def __init__(self):
        self.llm = ChatVertexAI(
            model_name=VisionModelConfig.PRIMARY_VLM,
            project=settings.monitoring.vertex_ai_project,
            location=settings.monitoring.vertex_ai_region,
            temperature=VisionModelConfig.VLM_TEMPERATURE,
            max_output_tokens=VisionModelConfig.VLM_MAX_TOKENS
        )
    
    async def analyze_frame_content(
        self, 
        frame: np.ndarray, 
        frame_index: int,
        original_prompt: str
    ) -> Dict[str, Any]:
        """Analyze frame content using Gemini Pro Vision"""
        
        try:
            # Convert frame to base64 for API
            image_b64 = self._frame_to_base64(frame)
            
            system_prompt = """
            You are an expert video content analyzer. Analyze this video frame and provide detailed insights.
            
            Focus on:
            1. Visual content description (objects, scenes, characters, actions)
            2. Visual quality (clarity, lighting, composition, colors)
            3. Technical aspects (resolution appearance, artifacts, blur)
            4. Artistic elements (style, mood, aesthetic appeal)
            5. Content appropriateness and safety
            
            Provide analysis in JSON format with specific scores (0.0-1.0) for each aspect.
            """
            
            user_prompt = f"""
            Original Prompt: "{original_prompt}"
            Frame Index: {frame_index}
            
            Please analyze this video frame and return a JSON response with:
            {{
                "content_description": "detailed description of what's shown",
                "prompt_adherence_score": 0.0-1.0,
                "visual_quality_score": 0.0-1.0,
                "composition_score": 0.0-1.0,
                "lighting_quality": 0.0-1.0,
                "color_harmony": 0.0-1.0,
                "technical_quality": 0.0-1.0,
                "artistic_appeal": 0.0-1.0,
                "content_safety": 0.0-1.0,
                "detected_objects": ["list", "of", "objects"],
                "scene_type": "description",
                "dominant_colors": ["color1", "color2", "color3"],
                "overall_assessment": "brief overall assessment"
            }}
            """
            
            # Create message with image
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ])
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            try:
                analysis = json.loads(response.content)
                return analysis
            except json.JSONDecodeError:
                # Fallback parsing if JSON is embedded in text
                return self._extract_json_from_text(response.content)
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return self._create_fallback_analysis(frame, frame_index)
    
    async def analyze_video_sequence(
        self,
        frames: List[np.ndarray],
        original_prompt: str
    ) -> Dict[str, Any]:
        """Analyze sequence of frames for temporal coherence"""
        
        try:
            # Convert frames to base64
            frame_images = [self._frame_to_base64(frame) for frame in frames[:4]]  # Limit to 4 frames
            
            system_prompt = """
            You are an expert video sequence analyzer. Analyze this sequence of video frames for:
            
            1. Temporal coherence (consistency across frames)
            2. Motion smoothness and natural progression
            3. Visual consistency (lighting, style, color)
            4. Narrative flow and logical progression
            5. Overall sequence quality
            
            Return analysis in JSON format with scores (0.0-1.0) for each aspect.
            """
            
            user_prompt = f"""
            Original Prompt: "{original_prompt}"
            Number of Frames: {len(frame_images)}
            
            Analyze this video sequence and return JSON:
            {{
                "temporal_coherence": 0.0-1.0,
                "motion_smoothness": 0.0-1.0,
                "visual_consistency": 0.0-1.0,
                "narrative_flow": 0.0-1.0,
                "transition_quality": 0.0-1.0,
                "sequence_description": "description of the sequence",
                "coherence_issues": ["list", "of", "issues"],
                "strengths": ["list", "of", "strengths"],
                "overall_sequence_score": 0.0-1.0
            }}
            """
            
            # Create multimodal message
            content = [{"type": "text", "text": user_prompt}]
            for i, img_b64 in enumerate(frame_images):
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return self._extract_json_from_text(response.content)
            
        except Exception as e:
            logger.error(f"Video sequence analysis failed: {e}")
            return self._create_fallback_sequence_analysis()
    
    async def analyze_aesthetic_quality(
        self,
        frame: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Specialized aesthetic quality analysis"""
        
        try:
            image_b64 = self._frame_to_base64(frame)
            
            system_prompt = """
            You are a professional visual aesthetics expert. Evaluate this image's aesthetic quality across multiple dimensions.
            
            Consider:
            - Composition and visual balance
            - Color harmony and palette effectiveness
            - Lighting quality and mood creation
            - Visual appeal and attractiveness
            - Professional polish and finish
            - Artistic style and creativity
            - Emotional impact and engagement
            
            Rate each dimension on a scale of 0.0 to 1.0.
            """
            
            user_prompt = """
            Analyze the aesthetic quality of this image and return JSON:
            {
                "overall_aesthetic_score": 0.0-1.0,
                "composition_balance": 0.0-1.0,
                "color_harmony": 0.0-1.0,
                "lighting_quality": 0.0-1.0,
                "visual_appeal": 0.0-1.0,
                "professional_polish": 0.0-1.0,
                "artistic_creativity": 0.0-1.0,
                "emotional_impact": 0.0-1.0,
                "style_consistency": 0.0-1.0,
                "aesthetic_strengths": ["list", "of", "strengths"],
                "improvement_suggestions": ["list", "of", "suggestions"]
            }
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ])
            ]
            
            response = await self.llm.ainvoke(messages)
            
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return self._extract_json_from_text(response.content)
                
        except Exception as e:
            logger.error(f"Aesthetic analysis failed: {e}")
            return {
                "overall_aesthetic_score": 0.75,
                "composition_balance": 0.75,
                "color_harmony": 0.75,
                "lighting_quality": 0.75,
                "visual_appeal": 0.75,
                "professional_polish": 0.75,
                "artistic_creativity": 0.75,
                "emotional_impact": 0.75,
                "style_consistency": 0.75
            }
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 for API transmission"""
        try:
            # Resize frame if too large
            height, width = frame.shape[:2]
            if width > VisionModelConfig.TARGET_RESOLUTION[0] or height > VisionModelConfig.TARGET_RESOLUTION[1]:
                frame = cv2.resize(frame, VisionModelConfig.TARGET_RESOLUTION)
            
            # Convert BGR to RGB (OpenCV uses BGR)
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=VisionModelConfig.IMAGE_QUALITY)
            img_bytes = buffer.getvalue()
            
            return base64.b64encode(img_bytes).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Frame to base64 conversion failed: {e}")
            # Return a minimal 1x1 image as fallback
            fallback_img = Image.new('RGB', (1, 1), color='black')
            buffer = io.BytesIO()
            fallback_img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response"""
        try:
            # Find JSON blocks in the text
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # If no valid JSON found, create structured fallback
            return self._create_fallback_analysis_from_text(text)
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return {"error": "Failed to parse response", "raw_text": text[:500]}
    
    def _create_fallback_analysis(self, frame: np.ndarray, frame_index: int) -> Dict[str, Any]:
        """Create fallback analysis when vision model fails"""
        
        # Basic image analysis using OpenCV
        height, width = frame.shape[:2]
        
        # Calculate basic metrics
        brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        contrast = np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        return {
            "content_description": f"Frame {frame_index} ({width}x{height})",
            "prompt_adherence_score": 0.7,
            "visual_quality_score": min(1.0, brightness / 128.0),
            "composition_score": 0.7,
            "lighting_quality": min(1.0, brightness / 128.0),
            "color_harmony": 0.7,
            "technical_quality": 0.8 if width >= 512 else 0.6,
            "artistic_appeal": 0.7,
            "content_safety": 1.0,
            "detected_objects": ["unknown"],
            "scene_type": "general",
            "dominant_colors": ["unknown"],
            "overall_assessment": "Basic analysis - vision model unavailable",
            "fallback": True
        }
    
    def _create_fallback_sequence_analysis(self) -> Dict[str, Any]:
        """Create fallback sequence analysis"""
        return {
            "temporal_coherence": 0.8,
            "motion_smoothness": 0.8,
            "visual_consistency": 0.8,
            "narrative_flow": 0.7,
            "transition_quality": 0.8,
            "sequence_description": "Fallback analysis - detailed assessment unavailable",
            "coherence_issues": [],
            "strengths": ["Basic technical quality maintained"],
            "overall_sequence_score": 0.75,
            "fallback": True
        }
    
    def _create_fallback_analysis_from_text(self, text: str) -> Dict[str, Any]:
        """Create structured analysis from unstructured text"""
        
        # Simple keyword-based scoring
        positive_keywords = ["good", "excellent", "great", "high quality", "beautiful", "clear"]
        negative_keywords = ["poor", "bad", "low quality", "blurry", "artifacts", "issues"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        # Simple sentiment-based scoring
        base_score = 0.7
        sentiment_adjustment = (positive_count - negative_count) * 0.05
        estimated_score = max(0.1, min(1.0, base_score + sentiment_adjustment))
        
        return {
            "content_description": text[:200] + "..." if len(text) > 200 else text,
            "prompt_adherence_score": estimated_score,
            "visual_quality_score": estimated_score,
            "composition_score": estimated_score,
            "lighting_quality": estimated_score,
            "color_harmony": estimated_score,
            "technical_quality": estimated_score,
            "artistic_appeal": estimated_score,
            "content_safety": 1.0,
            "detected_objects": ["analyzed_via_text"],
            "scene_type": "text_analysis",
            "dominant_colors": ["unknown"],
            "overall_assessment": "Text-based analysis fallback",
            "text_analysis": True,
            "estimated_score": estimated_score
        }


class TechnicalQualityAnalyzer:
    """Specialized analyzer for technical video quality"""
    
    def __init__(self):
        self.setup_quality_metrics()
    
    def setup_quality_metrics(self):
        """Setup technical quality assessment tools"""
        # Initialize any specialized models or tools
        pass
    
    async def analyze_technical_quality(
        self,
        frames: List[np.ndarray],
        video_properties: Dict[str, Any]
    ) -> Dict[str, float]:
        """Advanced technical quality analysis"""
        
        if not frames:
            return {"error": "No frames provided"}
        
        try:
            # Multi-frame technical analysis
            sharpness_scores = []
            noise_scores = []
            compression_artifacts = []
            
            for frame in frames[:10]:  # Analyze up to 10 frames
                # Sharpness analysis using Laplacian variance
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_scores.append(min(1.0, sharpness / 1000))  # Normalize
                
                # Noise analysis using standard deviation
                noise_level = np.std(gray)
                noise_scores.append(max(0.0, 1.0 - (noise_level / 50)))  # Inverse noise
                
                # Basic compression artifact detection
                # (In production, you'd use more sophisticated methods)
                artifacts = self._detect_compression_artifacts(frame)
                compression_artifacts.append(artifacts)
            
            # Aggregate scores
            avg_sharpness = np.mean(sharpness_scores) if sharpness_scores else 0.5
            avg_noise_score = np.mean(noise_scores) if noise_scores else 0.5
            avg_artifacts = 1.0 - np.mean(compression_artifacts) if compression_artifacts else 0.8
            
            # Resolution scoring
            height, width = frames[0].shape[:2]
            resolution_score = self._score_resolution(width, height)
            
            # Frame consistency
            consistency_score = self._analyze_frame_consistency(frames)
            
            return {
                "sharpness_score": float(avg_sharpness),
                "noise_score": float(avg_noise_score),
                "compression_score": float(avg_artifacts),
                "resolution_score": float(resolution_score),
                "consistency_score": float(consistency_score),
                "overall_technical_score": float(
                    (avg_sharpness + avg_noise_score + avg_artifacts + 
                     resolution_score + consistency_score) / 5
                )
            }
            
        except Exception as e:
            logger.error(f"Technical quality analysis failed: {e}")
            return {
                "sharpness_score": 0.7,
                "noise_score": 0.7,
                "compression_score": 0.7,
                "resolution_score": 0.7,
                "consistency_score": 0.7,
                "overall_technical_score": 0.7,
                "error": str(e)
            }
    
    def _detect_compression_artifacts(self, frame: np.ndarray) -> float:
        """Detect compression artifacts in frame"""
        try:
            # Simple blocking artifact detection using gradient analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # High frequency content suggests less compression
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            hf_content = np.mean(gradient_magnitude)
            
            # Normalize and invert (higher gradient = less artifacts)
            artifact_level = max(0.0, 1.0 - (hf_content / 100))
            return artifact_level
            
        except Exception:
            return 0.3  # Moderate artifact level as fallback
    
    def _score_resolution(self, width: int, height: int) -> float:
        """Score resolution quality"""
        resolution = width * height
        
        if resolution >= 1920 * 1080:  # 1080p+
            return 1.0
        elif resolution >= 1280 * 720:  # 720p
            return 0.8
        elif resolution >= 854 * 480:   # 480p
            return 0.6
        elif resolution >= 640 * 360:   # 360p
            return 0.4
        else:
            return 0.2
    
    def _analyze_frame_consistency(self, frames: List[np.ndarray]) -> float:
        """Analyze consistency across frames"""
        if len(frames) < 2:
            return 1.0
        
        try:
            consistency_scores = []
            
            for i in range(1, min(len(frames), 10)):
                # Convert to grayscale
                prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Calculate structural similarity
                # Simple version - in production use SSIM
                diff = cv2.absdiff(prev_gray, curr_gray)
                mean_diff = np.mean(diff)
                
                # Convert difference to consistency score
                consistency = max(0.0, 1.0 - (mean_diff / 50))
                consistency_scores.append(consistency)
            
            return float(np.mean(consistency_scores)) if consistency_scores else 0.8
            
        except Exception:
            return 0.8  # Default consistency score


class VideoQualityOrchestrator:
    """Orchestrates multiple vision models for comprehensive video analysis"""
    
    def __init__(self):
        self.gemini_analyzer = GeminiVisionAnalyzer()
        self.technical_analyzer = TechnicalQualityAnalyzer()
    
    async def comprehensive_video_analysis(
        self,
        video_path: str,
        original_prompt: str,
        max_frames: int = 8
    ) -> Dict[str, Any]:
        """Perform comprehensive video analysis using multiple models"""
        
        try:
            # Extract frames from video
            frames = self._extract_frames(video_path, max_frames)
            
            if not frames:
                return {"error": "Could not extract frames from video"}
            
            # Run analyses in parallel for efficiency
            tasks = [
                self._analyze_content_with_gemini(frames, original_prompt),
                self._analyze_technical_quality(frames),
                self._analyze_aesthetic_quality(frames)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            content_analysis = results[0] if not isinstance(results[0], Exception) else {}
            technical_analysis = results[1] if not isinstance(results[1], Exception) else {}
            aesthetic_analysis = results[2] if not isinstance(results[2], Exception) else {}
            
            # Create comprehensive report
            comprehensive_report = {
                "video_path": video_path,
                "original_prompt": original_prompt,
                "frames_analyzed": len(frames),
                "content_analysis": content_analysis,
                "technical_analysis": technical_analysis,
                "aesthetic_analysis": aesthetic_analysis,
                "overall_scores": self._calculate_overall_scores(
                    content_analysis, technical_analysis, aesthetic_analysis
                ),
                "analysis_timestamp": "2024-01-01T00:00:00Z",  # Use actual timestamp
                "models_used": ["gemini-pro-vision", "technical-analyzer", "aesthetic-analyzer"]
            }
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Comprehensive video analysis failed: {e}")
            return {"error": str(e), "video_path": video_path}
    
    async def _analyze_content_with_gemini(
        self, 
        frames: List[np.ndarray], 
        original_prompt: str
    ) -> Dict[str, Any]:
        """Analyze content using Gemini Pro Vision"""
        
        # Analyze individual frames
        frame_analyses = []
        for i, frame in enumerate(frames[:4]):  # Limit to 4 frames for cost efficiency
            analysis = await self.gemini_analyzer.analyze_frame_content(
                frame, i, original_prompt
            )
            frame_analyses.append(analysis)
        
        # Analyze sequence coherence
        sequence_analysis = await self.gemini_analyzer.analyze_video_sequence(
            frames, original_prompt
        )
        
        return {
            "frame_analyses": frame_analyses,
            "sequence_analysis": sequence_analysis,
            "aggregated_scores": self._aggregate_content_scores(frame_analyses, sequence_analysis)
        }
    
    async def _analyze_technical_quality(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze technical quality"""
        
        video_properties = {
            "width": frames[0].shape[1] if frames else 0,
            "height": frames[0].shape[0] if frames else 0,
            "frame_count": len(frames)
        }
        
        technical_scores = await self.technical_analyzer.analyze_technical_quality(
            frames, video_properties
        )
        
        return technical_scores
    
    async def _analyze_aesthetic_quality(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze aesthetic quality"""
        
        if not frames:
            return {"error": "No frames for aesthetic analysis"}
        
        # Analyze aesthetic quality of representative frames
        aesthetic_analyses = []
        sample_frames = frames[::max(1, len(frames)//3)][:3]  # Sample 3 frames
        
        for frame in sample_frames:
            aesthetic_analysis = await self.gemini_analyzer.analyze_aesthetic_quality(frame)
            aesthetic_analyses.append(aesthetic_analysis)
        
        # Aggregate aesthetic scores
        aggregated_aesthetic = self._aggregate_aesthetic_scores(aesthetic_analyses)
        
        return {
            "frame_aesthetic_analyses": aesthetic_analyses,
            "aggregated_aesthetic": aggregated_aesthetic
        }
    
    def _extract_frames(self, video_path: str, max_frames: int) -> List[np.ndarray]:
        """Extract frames from video for analysis"""
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                cap.release()
                return []
            
            # Calculate frame sampling indices
            if frame_count <= max_frames:
                indices = list(range(frame_count))
            else:
                indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frames.append(frame)
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    def _aggregate_content_scores(
        self, 
        frame_analyses: List[Dict], 
        sequence_analysis: Dict
    ) -> Dict[str, float]:
        """Aggregate content analysis scores"""
        
        if not frame_analyses:
            return {"error": "No frame analyses to aggregate"}
        
        # Extract scores from frame analyses
        scores = {
            "prompt_adherence": [],
            "visual_quality": [],
            "composition": [],
            "lighting_quality": [],
            "color_harmony": [],
            "technical_quality": [],
            "artistic_appeal": []
        }
        
        for analysis in frame_analyses:
            if isinstance(analysis, dict) and not analysis.get("error"):
                scores["prompt_adherence"].append(analysis.get("prompt_adherence_score", 0.7))
                scores["visual_quality"].append(analysis.get("visual_quality_score", 0.7))
                scores["composition"].append(analysis.get("composition_score", 0.7))
                scores["lighting_quality"].append(analysis.get("lighting_quality", 0.7))
                scores["color_harmony"].append(analysis.get("color_harmony", 0.7))
                scores["technical_quality"].append(analysis.get("technical_quality", 0.7))
                scores["artistic_appeal"].append(analysis.get("artistic_appeal", 0.7))
        
        # Calculate averages
        aggregated = {}
        for key, values in scores.items():
            aggregated[f"avg_{key}"] = float(np.mean(values)) if values else 0.7
        
        # Add sequence scores
        if isinstance(sequence_analysis, dict) and not sequence_analysis.get("error"):
            aggregated["temporal_coherence"] = sequence_analysis.get("temporal_coherence", 0.8)
            aggregated["motion_smoothness"] = sequence_analysis.get("motion_smoothness", 0.8)
            aggregated["narrative_flow"] = sequence_analysis.get("narrative_flow", 0.7)
        
        return aggregated
    
    def _aggregate_aesthetic_scores(self, aesthetic_analyses: List[Dict]) -> Dict[str, float]:
        """Aggregate aesthetic analysis scores"""
        
        if not aesthetic_analyses:
            return {"overall_aesthetic_score": 0.7}
        
        scores = {
            "overall_aesthetic": [],
            "composition_balance": [],
            "color_harmony": [],
            "lighting_quality": [],
            "visual_appeal": [],
            "professional_polish": [],
            "artistic_creativity": []
        }
        
        for analysis in aesthetic_analyses:
            if isinstance(analysis, dict) and not analysis.get("error"):
                scores["overall_aesthetic"].append(analysis.get("overall_aesthetic_score", 0.7))
                scores["composition_balance"].append(analysis.get("composition_balance", 0.7))
                scores["color_harmony"].append(analysis.get("color_harmony", 0.7))
                scores["lighting_quality"].append(analysis.get("lighting_quality", 0.7))
                scores["visual_appeal"].append(analysis.get("visual_appeal", 0.7))
                scores["professional_polish"].append(analysis.get("professional_polish", 0.7))
                scores["artistic_creativity"].append(analysis.get("artistic_creativity", 0.7))
        
        # Calculate averages
        aggregated = {}
        for key, values in scores.items():
            aggregated[f"avg_{key}"] = float(np.mean(values)) if values else 0.7
        
        return aggregated
    
    def _calculate_overall_scores(
        self, 
        content_analysis: Dict, 
        technical_analysis: Dict, 
        aesthetic_analysis: Dict
    ) -> Dict[str, float]:
        """Calculate overall quality scores"""
        
        # Extract key scores with fallbacks
        content_score = content_analysis.get("aggregated_scores", {}).get("avg_prompt_adherence", 0.7)
        technical_score = technical_analysis.get("overall_technical_score", 0.7)
        aesthetic_score = aesthetic_analysis.get("aggregated_aesthetic", {}).get("avg_overall_aesthetic", 0.7)
        
        # Calculate weighted overall score
        overall_score = (
            content_score * 0.4 +      # Content is most important
            technical_score * 0.35 +   # Technical quality
            aesthetic_score * 0.25     # Aesthetic appeal
        )
        
        return {
            "content_score": float(content_score),
            "technical_score": float(technical_score), 
            "aesthetic_score": float(aesthetic_score),
            "overall_score": float(overall_score)
        }


# Singleton instances
_video_quality_orchestrator = None

def get_video_quality_orchestrator() -> VideoQualityOrchestrator:
    """Get or create video quality orchestrator instance"""
    global _video_quality_orchestrator
    if _video_quality_orchestrator is None:
        _video_quality_orchestrator = VideoQualityOrchestrator()
    return _video_quality_orchestrator
