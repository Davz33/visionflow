"""Prompt optimization service for enhanced video generation."""

import re
from typing import Dict, List, Tuple

from ...shared.config import get_settings
from ...shared.models import IntentAnalysis, PromptOptimization, VideoGenerationRequest
from ...shared.monitoring import get_logger, track_request_metrics

logger = get_logger("prompt_service")


class PromptOptimizationService:
    """Service for optimizing prompts to improve video generation quality."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Quality enhancement patterns
        self.quality_enhancers = {
            "visual_quality": [
                "high quality", "ultra detailed", "8k resolution", "masterpiece",
                "photorealistic", "sharp focus", "professional lighting"
            ],
            "video_specific": [
                "smooth motion", "fluid animation", "cinematic", "stable video",
                "consistent frames", "seamless transition"
            ],
            "artistic_style": [
                "beautiful", "stunning", "award winning", "professional",
                "artistic composition", "perfect lighting"
            ]
        }
        
        # Negative prompts for better quality
        self.negative_prompts = [
            "blurry", "pixelated", "low quality", "distorted", "artifacts",
            "noise", "compression", "unstable", "flickering", "jittery",
            "inconsistent", "malformed", "deformed", "ugly", "worst quality"
        ]
        
        # Style-specific optimizations
        self.style_optimizations = {
            "realistic": {
                "enhancers": ["photorealistic", "realistic lighting", "natural colors", "real world"],
                "technical": ["DSLR", "professional photography", "high dynamic range"]
            },
            "cinematic": {
                "enhancers": ["cinematic lighting", "dramatic composition", "film grain"],
                "technical": ["35mm lens", "depth of field", "professional cinematography"]
            },
            "artistic": {
                "enhancers": ["artistic", "creative composition", "expressive"],
                "technical": ["digital art", "concept art", "illustration style"]
            },
            "animated": {
                "enhancers": ["smooth animation", "cartoon style", "vibrant colors"],
                "technical": ["3D rendered", "animated style", "character animation"]
            }
        }
        
        # Intent-specific optimizations
        self.intent_optimizations = {
            "scene_creation": {
                "focus": ["environment", "landscape", "scenery", "atmospheric"],
                "composition": ["wide shot", "establishing shot", "panoramic view"]
            },
            "character_animation": {
                "focus": ["character movement", "human motion", "expressive animation"],
                "composition": ["medium shot", "character focus", "action sequence"]
            },
            "object_focus": {
                "focus": ["product showcase", "detailed view", "object clarity"],
                "composition": ["close-up", "macro detail", "isolated subject"]
            },
            "abstract_artistic": {
                "focus": ["abstract art", "creative expression", "artistic interpretation"],
                "composition": ["creative composition", "artistic vision", "experimental"]
            },
            "nature_documentary": {
                "focus": ["natural behavior", "wildlife", "nature documentary style"],
                "composition": ["documentary shot", "natural lighting", "realistic movement"]
            }
        }
    
    @track_request_metrics("prompt_service")
    async def optimize_prompt(
        self,
        request: VideoGenerationRequest,
        intent: IntentAnalysis,
    ) -> PromptOptimization:
        """Optimize the prompt for better video generation quality."""
        
        original_prompt = request.prompt
        
        # Start with the original prompt
        optimized_prompt = original_prompt
        modifications = []
        
        # Apply intent-specific optimizations
        optimized_prompt, intent_mods = self._apply_intent_optimizations(
            optimized_prompt, intent
        )
        modifications.extend(intent_mods)
        
        # Apply style optimizations
        optimized_prompt, style_mods = self._apply_style_optimizations(
            optimized_prompt, intent.parameters
        )
        modifications.extend(style_mods)
        
        # Add quality enhancers
        optimized_prompt, quality_mods = self._add_quality_enhancers(
            optimized_prompt, request
        )
        modifications.extend(quality_mods)
        
        # Clean up and format
        optimized_prompt = self._clean_and_format_prompt(optimized_prompt)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            original_prompt, optimized_prompt, intent
        )
        
        # Determine optimization strategy
        strategy = self._determine_strategy(modifications)
        
        logger.info(
            "Prompt optimization completed",
            original_length=len(original_prompt),
            optimized_length=len(optimized_prompt),
            quality_score=quality_score,
            modifications_count=len(modifications),
            strategy=strategy,
        )
        
        return PromptOptimization(
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            optimization_strategy=strategy,
            quality_score=quality_score,
            modifications=modifications,
        )
    
    def _apply_intent_optimizations(
        self, prompt: str, intent: IntentAnalysis
    ) -> Tuple[str, List[str]]:
        """Apply optimizations based on detected intent."""
        modifications = []
        
        intent_config = self.intent_optimizations.get(intent.intent_type, {})
        
        # Add focus keywords
        focus_keywords = intent_config.get("focus", [])
        for keyword in focus_keywords[:2]:  # Limit to 2 keywords
            if keyword.lower() not in prompt.lower():
                prompt += f", {keyword}"
                modifications.append(f"Added intent focus: {keyword}")
        
        # Add composition keywords
        composition_keywords = intent_config.get("composition", [])
        for keyword in composition_keywords[:1]:  # Limit to 1 composition keyword
            if keyword.lower() not in prompt.lower():
                prompt += f", {keyword}"
                modifications.append(f"Added composition: {keyword}")
        
        return prompt, modifications
    
    def _apply_style_optimizations(
        self, prompt: str, parameters: Dict
    ) -> Tuple[str, List[str]]:
        """Apply style-specific optimizations."""
        modifications = []
        
        # Check if style preference is detected
        style_preferences = parameters.get("style_preference", [])
        
        for style in style_preferences:
            if style in self.style_optimizations:
                style_config = self.style_optimizations[style]
                
                # Add enhancers
                enhancers = style_config.get("enhancers", [])
                for enhancer in enhancers[:2]:  # Limit to 2 enhancers per style
                    if enhancer.lower() not in prompt.lower():
                        prompt += f", {enhancer}"
                        modifications.append(f"Added {style} enhancer: {enhancer}")
                
                # Add technical terms
                technical = style_config.get("technical", [])
                for tech_term in technical[:1]:  # Limit to 1 technical term per style
                    if tech_term.lower() not in prompt.lower():
                        prompt += f", {tech_term}"
                        modifications.append(f"Added {style} technical: {tech_term}")
        
        return prompt, modifications
    
    def _add_quality_enhancers(
        self, prompt: str, request: VideoGenerationRequest
    ) -> Tuple[str, List[str]]:
        """Add quality enhancement keywords."""
        modifications = []
        
        # Add video-specific quality enhancers
        video_enhancers = self.quality_enhancers["video_specific"]
        for enhancer in video_enhancers[:2]:  # Limit to 2 video enhancers
            if enhancer.lower() not in prompt.lower():
                prompt += f", {enhancer}"
                modifications.append(f"Added video quality: {enhancer}")
        
        # Add visual quality enhancers based on quality setting
        if request.quality.value in ["high", "ultra"]:
            visual_enhancers = self.quality_enhancers["visual_quality"]
            for enhancer in visual_enhancers[:3]:  # More enhancers for high quality
                if enhancer.lower() not in prompt.lower():
                    prompt += f", {enhancer}"
                    modifications.append(f"Added visual quality: {enhancer}")
        
        # Add artistic enhancers for creative prompts
        if len(prompt.split()) > 10:  # Detailed prompts get artistic enhancement
            artistic_enhancers = self.quality_enhancers["artistic_style"]
            for enhancer in artistic_enhancers[:1]:  # Limit to 1 artistic enhancer
                if enhancer.lower() not in prompt.lower():
                    prompt += f", {enhancer}"
                    modifications.append(f"Added artistic quality: {enhancer}")
        
        return prompt, modifications
    
    def _clean_and_format_prompt(self, prompt: str) -> str:
        """Clean up and format the optimized prompt."""
        # Remove duplicate commas and spaces
        prompt = re.sub(r",\s*,", ",", prompt)
        prompt = re.sub(r"\s+", " ", prompt)
        
        # Ensure proper comma spacing
        prompt = re.sub(r",(?!\s)", ", ", prompt)
        
        # Remove leading/trailing whitespace and commas
        prompt = prompt.strip().strip(",").strip()
        
        # Capitalize first letter
        if prompt:
            prompt = prompt[0].upper() + prompt[1:]
        
        # Ensure prompt doesn't exceed max length
        max_length = self.settings.model.max_length * 10  # Allow more for video prompts
        if len(prompt) > max_length:
            # Truncate at last complete word before limit
            truncated = prompt[:max_length].rsplit(" ", 1)[0]
            prompt = truncated
        
        return prompt
    
    def _calculate_quality_score(
        self, original: str, optimized: str, intent: IntentAnalysis
    ) -> float:
        """Calculate expected quality improvement score."""
        base_score = 0.5
        
        # Factor in intent confidence
        base_score += intent.confidence * 0.2
        
        # Factor in complexity (more complex prompts benefit more from optimization)
        base_score += intent.complexity_score * 0.1
        
        # Factor in length improvement
        length_improvement = (len(optimized) - len(original)) / len(original)
        if 0.1 <= length_improvement <= 0.5:  # Optimal range
            base_score += 0.15
        elif length_improvement > 0:
            base_score += 0.05
        
        # Factor in quality keywords added
        quality_keywords_count = sum(
            1 for enhancer_list in self.quality_enhancers.values()
            for enhancer in enhancer_list
            if enhancer.lower() in optimized.lower()
        )
        base_score += min(quality_keywords_count * 0.02, 0.1)
        
        # Factor in style keywords added
        style_keywords_count = sum(
            1 for style_config in self.style_optimizations.values()
            for enhancer_list in style_config.values()
            for enhancer in enhancer_list
            if enhancer.lower() in optimized.lower()
        )
        base_score += min(style_keywords_count * 0.01, 0.05)
        
        return max(0.0, min(1.0, base_score))
    
    def _determine_strategy(self, modifications: List[str]) -> str:
        """Determine the optimization strategy used."""
        if not modifications:
            return "minimal_optimization"
        
        strategies = []
        
        if any("intent" in mod.lower() for mod in modifications):
            strategies.append("intent_optimization")
        
        if any("style" in mod.lower() or "artistic" in mod.lower() for mod in modifications):
            strategies.append("style_enhancement")
        
        if any("quality" in mod.lower() for mod in modifications):
            strategies.append("quality_enhancement")
        
        if any("video" in mod.lower() for mod in modifications):
            strategies.append("video_optimization")
        
        if len(strategies) >= 3:
            return "comprehensive_optimization"
        elif len(strategies) >= 2:
            return "multi_aspect_optimization"
        elif strategies:
            return strategies[0]
        else:
            return "general_optimization"


# Service instance
prompt_service = PromptOptimizationService()
