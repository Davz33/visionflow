"""Intent analysis service for video generation requests."""

import re
from typing import Any, Dict, List

from ...shared.config import get_settings
from ...shared.models import IntentAnalysis, VideoGenerationRequest
from ...shared.monitoring import get_logger, track_request_metrics

logger = get_logger("intent_service")


class IntentAnalysisService:
    """Service for analyzing user intent in video generation requests."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Intent patterns for classification
        self.intent_patterns = {
            "scene_creation": [
                r"\b(create|generate|make|build)\b.*\b(scene|environment|landscape|setting)",
                r"\b(show|display|depict)\b.*\b(place|location|area)",
                r"\b(outdoor|indoor|forest|city|beach|mountain|space)\b",
            ],
            "character_animation": [
                r"\b(person|people|character|figure|human|man|woman|child)\b",
                r"\b(walking|running|dancing|moving|gesturing)\b",
                r"\b(actor|performer|model)\b",
            ],
            "object_focus": [
                r"\b(car|vehicle|animal|object|item|thing)\b",
                r"\b(product|device|tool|machine)\b",
                r"\b(close.?up|macro|detailed view)\b",
            ],
            "abstract_artistic": [
                r"\b(abstract|artistic|creative|surreal|dreamlike)\b",
                r"\b(pattern|texture|color|light|shadow)\b",
                r"\b(emotion|feeling|mood|atmosphere)\b",
            ],
            "nature_documentary": [
                r"\b(nature|wildlife|animal|plant|ocean|forest)\b",
                r"\b(documentary|realistic|natural)\b",
                r"\b(time.?lapse|slow.?motion)\b",
            ],
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            "high": [
                r"\b(complex|detailed|intricate|elaborate)\b",
                r"\bmultiple\b.*\b(characters|objects|scenes)\b",
                r"\b(interaction|dialogue|conversation)\b",
                r"\b(special effects|CGI|animation)\b",
            ],
            "medium": [
                r"\b(some|several|few)\b",
                r"\b(movement|action|motion)\b",
                r"\b(background|foreground)\b",
            ],
            "low": [
                r"\b(simple|basic|minimal|clean)\b",
                r"\b(static|still|peaceful|calm)\b",
                r"\b(single|one|solo)\b",
            ],
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            "objects": r"\b(car|house|tree|person|animal|building|mountain|ocean|sky|sun|moon)\b",
            "actions": r"\b(running|walking|flying|swimming|dancing|jumping|sitting|standing)\b",
            "colors": r"\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|brown)\b",
            "emotions": r"\b(happy|sad|angry|peaceful|excited|calm|dramatic|intense|serene)\b",
            "styles": r"\b(realistic|cartoon|anime|photorealistic|artistic|abstract|vintage|modern)\b",
        }
    
    @track_request_metrics("intent_service")
    async def analyze_intent(self, request: VideoGenerationRequest) -> IntentAnalysis:
        """Analyze the intent of a video generation request."""
        prompt = request.prompt.lower()
        
        # Classify intent type
        intent_type, confidence = self._classify_intent(prompt)
        
        # Extract parameters
        parameters = self._extract_parameters(request)
        
        # Extract entities
        entities = self._extract_entities(prompt)
        
        # Assess complexity
        complexity_score = self._assess_complexity(prompt, request)
        
        logger.info(
            "Intent analysis completed",
            intent_type=intent_type,
            confidence=confidence,
            complexity_score=complexity_score,
            entities_count=len(entities),
        )
        
        return IntentAnalysis(
            intent_type=intent_type,
            confidence=confidence,
            parameters=parameters,
            entities=entities,
            complexity_score=complexity_score,
        )
    
    def _classify_intent(self, prompt: str) -> tuple[str, float]:
        """Classify the primary intent of the prompt."""
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    matches += 1
                    score += 1.0 / len(patterns)
            
            if matches > 0:
                # Boost score based on number of pattern matches
                score = min(score * (1 + matches * 0.1), 1.0)
                intent_scores[intent_type] = score
        
        if not intent_scores:
            return "general", 0.5
        
        # Get the highest scoring intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1]
    
    def _extract_parameters(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """Extract relevant parameters from the request."""
        parameters = {
            "duration": request.duration,
            "quality": request.quality.value,
            "fps": request.fps,
            "resolution": request.resolution,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
        }
        
        if request.seed is not None:
            parameters["seed"] = request.seed
        
        # Extract style preferences from prompt
        prompt_lower = request.prompt.lower()
        
        # Style detection
        style_keywords = {
            "cinematic": ["cinematic", "movie", "film", "dramatic lighting"],
            "realistic": ["realistic", "photorealistic", "real", "photograph"],
            "artistic": ["artistic", "painting", "drawn", "illustrated"],
            "animated": ["animated", "cartoon", "anime", "3d rendered"],
        }
        
        detected_styles = []
        for style, keywords in style_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_styles.append(style)
        
        if detected_styles:
            parameters["style_preference"] = detected_styles
        
        # Time of day detection
        time_keywords = {
            "dawn": ["dawn", "sunrise", "early morning"],
            "day": ["day", "daytime", "noon", "bright"],
            "dusk": ["dusk", "sunset", "evening"],
            "night": ["night", "nighttime", "dark", "moonlight"],
        }
        
        for time_period, keywords in time_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                parameters["time_of_day"] = time_period
                break
        
        return parameters
    
    def _extract_entities(self, prompt: str) -> List[str]:
        """Extract named entities from the prompt."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                if match not in entities:
                    entities.append(match)
        
        return entities[:10]  # Limit to top 10 entities
    
    def _assess_complexity(self, prompt: str, request: VideoGenerationRequest) -> float:
        """Assess the complexity of the video generation request."""
        complexity_score = 0.5  # Base complexity
        
        # Check complexity indicators in prompt
        for complexity_level, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    if complexity_level == "high":
                        complexity_score += 0.2
                    elif complexity_level == "medium":
                        complexity_score += 0.1
                    elif complexity_level == "low":
                        complexity_score -= 0.1
        
        # Factor in technical parameters
        if request.duration > 10:
            complexity_score += 0.1
        if request.quality.value in ["high", "ultra"]:
            complexity_score += 0.1
        if request.fps > 30:
            complexity_score += 0.05
        if request.num_inference_steps > 50:
            complexity_score += 0.1
        
        # Factor in prompt length and detail
        word_count = len(prompt.split())
        if word_count > 20:
            complexity_score += 0.1
        if word_count > 50:
            complexity_score += 0.1
        
        return max(0.0, min(1.0, complexity_score))


# Service instance
intent_service = IntentAnalysisService()
