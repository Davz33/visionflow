"""
Quality Metrics and Data Models for Video Generation Evaluation
Human-centered evaluation framework with comprehensive quality dimensions
Enhanced with industry-standard metrics: LPIPS, FVMD, CLIP, ETVA
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field

from ...shared.monitoring import get_logger

logger = get_logger("quality_metrics")


class QualityDimension(str, Enum):
    """Core quality dimensions for video evaluation"""
    TECHNICAL = "technical"
    CONTENT = "content" 
    AESTHETIC = "aesthetic"
    USER_EXPERIENCE = "user_experience"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


class EvaluationLevel(str, Enum):
    """Evaluation complexity levels"""
    BASIC = "basic"           # Fast, automated checks
    STANDARD = "standard"     # Balanced automation + AI
    COMPREHENSIVE = "comprehensive"  # Deep AI analysis
    EXPERT = "expert"         # Human-level evaluation


class QualityGrade(str, Enum):
    """Human-readable quality grades"""
    EXCEPTIONAL = "exceptional"  # 0.95+
    EXCELLENT = "excellent"      # 0.85+
    GOOD = "good"               # 0.75+
    ACCEPTABLE = "acceptable"    # 0.65+
    POOR = "poor"               # 0.50+
    UNACCEPTABLE = "unacceptable"  # <0.50


class IndustryStandardMetrics(BaseModel):
    """Industry-standard video evaluation metrics based on latest research"""
    
    # LPIPS (Learned Perceptual Image Patch Similarity)
    lpips_score: Optional[float] = Field(None, description="LPIPS score (0.0-1.0+, lower is better)")
    lpips_confidence: Optional[float] = Field(None, description="Confidence in LPIPS assessment")
    lpips_threshold_met: Optional[bool] = Field(None, description="Whether 0.15 threshold is met for reliable judgment")
    
    # FVMD (Fréchet Video Motion Distance)
    fvmd_score: Optional[float] = Field(None, description="FVMD score (lower is better)")
    fvmd_motion_consistency: Optional[float] = Field(None, description="Motion consistency score (0-1)")
    fvmd_human_correlation: Optional[float] = Field(None, description="Correlation with human judgment")
    
    # CLIP (Contrastive Language-Image Pre-training)
    clip_alignment_score: Optional[float] = Field(None, description="CLIP text-video alignment (0-1)")
    clip_threshold_met: Optional[bool] = Field(None, description="Whether 0.7+ threshold is met for good alignment")
    clip_frame_scores: Optional[List[float]] = Field(None, description="Per-frame CLIP scores")
    
    # ETVA (Evaluation Through Video-specific Questions)
    etva_question_scores: Optional[Dict[str, float]] = Field(None, description="ETVA question category scores")
    etva_human_correlation: Optional[float] = Field(None, description="Correlation with human evaluation")
    etva_semantic_accuracy: Optional[float] = Field(None, description="Semantic accuracy score (0-1)")
    
    @property
    def overall_industry_score(self) -> Optional[float]:
        """Calculate weighted overall industry standard score"""
        scores = []
        weights = []
        
        if self.lpips_score is not None:
            # LPIPS: lower is better, so invert and normalize
            lpips_normalized = max(0, 1 - self.lpips_score)
            scores.append(lpips_normalized)
            weights.append(0.3)
        
        if self.fvmd_motion_consistency is not None:
            scores.append(self.fvmd_motion_consistency)
            weights.append(0.25)
        
        if self.clip_alignment_score is not None:
            scores.append(self.clip_alignment_score)
            weights.append(0.25)
        
        if self.etva_semantic_accuracy is not None:
            scores.append(self.etva_semantic_accuracy)
            weights.append(0.2)
        
        if not scores:
            return None
        
        # Calculate weighted average
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else None
    
    def get_industry_insights(self) -> List[str]:
        """Get insights based on industry standard metrics"""
        insights = []
        
        if self.lpips_score is not None:
            if self.lpips_score < 0.2:
                insights.append("Excellent perceptual quality (LPIPS < 0.2)")
            elif self.lpips_score < 0.6:
                insights.append("Good perceptual quality (LPIPS < 0.6)")
            else:
                insights.append("Perceptual quality needs improvement (LPIPS >= 0.6)")
            
            if self.lpips_threshold_met:
                insights.append("LPIPS threshold met (0.15) - reliable quality assessment")
            else:
                insights.append("LPIPS below threshold - consider human review")
        
        if self.fvmd_motion_consistency is not None:
            if self.fvmd_motion_consistency > 0.8:
                insights.append("Excellent motion consistency (FVMD correlation > 0.8)")
            elif self.fvmd_motion_consistency > 0.6:
                insights.append("Good motion consistency")
            else:
                insights.append("Motion consistency needs improvement")
        
        if self.clip_alignment_score is not None:
            if self.clip_alignment_score >= 0.7:
                insights.append("Strong text-video alignment (CLIP >= 0.7)")
            else:
                insights.append("Text-video alignment needs improvement (CLIP < 0.7)")
        
        if self.etva_semantic_accuracy is not None:
            if self.etva_semantic_accuracy > 0.5:
                insights.append("Good semantic accuracy (ETVA > 0.5)")
            else:
                insights.append("Semantic accuracy needs improvement")
        
        return insights


class RubricCriteria(BaseModel):
    """Qualitative rubric criteria for consistent evaluation"""
    criteria_name: str = Field(description="Name of the evaluation criteria")
    description: str = Field(description="Detailed description of what this criteria measures")
    examples: List[str] = Field(default_factory=list, description="Example scenarios for this criteria")
    thresholds: Dict[str, str] = Field(description="Qualitative thresholds (e.g., 'excellent', 'good', 'poor')")
    
    def get_qualitative_assessment(self, score: float) -> str:
        """Convert numeric score to qualitative assessment"""
        if score >= 0.9:
            return self.thresholds.get("excellent", "excellent")
        elif score >= 0.8:
            return self.thresholds.get("good", "good")
        elif score >= 0.7:
            return self.thresholds.get("acceptable", "acceptable")
        elif score >= 0.6:
            return self.thresholds.get("below_average", "below average")
        else:
            return self.thresholds.get("poor", "poor")


class QualitativeAssessment(BaseModel):
    """Qualitative assessment to complement numeric scores"""
    rubric_criteria: RubricCriteria
    numeric_score: float
    qualitative_score: str
    reasoning: str = Field(description="Explanation for the assessment")
    confidence: float = Field(description="Confidence in the assessment (0-1)")
    contextual_factors: List[str] = Field(default_factory=list, description="Context factors affecting assessment")
    
    def to_human_readable(self) -> str:
        """Generate human-readable assessment"""
        return f"{self.rubric_criteria.criteria_name}: {self.qualitative_score} ({self.numeric_score:.2f})\nReasoning: {self.reasoning}\nConfidence: {self.confidence:.1%}"


class TechnicalMetrics(BaseModel):
    """Technical quality assessment metrics"""
    resolution_score: float = Field(description="Resolution quality score (0-1)")
    framerate_consistency: float = Field(description="Frame rate consistency score (0-1)")
    encoding_quality: float = Field(description="Video encoding quality score (0-1)")
    duration_accuracy: float = Field(description="Duration accuracy vs request (0-1)")
    file_integrity: float = Field(description="File integrity and playability (0-1)")
    bitrate_efficiency: float = Field(description="Bitrate efficiency score (0-1)")
    compression_ratio: float = Field(description="Compression efficiency score (0-1)")
    
    # Advanced technical metrics
    motion_smoothness: Optional[float] = Field(None, description="Motion smoothness analysis")
    color_accuracy: Optional[float] = Field(None, description="Color reproduction quality")
    audio_sync: Optional[float] = Field(None, description="Audio-video synchronization")
    
    # Industry standard metrics
    industry_metrics: Optional[IndustryStandardMetrics] = Field(None, description="Industry standard evaluation metrics")
    
    # Qualitative assessments
    qualitative_assessments: List[QualitativeAssessment] = Field(default_factory=list, description="Qualitative technical assessments")
    
    @property
    def overall_technical_score(self) -> float:
        """Calculate overall technical quality score"""
        core_metrics = [
            self.resolution_score,
            self.framerate_consistency,
            self.encoding_quality,
            self.duration_accuracy,
            self.file_integrity,
            self.bitrate_efficiency
        ]
        
        # Add optional metrics if available
        optional_metrics = []
        if self.motion_smoothness is not None:
            optional_metrics.append(self.motion_smoothness)
        if self.color_accuracy is not None:
            optional_metrics.append(self.color_accuracy)
        if self.audio_sync is not None:
            optional_metrics.append(self.audio_sync)
        
        all_metrics = core_metrics + optional_metrics
        base_score = sum(all_metrics) / len(all_metrics)
        
        # Enhance with industry metrics if available
        if self.industry_metrics and self.industry_metrics.overall_industry_score is not None:
            # Blend traditional metrics with industry standards
            industry_weight = 0.4
            traditional_weight = 1 - industry_weight
            enhanced_score = (base_score * traditional_weight + 
                            self.industry_metrics.overall_industry_score * industry_weight)
            return enhanced_score
        
        return base_score
    
    def get_qualitative_summary(self) -> str:
        """Get qualitative summary of technical quality"""
        if not self.qualitative_assessments:
            return "No qualitative assessment available"
        
        # Group by quality level
        quality_groups = {}
        for assessment in self.qualitative_assessments:
            quality = assessment.qualitative_score
            if quality not in quality_groups:
                quality_groups[quality] = []
            quality_groups[quality].append(assessment.rubric_criteria.criteria_name)
        
        summary_parts = []
        for quality, criteria_list in quality_groups.items():
            summary_parts.append(f"{quality.title()}: {', '.join(criteria_list)}")
        
        return "; ".join(summary_parts)


class ContentMetrics(BaseModel):
    """Content quality assessment metrics"""
    prompt_adherence: float = Field(description="How well video matches the prompt (0-1)")
    visual_coherence: float = Field(description="Visual coherence and consistency (0-1)")
    narrative_flow: float = Field(description="Narrative flow and storytelling (0-1)")
    creativity_score: float = Field(description="Creative interpretation quality (0-1)")
    detail_richness: float = Field(description="Level of detail and richness (0-1)")
    scene_composition: float = Field(description="Scene composition quality (0-1)")
    
    # Content-specific analysis
    object_accuracy: Optional[float] = Field(None, description="Object detection accuracy")
    character_consistency: Optional[float] = Field(None, description="Character consistency")
    scene_transitions: Optional[float] = Field(None, description="Scene transition quality")
    text_readability: Optional[float] = Field(None, description="Text clarity if present")
    
    # Industry standard metrics for content evaluation
    industry_metrics: Optional[IndustryStandardMetrics] = Field(None, description="Industry standard content evaluation metrics")
    
    # Qualitative assessments
    qualitative_assessments: List[QualitativeAssessment] = Field(default_factory=list, description="Qualitative content assessments")
    
    @property
    def overall_content_score(self) -> float:
        """Calculate overall content quality score"""
        core_metrics = [
            self.prompt_adherence,
            self.visual_coherence,
            self.narrative_flow,
            self.creativity_score,
            self.detail_richness,
            self.scene_composition
        ]
        
        # Add optional metrics if available
        optional_metrics = []
        if self.object_accuracy is not None:
            optional_metrics.append(self.object_accuracy)
        if self.character_consistency is not None:
            optional_metrics.append(self.character_consistency)
        if self.scene_transitions is not None:
            optional_metrics.append(self.scene_transitions)
        if self.text_readability is not None:
            optional_metrics.append(self.text_readability)
        
        all_metrics = core_metrics + optional_metrics
        base_score = sum(all_metrics) / len(all_metrics)
        
        # Enhance with industry metrics if available
        if self.industry_metrics and self.industry_metrics.overall_industry_score is not None:
            # Blend traditional metrics with industry standards
            industry_weight = 0.4
            traditional_weight = 1 - industry_weight
            enhanced_score = (base_score * traditional_weight + 
                            self.industry_metrics.overall_industry_score * industry_weight)
            return enhanced_score
        
        return base_score
    
    def get_qualitative_summary(self) -> str:
        """Get qualitative summary of content quality"""
        if not self.qualitative_assessments:
            return "No qualitative assessment available"
        
        # Group by quality level
        quality_groups = {}
        for assessment in self.qualitative_assessments:
            quality = assessment.qualitative_score
            if quality not in quality_groups:
                quality_groups[quality] = []
            quality_groups[quality].append(assessment.rubric_criteria.criteria_name)
        
        summary_parts = []
        for quality, criteria_list in quality_groups.items():
            summary_parts.append(f"{quality.title()}: {', '.join(criteria_list)}")
        
        return "; ".join(summary_parts)


class AestheticMetrics(BaseModel):
    """Aesthetic and visual appeal metrics"""
    visual_appeal: float = Field(description="Overall visual appeal (0-1)")
    color_harmony: float = Field(description="Color harmony and palette (0-1)")
    lighting_quality: float = Field(description="Lighting quality and mood (0-1)")
    composition_balance: float = Field(description="Visual composition balance (0-1)")
    artistic_style: float = Field(description="Artistic style consistency (0-1)")
    professional_polish: float = Field(description="Professional finish quality (0-1)")
    
    # Advanced aesthetic analysis
    depth_perception: Optional[float] = Field(None, description="Depth and dimensionality")
    texture_quality: Optional[float] = Field(None, description="Texture detail and realism")
    atmosphere_mood: Optional[float] = Field(None, description="Atmosphere and mood creation")
    
    # Qualitative assessments
    qualitative_assessments: List[QualitativeAssessment] = Field(default_factory=list, description="Qualitative aesthetic assessments")
    
    @property
    def overall_aesthetic_score(self) -> float:
        """Calculate overall aesthetic quality score"""
        core_metrics = [
            self.visual_appeal,
            self.color_harmony,
            self.lighting_quality,
            self.composition_balance,
            self.artistic_style,
            self.professional_polish
        ]
        
        # Add optional metrics if available
        optional_metrics = []
        if self.depth_perception is not None:
            optional_metrics.append(self.depth_perception)
        if self.texture_quality is not None:
            optional_metrics.append(self.texture_quality)
        if self.atmosphere_mood is not None:
            optional_metrics.append(self.atmosphere_mood)
        
        all_metrics = core_metrics + optional_metrics
        return sum(all_metrics) / len(all_metrics)
    
    def get_qualitative_summary(self) -> str:
        """Get qualitative summary of aesthetic quality"""
        if not self.qualitative_assessments:
            return "No qualitative assessment available"
        
        # Group by quality level
        quality_groups = {}
        for assessment in self.qualitative_assessments:
            quality = assessment.qualitative_score
            if quality not in quality_groups:
                quality_groups[quality] = []
            quality_groups[quality].append(assessment.rubric_criteria.criteria_name)
        
        summary_parts = []
        for quality, criteria_list in quality_groups.items():
            summary_parts.append(f"{quality.title()}: {', '.join(criteria_list)}")
        
        return "; ".join(summary_parts)


class UserExperienceMetrics(BaseModel):
    """User experience and satisfaction metrics"""
    engagement_level: float = Field(description="Predicted user engagement (0-1)")
    satisfaction_prediction: float = Field(description="Predicted user satisfaction (0-1)")
    shareability_score: float = Field(description="Likelihood of sharing (0-1)")
    accessibility_score: float = Field(description="Accessibility compliance (0-1)")
    usability_rating: float = Field(description="Ease of viewing/consumption (0-1)")
    emotional_impact: float = Field(description="Emotional resonance (0-1)")
    
    # UX-specific metrics
    loading_experience: Optional[float] = Field(None, description="Loading and streaming quality")
    cross_platform_compatibility: Optional[float] = Field(None, description="Cross-platform playback")
    
    # Qualitative assessments
    qualitative_assessments: List[QualitativeAssessment] = Field(default_factory=list, description="Qualitative user experience assessments")
    
    @property
    def overall_ux_score(self) -> float:
        """Calculate overall user experience score"""
        core_metrics = [
            self.engagement_level,
            self.satisfaction_prediction,
            self.shareability_score,
            self.accessibility_score,
            self.usability_rating,
            self.emotional_impact
        ]
        
        # Add optional metrics if available
        optional_metrics = []
        if self.loading_experience is not None:
            optional_metrics.append(self.loading_experience)
        if self.cross_platform_compatibility is not None:
            optional_metrics.append(self.cross_platform_compatibility)
        
        all_metrics = core_metrics + optional_metrics
        return sum(all_metrics) / len(all_metrics)
    
    def get_qualitative_summary(self) -> str:
        """Get qualitative summary of user experience"""
        if not self.qualitative_assessments:
            return "No qualitative assessment available"
        
        # Group by quality level
        quality_groups = {}
        for assessment in self.qualitative_assessments:
            quality = assessment.qualitative_score
            if quality not in quality_groups:
                quality_groups[quality] = []
            quality_groups[quality].append(assessment.rubric_criteria.criteria_name)
        
        summary_parts = []
        for quality, criteria_list in quality_groups.items():
            summary_parts.append(f"{quality.title()}: {', '.join(criteria_list)}")
        
        return "; ".join(summary_parts)


class PerformanceMetrics(BaseModel):
    """Generation performance and efficiency metrics"""
    generation_speed: float = Field(description="Generation speed score (0-1)")
    resource_efficiency: float = Field(description="Resource utilization efficiency (0-1)")
    cost_effectiveness: float = Field(description="Cost per quality unit (0-1)")
    scalability_score: float = Field(description="Scalability potential (0-1)")
    reliability_rating: float = Field(description="Generation reliability (0-1)")
    
    # Performance details
    actual_generation_time: float = Field(description="Actual generation time in seconds")
    estimated_cost: float = Field(description="Estimated generation cost")
    resource_usage: Dict[str, float] = Field(default_factory=dict, description="Resource usage breakdown")
    
    # Qualitative assessments
    qualitative_assessments: List[QualitativeAssessment] = Field(default_factory=list, description="Qualitative performance assessments")
    
    @property
    def overall_performance_score(self) -> float:
        """Calculate overall performance score"""
        core_metrics = [
            self.generation_speed,
            self.resource_efficiency,
            self.cost_effectiveness,
            self.scalability_score,
            self.reliability_rating
        ]
        return sum(core_metrics) / len(core_metrics)
    
    def get_qualitative_summary(self) -> str:
        """Get qualitative summary of performance"""
        if not self.qualitative_assessments:
            return "No qualitative assessment available"
        
        # Group by quality level
        quality_groups = {}
        for assessment in self.qualitative_assessments:
            quality = assessment.qualitative_score
            if quality not in quality_groups:
                quality_groups[quality] = []
            quality_groups[quality].append(assessment.rubric_criteria.criteria_name)
        
        summary_parts = []
        for quality, criteria_list in quality_groups.items():
            summary_parts.append(f"{quality.title()}: {', '.join(criteria_list)}")
        
        return "; ".join(summary_parts)


class ComplianceMetrics(BaseModel):
    """Content compliance and safety metrics"""
    content_safety: float = Field(description="Content safety score (0-1)")
    policy_compliance: float = Field(description="Platform policy compliance (0-1)")
    copyright_safety: float = Field(description="Copyright safety assessment (0-1)")
    age_appropriateness: float = Field(description="Age appropriateness score (0-1)")
    cultural_sensitivity: float = Field(description="Cultural sensitivity rating (0-1)")
    
    # Compliance details
    flagged_content: List[str] = Field(default_factory=list, description="Flagged content issues")
    compliance_warnings: List[str] = Field(default_factory=list, description="Compliance warnings")
    
    # Qualitative assessments
    qualitative_assessments: List[QualitativeAssessment] = Field(default_factory=list, description="Qualitative compliance assessments")
    
    @property
    def overall_compliance_score(self) -> float:
        """Calculate overall compliance score"""
        core_metrics = [
            self.content_safety,
            self.policy_compliance,
            self.copyright_safety,
            self.age_appropriateness,
            self.cultural_sensitivity
        ]
        return sum(core_metrics) / len(core_metrics)
    
    def get_qualitative_summary(self) -> str:
        """Get qualitative summary of compliance"""
        if not self.qualitative_assessments:
            return "No qualitative assessment available"
        
        # Group by quality level
        quality_groups = {}
        for assessment in self.qualitative_assessments:
            quality = assessment.qualitative_score
            if quality not in quality_groups:
                quality_groups[quality] = []
            quality_groups[quality].append(assessment.rubric_criteria.criteria_name)
        
        summary_parts = []
        for quality, criteria_list in quality_groups.items():
            summary_parts.append(f"{quality.title()}: {', '.join(criteria_list)}")
        
        return "; ".join(summary_parts)


class QualityDimensions(BaseModel):
    """Comprehensive quality assessment across all dimensions"""
    technical: TechnicalMetrics
    content: ContentMetrics
    aesthetic: AestheticMetrics
    user_experience: UserExperienceMetrics
    performance: PerformanceMetrics
    compliance: ComplianceMetrics
    
    # Meta information
    evaluation_level: EvaluationLevel = Field(description="Level of evaluation performed")
    evaluator_version: str = Field(description="Version of evaluation system used")
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def overall_quality_score(self) -> float:
        """Calculate weighted overall quality score"""
        # Weighted importance of each dimension
        weights = {
            "technical": 0.20,      # Technical quality baseline
            "content": 0.25,        # Content is king
            "aesthetic": 0.20,      # Visual appeal matters
            "user_experience": 0.15, # User satisfaction
            "performance": 0.10,    # Efficiency
            "compliance": 0.10      # Safety and compliance
        }
        
        weighted_score = (
            self.technical.overall_technical_score * weights["technical"] +
            self.content.overall_content_score * weights["content"] +
            self.aesthetic.overall_aesthetic_score * weights["aesthetic"] +
            self.user_experience.overall_ux_score * weights["user_experience"] +
            self.performance.overall_performance_score * weights["performance"] +
            self.compliance.overall_compliance_score * weights["compliance"]
        )
        
        return weighted_score
    
    @property
    def quality_grade(self) -> QualityGrade:
        """Get human-readable quality grade"""
        score = self.overall_quality_score
        
        if score >= 0.95:
            return QualityGrade.EXCEPTIONAL
        elif score >= 0.85:
            return QualityGrade.EXCELLENT
        elif score >= 0.75:
            return QualityGrade.GOOD
        elif score >= 0.65:
            return QualityGrade.ACCEPTABLE
        elif score >= 0.50:
            return QualityGrade.POOR
        else:
            return QualityGrade.UNACCEPTABLE
    
    def get_dimension_scores(self) -> Dict[str, float]:
        """Get all dimension scores as a dictionary"""
        return {
            "technical": self.technical.overall_technical_score,
            "content": self.content.overall_content_score,
            "aesthetic": self.aesthetic.overall_aesthetic_score,
            "user_experience": self.user_experience.overall_ux_score,
            "performance": self.performance.overall_performance_score,
            "compliance": self.compliance.overall_compliance_score,
            "overall": self.overall_quality_score
        }
    
    def get_improvement_recommendations(self) -> List[str]:
        """Generate human-readable improvement recommendations"""
        recommendations = []
        dimension_scores = self.get_dimension_scores()
        
        # Find weakest dimensions (score < 0.75)
        weak_dimensions = {dim: score for dim, score in dimension_scores.items() 
                          if score < 0.75 and dim != "overall"}
        
        if weak_dimensions:
            sorted_weak = sorted(weak_dimensions.items(), key=lambda x: x[1])
            
            for dimension, score in sorted_weak:
                if dimension == "technical":
                    recommendations.append(f"Improve technical quality (current: {score:.2f}) - focus on resolution, encoding, and file integrity")
                elif dimension == "content":
                    recommendations.append(f"Enhance content quality (current: {score:.2f}) - better prompt adherence and visual coherence")
                elif dimension == "aesthetic":
                    recommendations.append(f"Refine aesthetic appeal (current: {score:.2f}) - improve composition, color harmony, and lighting")
                elif dimension == "user_experience":
                    recommendations.append(f"Boost user experience (current: {score:.2f}) - increase engagement and accessibility")
                elif dimension == "performance":
                    recommendations.append(f"Optimize performance (current: {score:.2f}) - reduce generation time and resource usage")
                elif dimension == "compliance":
                    recommendations.append(f"Address compliance issues (current: {score:.2f}) - ensure content safety and policy adherence")
        
        if not recommendations:
            recommendations.append("Excellent quality across all dimensions! Continue maintaining high standards.")
        
        return recommendations
    
    def get_qualitative_summary(self) -> Dict[str, str]:
        """Get qualitative summary across all dimensions"""
        return {
            "technical": self.technical.get_qualitative_summary(),
            "content": self.content.get_qualitative_summary(),
            "aesthetic": self.aesthetic.get_qualitative_summary(),
            "user_experience": self.user_experience.get_qualitative_summary(),
            "performance": self.performance.get_qualitative_summary(),
            "compliance": self.compliance.get_qualitative_summary()
        }
    
    def get_assessment_confidence(self) -> Dict[str, float]:
        """Get confidence levels for each dimension assessment"""
        confidence_scores = {}
        
        for dimension_name in ["technical", "content", "aesthetic", "user_experience", "performance", "compliance"]:
            dimension = getattr(self, dimension_name)
            if hasattr(dimension, 'qualitative_assessments') and dimension.qualitative_assessments:
                # Calculate average confidence from qualitative assessments
                avg_confidence = sum(assess.confidence for assess in dimension.qualitative_assessments) / len(dimension.qualitative_assessments)
                confidence_scores[dimension_name] = avg_confidence
            else:
                # Default confidence for numeric-only assessments
                confidence_scores[dimension_name] = 0.7  # Moderate confidence for numeric scores
        
        return confidence_scores


class EvaluationResult(BaseModel):
    """Complete evaluation result for a video generation"""
    
    # Identification
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = Field(description="Associated job ID")
    video_path: str = Field(description="Path to evaluated video")
    original_prompt: str = Field(description="Original generation prompt")
    
    # Quality assessment
    quality_dimensions: QualityDimensions
    
    # Evaluation metadata
    evaluation_level: EvaluationLevel
    evaluation_duration: float = Field(description="Time taken for evaluation in seconds")
    evaluator_agents: List[str] = Field(description="List of evaluator agents used")
    
    # Human feedback integration
    human_feedback: Optional[Dict[str, Any]] = Field(None, description="Human feedback if available")
    user_rating: Optional[float] = Field(None, description="User rating if provided")
    
    # Comparison data
    benchmark_comparison: Optional[Dict[str, float]] = Field(None, description="Comparison with benchmarks")
    similar_generations: Optional[List[str]] = Field(None, description="Similar generation IDs for comparison")
    
    # Improvement tracking
    previous_evaluation_id: Optional[str] = Field(None, description="Previous evaluation for tracking improvement")
    improvement_delta: Optional[Dict[str, float]] = Field(None, description="Improvement since last evaluation")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get evaluation summary for quick overview"""
        return {
            "evaluation_id": self.evaluation_id,
            "job_id": self.job_id,
            "overall_score": self.quality_dimensions.overall_quality_score,
            "quality_grade": self.quality_dimensions.quality_grade.value,
            "dimension_scores": self.quality_dimensions.get_dimension_scores(),
            "evaluation_level": self.evaluation_level.value,
            "evaluation_duration": self.evaluation_duration,
            "recommendations": self.quality_dimensions.get_improvement_recommendations()[:3],  # Top 3
            "created_at": self.created_at
        }
    
    def to_human_readable(self) -> str:
        """Generate human-readable evaluation report"""
        score = self.quality_dimensions.overall_quality_score
        grade = self.quality_dimensions.quality_grade.value
        
        report = f"""
🎬 Video Generation Evaluation Report
=====================================

📋 Basic Information:
   • Evaluation ID: {self.evaluation_id}
   • Job ID: {self.job_id}
   • Prompt: "{self.original_prompt[:100]}..."
   • Evaluation Level: {self.evaluation_level.value.title()}
   • Duration: {self.evaluation_duration:.2f} seconds

🎯 Overall Assessment:
   • Quality Score: {score:.2f}/1.00 ({score*100:.1f}%)
   • Quality Grade: {grade.upper()}
   
📊 Dimension Breakdown:
   • Technical Quality: {self.quality_dimensions.technical.overall_technical_score:.2f}
   • Content Quality: {self.quality_dimensions.content.overall_content_score:.2f}
   • Aesthetic Appeal: {self.quality_dimensions.aesthetic.overall_aesthetic_score:.2f}
   • User Experience: {self.quality_dimensions.user_experience.overall_ux_score:.2f}
   • Performance: {self.quality_dimensions.performance.overall_performance_score:.2f}
   • Compliance: {self.quality_dimensions.compliance.overall_compliance_score:.2f}

🔍 Qualitative Assessment:
"""
        
        # Add qualitative summaries for each dimension
        qualitative_summary = self.quality_dimensions.get_qualitative_summary()
        for dimension, summary in qualitative_summary.items():
            if summary != "No qualitative assessment available":
                report += f"   • {dimension.title()}: {summary}\n"
        
        # Add confidence levels
        confidence_scores = self.quality_dimensions.get_assessment_confidence()
        report += f"\n🎯 Assessment Confidence:\n"
        for dimension, confidence in confidence_scores.items():
            confidence_level = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low"
            report += f"   • {dimension.title()}: {confidence_level} ({confidence:.1%})\n"
        
        # Add industry standard insights
        industry_insights = self._get_industry_insights()
        if industry_insights:
            report += f"\n🏭 Industry Standard Insights:\n"
            for insight in industry_insights:
                report += f"   • {insight}\n"

        report += f"\n💡 Key Recommendations:\n"
        for i, rec in enumerate(self.quality_dimensions.get_improvement_recommendations()[:5], 1):
            report += f"   {i}. {rec}\n"
        
        if self.benchmark_comparison:
            report += f"\n📈 Benchmark Comparison:\n"
            for benchmark, score in self.benchmark_comparison.items():
                report += f"   • {benchmark}: {score:.2f}\n"
        
        report += f"\n⏰ Evaluated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return report
    
    def _get_industry_insights(self) -> List[str]:
        """Get industry standard insights from all dimensions"""
        insights = []
        
        # Collect insights from technical metrics
        if self.quality_dimensions.technical.industry_metrics:
            insights.extend(self.quality_dimensions.technical.industry_metrics.get_industry_insights())
        
        # Collect insights from content metrics
        if self.quality_dimensions.content.industry_metrics:
            insights.extend(self.quality_dimensions.content.industry_metrics.get_industry_insights())
        
        return insights


class BenchmarkScore(BaseModel):
    """Benchmark comparison score"""
    benchmark_name: str = Field(description="Name of the benchmark")
    benchmark_version: str = Field(description="Version of the benchmark")
    comparison_score: float = Field(description="Score compared to benchmark (0-1)")
    percentile_rank: float = Field(description="Percentile rank in benchmark dataset")
    category: str = Field(description="Benchmark category")
    
    # Detailed comparison
    strengths: List[str] = Field(default_factory=list, description="Areas where generation excels")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")
    
    # Metadata
    benchmark_date: datetime = Field(default_factory=datetime.utcnow)
    sample_size: int = Field(description="Number of samples in benchmark")


def create_mock_quality_dimensions() -> QualityDimensions:
    """Create mock quality dimensions for testing and development"""
    return QualityDimensions(
        technical=TechnicalMetrics(
            resolution_score=0.85,
            framerate_consistency=0.90,
            encoding_quality=0.88,
            duration_accuracy=0.95,
            file_integrity=1.0,
            bitrate_efficiency=0.82,
            compression_ratio=0.78
        ),
        content=ContentMetrics(
            prompt_adherence=0.92,
            visual_coherence=0.88,
            narrative_flow=0.85,
            creativity_score=0.90,
            detail_richness=0.87,
            scene_composition=0.89
        ),
        aesthetic=AestheticMetrics(
            visual_appeal=0.91,
            color_harmony=0.89,
            lighting_quality=0.87,
            composition_balance=0.90,
            artistic_style=0.85,
            professional_polish=0.88
        ),
        user_experience=UserExperienceMetrics(
            engagement_level=0.89,
            satisfaction_prediction=0.91,
            shareability_score=0.87,
            accessibility_score=0.92,
            usability_rating=0.90,
            emotional_impact=0.85
        ),
        performance=PerformanceMetrics(
            generation_speed=0.83,
            resource_efficiency=0.78,
            cost_effectiveness=0.85,
            scalability_score=0.80,
            reliability_rating=0.95,
            actual_generation_time=45.0,
            estimated_cost=0.50,
            resource_usage={"gpu": 0.75, "memory": 0.68, "cpu": 0.45}
        ),
        compliance=ComplianceMetrics(
            content_safety=1.0,
            policy_compliance=1.0,
            copyright_safety=0.95,
            age_appropriateness=1.0,
            cultural_sensitivity=0.92
        ),
        evaluation_level=EvaluationLevel.COMPREHENSIVE,
        evaluator_version="v2024.1.0"
    )


# Predefined rubric criteria for common video generation evaluation scenarios
def get_default_video_quality_rubrics() -> Dict[str, RubricCriteria]:
    """Get default rubric criteria for video quality evaluation"""
    return {
        "prompt_adherence": RubricCriteria(
            criteria_name="Prompt Adherence",
            description="How well the generated video matches the user's prompt and requirements",
            examples=[
                "Video shows exactly what was requested in the prompt",
                "Video captures the essence but misses some specific details",
                "Video is related but doesn't match the prompt well"
            ],
            thresholds={
                "excellent": "Video perfectly matches the prompt with all requested elements",
                "good": "Video captures the main elements of the prompt with minor variations",
                "acceptable": "Video is related to the prompt but has significant differences",
                "below_average": "Video has some connection to the prompt but misses key elements",
                "poor": "Video bears little resemblance to the requested prompt"
            }
        ),
        "visual_coherence": RubricCriteria(
            criteria_name="Visual Coherence",
            description="Consistency and logical flow of visual elements throughout the video",
            examples=[
                "All scenes flow naturally and maintain visual consistency",
                "Most scenes are coherent with occasional inconsistencies",
                "Visual elements change abruptly without logical progression"
            ],
            thresholds={
                "excellent": "Perfect visual consistency and logical scene progression",
                "good": "Strong visual coherence with minor inconsistencies",
                "acceptable": "Generally coherent with some visual jumps",
                "below_average": "Moderate coherence issues affecting viewing experience",
                "poor": "Severe visual inconsistencies and poor scene flow"
            }
        ),
        "technical_quality": RubricCriteria(
            criteria_name="Technical Quality",
            description="Resolution, encoding, and playback quality of the video",
            examples=[
                "High resolution with smooth playback and no artifacts",
                "Good quality with minor compression artifacts",
                "Low resolution or significant playback issues"
            ],
            thresholds={
                "excellent": "Professional-grade technical quality with no visible issues",
                "good": "High-quality video with minimal technical problems",
                "acceptable": "Decent quality with some noticeable technical issues",
                "below_average": "Quality issues that affect viewing experience",
                "poor": "Significant technical problems making video hard to watch"
            }
        )
    }


def create_qualitative_assessment(
    rubric: RubricCriteria,
    numeric_score: float,
    reasoning: str,
    confidence: float = 0.8,
    contextual_factors: List[str] = None
) -> QualitativeAssessment:
    """Create a qualitative assessment from rubric criteria and numeric score"""
    if contextual_factors is None:
        contextual_factors = []
    
    qualitative_score = rubric.get_qualitative_assessment(numeric_score)
    
    return QualitativeAssessment(
        rubric_criteria=rubric,
        numeric_score=numeric_score,
        qualitative_score=qualitative_score,
        reasoning=reasoning,
        confidence=confidence,
        contextual_factors=contextual_factors
    )


# Utility functions for working with quality metrics
def compare_evaluations(eval1: EvaluationResult, eval2: EvaluationResult) -> Dict[str, float]:
    """Compare two evaluations and return improvement deltas"""
    dims1 = eval1.quality_dimensions.get_dimension_scores()
    dims2 = eval2.quality_dimensions.get_dimension_scores()
    
    return {
        dimension: dims2[dimension] - dims1[dimension]
        for dimension in dims1.keys()
    }


def aggregate_evaluations(evaluations: List[EvaluationResult]) -> Dict[str, float]:
    """Aggregate multiple evaluations to get average scores"""
    if not evaluations:
        return {}
    
    all_scores = [eval_result.quality_dimensions.get_dimension_scores() 
                  for eval_result in evaluations]
    
    # Calculate averages
    aggregated = {}
    for dimension in all_scores[0].keys():
        aggregated[dimension] = sum(scores[dimension] for scores in all_scores) / len(all_scores)
    
    return aggregated


def validate_evaluation_consistency(evaluations: List[EvaluationResult]) -> Dict[str, Any]:
    """Validate consistency across multiple evaluations to detect LLM scoring inconsistencies"""
    if len(evaluations) < 2:
        return {"status": "insufficient_data", "message": "Need at least 2 evaluations for consistency check"}
    
    # Calculate standard deviation for each dimension
    dimension_scores = {}
    for dimension in ["technical", "content", "aesthetic", "user_experience", "performance", "compliance"]:
        scores = []
        for eval_result in evaluations:
            dim_scores = eval_result.quality_dimensions.get_dimension_scores()
            if dimension in dim_scores:
                scores.append(dim_scores[dimension])
        
        if scores:
            import statistics
            mean_score = statistics.mean(scores)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
            coefficient_of_variation = (std_dev / mean_score) if mean_score > 0 else 0
            
            dimension_scores[dimension] = {
                "mean": mean_score,
                "std_dev": std_dev,
                "coefficient_of_variation": coefficient_of_variation,
                "consistency_level": "high" if coefficient_of_variation < 0.1 else "medium" if coefficient_of_variation < 0.2 else "low"
            }
    
    # Overall consistency assessment
    consistency_scores = [dim_data["consistency_level"] for dim_data in dimension_scores.values()]
    overall_consistency = "high" if consistency_scores.count("high") >= len(consistency_scores) * 0.7 else "medium" if consistency_scores.count("low") < len(consistency_scores) * 0.3 else "low"
    
    return {
        "status": "success",
        "overall_consistency": overall_consistency,
        "dimension_consistency": dimension_scores,
        "recommendations": _generate_consistency_recommendations(overall_consistency, dimension_scores)
    }


def _generate_consistency_recommendations(overall_consistency: str, dimension_consistency: Dict[str, Any]) -> List[str]:
    """Generate recommendations for improving evaluation consistency"""
    recommendations = []
    
    if overall_consistency == "low":
        recommendations.append("Consider implementing more structured evaluation prompts with specific criteria")
        recommendations.append("Review and standardize evaluation rubrics across all dimensions")
        recommendations.append("Implement human review for inconsistent evaluation results")
    
    # Specific dimension recommendations
    for dimension, data in dimension_consistency.items():
        if data["consistency_level"] == "low":
            recommendations.append(f"Improve consistency in {dimension} evaluations - consider more specific criteria")
    
    if not recommendations:
        recommendations.append("Evaluation consistency is good - continue current practices")
    
    return recommendations
