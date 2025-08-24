"""
Continuous Learning and Improvement System for Autoraters and Autoevals
Human-in-the-loop feedback integration and adaptive evaluation improvement
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from enum import Enum
import numpy as np

from pydantic import BaseModel, Field

from .quality_metrics import EvaluationResult, QualityDimensions
from .benchmarks import BenchmarkComparison
from ...shared.config import get_settings
from ...shared.monitoring import get_logger
from ...shared.database import DatabaseManager

logger = get_logger("continuous_learning")
settings = get_settings()


class FeedbackType(str, Enum):
    """Types of feedback for continuous learning"""
    USER_RATING = "user_rating"
    HUMAN_EXPERT = "human_expert"
    A_B_TEST = "a_b_test"
    PRODUCTION_METRICS = "production_metrics"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


class FeedbackSource(str, Enum):
    """Sources of feedback"""
    END_USER = "end_user"
    CONTENT_CREATOR = "content_creator"
    QUALITY_EXPERT = "quality_expert"
    AUTOMATED_SYSTEM = "automated_system"
    PRODUCTION_ANALYTICS = "production_analytics"


class LearningPriority(str, Enum):
    """Priority levels for learning improvements"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FeedbackRecord(BaseModel):
    """Individual feedback record for learning"""
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evaluation_id: str = Field(description="Associated evaluation ID")
    job_id: str = Field(description="Associated job ID")
    
    # Feedback details
    feedback_type: FeedbackType = Field(description="Type of feedback")
    feedback_source: FeedbackSource = Field(description="Source of feedback")
    
    # Rating and assessment
    overall_rating: Optional[float] = Field(None, description="Overall user rating (0-1)")
    dimension_ratings: Dict[str, float] = Field(default_factory=dict, description="Ratings for each dimension")
    
    # Qualitative feedback
    comments: Optional[str] = Field(None, description="Written feedback comments")
    specific_issues: List[str] = Field(default_factory=list, description="Specific issues identified")
    positive_aspects: List[str] = Field(default_factory=list, description="Positive aspects highlighted")
    
    # Correction data
    expected_scores: Optional[Dict[str, float]] = Field(None, description="Expert-corrected scores")
    evaluation_discrepancies: Optional[Dict[str, float]] = Field(None, description="Discrepancies from evaluation")
    
    # Context
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context information")
    prompt_category: Optional[str] = Field(None, description="Category of the original prompt")
    complexity_level: Optional[str] = Field(None, description="Complexity level of the request")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = Field(default=False, description="Whether feedback has been processed")
    integration_notes: Optional[str] = Field(None, description="Notes on how feedback was integrated")


class LearningInsight(BaseModel):
    """Insight derived from continuous learning analysis"""
    insight_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: str = Field(description="Type of insight (bias, calibration, gap, etc.)")
    
    # Insight details
    title: str = Field(description="Human-readable insight title")
    description: str = Field(description="Detailed description of the insight")
    
    # Impact assessment
    affected_dimensions: List[str] = Field(description="Quality dimensions affected")
    confidence_level: float = Field(description="Confidence in this insight (0-1)")
    potential_impact: float = Field(description="Potential impact of addressing this (0-1)")
    
    # Supporting data
    sample_size: int = Field(description="Number of data points supporting this insight")
    statistical_significance: float = Field(description="Statistical significance level")
    
    # Recommendations
    recommended_actions: List[str] = Field(description="Recommended actions to address this insight")
    priority: LearningPriority = Field(description="Priority level for addressing this insight")
    
    # Implementation tracking
    implementation_status: str = Field(default="identified", description="Status of implementing fixes")
    estimated_effort: Optional[str] = Field(None, description="Estimated effort to implement")
    
    # Metadata
    discovered_date: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class CalibrationAnalysis(BaseModel):
    """Analysis of evaluator calibration with human feedback"""
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timeframe: str = Field(description="Timeframe of analysis")
    
    # Calibration metrics
    overall_calibration: float = Field(description="Overall calibration score (0-1, 1=perfect)")
    dimension_calibration: Dict[str, float] = Field(description="Calibration for each dimension")
    
    # Bias analysis
    systematic_bias: Dict[str, float] = Field(description="Systematic bias in each dimension")
    bias_direction: Dict[str, str] = Field(description="Direction of bias (over/under-estimate)")
    
    # Accuracy metrics
    mean_absolute_error: Dict[str, float] = Field(description="MAE for each dimension")
    root_mean_square_error: Dict[str, float] = Field(description="RMSE for each dimension")
    correlation_with_human: Dict[str, float] = Field(description="Correlation with human ratings")
    
    # Confidence analysis
    confidence_accuracy: float = Field(description="How well confidence predicts accuracy")
    overconfidence_bias: float = Field(description="Tendency to be overconfident")
    
    # Sample statistics
    feedback_count: int = Field(description="Number of feedback samples analyzed")
    coverage: Dict[str, int] = Field(description="Coverage across different categories")
    
    # Improvement recommendations
    calibration_improvements: List[str] = Field(description="Specific calibration improvements needed")
    training_recommendations: List[str] = Field(description="Training data improvements needed")


class LearningService:
    """Service for continuous learning and evaluation improvement"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.feedback_buffer: List[FeedbackRecord] = []
        self.learning_insights: List[LearningInsight] = []
        self.calibration_history: List[CalibrationAnalysis] = []
        
    async def collect_user_feedback(
        self,
        evaluation_id: str,
        job_id: str,
        user_rating: float,
        comments: Optional[str] = None,
        dimension_ratings: Optional[Dict[str, float]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> FeedbackRecord:
        """Collect feedback from end users"""
        
        feedback = FeedbackRecord(
            evaluation_id=evaluation_id,
            job_id=job_id,
            feedback_type=FeedbackType.USER_RATING,
            feedback_source=FeedbackSource.END_USER,
            overall_rating=user_rating,
            dimension_ratings=dimension_ratings or {},
            comments=comments,
            user_context=user_context
        )
        
        self.feedback_buffer.append(feedback)
        
        logger.info(f"Collected user feedback for evaluation {evaluation_id}: rating {user_rating}")
        
        return feedback
    
    async def collect_expert_feedback(
        self,
        evaluation_id: str,
        job_id: str,
        expert_scores: Dict[str, float],
        expert_comments: str,
        specific_issues: List[str] = None,
        positive_aspects: List[str] = None
    ) -> FeedbackRecord:
        """Collect feedback from human experts"""
        
        feedback = FeedbackRecord(
            evaluation_id=evaluation_id,
            job_id=job_id,
            feedback_type=FeedbackType.HUMAN_EXPERT,
            feedback_source=FeedbackSource.QUALITY_EXPERT,
            expected_scores=expert_scores,
            comments=expert_comments,
            specific_issues=specific_issues or [],
            positive_aspects=positive_aspects or []
        )
        
        self.feedback_buffer.append(feedback)
        
        logger.info(f"Collected expert feedback for evaluation {evaluation_id}")
        
        return feedback
    
    async def process_feedback_batch(self) -> List[LearningInsight]:
        """Process accumulated feedback to generate learning insights"""
        
        if len(self.feedback_buffer) < 10:  # Need minimum sample size
            logger.info("Insufficient feedback for batch processing")
            return []
        
        logger.info(f"Processing feedback batch of {len(self.feedback_buffer)} records")
        
        insights = []
        
        # Analyze different types of insights
        calibration_insights = await self._analyze_calibration_insights()
        bias_insights = await self._analyze_bias_insights()
        gap_insights = await self._analyze_performance_gaps()
        trend_insights = await self._analyze_feedback_trends()
        
        insights.extend(calibration_insights)
        insights.extend(bias_insights)
        insights.extend(gap_insights)
        insights.extend(trend_insights)
        
        # Mark feedback as processed
        for feedback in self.feedback_buffer:
            feedback.processed = True
        
        # Store insights
        self.learning_insights.extend(insights)
        
        # Clear processed feedback
        self.feedback_buffer = [f for f in self.feedback_buffer if not f.processed]
        
        logger.info(f"Generated {len(insights)} new learning insights")
        
        return insights
    
    async def analyze_evaluator_calibration(
        self,
        timeframe_days: int = 30
    ) -> CalibrationAnalysis:
        """Analyze how well the evaluator is calibrated with human feedback"""
        
        logger.info(f"Analyzing evaluator calibration over {timeframe_days} days")
        
        # Get feedback from the specified timeframe
        cutoff_date = datetime.utcnow() - timedelta(days=timeframe_days)
        recent_feedback = [f for f in self.feedback_buffer + 
                          [f for f in self.learning_insights if hasattr(f, 'timestamp')] 
                          if hasattr(f, 'timestamp') and f.timestamp > cutoff_date]
        
        # Perform calibration analysis
        calibration_analysis = await self._perform_calibration_analysis(recent_feedback)
        
        self.calibration_history.append(calibration_analysis)
        
        return calibration_analysis
    
    async def get_improvement_recommendations(
        self,
        priority_filter: Optional[LearningPriority] = None
    ) -> List[Dict[str, Any]]:
        """Get prioritized improvement recommendations"""
        
        insights = self.learning_insights
        
        if priority_filter:
            insights = [i for i in insights if i.priority == priority_filter]
        
        # Sort by potential impact and confidence
        insights.sort(key=lambda x: x.potential_impact * x.confidence_level, reverse=True)
        
        recommendations = []
        
        for insight in insights[:10]:  # Top 10 recommendations
            recommendation = {
                "insight_id": insight.insight_id,
                "title": insight.title,
                "description": insight.description,
                "priority": insight.priority.value,
                "potential_impact": insight.potential_impact,
                "confidence": insight.confidence_level,
                "affected_dimensions": insight.affected_dimensions,
                "recommended_actions": insight.recommended_actions,
                "implementation_status": insight.implementation_status,
                "estimated_effort": insight.estimated_effort
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    async def implement_improvement(
        self,
        insight_id: str,
        implementation_notes: str,
        effectiveness_score: Optional[float] = None
    ) -> bool:
        """Record implementation of an improvement based on learning insight"""
        
        insight = next((i for i in self.learning_insights if i.insight_id == insight_id), None)
        
        if not insight:
            logger.error(f"Insight {insight_id} not found")
            return False
        
        insight.implementation_status = "implemented"
        insight.integration_notes = implementation_notes
        insight.last_updated = datetime.utcnow()
        
        logger.info(f"Implemented improvement for insight: {insight.title}")
        
        return True
    
    async def _analyze_calibration_insights(self) -> List[LearningInsight]:
        """Analyze calibration-related insights"""
        
        insights = []
        
        # Example calibration insight
        if len(self.feedback_buffer) > 20:
            # Mock analysis - in production, this would analyze actual feedback data
            calibration_insight = LearningInsight(
                insight_type="calibration_bias",
                title="Systematic Over-estimation in Aesthetic Quality",
                description="The evaluator consistently rates aesthetic quality 15% higher than human experts, particularly for artistic and creative content.",
                affected_dimensions=["aesthetic"],
                confidence_level=0.85,
                potential_impact=0.7,
                sample_size=len(self.feedback_buffer),
                statistical_significance=0.02,
                recommended_actions=[
                    "Recalibrate aesthetic evaluation weights based on expert feedback",
                    "Increase sample diversity in aesthetic training data",
                    "Implement human-in-the-loop validation for aesthetic assessments"
                ],
                priority=LearningPriority.HIGH
            )
            insights.append(calibration_insight)
        
        return insights
    
    async def _analyze_bias_insights(self) -> List[LearningInsight]:
        """Analyze bias-related insights"""
        
        insights = []
        
        # Example bias insight
        bias_insight = LearningInsight(
            insight_type="demographic_bias",
            title="Content Bias Toward Western Cultural References",
            description="The evaluator shows higher content scores for prompts with Western cultural references compared to non-Western content of similar quality.",
            affected_dimensions=["content", "user_experience"],
            confidence_level=0.78,
            potential_impact=0.8,
            sample_size=45,
            statistical_significance=0.03,
            recommended_actions=[
                "Expand training data to include more diverse cultural content",
                "Implement cultural sensitivity scoring",
                "Add cultural diversity metrics to evaluation framework"
            ],
            priority=LearningPriority.HIGH
        )
        insights.append(bias_insight)
        
        return insights
    
    async def _analyze_performance_gaps(self) -> List[LearningInsight]:
        """Analyze performance gap insights"""
        
        insights = []
        
        # Example performance gap insight
        gap_insight = LearningInsight(
            insight_type="performance_gap",
            title="Technical Quality Assessment Gap for Complex Scenes",
            description="Technical quality evaluation accuracy drops by 25% for scenes with complex motion or multiple objects compared to simple scenes.",
            affected_dimensions=["technical"],
            confidence_level=0.82,
            potential_impact=0.6,
            sample_size=67,
            statistical_significance=0.01,
            recommended_actions=[
                "Enhance frame-level analysis for complex motion",
                "Implement object-aware technical quality metrics",
                "Add scene complexity factor to technical evaluation"
            ],
            priority=LearningPriority.MEDIUM
        )
        insights.append(gap_insight)
        
        return insights
    
    async def _analyze_feedback_trends(self) -> List[LearningInsight]:
        """Analyze trending feedback patterns"""
        
        insights = []
        
        # Example trend insight
        trend_insight = LearningInsight(
            insight_type="feedback_trend",
            title="Increasing User Expectations for Video Length",
            description="User satisfaction scores are declining for videos under 10 seconds, indicating shifting expectations for content duration.",
            affected_dimensions=["user_experience", "content"],
            confidence_level=0.75,
            potential_impact=0.5,
            sample_size=120,
            statistical_significance=0.05,
            recommended_actions=[
                "Adjust duration expectations in user experience scoring",
                "Consider video length in overall quality weighting",
                "Provide duration guidance in user interface"
            ],
            priority=LearningPriority.MEDIUM
        )
        insights.append(trend_insight)
        
        return insights
    
    async def _perform_calibration_analysis(
        self,
        feedback_data: List[Any]
    ) -> CalibrationAnalysis:
        """Perform detailed calibration analysis"""
        
        # Mock calibration analysis
        calibration_analysis = CalibrationAnalysis(
            timeframe="last_30_days",
            overall_calibration=0.82,
            dimension_calibration={
                "technical": 0.88,
                "content": 0.79,
                "aesthetic": 0.76,
                "user_experience": 0.84,
                "performance": 0.91,
                "compliance": 0.95
            },
            systematic_bias={
                "technical": 0.02,    # 2% over-estimation
                "content": -0.05,     # 5% under-estimation
                "aesthetic": 0.08,    # 8% over-estimation
                "user_experience": 0.01,
                "performance": -0.02,
                "compliance": 0.00
            },
            bias_direction={
                "technical": "over-estimate",
                "content": "under-estimate",
                "aesthetic": "over-estimate",
                "user_experience": "over-estimate",
                "performance": "under-estimate",
                "compliance": "well-calibrated"
            },
            mean_absolute_error={
                "technical": 0.08,
                "content": 0.12,
                "aesthetic": 0.15,
                "user_experience": 0.09,
                "performance": 0.06,
                "compliance": 0.03
            },
            root_mean_square_error={
                "technical": 0.11,
                "content": 0.16,
                "aesthetic": 0.19,
                "user_experience": 0.12,
                "performance": 0.08,
                "compliance": 0.04
            },
            correlation_with_human={
                "technical": 0.85,
                "content": 0.78,
                "aesthetic": 0.71,
                "user_experience": 0.82,
                "performance": 0.89,
                "compliance": 0.94
            },
            confidence_accuracy=0.77,
            overconfidence_bias=0.12,
            feedback_count=len(feedback_data),
            coverage={
                "simple": 45,
                "medium": 67,
                "complex": 23,
                "expert": 8
            },
            calibration_improvements=[
                "Reduce aesthetic over-estimation through additional training data",
                "Address content under-estimation with prompt-specific calibration",
                "Improve complex scene technical assessment accuracy"
            ],
            training_recommendations=[
                "Increase diversity in aesthetic training examples",
                "Add more expert-validated content assessments",
                "Include edge cases and challenging scenarios in training"
            ]
        )
        
        return calibration_analysis
    
    async def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning and improvement report"""
        
        recent_insights = [i for i in self.learning_insights 
                          if i.discovered_date > datetime.utcnow() - timedelta(days=30)]
        
        report = {
            "summary": {
                "total_feedback_collected": len([f for f in self.feedback_buffer if f.processed]) + len(self.learning_insights),
                "active_insights": len([i for i in self.learning_insights if i.implementation_status != "implemented"]),
                "recent_insights": len(recent_insights),
                "calibration_score": self.calibration_history[-1].overall_calibration if self.calibration_history else 0.8
            },
            "key_insights": [
                {
                    "title": insight.title,
                    "priority": insight.priority.value,
                    "potential_impact": insight.potential_impact,
                    "confidence": insight.confidence_level,
                    "implementation_status": insight.implementation_status
                }
                for insight in sorted(recent_insights, key=lambda x: x.potential_impact, reverse=True)[:5]
            ],
            "calibration_status": {
                "overall_calibration": self.calibration_history[-1].overall_calibration if self.calibration_history else 0.8,
                "dimensions_needing_attention": [
                    dim for dim, score in (self.calibration_history[-1].dimension_calibration.items() 
                                         if self.calibration_history else {}).items() 
                    if score < 0.8
                ],
                "systematic_biases": self.calibration_history[-1].systematic_bias if self.calibration_history else {}
            },
            "improvement_trajectory": {
                "implemented_improvements": len([i for i in self.learning_insights if i.implementation_status == "implemented"]),
                "pending_high_priority": len([i for i in self.learning_insights 
                                            if i.priority == LearningPriority.HIGH and i.implementation_status != "implemented"]),
                "estimated_impact": sum([i.potential_impact for i in self.learning_insights 
                                       if i.implementation_status != "implemented"])
            },
            "recommendations": await self.get_improvement_recommendations(LearningPriority.HIGH)
        }
        
        return report
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        
        all_feedback = self.feedback_buffer + [getattr(i, 'feedback_data', {}) for i in self.learning_insights if hasattr(i, 'feedback_data')]
        
        stats = {
            "total_feedback_records": len(all_feedback),
            "feedback_by_type": {},
            "feedback_by_source": {},
            "average_ratings": {},
            "feedback_frequency": {
                "daily_average": len(all_feedback) / 30,  # Assume 30-day window
                "recent_trend": "increasing"  # Mock trend
            }
        }
        
        # Calculate statistics
        for feedback in self.feedback_buffer:
            # Count by type
            feedback_type = feedback.feedback_type.value
            stats["feedback_by_type"][feedback_type] = stats["feedback_by_type"].get(feedback_type, 0) + 1
            
            # Count by source
            feedback_source = feedback.feedback_source.value
            stats["feedback_by_source"][feedback_source] = stats["feedback_by_source"].get(feedback_source, 0) + 1
            
            # Collect ratings
            if feedback.overall_rating is not None:
                if "overall" not in stats["average_ratings"]:
                    stats["average_ratings"]["overall"] = []
                stats["average_ratings"]["overall"].append(feedback.overall_rating)
        
        # Calculate averages
        for dimension, ratings in stats["average_ratings"].items():
            stats["average_ratings"][dimension] = sum(ratings) / len(ratings) if ratings else 0.0
        
        return stats


# Singleton instance
_learning_service = None

def get_learning_service() -> LearningService:
    """Get or create learning service instance"""
    global _learning_service
    if _learning_service is None:
        _learning_service = LearningService()
    return _learning_service
