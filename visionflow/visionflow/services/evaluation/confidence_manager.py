"""
Confidence Management System for Video Evaluation
Implements automated flagging and decision making based on confidence levels.

Based on slide 14: Confidence Management with 5 ranges and automated decisions.
"""

import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

from ...shared.monitoring import get_logger
from .video_evaluation_orchestrator import EvaluationResult, ConfidenceLevel

logger = get_logger(__name__)

class ReviewPriority(str, Enum):
    """Review priority levels"""
    NONE = "none"
    MONITOR = "monitor"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"

class ActionType(str, Enum):
    """Types of actions taken by confidence manager"""
    AUTO_APPROVE = "auto_approve"
    FLAG_MONITORING = "flag_monitoring"
    FLAG_REVIEW = "flag_review"
    QUEUE_REVIEW = "queue_review"
    IMMEDIATE_REVIEW = "immediate_review"

@dataclass
class ConfidenceAction:
    """Action taken based on confidence assessment"""
    evaluation_id: str
    action_type: ActionType
    confidence_level: ConfidenceLevel
    confidence_score: float
    review_priority: ReviewPriority
    requires_human_review: bool
    timestamp: datetime
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReviewQueueItem:
    """Item in the human review queue"""
    evaluation_id: str
    evaluation_result: EvaluationResult
    priority: ReviewPriority
    queued_at: datetime
    assigned_reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_outcome: Optional[str] = None
    review_notes: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance tracking for continuous learning"""
    total_evaluations: int
    auto_approved: int
    flagged_for_review: int
    immediate_reviews: int
    low_confidence_rate: float
    high_automation_rate: float
    avg_confidence: float
    period_start: datetime
    period_end: datetime

class ConfidenceManager:
    """
    Manages confidence-based decision making and automated flagging
    Implements the confidence management workflow from slide 14
    """
    
    def __init__(self, 
                 monitoring_window_hours: int = 24,
                 low_confidence_threshold: float = 0.3,  # 30% low confidence rate triggers retraining
                 high_automation_threshold: float = 0.9): # 90% automation rate triggers quality gates
        
        self.monitoring_window_hours = monitoring_window_hours
        self.low_confidence_threshold = low_confidence_threshold
        self.high_automation_threshold = high_automation_threshold
        
        # Action history for performance tracking
        self.action_history: deque = deque(maxlen=10000)
        
        # Review queues by priority
        self.review_queues: Dict[ReviewPriority, List[ReviewQueueItem]] = {
            priority: [] for priority in ReviewPriority
        }
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        
        # Confidence thresholds (from slide 14)
        self.confidence_config = {
            ConfidenceLevel.EXCELLENT: {
                "min_score": 0.9,
                "action": ActionType.AUTO_APPROVE,
                "priority": ReviewPriority.NONE,
                "requires_review": False,
                "description": "No review needed"
            },
            ConfidenceLevel.HIGH: {
                "min_score": 0.8,
                "action": ActionType.AUTO_APPROVE,  # Can be FLAG_MONITORING based on exact score
                "priority": ReviewPriority.MONITOR,
                "requires_review": False,
                "description": "Conditional approval"
            },
            ConfidenceLevel.MEDIUM: {
                "min_score": 0.6,
                "action": ActionType.FLAG_REVIEW,
                "priority": ReviewPriority.MEDIUM,
                "requires_review": True,
                "description": "Human review required"
            },
            ConfidenceLevel.LOW: {
                "min_score": 0.4,
                "action": ActionType.QUEUE_REVIEW,
                "priority": ReviewPriority.HIGH,
                "requires_review": True,
                "description": "High priority review"
            },
            ConfidenceLevel.CRITICAL: {
                "min_score": 0.0,
                "action": ActionType.IMMEDIATE_REVIEW,
                "priority": ReviewPriority.CRITICAL,
                "requires_review": True,
                "description": "Highest priority review"
            }
        }
        
        logger.info(f"🎯 Confidence Manager initialized")
        logger.info(f"   Monitoring window: {monitoring_window_hours}h")
        logger.info(f"   Low confidence threshold: {low_confidence_threshold}")
        logger.info(f"   High automation threshold: {high_automation_threshold}")
    
    async def process_evaluation(self, evaluation_result: EvaluationResult) -> ConfidenceAction:
        """
        Process an evaluation result and make confidence-based decisions
        Implements the decision tree from slide 14
        """
        logger.debug(f"🔍 Processing evaluation confidence: {evaluation_result.evaluation_id}")
        
        confidence_level = evaluation_result.confidence_level
        confidence_score = evaluation_result.overall_confidence
        
        # Get base configuration for this confidence level
        config = self.confidence_config[confidence_level]
        
        # Determine specific action (may override base config)
        action_type, priority, requires_review, reason = self._determine_action(
            confidence_level, confidence_score, evaluation_result
        )
        
        # Create confidence action
        action = ConfidenceAction(
            evaluation_id=evaluation_result.evaluation_id,
            action_type=action_type,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            review_priority=priority,
            requires_human_review=requires_review,
            timestamp=datetime.utcnow(),
            reason=reason,
            metadata={
                "overall_score": evaluation_result.overall_score,
                "frames_evaluated": evaluation_result.frames_evaluated,
                "evaluation_time": evaluation_result.evaluation_time
            }
        )
        
        # Log the action
        self.action_history.append(action)
        
        # Add to review queue if needed
        if requires_review:
            await self._add_to_review_queue(evaluation_result, priority)
        
        # Log decision
        logger.info(f"📋 Confidence decision: {evaluation_result.evaluation_id}")
        logger.info(f"   Confidence: {confidence_score:.3f} ({confidence_level})")
        logger.info(f"   Action: {action_type}")
        logger.info(f"   Priority: {priority}")
        logger.info(f"   Review needed: {requires_review}")
        
        return action
    
    def _determine_action(self, 
                         confidence_level: ConfidenceLevel, 
                         confidence_score: float,
                         evaluation_result: EvaluationResult) -> Tuple[ActionType, ReviewPriority, bool, str]:
        """
        Determine specific action based on confidence level and score
        Implements fine-grained decision logic
        """
        
        if confidence_level == ConfidenceLevel.EXCELLENT:
            return (ActionType.AUTO_APPROVE, ReviewPriority.NONE, False, 
                   "High confidence - automatic approval")
        
        elif confidence_level == ConfidenceLevel.HIGH:
            # Conditional approval based on exact score (slide 14)
            if confidence_score >= 0.85:
                return (ActionType.AUTO_APPROVE, ReviewPriority.MONITOR, False,
                       "High confidence - auto approve with monitoring")
            else:
                return (ActionType.FLAG_MONITORING, ReviewPriority.LOW, False,
                       "Moderate high confidence - flag for monitoring")
        
        elif confidence_level == ConfidenceLevel.MEDIUM:
            # Check if any critical dimensions are very low
            critical_dimensions = [
                score for score in evaluation_result.dimension_scores 
                if score.score < 0.4
            ]
            if critical_dimensions:
                return (ActionType.QUEUE_REVIEW, ReviewPriority.HIGH, True,
                       f"Medium confidence with {len(critical_dimensions)} critical dimensions")
            else:
                return (ActionType.FLAG_REVIEW, ReviewPriority.MEDIUM, True,
                       "Medium confidence - standard review")
        
        elif confidence_level == ConfidenceLevel.LOW:
            return (ActionType.QUEUE_REVIEW, ReviewPriority.HIGH, True,
                   "Low confidence - high priority review")
        
        else:  # CRITICAL
            return (ActionType.IMMEDIATE_REVIEW, ReviewPriority.CRITICAL, True,
                   "Critical confidence - immediate review required")
    
    async def _add_to_review_queue(self, evaluation_result: EvaluationResult, priority: ReviewPriority):
        """Add evaluation to appropriate review queue"""
        review_item = ReviewQueueItem(
            evaluation_id=evaluation_result.evaluation_id,
            evaluation_result=evaluation_result,
            priority=priority,
            queued_at=datetime.utcnow()
        )
        
        self.review_queues[priority].append(review_item)
        
        logger.info(f"📋 Added to {priority} review queue: {evaluation_result.evaluation_id}")
        logger.info(f"   Queue size: {len(self.review_queues[priority])}")
    
    async def get_review_queue(self, priority: Optional[ReviewPriority] = None) -> List[ReviewQueueItem]:
        """Get items from review queue(s)"""
        if priority:
            return self.review_queues[priority].copy()
        
        # Return all items sorted by priority
        all_items = []
        priority_order = [ReviewPriority.CRITICAL, ReviewPriority.HIGH, 
                         ReviewPriority.MEDIUM, ReviewPriority.LOW]
        
        for prio in priority_order:
            all_items.extend(self.review_queues[prio])
        
        return all_items
    
    async def complete_review(self, 
                            evaluation_id: str, 
                            reviewer: str,
                            outcome: str,
                            notes: Optional[str] = None) -> bool:
        """Mark a review as completed"""
        
        for priority_queue in self.review_queues.values():
            for item in priority_queue:
                if item.evaluation_id == evaluation_id:
                    item.assigned_reviewer = reviewer
                    item.reviewed_at = datetime.utcnow()
                    item.review_outcome = outcome
                    item.review_notes = notes
                    
                    logger.info(f"✅ Review completed: {evaluation_id}")
                    logger.info(f"   Reviewer: {reviewer}")
                    logger.info(f"   Outcome: {outcome}")
                    
                    return True
        
        logger.warning(f"⚠️ Review item not found: {evaluation_id}")
        return False
    
    async def get_performance_metrics(self, hours: Optional[int] = None) -> PerformanceMetrics:
        """
        Calculate performance metrics for continuous learning
        Implements metrics tracking from slide 13
        """
        if hours is None:
            hours = self.monitoring_window_hours
            
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_actions = [
            action for action in self.action_history 
            if action.timestamp >= cutoff_time
        ]
        
        if not recent_actions:
            return PerformanceMetrics(
                total_evaluations=0,
                auto_approved=0,
                flagged_for_review=0,
                immediate_reviews=0,
                low_confidence_rate=0.0,
                high_automation_rate=0.0,
                avg_confidence=0.0,
                period_start=cutoff_time,
                period_end=datetime.utcnow()
            )
        
        # Calculate metrics
        total_evaluations = len(recent_actions)
        auto_approved = len([a for a in recent_actions 
                           if a.action_type == ActionType.AUTO_APPROVE])
        flagged_for_review = len([a for a in recent_actions 
                                if a.requires_human_review])
        immediate_reviews = len([a for a in recent_actions 
                               if a.action_type == ActionType.IMMEDIATE_REVIEW])
        
        low_confidence_actions = len([a for a in recent_actions 
                                    if a.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.CRITICAL]])
        low_confidence_rate = low_confidence_actions / total_evaluations
        
        high_automation_rate = auto_approved / total_evaluations
        avg_confidence = sum(a.confidence_score for a in recent_actions) / total_evaluations
        
        metrics = PerformanceMetrics(
            total_evaluations=total_evaluations,
            auto_approved=auto_approved,
            flagged_for_review=flagged_for_review,
            immediate_reviews=immediate_reviews,
            low_confidence_rate=low_confidence_rate,
            high_automation_rate=high_automation_rate,
            avg_confidence=avg_confidence,
            period_start=cutoff_time,
            period_end=datetime.utcnow()
        )
        
        return metrics
    
    async def check_fine_tuning_triggers(self) -> Dict[str, Any]:
        """
        Check if fine-tuning triggers should be activated
        Implements continuous learning logic from slide 13
        """
        metrics = await self.get_performance_metrics()
        
        triggers = {
            "low_confidence_trigger": False,
            "high_automation_trigger": False,
            "recommendations": [],
            "metrics": metrics
        }
        
        # Low confidence rate trigger (>30% from slide 13)
        if metrics.low_confidence_rate > self.low_confidence_threshold:
            triggers["low_confidence_trigger"] = True
            triggers["recommendations"].append({
                "type": "model_retraining",
                "reason": f"Low confidence rate: {metrics.low_confidence_rate:.1%} > {self.low_confidence_threshold:.1%}",
                "priority": "high",
                "actions": [
                    "Add more training data",
                    "Improve aggregation methods",
                    "Adjust confidence thresholds",
                    "Review evaluation metrics"
                ]
            })
        
        # High automation rate trigger (>90% from slide 13)
        if metrics.high_automation_rate > self.high_automation_threshold:
            triggers["high_automation_trigger"] = True
            triggers["recommendations"].append({
                "type": "quality_gates",
                "reason": f"High automation rate: {metrics.high_automation_rate:.1%} > {self.high_automation_threshold:.1%}",
                "priority": "medium",
                "actions": [
                    "Tighten quality thresholds",
                    "Add additional quality gates",
                    "Increase sampling frequency for random audits",
                    "Implement stricter validation checks"
                ]
            })
        
        # Additional trigger: very low average confidence
        if metrics.avg_confidence < 0.6:
            triggers["recommendations"].append({
                "type": "system_review",
                "reason": f"Low average confidence: {metrics.avg_confidence:.3f}",
                "priority": "high",
                "actions": [
                    "Review evaluation pipeline",
                    "Check model performance",
                    "Validate input data quality",
                    "Calibrate confidence scoring"
                ]
            })
        
        logger.info(f"🔍 Fine-tuning trigger check completed")
        logger.info(f"   Low confidence trigger: {triggers['low_confidence_trigger']}")
        logger.info(f"   High automation trigger: {triggers['high_automation_trigger']}")
        logger.info(f"   Recommendations: {len(triggers['recommendations'])}")
        
        return triggers
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        metrics = await self.get_performance_metrics()
        
        return {
            "performance_metrics": metrics,
            "review_queue_status": {
                priority.value: len(queue) 
                for priority, queue in self.review_queues.items()
            },
            "recent_actions": len(self.action_history),
            "confidence_thresholds": {
                level.value: config["min_score"] 
                for level, config in self.confidence_config.items()
            },
            "system_health": {
                "status": "healthy" if metrics.avg_confidence > 0.6 else "needs_attention",
                "confidence_distribution": self._get_confidence_distribution(),
                "uptime_hours": self.monitoring_window_hours
            }
        }
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels in recent actions"""
        recent_actions = list(self.action_history)[-100:]  # Last 100 actions
        
        distribution = defaultdict(int)
        for action in recent_actions:
            distribution[action.confidence_level.value] += 1
            
        return dict(distribution)
