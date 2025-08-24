"""
Score Aggregation System for Multi-dimensional Video Evaluation
Implements weighted reliability, ensemble methods, and Bayesian uncertainty aggregation.

Based on slides 15-19: Score Aggregation with multiple complementary methods.
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

from ...shared.monitoring import get_logger
from .video_evaluation_orchestrator import DimensionScore, EvaluationDimension

logger = get_logger(__name__)

class AggregationMethod(str, Enum):
    """Score aggregation methods based on data quality"""
    WEIGHTED_RELIABILITY = "weighted_reliability"      # High-quality data (>0.8)
    ENSEMBLE_MULTI_METHOD = "ensemble_multi_method"   # Medium-quality data (0.6-0.8)
    CONSERVATIVE = "conservative"                      # Low-quality data (<0.6)

class DataQuality(str, Enum):
    """Data quality assessment levels"""
    HIGH = "high"       # >0.8 - High completeness and consistency
    MEDIUM = "medium"   # 0.6-0.8 - Moderate quality
    LOW = "low"         # <0.6 - Low quality or inconsistent

@dataclass
class QualityAssessment:
    """Assessment of data quality for aggregation method selection"""
    completeness_score: float      # How many metrics we have (0-1)
    consistency_score: float       # Low variance = high consistency (0-1)
    overall_quality: float         # Combined quality score (0-1)
    quality_level: DataQuality     # Categorical quality level
    missing_dimensions: List[EvaluationDimension]
    reliability_scores: Dict[EvaluationDimension, float]

@dataclass
class AggregationResult:
    """Result of score aggregation with uncertainty metrics"""
    final_score: float
    final_confidence: float
    method_used: AggregationMethod
    quality_assessment: QualityAssessment
    uncertainty_metrics: Dict[str, float]
    individual_method_results: Dict[str, float]
    weights_used: Dict[EvaluationDimension, float]

class ScoreAggregator:
    """
    Multi-method score aggregation system
    Implements ensemble approach from slides 15-19
    """
    
    def __init__(self):
        # Metric reliability weights from slide 15
        # (Higher weights for more reliable metrics)
        self.reliability_weights = {
            EvaluationDimension.PERCEPTUAL_QUALITY: 0.91,      # LPIPS reliability
            EvaluationDimension.MOTION_CONSISTENCY: 0.87,      # FVMD reliability  
            EvaluationDimension.TEXT_VIDEO_ALIGNMENT: 0.84,    # CLIP reliability
            EvaluationDimension.AESTHETIC_QUALITY: 0.82,       # LLaVA Content reliability
            EvaluationDimension.NARRATIVE_FLOW: 0.79,          # LLaVA Aesthetic reliability
            EvaluationDimension.VISUAL_QUALITY: 0.85           # Custom visual metrics
        }
        
        # Base importance weights for dimensions
        self.dimension_importance = {
            EvaluationDimension.VISUAL_QUALITY: 0.20,
            EvaluationDimension.PERCEPTUAL_QUALITY: 0.18,
            EvaluationDimension.MOTION_CONSISTENCY: 0.17,
            EvaluationDimension.TEXT_VIDEO_ALIGNMENT: 0.16,
            EvaluationDimension.AESTHETIC_QUALITY: 0.15,
            EvaluationDimension.NARRATIVE_FLOW: 0.14
        }
        
        logger.info("🔢 Score Aggregator initialized with ensemble methods")
    
    async def aggregate_scores(self, dimension_scores: List[DimensionScore]) -> AggregationResult:
        """
        Main aggregation entry point - dynamically selects method based on data quality
        Implements the workflow from slide 16: Input → Quality Assessment → Method Selection → Aggregation
        """
        logger.debug(f"🎯 Aggregating {len(dimension_scores)} dimension scores")
        
        # Step 1: Assess data quality
        quality_assessment = self._assess_data_quality(dimension_scores)
        
        # Step 2: Select aggregation method based on quality
        method = self._select_aggregation_method(quality_assessment)
        
        # Step 3: Apply selected aggregation method
        if method == AggregationMethod.WEIGHTED_RELIABILITY:
            result = await self._weighted_reliability_aggregation(dimension_scores, quality_assessment)
        elif method == AggregationMethod.ENSEMBLE_MULTI_METHOD:
            result = await self._ensemble_multi_method_aggregation(dimension_scores, quality_assessment)
        elif method == AggregationMethod.CONSERVATIVE:
            result = await self._conservative_aggregation(dimension_scores, quality_assessment)
        else:
            # Fallback to weighted reliability
            result = await self._weighted_reliability_aggregation(dimension_scores, quality_assessment)
        
        logger.info(f"✅ Score aggregation completed")
        logger.info(f"   Method used: {method}")
        logger.info(f"   Final score: {result.final_score:.3f}")
        logger.info(f"   Final confidence: {result.final_confidence:.3f}")
        logger.info(f"   Data quality: {quality_assessment.quality_level}")
        
        return result
    
    def _assess_data_quality(self, dimension_scores: List[DimensionScore]) -> QualityAssessment:
        """
        Assess data quality by checking completeness and consistency
        From slide 16: "data quality is assessed by checking score completeness and consistency"
        """
        total_dimensions = len(EvaluationDimension)
        available_dimensions = len(dimension_scores)
        
        # Completeness: how many dimensions we have
        completeness_score = available_dimensions / total_dimensions
        
        # Consistency: low variance = high quality
        if dimension_scores:
            scores = [ds.score for ds in dimension_scores]
            confidences = [ds.confidence for ds in dimension_scores]
            
            # Calculate consistency based on variance
            score_variance = np.var(scores)
            confidence_variance = np.var(confidences)
            
            # Normalize variance to 0-1 scale (lower variance = higher consistency)
            score_consistency = max(0.0, 1.0 - score_variance)
            confidence_consistency = max(0.0, 1.0 - confidence_variance)
            consistency_score = (score_consistency + confidence_consistency) / 2
        else:
            consistency_score = 0.0
        
        # Overall quality combines completeness and consistency
        overall_quality = (completeness_score + consistency_score) / 2
        
        # Determine quality level
        if overall_quality > 0.8:
            quality_level = DataQuality.HIGH
        elif overall_quality > 0.6:
            quality_level = DataQuality.MEDIUM
        else:
            quality_level = DataQuality.LOW
        
        # Find missing dimensions
        available_dims = {ds.dimension for ds in dimension_scores}
        missing_dimensions = [dim for dim in EvaluationDimension if dim not in available_dims]
        
        # Calculate reliability scores for available dimensions
        reliability_scores = {
            ds.dimension: self.reliability_weights.get(ds.dimension, 0.8) * ds.confidence
            for ds in dimension_scores
        }
        
        return QualityAssessment(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            overall_quality=overall_quality,
            quality_level=quality_level,
            missing_dimensions=missing_dimensions,
            reliability_scores=reliability_scores
        )
    
    def _select_aggregation_method(self, quality_assessment: QualityAssessment) -> AggregationMethod:
        """
        Select aggregation method based on data quality
        From slide 15: method selection based on data quality thresholds
        """
        if quality_assessment.quality_level == DataQuality.HIGH:
            return AggregationMethod.WEIGHTED_RELIABILITY
        elif quality_assessment.quality_level == DataQuality.MEDIUM:
            return AggregationMethod.ENSEMBLE_MULTI_METHOD
        else:
            return AggregationMethod.CONSERVATIVE
    
    async def _weighted_reliability_aggregation(self, 
                                              dimension_scores: List[DimensionScore],
                                              quality_assessment: QualityAssessment) -> AggregationResult:
        """
        Weighted reliability aggregation for high-quality data (>0.8)
        From slide 15: "Weight scores by metric reliability"
        """
        if not dimension_scores:
            return self._create_empty_result(AggregationMethod.WEIGHTED_RELIABILITY, quality_assessment)
        
        weighted_score = 0.0
        total_weight = 0.0
        weights_used = {}
        
        for ds in dimension_scores:
            # Combine reliability weight, importance weight, and confidence
            reliability = self.reliability_weights.get(ds.dimension, 0.8)
            importance = self.dimension_importance.get(ds.dimension, 1.0)
            
            # Final weight combines all factors
            weight = reliability * importance * ds.confidence
            weights_used[ds.dimension] = weight
            
            weighted_score += ds.score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
            # Confidence is weighted average of individual confidences
            final_confidence = sum(ds.confidence * weights_used[ds.dimension] for ds in dimension_scores) / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0
        
        # Calculate uncertainty metrics
        uncertainty_metrics = self._calculate_uncertainty_metrics(dimension_scores, weights_used)
        
        return AggregationResult(
            final_score=final_score,
            final_confidence=final_confidence,
            method_used=AggregationMethod.WEIGHTED_RELIABILITY,
            quality_assessment=quality_assessment,
            uncertainty_metrics=uncertainty_metrics,
            individual_method_results={"weighted_reliability": final_score},
            weights_used=weights_used
        )
    
    async def _ensemble_multi_method_aggregation(self,
                                               dimension_scores: List[DimensionScore],
                                               quality_assessment: QualityAssessment) -> AggregationResult:
        """
        Ensemble multi-method aggregation for medium-quality data (0.6-0.8)
        From slide 15: "Run all methods and combine results → weighted average with higher weights for higher confidence methods"
        Includes Bayesian uncertainty aggregation
        """
        if not dimension_scores:
            return self._create_empty_result(AggregationMethod.ENSEMBLE_MULTI_METHOD, quality_assessment)
        
        # Method 1: Weighted reliability
        weighted_result = await self._weighted_reliability_aggregation(dimension_scores, quality_assessment)
        
        # Method 2: Simple average
        simple_avg_score = np.mean([ds.score for ds in dimension_scores])
        simple_avg_confidence = np.mean([ds.confidence for ds in dimension_scores])
        
        # Method 3: Confidence-weighted average
        confidence_weights = [ds.confidence for ds in dimension_scores]
        if sum(confidence_weights) > 0:
            conf_weighted_score = np.average([ds.score for ds in dimension_scores], weights=confidence_weights)
        else:
            conf_weighted_score = simple_avg_score
        
        # Method 4: Geometric mean (for robustness)
        positive_scores = [max(0.001, ds.score) for ds in dimension_scores]  # Avoid log(0)
        geometric_mean_score = np.exp(np.mean(np.log(positive_scores)))
        
        # Combine methods with confidence-based weights
        method_scores = [
            weighted_result.final_score,
            simple_avg_score,
            conf_weighted_score,
            geometric_mean_score
        ]
        
        method_confidences = [
            weighted_result.final_confidence,
            simple_avg_confidence,
            simple_avg_confidence,
            simple_avg_confidence * 0.9  # Slightly lower confidence for geometric mean
        ]
        
        # Ensemble weighting: higher confidence methods get more weight
        ensemble_weights = np.array(method_confidences)
        ensemble_weights = ensemble_weights / np.sum(ensemble_weights)
        
        final_score = np.average(method_scores, weights=ensemble_weights)
        
        # Bayesian uncertainty aggregation (from slide 15)
        uncertainty_metrics = self._bayesian_uncertainty_aggregation(dimension_scores, method_scores, method_confidences)
        
        # Final confidence incorporates uncertainty
        final_confidence = uncertainty_metrics.get("adjusted_confidence", np.mean(method_confidences))
        
        return AggregationResult(
            final_score=final_score,
            final_confidence=final_confidence,
            method_used=AggregationMethod.ENSEMBLE_MULTI_METHOD,
            quality_assessment=quality_assessment,
            uncertainty_metrics=uncertainty_metrics,
            individual_method_results={
                "weighted_reliability": method_scores[0],
                "simple_average": method_scores[1],
                "confidence_weighted": method_scores[2],
                "geometric_mean": method_scores[3]
            },
            weights_used={ds.dimension: 1.0/len(dimension_scores) for ds in dimension_scores}
        )
    
    async def _conservative_aggregation(self,
                                      dimension_scores: List[DimensionScore],
                                      quality_assessment: QualityAssessment) -> AggregationResult:
        """
        Conservative aggregation for low-quality data (<0.6)
        From slide 15: "Use minimum scores across all dimensions"
        """
        if not dimension_scores:
            return self._create_empty_result(AggregationMethod.CONSERVATIVE, quality_assessment)
        
        # Take minimum scores across all dimensions (conservative approach)
        scores = [ds.score for ds in dimension_scores]
        confidences = [ds.confidence for ds in dimension_scores]
        
        final_score = min(scores)
        final_confidence = min(confidences)
        
        # Add penalty for missing dimensions
        completeness_penalty = 1.0 - (len(quality_assessment.missing_dimensions) / len(EvaluationDimension))
        final_score *= completeness_penalty
        final_confidence *= completeness_penalty
        
        uncertainty_metrics = {
            "conservative_penalty": 1.0 - completeness_penalty,
            "score_range": max(scores) - min(scores),
            "confidence_range": max(confidences) - min(confidences),
            "data_completeness": quality_assessment.completeness_score
        }
        
        return AggregationResult(
            final_score=final_score,
            final_confidence=final_confidence,
            method_used=AggregationMethod.CONSERVATIVE,
            quality_assessment=quality_assessment,
            uncertainty_metrics=uncertainty_metrics,
            individual_method_results={"conservative_min": final_score},
            weights_used={ds.dimension: 1.0 for ds in dimension_scores}
        )
    
    def _bayesian_uncertainty_aggregation(self, 
                                        dimension_scores: List[DimensionScore],
                                        method_scores: List[float],
                                        method_confidences: List[float]) -> Dict[str, float]:
        """
        Bayesian uncertainty aggregation with error propagation
        From slide 15: "Standard Error of Mean (SEM) + error propagation"
        "Uncertainty Propagation: σy = √(a²σx1² + b²σx2² + c²σx3²)"
        """
        
        # Standard Error of Mean for dimension scores
        if len(dimension_scores) > 1:
            scores = [ds.score for ds in dimension_scores]
            sem = np.std(scores, ddof=1) / np.sqrt(len(scores))
        else:
            sem = 0.0
        
        # Uncertainty propagation for method combination
        # Each method contributes uncertainty based on its weight and confidence
        method_weights = np.array(method_confidences)
        method_weights = method_weights / np.sum(method_weights)
        
        # Calculate propagated uncertainty
        uncertainty_contributions = []
        for i, (score, confidence, weight) in enumerate(zip(method_scores, method_confidences, method_weights)):
            # Uncertainty = (1 - confidence) for each method
            method_uncertainty = 1.0 - confidence
            # Weighted contribution to total uncertainty
            weighted_uncertainty = weight * method_uncertainty
            uncertainty_contributions.append(weighted_uncertainty)
        
        # Propagated uncertainty (quadrature sum)
        propagated_uncertainty = np.sqrt(sum(uc**2 for uc in uncertainty_contributions))
        
        # Adjusted confidence accounting for uncertainty
        base_confidence = np.average(method_confidences, weights=method_weights)
        adjusted_confidence = base_confidence * (1.0 - propagated_uncertainty)
        
        return {
            "standard_error_mean": sem,
            "propagated_uncertainty": propagated_uncertainty,
            "method_disagreement": np.std(method_scores),
            "confidence_spread": np.std(method_confidences),
            "adjusted_confidence": max(0.0, min(1.0, adjusted_confidence)),
            "uncertainty_contributions": uncertainty_contributions
        }
    
    def _calculate_uncertainty_metrics(self, 
                                     dimension_scores: List[DimensionScore],
                                     weights: Dict[EvaluationDimension, float]) -> Dict[str, float]:
        """Calculate uncertainty metrics for aggregation result"""
        
        if not dimension_scores:
            return {"total_uncertainty": 1.0}
        
        scores = [ds.score for ds in dimension_scores]
        confidences = [ds.confidence for ds in dimension_scores]
        weight_values = [weights.get(ds.dimension, 1.0) for ds in dimension_scores]
        
        return {
            "score_variance": np.var(scores),
            "confidence_variance": np.var(confidences),
            "weight_distribution": np.std(weight_values),
            "total_uncertainty": 1.0 - np.mean(confidences),
            "dimension_agreement": 1.0 - np.std(scores),
            "missing_dimension_penalty": len([d for d in EvaluationDimension if d not in weights]) / len(EvaluationDimension)
        }
    
    def _create_empty_result(self, 
                           method: AggregationMethod,
                           quality_assessment: QualityAssessment) -> AggregationResult:
        """Create empty result for edge cases"""
        return AggregationResult(
            final_score=0.0,
            final_confidence=0.0,
            method_used=method,
            quality_assessment=quality_assessment,
            uncertainty_metrics={"total_uncertainty": 1.0},
            individual_method_results={},
            weights_used={}
        )
