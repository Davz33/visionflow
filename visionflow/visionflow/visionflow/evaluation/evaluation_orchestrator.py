"""
Evaluation Orchestrator - Main coordination hub for autoraters and autoevals
Integrates seamlessly with the multi-agent orchestration system
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field

from .autoraters import AutoraterService, AutoraterConfig, EvaluationLevel
from .benchmarks import BenchmarkService, BenchmarkComparison
from .continuous_learning import LearningService, FeedbackRecord
from .quality_metrics import EvaluationResult, QualityDimensions
from ...shared.config import get_settings
from ...shared.monitoring import get_logger
from ...shared.database import DatabaseManager

logger = get_logger("evaluation_orchestrator")
settings = get_settings()


class EvaluationRequest(BaseModel):
    """Request for comprehensive video evaluation"""
    job_id: str = Field(description="Associated job ID")
    video_path: str = Field(description="Path to video file")
    original_prompt: str = Field(description="Original generation prompt")
    
    # Evaluation configuration
    evaluation_level: EvaluationLevel = Field(default=EvaluationLevel.STANDARD, description="Depth of evaluation")
    include_benchmarks: bool = Field(default=True, description="Include benchmark comparisons")
    benchmark_names: Optional[List[str]] = Field(None, description="Specific benchmarks to use")
    
    # Generation metadata
    generation_metadata: Optional[Dict[str, Any]] = Field(None, description="Generation process metadata")
    
    # User context
    user_id: Optional[str] = Field(None, description="User ID for personalized evaluation")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    
    # Advanced options
    enable_learning: bool = Field(default=True, description="Enable continuous learning from this evaluation")
    priority: str = Field(default="normal", description="Evaluation priority (low, normal, high)")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")


class EvaluationResponse(BaseModel):
    """Comprehensive evaluation response"""
    evaluation_id: str = Field(description="Unique evaluation identifier")
    job_id: str = Field(description="Associated job ID")
    
    # Core evaluation
    evaluation_result: EvaluationResult = Field(description="Detailed evaluation results")
    
    # Benchmark comparisons
    benchmark_comparisons: Optional[Dict[str, BenchmarkComparison]] = Field(
        None, description="Benchmark comparison results"
    )
    
    # Learning insights
    learning_feedback_id: Optional[str] = Field(None, description="Learning system feedback ID")
    
    # Human-readable summary
    executive_summary: str = Field(description="Executive summary of evaluation")
    key_insights: List[str] = Field(description="Key insights from evaluation")
    improvement_recommendations: List[str] = Field(description="Specific improvement recommendations")
    
    # Performance metadata
    processing_time: float = Field(description="Total processing time in seconds")
    evaluator_agents_used: List[str] = Field(description="List of evaluator agents used")
    
    # Status
    status: str = Field(description="Evaluation status")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationOrchestrator:
    """Main orchestrator for comprehensive video evaluation"""
    
    def __init__(self):
        self.autorater_service = AutoraterService()
        self.benchmark_service = BenchmarkService()
        self.learning_service = LearningService()
        self.db_manager = DatabaseManager()
        
        # Configuration
        self.config = self._load_configuration()
        
        # Active evaluations tracking
        self.active_evaluations: Dict[str, asyncio.Task] = {}
        
        logger.info("Evaluation orchestrator initialized with comprehensive services")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load evaluation orchestrator configuration"""
        return {
            "max_concurrent_evaluations": 10,
            "default_timeout": 300,  # 5 minutes
            "enable_async_processing": True,
            "automatic_learning": True,
            "benchmark_cache_ttl": 3600,  # 1 hour
            "quality_thresholds": {
                "minimum_acceptable": 0.60,
                "good_quality": 0.75,
                "excellent_quality": 0.90
            }
        }
    
    async def evaluate_video_comprehensive(
        self,
        request: EvaluationRequest
    ) -> EvaluationResponse:
        """Perform comprehensive video evaluation with all features"""
        
        start_time = datetime.utcnow()
        evaluation_id = str(uuid.uuid4())
        
        logger.info(f"Starting comprehensive evaluation {evaluation_id} for job {request.job_id}")
        
        try:
            # Create evaluation task
            evaluation_task = asyncio.create_task(
                self._execute_comprehensive_evaluation(evaluation_id, request)
            )
            
            # Track active evaluation
            self.active_evaluations[evaluation_id] = evaluation_task
            
            # Execute evaluation
            result = await evaluation_task
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create comprehensive response
            response = EvaluationResponse(
                evaluation_id=evaluation_id,
                job_id=request.job_id,
                evaluation_result=result["evaluation_result"],
                benchmark_comparisons=result.get("benchmark_comparisons"),
                learning_feedback_id=result.get("learning_feedback_id"),
                executive_summary=self._generate_executive_summary(result),
                key_insights=self._extract_key_insights(result),
                improvement_recommendations=self._generate_improvement_recommendations(result),
                processing_time=processing_time,
                evaluator_agents_used=result["evaluator_agents_used"],
                status="completed"
            )
            
            logger.info(f"Comprehensive evaluation {evaluation_id} completed in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation {evaluation_id} failed: {e}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Return error response with minimal evaluation
            fallback_result = await self._create_fallback_evaluation(request)
            
            return EvaluationResponse(
                evaluation_id=evaluation_id,
                job_id=request.job_id,
                evaluation_result=fallback_result,
                executive_summary=f"Evaluation failed: {str(e)}",
                key_insights=["Evaluation system encountered an error"],
                improvement_recommendations=["Please retry evaluation or contact support"],
                processing_time=processing_time,
                evaluator_agents_used=["fallback_evaluator"],
                status="failed"
            )
        
        finally:
            # Clean up active evaluation tracking
            if evaluation_id in self.active_evaluations:
                del self.active_evaluations[evaluation_id]
    
    async def evaluate_video_quick(
        self,
        job_id: str,
        video_path: str,
        original_prompt: str,
        generation_metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResponse:
        """Perform quick evaluation for real-time feedback"""
        
        request = EvaluationRequest(
            job_id=job_id,
            video_path=video_path,
            original_prompt=original_prompt,
            evaluation_level=EvaluationLevel.BASIC,
            include_benchmarks=False,
            generation_metadata=generation_metadata,
            enable_learning=False
        )
        
        return await self.evaluate_video_comprehensive(request)
    
    async def _execute_comprehensive_evaluation(
        self,
        evaluation_id: str,
        request: EvaluationRequest
    ) -> Dict[str, Any]:
        """Execute the full comprehensive evaluation workflow"""
        
        results = {
            "evaluation_id": evaluation_id,
            "evaluator_agents_used": []
        }
        
        # Step 1: Core autorater evaluation
        logger.info(f"Step 1: Running autorater evaluation for {evaluation_id}")
        
        evaluation_result = await self.autorater_service.evaluate_video_comprehensive(
            video_path=request.video_path,
            original_prompt=request.original_prompt,
            job_id=request.job_id,
            generation_metadata=request.generation_metadata,
            evaluation_level=request.evaluation_level
        )
        
        results["evaluation_result"] = evaluation_result
        results["evaluator_agents_used"].extend(evaluation_result.evaluator_agents)
        
        # Step 2: Benchmark comparisons (if enabled)
        if request.include_benchmarks:
            logger.info(f"Step 2: Running benchmark comparisons for {evaluation_id}")
            
            benchmark_names = request.benchmark_names or ["industry_standard", "internal_baseline"]
            benchmark_comparisons = await self.benchmark_service.compare_against_multiple_benchmarks(
                evaluation_result, benchmark_names
            )
            
            results["benchmark_comparisons"] = benchmark_comparisons
            results["evaluator_agents_used"].append("benchmark_analyzer")
        
        # Step 3: Learning integration (if enabled)
        if request.enable_learning:
            logger.info(f"Step 3: Integrating with continuous learning for {evaluation_id}")
            
            # Create initial feedback record for learning system
            learning_feedback = await self.learning_service.collect_user_feedback(
                evaluation_id=evaluation_result.evaluation_id,
                job_id=request.job_id,
                user_rating=evaluation_result.quality_dimensions.overall_quality_score,
                comments=f"Automated evaluation: {evaluation_result.quality_dimensions.quality_grade.value}",
                user_context={"evaluation_level": request.evaluation_level.value}
            )
            
            results["learning_feedback_id"] = learning_feedback.feedback_id
            results["evaluator_agents_used"].append("learning_integrator")
        
        # Step 4: Advanced analytics (for high-level evaluations)
        if request.evaluation_level in [EvaluationLevel.COMPREHENSIVE, EvaluationLevel.EXPERT]:
            logger.info(f"Step 4: Running advanced analytics for {evaluation_id}")
            
            # Add competitive insights if available
            if "competitive" in (request.benchmark_names or []):
                competitive_insights = await self.benchmark_service.get_competitive_insights(
                    evaluation_result
                )
                results["competitive_insights"] = competitive_insights
                results["evaluator_agents_used"].append("competitive_analyzer")
            
            # Add trend analysis
            trend_analysis = await self.benchmark_service.analyze_performance_trends(
                timeframe_days=7,  # Short-term for individual evaluation
                job_ids=[request.job_id]
            )
            results["trend_analysis"] = trend_analysis
            results["evaluator_agents_used"].append("trend_analyzer")
        
        return results
    
    async def _create_fallback_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        """Create fallback evaluation when main evaluation fails"""
        
        return await self.autorater_service.evaluate_video_quick(
            video_path=request.video_path,
            original_prompt=request.original_prompt,
            job_id=request.job_id
        )
    
    def _generate_executive_summary(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate executive summary of evaluation results"""
        
        eval_result = evaluation_results["evaluation_result"]
        quality_score = eval_result.quality_dimensions.overall_quality_score
        quality_grade = eval_result.quality_dimensions.quality_grade.value
        
        summary = f"""
🎬 Video Generation Evaluation Summary

📊 Overall Assessment: {quality_score:.1%} ({quality_grade.upper()})

🎯 Performance Highlights:
• Generated video achieves {quality_grade} quality standards
• Evaluation completed using {len(eval_result.evaluator_agents)} specialized evaluators
• Processing time: {eval_result.evaluation_duration:.1f} seconds
        """.strip()
        
        # Add benchmark context if available
        if "benchmark_comparisons" in evaluation_results:
            benchmarks = evaluation_results["benchmark_comparisons"]
            if "industry_standard" in benchmarks:
                industry_comparison = benchmarks["industry_standard"]
                summary += f"\n• Industry benchmark: {industry_comparison.percentile_rank:.0f}th percentile"
        
        # Add key strengths
        strengths = eval_result.quality_dimensions.get_dimension_scores()
        top_dimension = max(strengths.items(), key=lambda x: x[1])
        summary += f"\n• Strongest dimension: {top_dimension[0].replace('_', ' ').title()} ({top_dimension[1]:.1%})"
        
        return summary
    
    def _extract_key_insights(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Extract key insights from evaluation results"""
        
        eval_result = evaluation_results["evaluation_result"]
        insights = []
        
        # Quality dimension insights
        dimension_scores = eval_result.quality_dimensions.get_dimension_scores()
        
        # Find strongest and weakest dimensions
        strongest = max(dimension_scores.items(), key=lambda x: x[1] if x[0] != "overall" else 0)
        weakest = min(dimension_scores.items(), key=lambda x: x[1] if x[0] != "overall" else 1)
        
        if strongest[0] != "overall":
            insights.append(f"Exceptional {strongest[0].replace('_', ' ')} quality ({strongest[1]:.1%})")
        
        if weakest[0] != "overall" and weakest[1] < 0.75:
            insights.append(f"{weakest[0].replace('_', ' ').title()} needs improvement ({weakest[1]:.1%})")
        
        # Benchmark insights
        if "benchmark_comparisons" in evaluation_results:
            benchmarks = evaluation_results["benchmark_comparisons"]
            
            for benchmark_name, comparison in benchmarks.items():
                if comparison.percentile_rank >= 90:
                    insights.append(f"Outperforms {benchmark_name.replace('_', ' ')} in {comparison.percentile_rank:.0f}th percentile")
                elif comparison.percentile_rank <= 25:
                    insights.append(f"Below average for {benchmark_name.replace('_', ' ')} ({comparison.percentile_rank:.0f}th percentile)")
        
        # Performance insights
        if eval_result.evaluation_duration < 10:
            insights.append("Rapid evaluation processing achieved")
        
        return insights[:5]  # Top 5 insights
    
    def _generate_improvement_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate specific improvement recommendations"""
        
        eval_result = evaluation_results["evaluation_result"]
        recommendations = eval_result.quality_dimensions.get_improvement_recommendations()
        
        # Add benchmark-based recommendations
        if "benchmark_comparisons" in evaluation_results:
            benchmarks = evaluation_results["benchmark_comparisons"]
            
            for benchmark_name, comparison in benchmarks.items():
                recommendations.extend(comparison.improvement_areas[:2])  # Top 2 from each benchmark
        
        # Add competitive recommendations
        if "competitive_insights" in evaluation_results:
            competitive = evaluation_results["competitive_insights"]
            recommendations.extend(competitive.get("strategic_recommendations", [])[:2])
        
        # Deduplicate and limit
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations[:7]  # Top 7 recommendations
    
    async def collect_user_feedback(
        self,
        evaluation_id: str,
        user_rating: float,
        comments: Optional[str] = None,
        dimension_ratings: Optional[Dict[str, float]] = None
    ) -> bool:
        """Collect user feedback for continuous learning"""
        
        try:
            # Find the job_id from evaluation_id (this would be a database lookup in production)
            job_id = "unknown"  # In production, lookup from evaluation database
            
            await self.learning_service.collect_user_feedback(
                evaluation_id=evaluation_id,
                job_id=job_id,
                user_rating=user_rating,
                comments=comments,
                dimension_ratings=dimension_ratings
            )
            
            logger.info(f"Collected user feedback for evaluation {evaluation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect user feedback: {e}")
            return False
    
    async def get_evaluation_status(self, evaluation_id: str) -> Dict[str, Any]:
        """Get status of an ongoing or completed evaluation"""
        
        if evaluation_id in self.active_evaluations:
            task = self.active_evaluations[evaluation_id]
            
            if task.done():
                try:
                    result = task.result()
                    return {"status": "completed", "result": result}
                except Exception as e:
                    return {"status": "failed", "error": str(e)}
            else:
                return {"status": "processing", "progress": "in_progress"}
        
        # Check database for completed evaluations
        # This would be implemented with actual database lookup
        return {"status": "not_found"}
    
    async def get_system_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        
        # Get learning system report
        learning_report = await self.learning_service.generate_learning_report()
        
        # Get benchmark summary
        benchmark_summary = self.benchmark_service.get_benchmark_summary()
        
        # Get feedback statistics
        feedback_stats = self.learning_service.get_feedback_statistics()
        
        # System metrics
        system_metrics = {
            "active_evaluations": len(self.active_evaluations),
            "evaluations_completed_today": 42,  # Mock metric
            "average_processing_time": 25.3,    # Mock metric
            "system_uptime": "99.8%",           # Mock metric
            "quality_improvement_trend": "+12% over last month"  # Mock metric
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": system_metrics,
            "learning_insights": learning_report,
            "benchmark_performance": benchmark_summary,
            "user_feedback": feedback_stats,
            "recommendations": [
                "Continue focus on aesthetic quality improvements",
                "Expand benchmark datasets for better calibration",
                "Implement real-time feedback collection improvements"
            ]
        }
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status and configuration"""
        
        return {
            "orchestrator_version": "v2024.1.0",
            "status": "healthy",
            "configuration": self.config,
            "active_evaluations": len(self.active_evaluations),
            "services_status": {
                "autorater_service": "active",
                "benchmark_service": "active",
                "learning_service": "active"
            },
            "capabilities": [
                "comprehensive_evaluation",
                "benchmark_comparison",
                "continuous_learning",
                "trend_analysis",
                "competitive_insights",
                "real_time_feedback"
            ]
        }


# Singleton instance
_evaluation_orchestrator = None

def get_evaluation_orchestrator() -> EvaluationOrchestrator:
    """Get or create evaluation orchestrator instance"""
    global _evaluation_orchestrator
    if _evaluation_orchestrator is None:
        _evaluation_orchestrator = EvaluationOrchestrator()
    return _evaluation_orchestrator
