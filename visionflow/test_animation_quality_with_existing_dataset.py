"""
Test Script for Animation Quality Assessment using Existing Evaluation Datasets
Tests the enhanced VideoEvaluationOrchestrator with real video content from evaluation datasets
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add the visionflow package to the path
sys.path.append(str(Path(__file__).parent / "visionflow"))

from visionflow.services.evaluation.video_evaluation_orchestrator import (
    VideoEvaluationOrchestrator,
    EvaluationDimension,
    ConfidenceLevel
)
from visionflow.shared.monitoring import get_logger

logger = get_logger("test_animation_quality_existing_dataset")


class ExistingDatasetAnimationQualityTestRunner:
    """
    Test runner for animation quality assessment using existing evaluation datasets
    """
    
    def __init__(self):
        self.orchestrator = VideoEvaluationOrchestrator()
        self.results = []
        
        logger.info("🧪 Existing Dataset Animation Quality Test Runner initialized")
    
    async def run_animation_quality_tests(self) -> Dict[str, Any]:
        """Run animation quality assessment on existing evaluation datasets"""
        
        logger.info("🚀 Starting Animation Quality Assessment with Existing Datasets")
        
        # Test datasets to evaluate
        test_datasets = [
            {
                "name": "Large-Scale Samples",
                "path": "evaluation_datasets/large_scale_samples/dataset_manifest.json",
                "video_dir": "evaluation_datasets/large_scale_samples",
                "focus_categories": ["animation", "action", "human_activity", "dramatic"]
            },
            {
                "name": "Reliable Sources",
                "path": "evaluation_datasets/reliable_sources/sample_evaluation_dataset.json", 
                "video_dir": "evaluation_datasets/reliable_sources",
                "focus_categories": ["action", "human_activity", "object"]
            }
        ]
        
        all_results = []
        
        for dataset_info in test_datasets:
            logger.info(f"📊 Testing dataset: {dataset_info['name']}")
            
            # Load dataset manifest
            manifest_path = Path(dataset_info["path"])
            if not manifest_path.exists():
                logger.warning(f"⚠️  Dataset manifest not found: {manifest_path}")
                continue
                
            with open(manifest_path, 'r') as f:
                dataset = json.load(f)
            
            # Filter for animation-relevant videos
            animation_videos = self._filter_animation_relevant_videos(
                dataset["videos"], 
                dataset_info["focus_categories"]
            )
            
            logger.info(f"🎬 Found {len(animation_videos)} animation-relevant videos")
            
            # Evaluate each video
            dataset_results = []
            for i, video_info in enumerate(animation_videos):
                logger.info(f"📹 Evaluating video {i+1}/{len(animation_videos)}: {video_info.get('filename', 'unknown')}")
                
                try:
                    # Construct video path
                    video_path = Path(dataset_info["video_dir"]) / video_info.get("filename", "")
                    
                    if not video_path.exists():
                        logger.warning(f"⚠️  Video file not found: {video_path}")
                        continue
                    
                    # Run evaluation
                    start_time = time.time()
                    evaluation_result = await self.orchestrator.evaluate_video(
                        video_path=str(video_path),
                        prompt=video_info.get("prompt", "Animation quality assessment"),
                        evaluation_id=f"{dataset_info['name']}_{i+1}"
                    )
                    evaluation_time = time.time() - start_time
                    
                    # Analyze animation-specific dimensions
                    animation_analysis = self._analyze_animation_quality(evaluation_result)
                    
                    # Store results
                    result = {
                        "dataset": dataset_info["name"],
                        "video_info": video_info,
                        "video_path": str(video_path),
                        "evaluation_result": evaluation_result,
                        "animation_analysis": animation_analysis,
                        "evaluation_time": evaluation_time
                    }
                    dataset_results.append(result)
                    
                    logger.info(f"✅ Evaluation {i+1} completed in {evaluation_time:.2f}s")
                    logger.info(f"   📊 Overall Score: {evaluation_result.overall_score:.3f}")
                    logger.info(f"   🎭 Character Consistency: {animation_analysis['character_consistency_score']:.3f}")
                    logger.info(f"   🔧 Technical Artifacts: {animation_analysis['technical_artifacts_score']:.3f}")
                    logger.info(f"   🎬 Animation Quality: {animation_analysis['animation_quality_score']:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Evaluation {i+1} failed: {e}")
                    dataset_results.append({
                        "dataset": dataset_info["name"],
                        "video_info": video_info,
                        "video_path": str(video_path) if 'video_path' in locals() else "unknown",
                        "evaluation_result": None,
                        "animation_analysis": None,
                        "evaluation_time": 0,
                        "error": str(e)
                    })
            
            all_results.extend(dataset_results)
        
        # Generate comprehensive test report
        test_report = self._generate_animation_quality_report(all_results)
        
        # Save results
        await self._save_test_results(test_report)
        
        logger.info("🎉 Animation Quality Assessment with Existing Datasets completed!")
        return test_report
    
    def _filter_animation_relevant_videos(self, videos: List[Dict], focus_categories: List[str]) -> List[Dict]:
        """Filter videos that are relevant for animation quality assessment"""
        
        animation_relevant = []
        
        for video in videos:
            # Check category
            category = video.get("category", "").lower()
            if category in focus_categories:
                animation_relevant.append(video)
                continue
            
            # Check prompt for animation keywords
            prompt = video.get("prompt", "").lower()
            animation_keywords = [
                "animation", "animated", "character", "cartoon", "motion", "movement",
                "dance", "action", "dramatic", "human", "person", "people"
            ]
            
            if any(keyword in prompt for keyword in animation_keywords):
                animation_relevant.append(video)
                continue
            
            # Check tags for animation content
            tags = video.get("tags", [])
            if any(tag.lower() in ["animation", "action", "human_activity"] for tag in tags):
                animation_relevant.append(video)
        
        return animation_relevant
    
    def _analyze_animation_quality(self, evaluation_result) -> Dict[str, Any]:
        """Analyze animation-specific quality dimensions from evaluation result"""
        
        if not evaluation_result:
            return {
                "character_consistency_score": 0.0,
                "technical_artifacts_score": 0.0,
                "animation_quality_score": 0.0,
                "character_consistency_confidence": 0.0,
                "technical_artifacts_confidence": 0.0,
                "animation_quality_confidence": 0.0,
                "animation_issues_detected": [],
                "quality_assessment": "unknown"
            }
        
        # Extract dimension scores
        dimension_scores = {ds.dimension: ds.score for ds in evaluation_result.dimension_scores}
        dimension_confidences = {ds.dimension: ds.confidence for ds in evaluation_result.dimension_scores}
        
        # Get animation-specific scores
        char_score = dimension_scores.get(EvaluationDimension.CHARACTER_CONSISTENCY, 0.0)
        tech_score = dimension_scores.get(EvaluationDimension.TECHNICAL_ARTIFACTS, 0.0)
        anim_score = dimension_scores.get(EvaluationDimension.ANIMATION_QUALITY, 0.0)
        
        char_conf = dimension_confidences.get(EvaluationDimension.CHARACTER_CONSISTENCY, 0.0)
        tech_conf = dimension_confidences.get(EvaluationDimension.TECHNICAL_ARTIFACTS, 0.0)
        anim_conf = dimension_confidences.get(EvaluationDimension.ANIMATION_QUALITY, 0.0)
        
        # Detect animation issues
        issues_detected = []
        if char_score < 0.6:
            issues_detected.append("character_consistency")
        if tech_score < 0.6:
            issues_detected.append("technical_artifacts")
        if anim_score < 0.6:
            issues_detected.append("overall_animation_quality")
        
        # Overall quality assessment
        avg_score = (char_score + tech_score + anim_score) / 3
        if avg_score >= 0.8:
            quality_assessment = "excellent"
        elif avg_score >= 0.6:
            quality_assessment = "good"
        elif avg_score >= 0.4:
            quality_assessment = "fair"
        else:
            quality_assessment = "poor"
        
        return {
            "character_consistency_score": char_score,
            "technical_artifacts_score": tech_score,
            "animation_quality_score": anim_score,
            "character_consistency_confidence": char_conf,
            "technical_artifacts_confidence": tech_conf,
            "animation_quality_confidence": anim_conf,
            "animation_issues_detected": issues_detected,
            "quality_assessment": quality_assessment,
            "average_animation_score": avg_score
        }
    
    def _generate_animation_quality_report(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive animation quality assessment report"""
        
        total_videos = len(all_results)
        successful_evaluations = sum(1 for r in all_results if r["evaluation_result"] is not None)
        failed_evaluations = total_videos - successful_evaluations
        
        # Calculate average scores
        successful_results = [r for r in all_results if r["evaluation_result"] is not None]
        
        if successful_results:
            avg_character_consistency = sum(r["animation_analysis"]["character_consistency_score"] for r in successful_results) / len(successful_results)
            avg_technical_artifacts = sum(r["animation_analysis"]["technical_artifacts_score"] for r in successful_results) / len(successful_results)
            avg_animation_quality = sum(r["animation_analysis"]["animation_quality_score"] for r in successful_results) / len(successful_results)
            avg_overall_score = sum(r["evaluation_result"].overall_score for r in successful_results) / len(successful_results)
            avg_evaluation_time = sum(r["evaluation_time"] for r in successful_results) / len(successful_results)
        else:
            avg_character_consistency = 0.0
            avg_technical_artifacts = 0.0
            avg_animation_quality = 0.0
            avg_overall_score = 0.0
            avg_evaluation_time = 0.0
        
        # Quality distribution
        quality_distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for result in successful_results:
            quality = result["animation_analysis"]["quality_assessment"]
            quality_distribution[quality] += 1
        
        # Issue detection statistics
        issue_stats = {
            "character_consistency": 0,
            "technical_artifacts": 0,
            "overall_animation_quality": 0
        }
        for result in successful_results:
            for issue in result["animation_analysis"]["animation_issues_detected"]:
                if issue in issue_stats:
                    issue_stats[issue] += 1
        
        # Dataset breakdown
        dataset_breakdown = {}
        for result in all_results:
            dataset_name = result["dataset"]
            if dataset_name not in dataset_breakdown:
                dataset_breakdown[dataset_name] = {
                    "total_videos": 0,
                    "successful_evaluations": 0,
                    "avg_character_consistency": 0.0,
                    "avg_technical_artifacts": 0.0,
                    "avg_animation_quality": 0.0
                }
            
            dataset_breakdown[dataset_name]["total_videos"] += 1
            if result["evaluation_result"] is not None:
                dataset_breakdown[dataset_name]["successful_evaluations"] += 1
        
        # Calculate dataset averages
        for dataset_name, stats in dataset_breakdown.items():
            dataset_results = [r for r in all_results if r["dataset"] == dataset_name and r["evaluation_result"] is not None]
            if dataset_results:
                stats["avg_character_consistency"] = sum(r["animation_analysis"]["character_consistency_score"] for r in dataset_results) / len(dataset_results)
                stats["avg_technical_artifacts"] = sum(r["animation_analysis"]["technical_artifacts_score"] for r in dataset_results) / len(dataset_results)
                stats["avg_animation_quality"] = sum(r["animation_analysis"]["animation_quality_score"] for r in dataset_results) / len(dataset_results)
        
        report = {
            "test_summary": {
                "total_videos": total_videos,
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": failed_evaluations,
                "success_rate": successful_evaluations / total_videos if total_videos > 0 else 0,
                "average_character_consistency": avg_character_consistency,
                "average_technical_artifacts": avg_technical_artifacts,
                "average_animation_quality": avg_animation_quality,
                "average_overall_score": avg_overall_score,
                "average_evaluation_time": avg_evaluation_time
            },
            "quality_distribution": quality_distribution,
            "issue_detection_stats": issue_stats,
            "dataset_breakdown": dataset_breakdown,
            "detailed_results": all_results,
            "test_metadata": {
                "test_scenario": "Animation Quality Assessment with Existing Datasets",
                "test_timestamp": time.time(),
                "framework_version": "1.0.0",
                "datasets_tested": list(dataset_breakdown.keys())
            }
        }
        
        return report
    
    async def _save_test_results(self, test_report: Dict[str, Any]):
        """Save test results to file"""
        
        # Create results directory
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed report
        report_path = results_dir / "animation_quality_existing_dataset_report.json"
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        # Save summary report
        summary = {
            "test_summary": test_report["test_summary"],
            "quality_distribution": test_report["quality_distribution"],
            "issue_detection_stats": test_report["issue_detection_stats"],
            "dataset_breakdown": test_report["dataset_breakdown"]
        }
        
        summary_path = results_dir / "animation_quality_existing_dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📄 Test results saved to {results_dir}")
        logger.info(f"📊 Detailed report: {report_path}")
        logger.info(f"📋 Summary report: {summary_path}")


async def main():
    """Main function to run the animation quality assessment with existing datasets"""
    
    print("🎬 Animation Quality Assessment with Existing Datasets")
    print("=" * 60)
    
    try:
        # Create test runner
        test_runner = ExistingDatasetAnimationQualityTestRunner()
        
        # Run animation quality tests
        test_report = await test_runner.run_animation_quality_tests()
        
        # Print summary
        print("\n📊 ANIMATION QUALITY ASSESSMENT RESULTS")
        print("=" * 40)
        
        summary = test_report["test_summary"]
        print(f"Total Videos: {summary['total_videos']}")
        print(f"Successful Evaluations: {summary['successful_evaluations']}")
        print(f"Failed Evaluations: {summary['failed_evaluations']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Average Character Consistency: {summary['average_character_consistency']:.3f}")
        print(f"Average Technical Artifacts: {summary['average_technical_artifacts']:.3f}")
        print(f"Average Animation Quality: {summary['average_animation_quality']:.3f}")
        print(f"Average Overall Score: {summary['average_overall_score']:.3f}")
        print(f"Average Evaluation Time: {summary['average_evaluation_time']:.2f}s")
        
        print("\n🎭 QUALITY DISTRIBUTION")
        print("=" * 25)
        for quality, count in test_report["quality_distribution"].items():
            print(f"{quality.title()}: {count} videos")
        
        print("\n🔍 ISSUE DETECTION STATISTICS")
        print("=" * 35)
        for issue, count in test_report["issue_detection_stats"].items():
            print(f"{issue.replace('_', ' ').title()}: {count} videos")
        
        print("\n📊 DATASET BREAKDOWN")
        print("=" * 20)
        for dataset, stats in test_report["dataset_breakdown"].items():
            print(f"\n{dataset}:")
            print(f"  Total Videos: {stats['total_videos']}")
            print(f"  Successful: {stats['successful_evaluations']}")
            print(f"  Avg Character Consistency: {stats['avg_character_consistency']:.3f}")
            print(f"  Avg Technical Artifacts: {stats['avg_technical_artifacts']:.3f}")
            print(f"  Avg Animation Quality: {stats['avg_animation_quality']:.3f}")
        
        print(f"\n✅ Animation quality assessment completed! Results saved to test_results/")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)




