"""
Test Script for Animation Quality Assessment (Test Scenario 1)
Comprehensive testing of the animation quality assessment framework
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
from visionflow.services.evaluation.test_animation_quality_dataset import (
    get_test_animation_dataset,
    generate_test_dataset
)
from visionflow.shared.monitoring import get_logger

logger = get_logger("test_animation_quality_scenario1")


class AnimationQualityTestRunner:
    """
    Test runner for animation quality assessment (Test Scenario 1)
    """
    
    def __init__(self):
        self.orchestrator = VideoEvaluationOrchestrator()
        self.test_dataset = get_test_animation_dataset()
        self.results = []
        
        logger.info("🧪 Animation Quality Test Runner initialized")
    
    async def run_complete_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite for animation quality assessment"""
        
        logger.info("🚀 Starting Animation Quality Assessment Test Suite (Scenario 1)")
        
        # Initialize the orchestrator (no async init needed)
        logger.info("🎯 Using enhanced VideoEvaluationOrchestrator with animation-specific dimensions")
        
        # Generate test dataset
        logger.info("📹 Generating test dataset...")
        test_videos = await generate_test_dataset()
        
        # Run assessments on all test videos
        logger.info("🔍 Running quality assessments...")
        assessment_results = []
        
        for i, video_metadata in enumerate(test_videos):
            logger.info(f"📊 Assessing video {i+1}/{len(test_videos)}: {video_metadata.description}")
            
            try:
                # Run evaluation using enhanced orchestrator
                start_time = time.time()
                evaluation_result = await self.orchestrator.evaluate_video(
                    video_path=video_metadata.video_path,
                    prompt=video_metadata.original_prompt,
                    evaluation_id=f"test_{i+1}"
                )
                assessment_time = time.time() - start_time
                
                # Compare with expected results
                validation_result = self._validate_evaluation(evaluation_result, video_metadata)
                
                # Store results
                result = {
                    "video_metadata": video_metadata,
                    "evaluation_result": evaluation_result,
                    "validation": validation_result,
                    "assessment_time": assessment_time
                }
                assessment_results.append(result)
                
                logger.info(f"✅ Assessment {i+1} completed in {assessment_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Assessment {i+1} failed: {e}")
                assessment_results.append({
                    "video_metadata": video_metadata,
                    "evaluation_result": None,
                    "validation": {"valid": False, "error": str(e)},
                    "assessment_time": 0
                })
        
        # Generate comprehensive test report
        test_report = self._generate_test_report(assessment_results)
        
        # Save results
        await self._save_test_results(test_report)
        
        logger.info("🎉 Animation Quality Assessment Test Suite completed!")
        return test_report
    
    def _validate_evaluation(self, evaluation_result, expected_metadata) -> Dict[str, Any]:
        """Validate evaluation results against expected issues"""
        
        validation = {
            "valid": True,
            "accuracy_score": 0.0,
            "detected_issues": [],
            "missed_issues": [],
            "false_positives": [],
            "severity_match": False,
            "overall_score_reasonable": False
        }
        
        try:
            if not evaluation_result:
                validation["valid"] = False
                validation["error"] = "No evaluation result"
                return validation
            
            # Extract scores from evaluation result
            dimension_scores = {ds.dimension: ds.score for ds in evaluation_result.dimension_scores}
            
            # Map expected issues to evaluation dimensions
            expected_issue_mapping = {
                "character_drift": EvaluationDimension.CHARACTER_CONSISTENCY,
                "character_inconsistency": EvaluationDimension.CHARACTER_CONSISTENCY,
                "technical_artifact": EvaluationDimension.TECHNICAL_ARTIFACTS,
                "generation_failure": EvaluationDimension.TECHNICAL_ARTIFACTS,
                "motion_inconsistency": EvaluationDimension.MOTION_CONSISTENCY,
                "color_inconsistency": EvaluationDimension.VISUAL_QUALITY,
                "resolution_drop": EvaluationDimension.TECHNICAL_ARTIFACTS,
                "frame_corruption": EvaluationDimension.TECHNICAL_ARTIFACTS
            }
            
            # Check which expected issues are detected based on low scores
            detected_issues = []
            for expected_issue in expected_metadata.expected_issues:
                if expected_issue.value in expected_issue_mapping:
                    dimension = expected_issue_mapping[expected_issue.value]
                    if dimension in dimension_scores and dimension_scores[dimension] < 0.6:
                        detected_issues.append(expected_issue.value)
            
            # Calculate accuracy
            expected_issue_names = [issue.value for issue in expected_metadata.expected_issues]
            true_positives = set(detected_issues).intersection(set(expected_issue_names))
            false_negatives = set(expected_issue_names) - set(detected_issues)
            
            validation["detected_issues"] = list(true_positives)
            validation["missed_issues"] = list(false_negatives)
            
            # Calculate accuracy score
            if expected_issue_names:
                precision = len(true_positives) / len(detected_issues) if detected_issues else 0
                recall = len(true_positives) / len(expected_issue_names)
                validation["accuracy_score"] = (precision + recall) / 2 if (precision + recall) > 0 else 0
            else:
                # No issues expected - check if scores are high
                low_scores = [score for score in dimension_scores.values() if score < 0.6]
                validation["accuracy_score"] = 1.0 if not low_scores else 0.0
            
            # Check if overall score is reasonable
            overall_score = evaluation_result.overall_score
            if expected_issue_names:
                # Issues expected - score should be lower
                validation["overall_score_reasonable"] = overall_score < 0.8
            else:
                # No issues expected - score should be higher
                validation["overall_score_reasonable"] = overall_score > 0.7
            
            # Overall validation
            validation["valid"] = (
                validation["accuracy_score"] > 0.5 and
                validation["overall_score_reasonable"]
            )
            
        except Exception as e:
            validation["valid"] = False
            validation["error"] = str(e)
        
        return validation
    
    def _generate_test_report(self, assessment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_tests = len(assessment_results)
        successful_tests = sum(1 for r in assessment_results if r["validation"]["valid"])
        failed_tests = total_tests - successful_tests
        
        # Calculate average metrics
        accuracy_scores = [r["validation"]["accuracy_score"] for r in assessment_results if r["validation"]["valid"]]
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        
        assessment_times = [r["assessment_time"] for r in assessment_results]
        avg_assessment_time = sum(assessment_times) / len(assessment_times) if assessment_times else 0
        
        # Issue detection statistics
        issue_detection_stats = self._calculate_issue_detection_stats(assessment_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(assessment_results)
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "average_accuracy": avg_accuracy,
                "average_assessment_time": avg_assessment_time
            },
            "issue_detection_stats": issue_detection_stats,
            "detailed_results": assessment_results,
            "recommendations": recommendations,
            "test_metadata": {
                "test_scenario": "Animation Quality Assessment Data (Test Scenario 1)",
                "test_timestamp": time.time(),
                "framework_version": "1.0.0"
            }
        }
        
        return report
    
    def _calculate_issue_detection_stats(self, assessment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for issue detection performance"""
        
        stats = {
            "character_drift": {"detected": 0, "expected": 0, "accuracy": 0.0},
            "technical_artifacts": {"detected": 0, "expected": 0, "accuracy": 0.0},
            "color_inconsistency": {"detected": 0, "expected": 0, "accuracy": 0.0},
            "resolution_drop": {"detected": 0, "expected": 0, "accuracy": 0.0},
            "frame_corruption": {"detected": 0, "expected": 0, "accuracy": 0.0},
            "motion_inconsistency": {"detected": 0, "expected": 0, "accuracy": 0.0}
        }
        
        for result in assessment_results:
            if not result["validation"]["valid"]:
                continue
                
            metadata = result["video_metadata"]
            validation = result["validation"]
            
            # Count expected issues
            for issue_type in metadata.expected_issues:
                issue_name = issue_type.value
                if issue_name in stats:
                    stats[issue_name]["expected"] += 1
            
            # Count detected issues
            for detected_issue in validation["detected_issues"]:
                if detected_issue in stats:
                    stats[detected_issue]["detected"] += 1
        
        # Calculate accuracy for each issue type
        for issue_type, data in stats.items():
            if data["expected"] > 0:
                data["accuracy"] = data["detected"] / data["expected"]
            else:
                data["accuracy"] = 1.0  # No issues expected
        
        return stats
    
    def _generate_recommendations(self, assessment_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Analyze overall performance
        successful_tests = [r for r in assessment_results if r["validation"]["valid"]]
        success_rate = len(successful_tests) / len(assessment_results) if assessment_results else 0
        
        if success_rate < 0.8:
            recommendations.append("Overall accuracy is below 80%. Consider improving issue detection algorithms.")
        
        # Analyze specific issue types
        issue_stats = self._calculate_issue_detection_stats(assessment_results)
        
        for issue_type, stats in issue_stats.items():
            if stats["accuracy"] < 0.7:
                recommendations.append(f"Low accuracy for {issue_type} detection ({stats['accuracy']:.2f}). Review detection logic.")
        
        # Analyze assessment time
        avg_time = sum(r["assessment_time"] for r in assessment_results) / len(assessment_results)
        if avg_time > 30:  # More than 30 seconds per assessment
            recommendations.append("Assessment time is high. Consider optimizing VLM model usage or frame sampling.")
        
        # General recommendations
        recommendations.extend([
            "Consider implementing confidence thresholds for issue detection to reduce false positives.",
            "Add more diverse test cases to improve robustness.",
            "Implement real-time quality monitoring for production use.",
            "Consider ensemble methods combining multiple VLM models for better accuracy."
        ])
        
        return recommendations
    
    async def _save_test_results(self, test_report: Dict[str, Any]):
        """Save test results to file"""
        
        # Create results directory
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed report
        report_path = results_dir / "animation_quality_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        # Save summary report
        summary = {
            "test_summary": test_report["test_summary"],
            "issue_detection_stats": test_report["issue_detection_stats"],
            "recommendations": test_report["recommendations"]
        }
        
        summary_path = results_dir / "animation_quality_test_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📄 Test results saved to {results_dir}")
        logger.info(f"📊 Detailed report: {report_path}")
        logger.info(f"📋 Summary report: {summary_path}")


async def main():
    """Main function to run the animation quality assessment test suite"""
    
    print("🎬 Animation Quality Assessment Test Suite (Test Scenario 1)")
    print("=" * 60)
    
    try:
        # Create test runner
        test_runner = AnimationQualityTestRunner()
        
        # Run complete test suite
        test_report = await test_runner.run_complete_test_suite()
        
        # Print summary
        print("\n📊 TEST RESULTS SUMMARY")
        print("=" * 30)
        
        summary = test_report["test_summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Average Accuracy: {summary['average_accuracy']:.2f}")
        print(f"Average Assessment Time: {summary['average_assessment_time']:.2f}s")
        
        print("\n🔍 ISSUE DETECTION STATISTICS")
        print("=" * 35)
        
        for issue_type, stats in test_report["issue_detection_stats"].items():
            print(f"{issue_type.replace('_', ' ').title()}:")
            print(f"  Expected: {stats['expected']}")
            print(f"  Detected: {stats['detected']}")
            print(f"  Accuracy: {stats['accuracy']:.2f}")
            print()
        
        print("💡 RECOMMENDATIONS")
        print("=" * 20)
        for i, rec in enumerate(test_report["recommendations"], 1):
            print(f"{i}. {rec}")
        
        print(f"\n✅ Test suite completed! Results saved to test_results/")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        print(f"❌ Test suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
