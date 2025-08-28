#!/usr/bin/env python3
"""
Test the evaluation system with existing videos.
This script tests the evaluation system using videos that are already available.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visionflow.services.evaluation.video_evaluation_orchestrator import (
    VideoEvaluationOrchestrator, SamplingStrategy
)
from visionflow.services.evaluation.confidence_manager import ConfidenceManager
from visionflow.services.evaluation.industry_metrics import IndustryMetrics

# Test video configuration
TEST_VIDEOS = [
    {
        "name": "wan_generated_01",
        "path": "generated/wan_video_d954a5bf.mp4",
        "prompt": "A beautiful sunset over a calm ocean with gentle waves, cinematic lighting, peaceful atmosphere",
        "category": "nature",
        "expected_duration": "5s",
        "evaluation_focus": ["visual_quality", "motion_smoothness", "lighting_realism"]
    },
    {
        "name": "test_dataset_01", 
        "path": "generated/test_dataset/test_video_01.mp4",
        "prompt": "Test video with consistent motion and spatial coherence",
        "category": "test",
        "expected_duration": "10s",
        "evaluation_focus": ["spatial_consistency", "temporal_smoothness", "object_stability"]
    },
    {
        "name": "test_dataset_02",
        "path": "generated/test_dataset/test_video_02.mp4", 
        "prompt": "Test video with natural movement and realistic physics",
        "category": "test",
        "expected_duration": "10s",
        "evaluation_focus": ["physics_realism", "motion_naturalness", "temporal_consistency"]
    },
    {
        "name": "diverse_sample_01",
        "path": "evaluation_datasets/large_scale_samples/diverse_01.mp4",
        "prompt": "Diverse content sample with multiple evaluation dimensions",
        "category": "diverse",
        "expected_duration": "10s",
        "evaluation_focus": ["overall_quality", "content_diversity", "technical_consistency"]
    }
]

async def test_evaluation_with_existing_videos():
    """Test the evaluation system with existing videos"""
    
    print("🎬 Testing Video Evaluation System with Existing Videos")
    print("=" * 60)
    
    # Initialize the evaluation system
    print("🚀 Initializing evaluation system...")
    orchestrator = VideoEvaluationOrchestrator(
        sampling_strategy=SamplingStrategy.ADAPTIVE,
        max_frames_per_video=20
    )
    
    confidence_manager = ConfidenceManager()
    industry_metrics = IndustryMetrics()
    
    print("✅ System initialized")
    print()
    
    # Test each video
    results = []
    total_videos = len(TEST_VIDEOS)
    
    for i, video_info in enumerate(TEST_VIDEOS, 1):
        video_path = Path(video_info["path"])
        
        if not video_path.exists():
            print(f"⚠️  Video file not found: {video_path}")
            continue
        
        print(f"📹 Evaluating {i}/{total_videos}: {video_info['name']}")
        print(f"   Path: {video_info['path']}")
        print(f"   Category: {video_info['category']}")
        print(f"   Prompt: {video_info['prompt']}")
        print(f"   Expected Duration: {video_info['expected_duration']}")
        
        try:
            # Run evaluation
            result = await orchestrator.evaluate_video(
                video_path=str(video_path),
                prompt=video_info['prompt']
            )
            
            # Store result
            video_result = {
                "name": video_info['name'],
                "category": video_info['category'],
                "path": video_info['path'],
                "prompt": video_info['prompt'],
                "overall_score": result.overall_score,
                "overall_confidence": result.overall_confidence,
                "decision": str(result.decision),
                "dimension_scores": {
                    "visual_quality": getattr(result.dimension_scores, 'visual_quality', 0.0),
                    "perceptual_quality": getattr(result.dimension_scores, 'perceptual_quality', 0.0),
                    "motion_consistency": getattr(result.dimension_scores, 'motion_consistency', 0.0),
                    "text_video_alignment": getattr(result.dimension_scores, 'text_video_alignment', 0.0),
                    "aesthetic_quality": getattr(result.dimension_scores, 'aesthetic_quality', 0.0),
                    "narrative_flow": getattr(result.dimension_scores, 'narrative_flow', 0.0)
                },
                "evaluation_time": getattr(result, 'evaluation_time', 0.0),
                "frames_evaluated": getattr(result, 'frames_evaluated', 0)
            }
            
            results.append(video_result)
            
            print(f"   ✅ Evaluation completed")
            print(f"   📊 Overall Score: {result.overall_score:.3f}")
            print(f"   🔍 Confidence: {result.overall_confidence:.3f}")
            print(f"   📋 Decision: {result.decision}")
            print(f"   ⏱️  Time: {getattr(result, 'evaluation_time', 0.0):.2f}s")
            print(f"   🎞️  Frames: {getattr(result, 'frames_evaluated', 0)}")
            
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            results.append({
                "name": video_info['name'],
                "category": video_info['category'],
                "path": video_info['path'],
                "error": str(e)
            })
        
        print()
    
    # Generate evaluation report
    generate_evaluation_report(results)
    
    print("🎉 Test evaluation completed!")
    print(f"📊 Total videos tested: {len(results)}")
    print(f"✅ Successful evaluations: {len([r for r in results if 'error' not in r])}")
    print(f"❌ Failed evaluations: {len([r for r in results if 'error' in r])}")

def generate_evaluation_report(results):
    """Generate a comprehensive evaluation report"""
    
    # Calculate statistics
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    if successful_results:
        avg_score = sum(r['overall_score'] for r in successful_results) / len(successful_results)
        avg_confidence = sum(r['overall_confidence'] for r in successful_results) / len(successful_results)
        avg_time = sum(r.get('evaluation_time', 0) for r in successful_results) / len(successful_results)
    else:
        avg_score = avg_confidence = avg_time = 0.0
    
    # Create report
    report = {
        "metadata": {
            "test_name": "Existing Videos Evaluation Test",
            "timestamp": asyncio.get_event_loop().time(),
            "total_videos": len(results),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(failed_results)
        },
        "summary": {
            "average_overall_score": avg_score,
            "average_confidence": avg_confidence,
            "average_evaluation_time": avg_time,
            "success_rate": len(successful_results) / len(results) if results else 0.0
        },
        "results": results,
        "analysis": {
            "score_distribution": analyze_score_distribution(successful_results),
            "confidence_distribution": analyze_confidence_distribution(successful_results),
            "category_performance": analyze_category_performance(successful_results)
        }
    }
    
    # Save report
    report_file = Path("evaluation_datasets/evaluation_test_report.json")
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Evaluation report saved: {report_file}")
    
    # Print summary
    print("\n📊 EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Total Videos: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Success Rate: {len(successful_results) / len(results) * 100:.1f}%")
    
    if successful_results:
        print(f"Average Score: {avg_score:.3f}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Average Time: {avg_time:.2f}s")
    
    # Print category performance
    if successful_results:
        print("\n📈 CATEGORY PERFORMANCE")
        print("-" * 30)
        category_perf = analyze_category_performance(successful_results)
        for category, perf in category_perf.items():
            print(f"{category}: {perf['avg_score']:.3f} (n={perf['count']})")

def analyze_score_distribution(results):
    """Analyze the distribution of overall scores"""
    if not results:
        return {}
    
    scores = [r['overall_score'] for r in results]
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "count": len(scores)
    }

def analyze_confidence_distribution(results):
    """Analyze the distribution of confidence scores"""
    if not results:
        return {}
    
    confidences = [r['overall_confidence'] for r in results]
    return {
        "min": min(confidences),
        "max": max(confidences),
        "mean": sum(confidences) / len(confidences),
        "count": len(confidences)
    }

def analyze_category_performance(results):
    """Analyze performance by category"""
    if not results:
        return {}
    
    category_data = {}
    for result in results:
        category = result['category']
        if category not in category_data:
            category_data[category] = []
        category_data[category].append(result['overall_score'])
    
    category_performance = {}
    for category, scores in category_data.items():
        category_performance[category] = {
            "avg_score": sum(scores) / len(scores),
            "count": len(scores),
            "min_score": min(scores),
            "max_score": max(scores)
        }
    
    return category_performance

if __name__ == "__main__":
    asyncio.run(test_evaluation_with_existing_videos())
