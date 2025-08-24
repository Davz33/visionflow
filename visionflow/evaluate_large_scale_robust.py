#!/usr/bin/env python3
"""
Robust Large-Scale Dataset Evaluation Script with Progressive Saving
Implements fail-safe evaluation with individual video error handling and progressive result saving.
"""

import asyncio
import json
import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to Python path - works in both local and Docker
if os.path.exists('/app'):
    # Docker environment
    project_root = Path('/app')
    sys.path.insert(0, '/app')
else:
    # Local environment
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

from visionflow.services.evaluation.video_evaluation_orchestrator import (
    VideoEvaluationOrchestrator, SamplingStrategy
)
from visionflow.services.evaluation.confidence_manager import ConfidenceManager
from visionflow.services.generation.video_metadata_tracker import metadata_tracker


async def evaluate_single_video(
    orchestrator: VideoEvaluationOrchestrator,
    confidence_manager: ConfidenceManager,
    video_info: Dict[str, Any],
    dataset_path: Path,
    video_index: int,
    total_videos: int
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single video with comprehensive error handling.
    Returns evaluation result or None if evaluation fails.
    """
    filename = video_info['filename']
    video_path = video_info.get('path', str(dataset_path / filename))
    prompt = video_info['prompt']
    category = video_info.get('category', 'unknown')
    source_dataset = video_info.get('source_dataset', 'unknown')
    
    print(f"📹 EVALUATING VIDEO {video_index}/{total_videos}")
    print("=" * 50)
    print(f"File: {filename}")
    print(f"Category: {category}")
    print(f"Source Style: {source_dataset}")
    print(f"Prompt: {prompt}")
    print()
    
    eval_start = time.time()
    
    try:
        # Ensure absolute path
        if not Path(video_path).is_absolute():
            video_path = dataset_path / filename
        
        # Verify video file exists and is readable
        if not Path(video_path).exists():
            print(f"❌ Video file not found: {video_path}")
            return None
        
        # Quick video verification
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"❌ Cannot open video file: {video_path}")
                cap.release()
                return None
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            size = Path(video_path).stat().st_size
            
            print(f"✅ Video verified: {frame_count} frames, {fps:.1f} FPS, {duration:.1f}s, {size/(1024*1024):.1f}MB")
            cap.release()
            
        except Exception as e:
            print(f"⚠️ Video verification failed: {e}")
            # Continue anyway - orchestrator might handle it
        
        # Run evaluation with the prompt
        print("🔄 Running evaluation...")
        evaluation_result = await orchestrator.evaluate_video(
            video_path=str(video_path),
            prompt=prompt
        )
        
        # Process through confidence manager
        print("🤖 Processing confidence assessment...")
        confidence_result = await confidence_manager.process_evaluation(evaluation_result)
        
        eval_time = time.time() - eval_start
        
        # Display results safely
        try:
            print(f"📊 EVALUATION RESULTS")
            print("-" * 30)
            print(f"Overall Score: {evaluation_result.overall_score:.3f}")
            print(f"Confidence: {evaluation_result.overall_confidence:.3f}")
            print(f"Level: {evaluation_result.confidence_level.value.upper()}")
            print(f"Decision: {confidence_result.action_type.value}")
            print(f"Review Needed: {confidence_result.requires_human_review}")
            print(f"Processing Time: {eval_time:.2f}s")
            print()
            
            # Show simplified dimension breakdown with error handling
            try:
                print("📈 Top Dimensions:")
                # dimension_scores is a List[DimensionScore], not a dict
                sorted_dims = sorted(
                    evaluation_result.dimension_scores,
                    key=lambda x: x.score,
                    reverse=True
                )
                
                for i, dim_score in enumerate(sorted_dims[:3]):
                    try:
                        dim_name = dim_score.dimension.value.replace('_', ' ').title()
                        score_str = f"{dim_score.score:.3f}"
                        status = "✅" if dim_score.score >= 0.7 else "⚠️"
                        print(f"  {status} {dim_name}: {score_str}")
                    except Exception as e:
                        print(f"  ⚠️ Dimension display error: {e}")
                        
            except Exception as e:
                print(f"⚠️ Could not display dimension breakdown: {e}")
            
            print()
            
            # Approval status
            try:
                if confidence_result.action_type.value == 'auto_approve':
                    print("✅ APPROVED")
                elif confidence_result.action_type.value == 'flag_monitoring':
                    print("🔍 FLAGGED FOR MONITORING")
                else:
                    print("👥 REQUIRES HUMAN REVIEW")
            except Exception as e:
                print(f"⚠️ Could not determine approval status: {e}")
            
        except Exception as e:
            print(f"⚠️ Error displaying results: {e}")
        
        print()
        print("=" * 60)
        print()
        
        # Create result record with safe attribute access
        result_record = {
            'filename': filename,
            'category': category,
            'source_dataset': source_dataset,
            'prompt': prompt,
            'processing_time': eval_time,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'success': True
        }
        
        # Safely extract evaluation results
        try:
            result_record.update({
                'overall_score': evaluation_result.overall_score,
                'confidence': evaluation_result.overall_confidence,
                'confidence_level': evaluation_result.confidence_level.value,
            })
        except Exception as e:
            print(f"⚠️ Error extracting evaluation scores: {e}")
            result_record.update({
                'overall_score': 0.0,
                'confidence': 0.0,
                'confidence_level': 'unknown',
                'extraction_error': str(e)
            })
        
        # Safely extract confidence manager results
        try:
            result_record.update({
                'decision': confidence_result.action_type.value,
                'requires_review': confidence_result.requires_human_review,
            })
        except Exception as e:
            print(f"⚠️ Error extracting confidence results: {e}")
            result_record.update({
                'decision': 'unknown',
                'requires_review': True,
                'confidence_error': str(e)
            })
        
        # Safely extract dimension scores
        try:
            result_record['dimension_scores'] = {
                dim_score.dimension.value: dim_score.score 
                for dim_score in evaluation_result.dimension_scores
            }
        except Exception as e:
            print(f"⚠️ Error extracting dimension scores: {e}")
            result_record['dimension_scores'] = {}
            result_record['dimension_error'] = str(e)
        
        return result_record
        
    except Exception as e:
        print(f"❌ Error evaluating {filename}: {e}")
        print(f"📋 Traceback:")
        traceback.print_exc()
        print("=" * 60)
        print()
        
        # Return error record
        return {
            'filename': filename,
            'category': category,
            'source_dataset': source_dataset,
            'prompt': prompt,
            'processing_time': time.time() - eval_start,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'success': False,
            'error': str(e),
            'overall_score': 0.0,
            'confidence': 0.0,
            'confidence_level': 'error',
            'decision': 'error',
            'requires_review': True,
            'dimension_scores': {}
        }


def save_progress(
    results: List[Dict[str, Any]], 
    dataset_path: Path, 
    total_videos: int, 
    current_index: int,
    start_time: float,
    failed_videos: List[str]
) -> None:
    """Save current progress to prevent data loss"""
    try:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Calculate statistics for successful evaluations
        successful_results = [r for r in results if r.get('success', False)]
        
        progress_data = {
            'metadata': {
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_videos': total_videos,
                'processed_videos': len(results),
                'successful_videos': len(successful_results),
                'failed_videos': len(failed_videos),
                'current_index': current_index,
                'dataset_path': str(dataset_path),
                'elapsed_time': elapsed_time,
                'estimated_total_time': elapsed_time * total_videos / max(current_index, 1),
                'last_update': current_time,
                'failed_video_list': failed_videos
            },
            'results': results
        }
        
        # Add summary statistics if we have successful results
        if successful_results:
            avg_score = sum(r['overall_score'] for r in successful_results) / len(successful_results)
            avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
            avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            
            progress_data['summary'] = {
                'avg_score': avg_score,
                'avg_confidence': avg_confidence,
                'avg_processing_time': avg_time,
                'success_rate': len(successful_results) / len(results),
            }
        
        # Save to both progress and results files
        results_path = dataset_path / "evaluation_results.json"
        progress_path = dataset_path / "evaluation_progress.json"
        
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        with open(results_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"💾 Progress saved: {len(results)}/{total_videos} videos ({len(successful_results)} successful)")
            
    except Exception as e:
        print(f"⚠️ Failed to save progress: {e}")


async def evaluate_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    """Evaluate all videos in the large-scale dataset with robust error handling"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return []
    
    manifest_path = dataset_path / "dataset_manifest.json"
    if not manifest_path.exists():
        print(f"❌ Dataset manifest not found: {manifest_path}")
        return []
    
    print(f"📁 Loading dataset from: {dataset_path}")
    
    with open(manifest_path, 'r') as f:
        dataset = json.load(f)
    
    videos = dataset.get('videos', [])
    if not videos:
        print("❌ No videos found in dataset")
        return []
    
    print(f"📊 Found {len(videos)} videos to evaluate")
    print()
    
    # Check for existing results to resume from
    results_path = dataset_path / "evaluation_results.json"
    evaluation_results = []
    processed_files = set()
    
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                existing_data = json.load(f)
                evaluation_results = existing_data.get('results', [])
                processed_files = {r['filename'] for r in evaluation_results}
                print(f"📋 Found existing results - {len(processed_files)} videos already processed")
        except Exception as e:
            print(f"⚠️ Could not load existing results: {e}")
    
    # Filter out already processed videos
    videos_to_process = [v for v in videos if v['filename'] not in processed_files]
    
    if not videos_to_process:
        print("✅ All videos already processed!")
        return evaluation_results
    
    print(f"🔄 Processing {len(videos_to_process)} remaining videos")
    print()
    
    # Initialize the evaluation system
    print("🔧 Initializing WAN Video Evaluation System...")
    
    try:
        orchestrator = VideoEvaluationOrchestrator(
            sampling_strategy=SamplingStrategy.ADAPTIVE,
            max_frames_per_video=20
        )
        
        confidence_manager = ConfidenceManager(
            monitoring_window_hours=1,
            low_confidence_threshold=0.3,
            high_automation_threshold=0.9
        )
        
        print("✅ System ready")
        print()
        
    except Exception as e:
        print(f"❌ Failed to initialize evaluation system: {e}")
        return evaluation_results
    
    # Process videos with progressive saving
    start_time = time.time()
    failed_videos = []
    
    for i, video_info in enumerate(videos_to_process, 1):
        print(f"\n🎯 PROCESSING VIDEO {len(evaluation_results) + 1}/{len(videos)} (Batch {i}/{len(videos_to_process)})")
        
        # Evaluate single video
        result = await evaluate_single_video(
            orchestrator=orchestrator,
            confidence_manager=confidence_manager,
            video_info=video_info,
            dataset_path=dataset_path,
            video_index=len(evaluation_results) + 1,
            total_videos=len(videos)
        )
        
        if result:
            evaluation_results.append(result)
            
            if not result.get('success', False):
                failed_videos.append(result['filename'])
        else:
            failed_videos.append(video_info['filename'])
        
        # Save progress after each video
        save_progress(
            results=evaluation_results,
            dataset_path=dataset_path,
            total_videos=len(videos),
            current_index=len(evaluation_results),
            start_time=start_time,
            failed_videos=failed_videos
        )
        
        # Brief pause to prevent system overload
        await asyncio.sleep(0.5)
    
    # Generate final summary
    total_time = time.time() - start_time
    successful_results = [r for r in evaluation_results if r.get('success', False)]
    
    print("\n" + "="*70)
    print("📊 LARGE-SCALE DATASET EVALUATION COMPLETE")
    print("="*70)
    
    if successful_results:
        avg_score = sum(r['overall_score'] for r in successful_results) / len(successful_results)
        avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        
        print(f"🎯 OVERALL PERFORMANCE:")
        print(f"   📊 Videos Processed: {len(evaluation_results)}/{len(videos)}")
        print(f"   ✅ Successful: {len(successful_results)} ({len(successful_results)/len(evaluation_results)*100:.1f}%)")
        print(f"   ❌ Failed: {len(failed_videos)} ({len(failed_videos)/len(evaluation_results)*100:.1f}%)")
        print(f"   ⭐ Average Score: {avg_score:.3f}")
        print(f"   🎯 Average Confidence: {avg_confidence:.3f}")
        print(f"   ⏱️  Average Processing Time: {avg_time:.2f}s")
        print(f"   🕒 Total Evaluation Time: {total_time:.1f}s")
        print()
        
        # Show failed videos if any
        if failed_videos:
            print(f"⚠️ FAILED VIDEOS ({len(failed_videos)}):")
            for filename in failed_videos:
                print(f"   ❌ {filename}")
            print()
        
        print("✅ Evaluation completed with progressive saving!")
        print(f"📄 Results saved to: {results_path}")
        
    else:
        print("❌ No successful evaluations completed")
    
    return evaluation_results


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Robust evaluation of large-scale text-to-video datasets")
    parser.add_argument("--dataset-dir", type=str, 
                       default="evaluation_datasets/large_scale_samples",
                       help="Directory containing the dataset to evaluate")
    
    args = parser.parse_args()
    
    # Environment detection
    env_name = "DOCKER" if os.path.exists('/app') else "LOCAL"
    
    print(f"🎬 WAN VIDEO EVALUATION SYSTEM - ROBUST LARGE-SCALE EVALUATION ({env_name})")
    print("Progressive saving with individual video error handling")
    print("=" * 80)
    print()
    
    try:
        results = asyncio.run(evaluate_dataset(args.dataset_dir))
        
        if results:
            successful_count = len([r for r in results if r.get('success', False)])
            print(f"\n🚀 EVALUATION SUMMARY:")
            print(f"✅ Total videos processed: {len(results)}")
            print(f"✅ Successful evaluations: {successful_count}")
            print(f"✅ Progressive saving: Enabled")
            print(f"✅ Error resilience: Demonstrated")
            
        else:
            print("\n⚠️ No results - check dataset path and format")
    
    except Exception as e:
        print(f"\n❌ Evaluation system failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
