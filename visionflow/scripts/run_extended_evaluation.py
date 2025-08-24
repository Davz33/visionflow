#!/usr/bin/env python3
"""
Extended Evaluation Script
Evaluates videos using the comprehensive test prompt dataset.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visionflow.services.evaluation.video_evaluation_orchestrator import (
    VideoEvaluationOrchestrator, SamplingStrategy
)
from visionflow.services.evaluation.confidence_manager import ConfidenceManager

async def run_extended_evaluation():
    """Run evaluation using extended test prompts"""
    
    # Load extended manifest
    manifest_path = Path("generated/test_dataset/extended_evaluation_manifest.json")
    
    if not manifest_path.exists():
        print("❌ Extended evaluation manifest not found!")
        print("💡 Run: python scripts/expand_test_dataset.py --create-extended")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print("🎯 EXTENDED EVALUATION SYSTEM")
    print("=" * 50)
    print(f"📋 Test prompts: {len(manifest['extended_test_prompts'])}")
    print(f"📹 Existing videos: {len(manifest['existing_videos'])}")
    print()
    
    # Initialize evaluation system
    orchestrator = VideoEvaluationOrchestrator(
        sampling_strategy=SamplingStrategy.ADAPTIVE,
        max_frames_per_video=15
    )
    
    confidence_manager = ConfidenceManager()
    
    # Evaluate existing videos first
    print("🔍 EVALUATING EXISTING VIDEOS")
    print("-" * 40)
    
    for video in manifest['existing_videos']:
        try:
            print(f"📹 {video['filename']}")
            evaluation_result = await orchestrator.evaluate_video(
                video_path=video['path'],
                prompt=video['prompt']
            )
            
            confidence_action = await confidence_manager.process_evaluation(evaluation_result)
            
            print(f"   Score: {evaluation_result.overall_score:.3f}")
            print(f"   Decision: {evaluation_result.decision}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Show extended test prompts for future generation
    print("🚀 EXTENDED TEST PROMPTS FOR FUTURE EVALUATION")
    print("-" * 50)
    
    for i, prompt_data in enumerate(manifest['extended_test_prompts'][:5], 1):
        print(f"{i}. [{prompt_data['category']}] {prompt_data['prompt']}")
        print(f"   Focus: {', '.join(prompt_data['evaluation_focus'])}")
        print(f"   Difficulty: {prompt_data['difficulty']}")
        print()
    
    print(f"... and {len(manifest['extended_test_prompts']) - 5} more prompts")
    print()
    print("💡 TO GENERATE VIDEOS FOR EXTENDED PROMPTS:")
    print("   Use these prompts with your WAN video generation system")
    print("   Then evaluate with the standard evaluation pipeline")

if __name__ == "__main__":
    asyncio.run(run_extended_evaluation())
