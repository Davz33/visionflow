#!/usr/bin/env python3
"""
Expand test dataset with more diverse prompts and create comprehensive evaluation sets.
Works with existing local data and provides guidance for external datasets.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import random
from datetime import datetime

@dataclass
class TestPrompt:
    """A test prompt with metadata"""
    text: str
    category: str
    difficulty: str
    evaluation_focus: List[str]

class TestDatasetExpander:
    """Expand existing test dataset with more comprehensive test cases"""
    
    def __init__(self, base_dir: str = "generated"):
        self.base_dir = Path(base_dir)
        self.test_dataset_dir = self.base_dir / "test_dataset"
        
        # Comprehensive test prompts categorized by evaluation focus
        self.prompt_categories = {
            "visual_quality": [
                TestPrompt(
                    "A crystal clear mountain lake reflecting snow-capped peaks at golden hour",
                    "landscape", "high", ["visual_quality", "aesthetic_quality"]
                ),
                TestPrompt(
                    "Extreme close-up of dewdrops on a spider web with morning sunlight",
                    "macro", "high", ["visual_quality", "motion_consistency"]
                ),
                TestPrompt(
                    "Professional photography studio with dramatic lighting and shadows",
                    "studio", "medium", ["visual_quality", "aesthetic_quality"]
                )
            ],
            "motion_consistency": [
                TestPrompt(
                    "A pendulum swinging in perfect rhythm with consistent motion",
                    "mechanical", "high", ["motion_consistency", "visual_quality"]
                ),
                TestPrompt(
                    "Leaves falling from a tree in autumn wind, natural movement",
                    "nature", "medium", ["motion_consistency", "aesthetic_quality"]
                ),
                TestPrompt(
                    "A dancer performing fluid movements in slow motion",
                    "performance", "high", ["motion_consistency", "aesthetic_quality"]
                )
            ],
            "text_alignment": [
                TestPrompt(
                    "A red Ferrari sports car driving on a winding mountain road",
                    "vehicles", "medium", ["text_video_alignment", "motion_consistency"]
                ),
                TestPrompt(
                    "A chef chopping vegetables with a sharp knife on wooden cutting board",
                    "cooking", "medium", ["text_video_alignment", "motion_consistency"]
                ),
                TestPrompt(
                    "Three golden retriever puppies playing with a blue frisbee in green grass",
                    "animals", "high", ["text_video_alignment", "motion_consistency"]
                )
            ],
            "aesthetic_quality": [
                TestPrompt(
                    "Minimalist zen garden with raked sand patterns and single stone",
                    "artistic", "high", ["aesthetic_quality", "visual_quality"]
                ),
                TestPrompt(
                    "Abstract watercolor painting coming to life with flowing colors",
                    "abstract", "high", ["aesthetic_quality", "motion_consistency"]
                ),
                TestPrompt(
                    "Elegant calligraphy being written with traditional brush and ink",
                    "calligraphy", "medium", ["aesthetic_quality", "motion_consistency"]
                )
            ],
            "narrative_flow": [
                TestPrompt(
                    "A story unfolds: morning coffee, commute, office work, evening return",
                    "daily_life", "high", ["narrative_flow", "text_video_alignment"]
                ),
                TestPrompt(
                    "Seasons changing from spring bloom to winter snow in same location",
                    "temporal", "high", ["narrative_flow", "visual_quality"]
                ),
                TestPrompt(
                    "A seed growing into a flower: planting, sprouting, blooming sequence",
                    "growth", "medium", ["narrative_flow", "motion_consistency"]
                )
            ]
        }
    
    def create_comprehensive_prompts(self) -> List[Dict[str, Any]]:
        """Create a comprehensive set of test prompts"""
        all_prompts = []
        
        for category, prompts in self.prompt_categories.items():
            for i, prompt in enumerate(prompts, 1):
                all_prompts.append({
                    "id": f"{category}_{i:02d}",
                    "prompt": prompt.text,
                    "category": prompt.category,
                    "difficulty": prompt.difficulty,
                    "evaluation_focus": prompt.evaluation_focus,
                    "expected_duration": "3-6 seconds",
                    "quality_setting": "high" if prompt.difficulty == "high" else "medium"
                })
        
        return all_prompts
    
    def create_evaluation_benchmark(self) -> Dict[str, Any]:
        """Create a benchmark configuration for systematic evaluation"""
        
        benchmark = {
            "metadata": {
                "name": "WAN Video Evaluation Benchmark",
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "description": "Comprehensive benchmark for evaluating WAN video generation quality"
            },
            "evaluation_dimensions": {
                "visual_quality": {
                    "weight": 0.25,
                    "metrics": ["sharpness", "clarity", "color_accuracy", "noise_level"],
                    "threshold_excellent": 0.9,
                    "threshold_good": 0.7,
                    "threshold_acceptable": 0.5
                },
                "motion_consistency": {
                    "weight": 0.20,
                    "metrics": ["temporal_coherence", "motion_smoothness", "flicker_detection"],
                    "threshold_excellent": 0.85,
                    "threshold_good": 0.65,
                    "threshold_acceptable": 0.45
                },
                "text_video_alignment": {
                    "weight": 0.25,
                    "metrics": ["semantic_alignment", "object_presence", "scene_accuracy"],
                    "threshold_excellent": 0.9,
                    "threshold_good": 0.75,
                    "threshold_acceptable": 0.6
                },
                "aesthetic_quality": {
                    "weight": 0.15,
                    "metrics": ["composition", "lighting", "color_harmony"],
                    "threshold_excellent": 0.85,
                    "threshold_good": 0.7,
                    "threshold_acceptable": 0.55
                },
                "narrative_flow": {
                    "weight": 0.15,
                    "metrics": ["coherence", "progression", "storytelling"],
                    "threshold_excellent": 0.8,
                    "threshold_good": 0.65,
                    "threshold_acceptable": 0.5
                }
            },
            "sampling_strategies": {
                "quick_test": {
                    "max_frames": 5,
                    "strategy": "keyframe_only",
                    "description": "Fast evaluation for development"
                },
                "standard_test": {
                    "max_frames": 10,
                    "strategy": "adaptive",
                    "description": "Standard evaluation for most cases"
                },
                "comprehensive_test": {
                    "max_frames": 20,
                    "strategy": "temporal_stratified", 
                    "description": "Thorough evaluation for benchmarking"
                }
            },
            "test_prompts": self.create_comprehensive_prompts()
        }
        
        return benchmark
    
    def generate_extended_manifest(self) -> bool:
        """Generate an extended manifest with comprehensive test cases"""
        
        # Load existing manifest if it exists
        existing_manifest_path = self.test_dataset_dir / "dataset_manifest.json"
        existing_videos = []
        
        if existing_manifest_path.exists():
            with open(existing_manifest_path, 'r') as f:
                existing_data = json.load(f)
                existing_videos = existing_data.get('videos', [])
        
        # Create comprehensive benchmark
        benchmark = self.create_evaluation_benchmark()
        
        # Extended manifest
        extended_manifest = {
            "metadata": {
                "name": "Extended WAN Video Evaluation Dataset",
                "description": "Comprehensive dataset with existing videos and extended test prompts",
                "created_at": datetime.now().isoformat(),
                "existing_videos": len(existing_videos),
                "extended_prompts": len(benchmark["test_prompts"]),
                "total_test_cases": len(existing_videos) + len(benchmark["test_prompts"])
            },
            "existing_videos": existing_videos,
            "extended_test_prompts": benchmark["test_prompts"],
            "evaluation_benchmark": benchmark["evaluation_dimensions"],
            "sampling_strategies": benchmark["sampling_strategies"],
            "usage_instructions": {
                "basic_evaluation": "Use existing_videos for current functionality testing",
                "comprehensive_evaluation": "Generate videos for extended_test_prompts for full benchmarking",
                "benchmark_comparison": "Compare results against evaluation_benchmark thresholds",
                "sampling_strategy": "Choose appropriate strategy based on testing needs"
            }
        }
        
        # Save extended manifest
        extended_manifest_path = self.test_dataset_dir / "extended_evaluation_manifest.json"
        with open(extended_manifest_path, 'w') as f:
            json.dump(extended_manifest, f, indent=2)
        
        print(f"📋 Extended evaluation manifest created: {extended_manifest_path}")
        return True
    
    def suggest_external_datasets(self) -> None:
        """Provide guidance on obtaining external datasets"""
        
        print("\n🌐 EXTERNAL DATASET RECOMMENDATIONS")
        print("=" * 60)
        
        external_datasets = [
            {
                "name": "WebVid-10M",
                "url": "https://github.com/m-bain/webvid",
                "description": "Large-scale text-video dataset", 
                "size": "10M text-video pairs",
                "use_case": "Text-to-video alignment evaluation",
                "access": "Download CSV with video URLs"
            },
            {
                "name": "MSR-VTT",
                "url": "https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/",
                "description": "Video description dataset",
                "size": "10K videos, 200K captions", 
                "use_case": "Video captioning and alignment",
                "access": "Request from Microsoft Research"
            },
            {
                "name": "UCF-101",
                "url": "https://www.crcv.ucf.edu/data/UCF101.php",
                "description": "Action recognition dataset",
                "size": "13K videos, 101 action classes",
                "use_case": "Motion consistency evaluation", 
                "access": "Direct download available"
            },
            {
                "name": "Kinetics-700",
                "url": "https://deepmind.com/research/open-source/kinetics",
                "description": "Large-scale action recognition",
                "size": "650K videos, 700 classes",
                "use_case": "Comprehensive motion evaluation",
                "access": "YouTube URLs provided"
            }
        ]
        
        for dataset in external_datasets:
            print(f"\n📹 {dataset['name']}")
            print(f"   🔗 URL: {dataset['url']}")
            print(f"   📝 Description: {dataset['description']}")
            print(f"   📊 Size: {dataset['size']}")
            print(f"   🎯 Use case: {dataset['use_case']}")
            print(f"   📥 Access: {dataset['access']}")
    
    def create_evaluation_script(self) -> None:
        """Create a script to evaluate using extended prompts"""
        
        script_content = '''#!/usr/bin/env python3
"""
Extended Evaluation Script
Evaluates videos using the comprehensive test prompt dataset.
"""

import asyncio
import json
from pathlib import Path
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
'''
        
        script_path = Path("scripts/run_extended_evaluation.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        print(f"📝 Extended evaluation script created: {script_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Expand test dataset with comprehensive evaluation prompts")
    parser.add_argument("--create-extended", action="store_true", help="Create extended evaluation manifest")
    parser.add_argument("--suggest-external", action="store_true", help="Show external dataset recommendations") 
    parser.add_argument("--create-script", action="store_true", help="Create extended evaluation script")
    parser.add_argument("--all", action="store_true", help="Create all extended evaluation resources")
    
    args = parser.parse_args()
    
    expander = TestDatasetExpander()
    
    if args.all:
        print("🚀 Creating all extended evaluation resources...")
        expander.generate_extended_manifest()
        expander.create_evaluation_script()
        expander.suggest_external_datasets()
    elif args.create_extended:
        expander.generate_extended_manifest()
    elif args.suggest_external:
        expander.suggest_external_datasets()
    elif args.create_script:
        expander.create_evaluation_script()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
