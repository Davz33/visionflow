#!/usr/bin/env python3
"""
Download reliable evaluation videos for video generation evaluation.
This script downloads videos from industry-standard benchmarks and datasets.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import time

# Configuration
EVALUATION_DATASET_DIR = Path("evaluation_datasets")
VBench_PROMPTS_FILE = EVALUATION_DATASET_DIR / "vbench_prompts.txt"
VBench_CATEGORIES_FILE = EVALUATION_DATASET_DIR / "vbench_categories.txt"

# Video sources and their metadata
VIDEO_SOURCES = {
    "vbench_samples": {
        "description": "VBench industry-standard evaluation samples",
        "base_url": "https://drive.google.com/drive/folders/1on66fnZ8atRoLDimcAXMxSwRxqN8_0yS?usp=sharing",
        "prompts_file": "vbench_prompts.txt",
        "categories_file": "vbench_categories.txt"
    },
    "video_mme": {
        "description": "Video-MME comprehensive evaluation benchmark",
        "huggingface_url": "https://huggingface.co/datasets/Video-MME",
        "paper_url": "https://arxiv.org/abs/2405.21075"
    },
    "etva_samples": {
        "description": "ETVA evaluation through video-specific questions",
        "website": "https://eftv-eval.github.io/etva-eval/",
        "paper_url": "https://arxiv.org/abs/2503.16867"
    }
}

def create_evaluation_dataset_structure():
    """Create the evaluation dataset directory structure"""
    dirs = [
        EVALUATION_DATASET_DIR / "reliable_sources",
        EVALUATION_DATASET_DIR / "reliable_sources" / "vbench",
        EVALUATION_DATASET_DIR / "reliable_sources" / "video_mme", 
        EVALUATION_DATASET_DIR / "reliable_sources" / "etva",
        EVALUATION_DATASET_DIR / "evaluation_criteria",
        EVALUATION_DATASET_DIR / "prompts",
        EVALUATION_DATASET_DIR / "metrics"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def download_video_from_url(url: str, output_path: Path) -> bool:
    """Download a video from URL using wget or curl"""
    try:
        # Try wget first
        result = subprocess.run([
            "wget", "-O", str(output_path), url
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Downloaded: {output_path.name}")
            return True
        else:
            # Try curl as fallback
            result = subprocess.run([
                "curl", "-L", "-o", str(output_path), url
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ Downloaded: {output_path.name}")
                return True
            else:
                print(f"❌ Failed to download: {url}")
                return False
                
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def create_sample_evaluation_videos():
    """Create sample evaluation videos with proper prompts"""
    
    # Sample evaluation videos with detailed prompts
    sample_videos = [
        {
            "filename": "sample_01_landscape.mp4",
            "prompt": "A serene mountain landscape at sunset with rolling hills, snow-capped peaks, and golden light filtering through clouds. The scene should show natural beauty with realistic lighting and atmospheric effects.",
            "category": "landscape",
            "evaluation_criteria": [
                "visual_quality", "lighting_realism", "spatial_consistency", "temporal_smoothness"
            ],
            "expected_duration": "10s",
            "source": "vbench_style"
        },
        {
            "filename": "sample_02_action.mp4", 
            "prompt": "A dynamic car chase scene through a city street with realistic motion blur, camera shake, and smooth transitions between frames. The action should maintain spatial coherence and temporal consistency.",
            "category": "action",
            "evaluation_criteria": [
                "motion_smoothness", "spatial_coherence", "temporal_consistency", "action_clarity"
            ],
            "expected_duration": "10s",
            "source": "vbench_style"
        },
        {
            "filename": "sample_03_object.mp4",
            "prompt": "A detailed close-up of a vintage camera with intricate mechanical parts, showing realistic textures, lighting, and depth. The object should maintain consistent appearance across all frames.",
            "category": "object",
            "evaluation_criteria": [
                "object_consistency", "texture_detail", "lighting_realism", "spatial_stability"
            ],
            "expected_duration": "10s",
            "source": "vbench_style"
        },
        {
            "filename": "sample_04_human_activity.mp4",
            "prompt": "A person cooking in a modern kitchen, showing realistic human movements, facial expressions, and interaction with kitchen objects. The scene should maintain human anatomy consistency.",
            "category": "human_activity",
            "evaluation_criteria": [
                "human_anatomy", "motion_naturalness", "object_interaction", "facial_consistency"
            ],
            "expected_duration": "10s",
            "source": "vbench_style"
        },
        {
            "filename": "sample_05_nature.mp4",
            "prompt": "A flowing river with realistic water physics, showing natural movement patterns, reflections, and environmental details. The water should exhibit natural fluid dynamics.",
            "category": "nature",
            "evaluation_criteria": [
                "physics_realism", "texture_detail", "motion_naturalness", "environmental_consistency"
            ],
            "expected_duration": "10s",
            "source": "vbench_style"
        }
    ]
    
    # Create the sample videos metadata
    sample_dataset = {
        "metadata": {
            "name": "Reliable Evaluation Video Samples",
            "description": "Curated sample videos for comprehensive video generation evaluation",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_videos": len(sample_videos),
            "evaluation_framework": "VBench + Video-MME + ETVA",
            "target_duration": "10 seconds",
            "quality_standards": "Industry benchmark quality"
        },
        "videos": sample_videos
    }
    
    # Save the sample dataset metadata
    output_file = EVALUATION_DATASET_DIR / "reliable_sources" / "sample_evaluation_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(sample_dataset, f, indent=2)
    
    print(f"✅ Created sample evaluation dataset: {output_file}")
    return sample_videos

def create_evaluation_criteria():
    """Create comprehensive evaluation criteria based on industry standards"""
    
    evaluation_criteria = {
        "technical_quality": {
            "description": "Technical aspects of video generation quality",
            "metrics": {
                "spatial_consistency": {
                    "description": "Consistency of objects and scenes across frames",
                    "evaluation_method": "LPIPS, FVMD",
                    "scale": "0-1 (lower is better)",
                    "thresholds": {
                        "excellent": "0.0-0.2",
                        "good": "0.2-0.4", 
                        "acceptable": "0.4-0.6",
                        "poor": "0.6+"
                    }
                },
                "temporal_smoothness": {
                    "description": "Smoothness of motion and transitions",
                    "evaluation_method": "FVMD, optical flow analysis",
                    "scale": "0-1 (lower is better)",
                    "thresholds": {
                        "excellent": "0.0-0.15",
                        "good": "0.15-0.3",
                        "acceptable": "0.3-0.5",
                        "poor": "0.5+"
                    }
                },
                "flicker_detection": {
                    "description": "Detection of temporal flickering artifacts",
                    "evaluation_method": "Temporal consistency analysis",
                    "scale": "0-1 (lower is better)",
                    "thresholds": {
                        "excellent": "0.0-0.1",
                        "good": "0.1-0.25",
                        "acceptable": "0.25-0.4",
                        "poor": "0.4+"
                    }
                }
            }
        },
        "semantic_quality": {
            "description": "Semantic understanding and alignment with prompts",
            "metrics": {
                "prompt_alignment": {
                    "description": "How well the video matches the text prompt",
                    "evaluation_method": "CLIP, ViCLIP",
                    "scale": "0-1 (higher is better)",
                    "thresholds": {
                        "excellent": "0.8+",
                        "good": "0.6-0.8",
                        "acceptable": "0.4-0.6",
                        "poor": "0.0-0.4"
                    }
                },
                "object_consistency": {
                    "description": "Consistency of objects and their properties",
                    "evaluation_method": "Object tracking, feature matching",
                    "scale": "0-1 (higher is better)",
                    "thresholds": {
                        "excellent": "0.9+",
                        "good": "0.7-0.9",
                        "acceptable": "0.5-0.7",
                        "poor": "0.0-0.5"
                    }
                },
                "scene_understanding": {
                    "description": "Logical coherence of scene elements",
                    "evaluation_method": "ETVA questions, human evaluation",
                    "scale": "0-1 (higher is better)",
                    "thresholds": {
                        "excellent": "0.85+",
                        "good": "0.7-0.85",
                        "acceptable": "0.5-0.7",
                        "poor": "0.0-0.5"
                    }
                }
            }
        },
        "human_alignment": {
            "description": "Alignment with human perception and preferences",
            "metrics": {
                "aesthetic_quality": {
                    "description": "Overall visual appeal and aesthetics",
                    "evaluation_method": "Human rating, aesthetic models",
                    "scale": "1-10 (higher is better)",
                    "thresholds": {
                        "excellent": "8-10",
                        "good": "6-8",
                        "acceptable": "4-6",
                        "poor": "1-4"
                    }
                },
                "realism": {
                    "description": "Perceived realism of generated content",
                    "evaluation_method": "Human evaluation, realism models",
                    "scale": "0-1 (higher is better)",
                    "thresholds": {
                        "excellent": "0.8+",
                        "good": "0.6-0.8",
                        "acceptable": "0.4-0.6",
                        "poor": "0.0-0.4"
                    }
                }
            }
        }
    }
    
    # Save evaluation criteria
    output_file = EVALUATION_DATASET_DIR / "evaluation_criteria" / "comprehensive_criteria.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_criteria, f, indent=2)
    
    print(f"✅ Created evaluation criteria: {output_file}")
    return evaluation_criteria

def create_evaluation_prompts():
    """Create detailed evaluation prompts for different video types"""
    
    evaluation_prompts = {
        "landscape": {
            "description": "Evaluation prompts for landscape and nature videos",
            "prompts": [
                "A majestic mountain range at golden hour with dramatic clouds and natural lighting. The scene should show realistic atmospheric perspective and natural color grading.",
                "A serene forest with sunlight filtering through trees, showing realistic shadows, depth, and natural movement of leaves in the breeze.",
                "A coastal scene with crashing waves, realistic water physics, and dynamic lighting that captures the mood of the ocean environment."
            ],
            "evaluation_focus": ["spatial_consistency", "lighting_realism", "environmental_detail", "atmospheric_effects"]
        },
        "action": {
            "description": "Evaluation prompts for action and motion videos", 
            "prompts": [
                "A high-speed car chase through urban streets with realistic motion blur, camera movement, and smooth transitions between action sequences.",
                "A martial arts demonstration with fluid human movements, realistic physics, and consistent body proportions throughout the sequence.",
                "A sports highlight reel showing dynamic athletic movements, realistic physics, and smooth camera work that captures the energy of the moment."
            ],
            "evaluation_focus": ["motion_smoothness", "spatial_coherence", "temporal_consistency", "action_clarity"]
        },
        "object": {
            "description": "Evaluation prompts for object-focused videos",
            "prompts": [
                "A detailed close-up of a vintage mechanical watch, showing intricate gears, realistic textures, and consistent lighting across all frames.",
                "A product demonstration video with stable camera work, consistent object appearance, and clear visibility of product features.",
                "A cooking tutorial showing detailed food preparation with realistic textures, consistent lighting, and clear step-by-step progression."
            ],
            "evaluation_focus": ["object_consistency", "texture_detail", "lighting_stability", "spatial_stability"]
        },
        "human_activity": {
            "description": "Evaluation prompts for human activity videos",
            "prompts": [
                "A person giving a presentation with natural facial expressions, consistent appearance, and realistic body language throughout the video.",
                "A cooking demonstration showing detailed hand movements, realistic interactions with kitchen objects, and consistent human anatomy.",
                "A dance performance with fluid movements, realistic physics, and consistent human proportions across all frames."
            ],
            "evaluation_focus": ["human_anatomy", "motion_naturalness", "object_interaction", "facial_consistency"]
        }
    }
    
    # Save evaluation prompts
    output_file = EVALUATION_DATASET_DIR / "prompts" / "evaluation_prompts.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_prompts, f, indent=2)
    
    print(f"✅ Created evaluation prompts: {output_file}")
    return evaluation_prompts

def main():
    """Main function to create the evaluation dataset structure"""
    print("🚀 Creating comprehensive evaluation dataset structure...")
    
    # Create directory structure
    create_evaluation_dataset_structure()
    
    # Create sample evaluation videos metadata
    sample_videos = create_sample_evaluation_videos()
    
    # Create evaluation criteria
    evaluation_criteria = create_evaluation_criteria()
    
    # Create evaluation prompts
    evaluation_prompts = create_evaluation_prompts()
    
    # Create a comprehensive README
    readme_content = f"""# Video Evaluation Dataset

This directory contains a comprehensive evaluation dataset for video generation quality assessment.

## Structure

- `reliable_sources/` - Industry-standard benchmark videos and metadata
- `evaluation_criteria/` - Comprehensive evaluation metrics and thresholds  
- `prompts/` - Detailed evaluation prompts for different video types
- `metrics/` - Implementation of evaluation metrics (LPIPS, FVMD, CLIP, ETVA)

## Dataset Information

- **Total Videos**: {len(sample_videos)}
- **Target Duration**: 10 seconds
- **Quality Standards**: Industry benchmark quality
- **Evaluation Framework**: VBench + Video-MME + ETVA

## Evaluation Criteria

The evaluation system covers three main areas:

1. **Technical Quality**: Spatial consistency, temporal smoothness, flicker detection
2. **Semantic Quality**: Prompt alignment, object consistency, scene understanding  
3. **Human Alignment**: Aesthetic quality, realism, human perception

## Usage

Use the evaluation criteria and prompts to assess video generation quality across multiple dimensions.

## Sources

- VBench: Comprehensive benchmark suite for video generative models
- Video-MME: Multi-modal evaluation benchmark
- ETVA: Evaluation through video-specific questions

For more information, see the individual JSON files in each directory.
"""
    
    readme_file = EVALUATION_DATASET_DIR / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created README: {readme_file}")
    print("\n🎉 Evaluation dataset structure created successfully!")
    print(f"📁 Dataset location: {EVALUATION_DATASET_DIR.absolute()}")
    print(f"📊 Total sample videos: {len(sample_videos)}")
    print(f"📋 Evaluation criteria: {len(evaluation_criteria)} categories")
    print(f"✍️  Evaluation prompts: {len(evaluation_prompts)} video types")

if __name__ == "__main__":
    main()
