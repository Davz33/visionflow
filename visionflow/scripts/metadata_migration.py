#!/usr/bin/env python3
"""
Metadata Migration Utility
Retroactively add metadata to existing video files that don't have tracking.

This addresses the common scenario where videos were generated before
implementing metadata tracking.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visionflow.services.generation.video_metadata_tracker import (
    metadata_tracker, VideoGenerationMetadata
)
from visionflow.shared.models import VideoQuality

async def analyze_existing_videos():
    """Analyze existing videos and their metadata status"""
    
    print("🔍 ANALYZING EXISTING VIDEOS")
    print("=" * 50)
    
    discovered = await metadata_tracker.discover_existing_videos()
    
    with_metadata = [v for v in discovered if v['has_metadata']]
    without_metadata = [v for v in discovered if not v['has_metadata']]
    
    print(f"Total videos found: {len(discovered)}")
    print(f"With metadata: {len(with_metadata)}")
    print(f"Without metadata: {len(without_metadata)}")
    print(f"Coverage: {len(with_metadata)/len(discovered)*100:.1f}%")
    print()
    
    if without_metadata:
        print("📁 Videos without metadata:")
        for video in without_metadata:
            print(f"  ❌ {video['filename']} ({video['file_size_mb']:.1f}MB)")
        print()
    
    return with_metadata, without_metadata

async def create_metadata_from_filename(video_path: str) -> Optional[VideoGenerationMetadata]:
    """
    Create metadata by inferring from filename patterns
    This is a fallback for videos generated without tracking
    """
    
    video_path = Path(video_path)
    filename = video_path.name
    
    # Try to extract info from filename patterns
    # Common patterns: wan_video_<uuid>.mp4, video_<timestamp>.mp4, etc.
    
    base_metadata = {
        'generation_id': f"retro_{filename.split('.')[0]}",
        'video_id': f"video_{filename.split('.')[0]}",
        'video_path': str(video_path.absolute()),
        'filename': filename,
        'file_size_bytes': video_path.stat().st_size,
        'file_created_at': datetime.fromtimestamp(video_path.stat().st_mtime),
        
        # Default values for unknown parameters
        'prompt': "Unknown prompt - please update manually",
        'negative_prompt': None,
        'duration': 3.0,  # Common default
        'fps': 24,
        'resolution': "832x480",  # Common WAN resolution
        'quality': VideoQuality.MEDIUM.value,
        'seed': None,
        'guidance_scale': 7.5,
        'num_inference_steps': 25,
        
        'model_name': "wan2.1-t2v-1.3b",  # Most common model
        'model_version': None,
        'device': "mps",  # Based on user's Mac setup
        
        'actual_duration': None,
        'actual_resolution': None,
        'actual_fps': None,
        'num_frames': None,
        'generation_time': 0.0,
        
        'created_at': datetime.fromtimestamp(video_path.stat().st_mtime),
        'completed_at': datetime.fromtimestamp(video_path.stat().st_mtime),
        'user_id': None,
        'session_id': None,
        
        'tags': ["retroactive", "needs_prompt_update"],
        'notes': f"Retroactively created metadata for {filename}. Please update prompt and parameters.",
        'metadata_version': "1.0"
    }
    
    return VideoGenerationMetadata(**base_metadata)

async def create_metadata_from_prompt_mapping(video_path: str, prompt_mapping: Dict[str, str]) -> Optional[VideoGenerationMetadata]:
    """Create metadata using a provided prompt mapping"""
    
    filename = Path(video_path).name
    
    if filename not in prompt_mapping:
        return None
    
    prompt = prompt_mapping[filename]
    
    # Create metadata with the provided prompt
    base_metadata = await create_metadata_from_filename(video_path)
    base_metadata.prompt = prompt
    base_metadata.tags = ["prompt_provided", "manual_mapping"]
    base_metadata.notes = f"Created with provided prompt mapping"
    
    return base_metadata

async def interactive_metadata_creation(video_path: str) -> Optional[VideoGenerationMetadata]:
    """Interactively create metadata for a video"""
    
    filename = Path(video_path).name
    
    print(f"\n📹 Creating metadata for: {filename}")
    print("-" * 40)
    
    # Get prompt from user
    prompt = input("Enter the original prompt (or 'skip' to use default): ").strip()
    if prompt.lower() == 'skip' or not prompt:
        prompt = "Unknown prompt - please update manually"
    
    # Get quality
    quality_input = input("Enter quality (low/medium/high) [default: medium]: ").strip().lower()
    if quality_input in ['low', 'medium', 'high']:
        quality = quality_input
    else:
        quality = 'medium'
    
    # Get duration
    duration_input = input("Enter duration in seconds [default: 3.0]: ").strip()
    try:
        duration = float(duration_input) if duration_input else 3.0
    except ValueError:
        duration = 3.0
    
    # Create metadata
    base_metadata = await create_metadata_from_filename(video_path)
    base_metadata.prompt = prompt
    base_metadata.quality = quality
    base_metadata.duration = duration
    base_metadata.tags = ["interactive_creation", "user_provided"]
    base_metadata.notes = f"Interactively created metadata"
    
    return base_metadata

async def batch_migrate_metadata(strategy: str = "filename", 
                                prompt_mapping: Optional[Dict[str, str]] = None,
                                interactive: bool = False):
    """
    Batch migrate metadata for videos without tracking
    
    Strategies:
    - 'filename': Infer from filename patterns
    - 'mapping': Use provided prompt mapping
    - 'interactive': Ask user for each video
    """
    
    print(f"🔄 BATCH METADATA MIGRATION ({strategy})")
    print("=" * 50)
    
    # Find videos without metadata
    _, without_metadata = await analyze_existing_videos()
    
    if not without_metadata:
        print("✅ All videos already have metadata!")
        return
    
    print(f"Migrating {len(without_metadata)} videos...")
    print()
    
    migrated_count = 0
    skipped_count = 0
    
    for video_info in without_metadata:
        video_path = video_info['video_path']
        filename = video_info['filename']
        
        try:
            metadata = None
            
            if strategy == "filename":
                metadata = await create_metadata_from_filename(video_path)
                print(f"📝 Created metadata from filename: {filename}")
                
            elif strategy == "mapping" and prompt_mapping:
                metadata = await create_metadata_from_prompt_mapping(video_path, prompt_mapping)
                if metadata:
                    print(f"📝 Created metadata from mapping: {filename}")
                else:
                    print(f"⚠️ No mapping found for: {filename}")
                    skipped_count += 1
                    continue
                    
            elif strategy == "interactive" and interactive:
                metadata = await interactive_metadata_creation(video_path)
                
            if metadata:
                # Store metadata in all backends
                success = await store_metadata_safely(metadata)
                if success:
                    migrated_count += 1
                    print(f"   ✅ Stored successfully")
                else:
                    print(f"   ❌ Storage failed")
                    skipped_count += 1
            else:
                skipped_count += 1
                print(f"   ⚠️ Skipped")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            skipped_count += 1
        
        print()
    
    print("📊 MIGRATION SUMMARY")
    print("-" * 30)
    print(f"Migrated: {migrated_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Success rate: {migrated_count/(migrated_count+skipped_count)*100:.1f}%")

async def store_metadata_safely(metadata: VideoGenerationMetadata) -> bool:
    """Safely store metadata with error handling"""
    
    try:
        # Store in all available backends
        success_count = 0
        
        for backend in metadata_tracker.storage_backends:
            try:
                if await backend.store_metadata(metadata):
                    success_count += 1
            except Exception as e:
                print(f"      ⚠️ Backend {type(backend).__name__} failed: {e}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"      ❌ Storage failed: {e}")
        return False

async def export_prompt_mapping_template():
    """Export a template for manual prompt mapping"""
    
    print("📄 CREATING PROMPT MAPPING TEMPLATE")
    print("=" * 50)
    
    _, without_metadata = await analyze_existing_videos()
    
    if not without_metadata:
        print("✅ All videos have metadata - no template needed!")
        return
    
    # Create mapping template
    mapping_template = {}
    for video_info in without_metadata:
        filename = video_info['filename']
        mapping_template[filename] = "Enter the original prompt here"
    
    # Save template
    template_path = Path("prompt_mapping_template.json")
    with open(template_path, 'w') as f:
        json.dump(mapping_template, f, indent=2)
    
    print(f"✅ Template created: {template_path}")
    print(f"📝 Edit this file to add prompts for {len(without_metadata)} videos")
    print(f"💡 Then run: python scripts/metadata_migration.py --mapping prompt_mapping_template.json")

async def load_prompt_mapping(mapping_file: str) -> Dict[str, str]:
    """Load prompt mapping from JSON file"""
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        # Filter out template entries
        filtered_mapping = {
            filename: prompt for filename, prompt in mapping.items()
            if prompt and prompt != "Enter the original prompt here"
        }
        
        print(f"📖 Loaded {len(filtered_mapping)} prompt mappings from {mapping_file}")
        return filtered_mapping
        
    except Exception as e:
        print(f"❌ Failed to load mapping file: {e}")
        return {}

async def main():
    """Main migration utility"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Video metadata migration utility")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing videos")
    parser.add_argument("--template", action="store_true", help="Create prompt mapping template")
    parser.add_argument("--migrate", choices=["filename", "mapping", "interactive"], 
                       help="Migration strategy")
    parser.add_argument("--mapping-file", help="JSON file with filename->prompt mapping")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    print("🗃️ VIDEO METADATA MIGRATION UTILITY")
    print("=" * 50)
    print("Retroactively add metadata to videos generated without tracking")
    print()
    
    if args.analyze or not any(vars(args).values()):
        await analyze_existing_videos()
    
    if args.template:
        await export_prompt_mapping_template()
    
    if args.migrate:
        prompt_mapping = None
        
        if args.migrate == "mapping":
            if args.mapping_file:
                prompt_mapping = await load_prompt_mapping(args.mapping_file)
            else:
                print("❌ --mapping-file required for mapping strategy")
                return
        
        await batch_migrate_metadata(
            strategy=args.migrate,
            prompt_mapping=prompt_mapping,
            interactive=args.interactive or args.migrate == "interactive"
        )
    
    if not any(vars(args).values()):
        print("\n💡 Usage examples:")
        print("  python scripts/metadata_migration.py --analyze")
        print("  python scripts/metadata_migration.py --template")
        print("  python scripts/metadata_migration.py --migrate filename")
        print("  python scripts/metadata_migration.py --migrate mapping --mapping-file prompts.json")
        print("  python scripts/metadata_migration.py --migrate interactive")

if __name__ == "__main__":
    asyncio.run(main())
