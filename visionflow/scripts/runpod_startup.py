#!/usr/bin/env python3
"""
RunPod Startup Script
Automatically restores the latest backup when the pod starts.
"""

import os
import sys
import time
from pathlib import Path

# Add the visionflow directory to Python path
visionflow_dir = Path(__file__).parent.parent
sys.path.insert(0, str(visionflow_dir))

from scripts.runpod_data_persistence import RunPodDataPersistence

def main():
    """Main startup function."""
    print("🚀 RunPod Startup - Data Restoration")
    print("=" * 50)
    
    # Check if we're in a fresh pod
    if not os.path.exists('/workspace/visionflow/generated') or not os.listdir('/workspace/visionflow/generated'):
        print("📋 Fresh pod detected - attempting data restoration...")
        
        try:
            persistence = RunPodDataPersistence()
            
            # List available backups
            backups = persistence.list_backups()
            
            if backups:
                latest_backup = backups[0]  # Most recent backup
                print(f"🔄 Restoring latest backup: {latest_backup}")
                
                # Restore the backup
                success = persistence.restore_backup(latest_backup)
                
                if success:
                    print(f"✅ Data restoration completed: {latest_backup}")
                    
                    # Verify restored data
                    print("\n📊 Restored Data Summary:")
                    if os.path.exists('/workspace/visionflow/generated'):
                        video_count = len([f for f in os.listdir('/workspace/visionflow/generated') if f.endswith('.mp4')])
                        print(f"   🎬 Generated videos: {video_count}")
                    
                    if os.path.exists('/workspace/hf_cache'):
                        hf_size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, dirnames, filenames in os.walk('/workspace/hf_cache')
                            for filename in filenames
                        ) / (1024**3)  # GB
                        print(f"   🧠 HuggingFace cache: {hf_size:.1f} GB")
                        
                else:
                    print(f"❌ Data restoration failed: {latest_backup}")
                    
            else:
                print("📋 No backups found - starting with fresh data")
                
        except Exception as e:
            print(f"❌ Data restoration error: {e}")
            print("📋 Continuing with fresh pod...")
    
    else:
        print("📋 Existing data detected - no restoration needed")
    
    print("\n🚀 Pod startup completed!")
    print("💡 To create a backup: python scripts/runpod_data_persistence.py backup")
    print("💡 To restore manually: python scripts/runpod_data_persistence.py restore <backup_name>")

if __name__ == "__main__":
    main()
