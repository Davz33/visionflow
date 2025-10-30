#!/usr/bin/env python3
"""
RunPod Startup Script
Automatically restores the latest backup when the pod starts.
"""

import os
import sys
from pathlib import Path
import json

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
            
            # Test S3 connection first
            print("🔍 Testing S3 connection...")
            try:
                persistence._test_s3_connection()
                print("✅ S3 connection successful")
            except Exception as e:
                print(f"❌ S3 connection failed: {e}")
                print("📋 Continuing with fresh pod...")
                return
            
            # List available backups
            backups = persistence.list_backups()
            
            if backups:
                latest_backup = backups[0]  # Most recent backup
                print(f"🔄 Restoring latest backup: {latest_backup}")
                
                # Get backup info before restoration
                try:
                    manifest_obj = persistence.s3_client.get_object(
                        Bucket=persistence.bucket_name,
                        Key=f"{persistence.backup_prefix}/{latest_backup}/manifest.json"
                    )
                    manifest = json.loads(manifest_obj['Body'].read())
                    print(f"📊 Backup size: {manifest.get('total_size_bytes', 0) / (1024**3):.1f} GB")
                except:
                    print("📊 Backup size: Unknown")
                
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
        
        # Check if we should create a backup of current state
        try:
            persistence = RunPodDataPersistence()
            
            # Check if S3 is accessible
            try:
                persistence._test_s3_connection()
                
                # Ask if user wants to create a backup
                print("\n💡 Current data detected. Consider creating a backup:")
                print("   python scripts/runpod_data_persistence.py backup")
                
            except Exception as e:
                print(f"⚠️  S3 not accessible: {e}")
                
        except Exception as e:
            print(f"⚠️  Could not initialize persistence: {e}")
    
    print("\n🚀 Pod startup completed!")
    print("\n💡 Available commands:")
    print("   📦 Create backup: python scripts/runpod_data_persistence.py backup")
    print("   🔄 Restore backup: python scripts/runpod_data_persistence.py restore <backup_name>")
    print("   📋 List backups: python scripts/runpod_data_persistence.py list")
    print("   🗑️  Delete backup: python scripts/runpod_data_persistence.py delete <backup_name>")
    print("   🧹 Cleanup old: python scripts/runpod_data_persistence.py cleanup [keep_count]")
    print("   📊 Storage info: python scripts/runpod_data_persistence.py info")
    print("   🧪 Test S3: python scripts/runpod_data_persistence.py test")

if __name__ == "__main__":
    main()
