#!/usr/bin/env python3
"""
RunPod Data Persistence Script
Handles S3 backup and restore operations to ensure data survives pod restarts.
"""

import os
import sys
import boto3
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RunPodDataPersistence:
    """Manages data persistence for RunPod using S3."""
    
    def __init__(self):
        # S3 Configuration for RunPod
        self.s3_client = boto3.client(
            's3',
            region_name='eu-ro-1',
            endpoint_url='https://s3api-eu-ro-1.runpod.io'
        )
        self.bucket_name = 'diaebtidre'
        self.backup_prefix = 'runpod-backups'
        
        # Critical paths to persist (these are the large files we want to backup)
        self.critical_paths = {
            'generated_videos': '/workspace/visionflow/generated',
            'hf_cache': '/workspace/hf_cache',
            'metadata_db': '/workspace/visionflow/generated/metadata.db'
        }
        
        # Test if S3 connection works
        self._test_s3_connection()
    
    def _test_s3_connection(self):
        """Test S3 connection and bucket access."""
        try:
            # Test bucket access
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ S3 connection successful - bucket '{self.bucket_name}' accessible")
            
            # List existing backups
            existing_backups = self.list_backups()
            logger.info(f"📋 Found {len(existing_backups)} existing backups")
            
        except Exception as e:
            logger.error(f"❌ S3 connection failed: {e}")
            logger.warning("⚠️  Data persistence will not work without S3 access")
    
    def create_backup(self, backup_name: str = None) -> str:
        """Create a comprehensive backup of all critical data."""
        if not backup_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{timestamp}"
        
        backup_path = f"{self.backup_prefix}/{backup_name}"
        logger.info(f"🚀 Starting backup: {backup_name}")
        
        try:
            # 1. Backup generated videos (large files)
            self._backup_directory(
                local_path=self.critical_paths['generated_videos'],
                s3_key=f"{backup_path}/generated_videos"
            )
            
            # 2. Backup HuggingFace cache (very large - models)
            self._backup_directory(
                local_path=self.critical_paths['hf_cache'],
                s3_key=f"{backup_path}/hf_cache"
            )
            
            # 3. Create backup manifest
            manifest = self._create_backup_manifest(backup_name)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f"{backup_path}/manifest.json",
                Body=json.dumps(manifest, indent=2)
            )
            
            logger.info(f"✅ Backup completed: {backup_name}")
            logger.info(f"📊 Backup size: {self._format_size(manifest.get('total_size_bytes', 0))}")
            return backup_name
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore data from a specific backup."""
        backup_path = f"{self.backup_prefix}/{backup_name}"
        logger.info(f"🔄 Starting restore from: {backup_name}")
        
        try:
            # Check if backup exists
            try:
                manifest_obj = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=f"{backup_path}/manifest.json"
                )
                manifest = json.loads(manifest_obj['Body'].read())
                logger.info(f"📋 Backup manifest: {manifest}")
            except Exception as e:
                logger.error(f"❌ Cannot read backup manifest: {e}")
                return False
            
            # 1. Restore generated videos
            self._restore_directory(
                s3_key=f"{backup_path}/generated_videos",
                local_path=self.critical_paths['generated_videos']
            )
            
            # 2. Restore HuggingFace cache
            self._restore_directory(
                s3_key=f"{backup_path}/hf_cache",
                local_path=self.critical_paths['hf_cache']
            )
            
            logger.info(f"✅ Restore completed: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            return False
    
    def list_backups(self) -> list:
        """List all available backups."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.backup_prefix
            )
            
            backups = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('/manifest.json'):
                    backup_name = obj['Key'].split('/')[-2]
                    backups.append(backup_name)
            
            return sorted(backups, reverse=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to list backups: {e}")
            return []
    
    def delete_backup(self, backup_name: str) -> bool:
        """Delete a specific backup to free up S3 space."""
        backup_path = f"{self.backup_prefix}/{backup_name}"
        logger.info(f"🗑️  Deleting backup: {backup_name}")
        
        try:
            # List all objects in the backup
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=backup_path
            )
            
            # Delete all objects
            objects_to_delete = [{'Key': obj['Key']} for obj in response.get('Contents', [])]
            if objects_to_delete:
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': objects_to_delete}
                )
            
            logger.info(f"✅ Backup deleted: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete backup: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 3) -> bool:
        """Keep only the most recent backups to save S3 space."""
        backups = self.list_backups()
        if len(backups) <= keep_count:
            logger.info(f"📋 Only {len(backups)} backups exist, no cleanup needed")
            return True
        
        backups_to_delete = backups[keep_count:]
        logger.info(f"🧹 Cleaning up {len(backups_to_delete)} old backups")
        
        for backup_name in backups_to_delete:
            self.delete_backup(backup_name)
        
        return True
    
    def _backup_directory(self, local_path: str, s3_key: str):
        """Backup a directory to S3."""
        if not os.path.exists(local_path):
            logger.warning(f"⚠️  Path does not exist: {local_path}")
            return
        
        logger.info(f"📁 Backing up: {local_path} -> {s3_key}")
        
        total_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, local_path)
                s3_file_key = f"{s3_key}/{relative_path}"
                
                try:
                    file_size = os.path.getsize(file_path)
                    with open(file_path, 'rb') as f:
                        self.s3_client.put_object(
                            Bucket=self.bucket_name,
                            Key=s3_file_key,
                            Body=f.read()
                        )
                    total_files += 1
                    total_size += file_size
                    logger.debug(f"  ✅ {relative_path} ({self._format_size(file_size)})")
                except Exception as e:
                    logger.error(f"  ❌ Failed to backup {relative_path}: {e}")
        
        logger.info(f"📊 Backup summary: {total_files} files, {self._format_size(total_size)}")
    
    def _restore_directory(self, s3_key: str, local_path: str):
        """Restore a directory from S3."""
        logger.info(f"📁 Restoring: {s3_key} -> {local_path}")
        
        # Create local directory
        os.makedirs(local_path, exist_ok=True)
        
        try:
            # List all objects in the backup directory
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=s3_key
            )
            
            total_files = 0
            total_size = 0
            
            for obj in response.get('Contents', []):
                if obj['Key'] == s3_key:  # Skip the directory itself
                    continue
                
                # Calculate relative path
                relative_path = obj['Key'].replace(f"{s3_key}/", "", 1)
                local_file_path = os.path.join(local_path, relative_path)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download file
                try:
                    file_obj = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    with open(local_file_path, 'wb') as f:
                        f.write(file_obj['Body'].read())
                    total_files += 1
                    total_size += obj['Size']
                    logger.debug(f"  ✅ {relative_path}")
                except Exception as e:
                    logger.error(f"  ❌ Failed to restore {relative_path}: {e}")
            
            logger.info(f"📊 Restore summary: {total_files} files, {self._format_size(total_size)}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to list S3 objects: {e}")
    
    def _create_backup_manifest(self, backup_name: str) -> dict:
        """Create a manifest of the backup contents."""
        manifest = {
            'backup_name': backup_name,
            'timestamp': datetime.now().isoformat(),
            'pod_id': os.getenv('RUNPOD_POD_ID', 'unknown'),
            'contents': {},
            'total_size_bytes': 0
        }
        
        for name, path in self.critical_paths.items():
            if os.path.exists(path):
                if os.path.isdir(path):
                    total_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(path)
                        for filename in filenames
                    )
                    file_count = sum(
                        len(filenames)
                        for dirpath, dirnames, filenames in os.walk(path)
                    )
                    manifest['contents'][name] = {
                        'type': 'directory',
                        'size_bytes': total_size,
                        'file_count': file_count
                    }
                    manifest['total_size_bytes'] += total_size
                else:
                    file_size = os.path.getsize(path)
                    manifest['contents'][name] = {
                        'type': 'file',
                        'size_bytes': file_size
                    }
                    manifest['total_size_bytes'] += file_size
        
        return manifest
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes into human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def get_storage_info(self) -> dict:
        """Get information about S3 storage usage."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.backup_prefix
            )
            
            total_size = sum(obj['Size'] for obj in response.get('Contents', []))
            total_objects = len(response.get('Contents', []))
            
            return {
                'total_size_bytes': total_size,
                'total_size_formatted': self._format_size(total_size),
                'total_objects': total_objects,
                'backup_count': len(self.list_backups())
            }
        except Exception as e:
            logger.error(f"❌ Failed to get storage info: {e}")
            return {}

def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("🚀 RunPod Data Persistence Script")
        print("=" * 50)
        print("Usage:")
        print("  python runpod_data_persistence.py backup [backup_name]")
        print("  python runpod_data_persistence.py restore <backup_name>")
        print("  python runpod_data_persistence.py list")
        print("  python runpod_data_persistence.py delete <backup_name>")
        print("  python runpod_data_persistence.py cleanup [keep_count]")
        print("  python runpod_data_persistence.py info")
        print("  python runpod_data_persistence.py test")
        return
    
    persistence = RunPodDataPersistence()
    command = sys.argv[1]
    
    if command == 'backup':
        backup_name = sys.argv[2] if len(sys.argv) > 2 else None
        backup_name = persistence.create_backup(backup_name)
        print(f"✅ Backup created: {backup_name}")
        
    elif command == 'restore':
        if len(sys.argv) < 3:
            print("❌ Please specify backup name to restore")
            return
        backup_name = sys.argv[2]
        success = persistence.restore_backup(backup_name)
        if success:
            print(f"✅ Restore completed: {backup_name}")
        else:
            print(f"❌ Restore failed: {backup_name}")
    
    elif command == 'list':
        backups = persistence.list_backups()
        if backups:
            print("📋 Available backups:")
            for backup in backups:
                print(f"  - {backup}")
        else:
            print("📋 No backups found")
    
    elif command == 'delete':
        if len(sys.argv) < 3:
            print("❌ Please specify backup name to delete")
            return
        backup_name = sys.argv[2]
        success = persistence.delete_backup(backup_name)
        if success:
            print(f"✅ Backup deleted: {backup_name}")
        else:
            print(f"❌ Failed to delete backup: {backup_name}")
    
    elif command == 'cleanup':
        keep_count = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        success = persistence.cleanup_old_backups(keep_count)
        if success:
            print(f"✅ Cleanup completed, keeping {keep_count} most recent backups")
        else:
            print(f"❌ Cleanup failed")
    
    elif command == 'info':
        info = persistence.get_storage_info()
        if info:
            print("📊 Storage Information:")
            print(f"   Total size: {info.get('total_size_formatted', 'N/A')}")
            print(f"   Total objects: {info.get('total_objects', 'N/A')}")
            print(f"   Backup count: {info.get('backup_count', 'N/A')}")
        else:
            print("❌ Failed to get storage information")
    
    elif command == 'test':
        print("🧪 Testing S3 connection...")
        try:
            persistence._test_s3_connection()
            print("✅ S3 connection test passed")
        except Exception as e:
            print(f"❌ S3 connection test failed: {e}")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python runpod_data_persistence.py' for help")

if __name__ == "__main__":
    main()
