"""
Video Generation Metadata Tracker
Industry best practices for tracking video generation parameters and enabling proper evaluation.

Implements multiple storage backends:
1. Database storage (primary) - for queries and analytics
2. JSON sidecar files (backup) - for file portability
3. Video metadata embedding (optional) - for self-contained files
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from ...shared.monitoring import get_logger
from ...shared.models import VideoGenerationRequest, VideoQuality

logger = get_logger(__name__)

@dataclass
class VideoGenerationMetadata:
    """Complete metadata for a generated video following MLOps best practices"""
    
    # Unique identifiers
    generation_id: str              # Unique ID for this generation
    video_id: str                  # Unique ID for the video file
    
    # File information
    video_path: str                # Path to video file
    filename: str                  # Video filename
    file_size_bytes: int           # File size in bytes
    file_created_at: datetime      # File creation timestamp
    
    # Generation parameters (input)
    prompt: str                    # Original text prompt
    negative_prompt: Optional[str] # Negative prompt if used
    duration: float               # Video duration in seconds
    fps: int                      # Frames per second
    resolution: str               # Video resolution (e.g., "832x480")
    quality: str                  # Quality setting (e.g., "high")
    seed: Optional[int]           # Random seed for reproducibility
    guidance_scale: float         # Guidance scale parameter
    num_inference_steps: int      # Number of inference steps
    
    # Model information
    model_name: str               # Model used (e.g., "wan2.1-t2v-1.3b")
    model_version: Optional[str]  # Model version if tracked
    device: str                   # Device used (e.g., "mps", "cuda")
    
    # Generation results
    actual_duration: Optional[float]    # Actual video duration
    actual_resolution: Optional[str]    # Actual video resolution
    actual_fps: Optional[float]         # Actual FPS
    num_frames: Optional[int]           # Total number of frames
    generation_time: float              # Time taken to generate
    
    # System metadata
    created_at: datetime                # When generation started
    completed_at: Optional[datetime]    # When generation completed
    user_id: Optional[str]             # User who requested generation
    session_id: Optional[str]          # Session ID for tracking
    
    # Evaluation metadata (populated later)
    evaluation_id: Optional[str] = None       # Link to evaluation results
    overall_score: Optional[float] = None     # Overall quality score
    confidence_level: Optional[str] = None    # Confidence level
    requires_review: Optional[bool] = None    # Whether human review needed
    
    # Additional metadata
    tags: Optional[List[str]] = None          # Custom tags for organization
    notes: Optional[str] = None               # Additional notes
    metadata_version: str = "1.0"             # Metadata schema version

class VideoMetadataStorage:
    """Abstract base for metadata storage backends"""
    
    async def store_metadata(self, metadata: VideoGenerationMetadata) -> bool:
        raise NotImplementedError
    
    async def get_metadata(self, video_path: str) -> Optional[VideoGenerationMetadata]:
        raise NotImplementedError
    
    async def get_metadata_by_id(self, generation_id: str) -> Optional[VideoGenerationMetadata]:
        raise NotImplementedError
    
    async def list_all_metadata(self) -> List[VideoGenerationMetadata]:
        raise NotImplementedError
    
    async def update_evaluation_results(self, generation_id: str, evaluation_data: Dict[str, Any]) -> bool:
        raise NotImplementedError

class SQLiteMetadataStorage(VideoMetadataStorage):
    """SQLite-based metadata storage for local development and small-scale deployment"""
    
    def __init__(self, db_path: str = "video_metadata.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with proper schema"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS video_metadata (
                    generation_id TEXT PRIMARY KEY,
                    video_id TEXT UNIQUE,
                    video_path TEXT UNIQUE,
                    filename TEXT,
                    file_size_bytes INTEGER,
                    file_created_at TEXT,
                    
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT,
                    duration REAL,
                    fps INTEGER,
                    resolution TEXT,
                    quality TEXT,
                    seed INTEGER,
                    guidance_scale REAL,
                    num_inference_steps INTEGER,
                    
                    model_name TEXT,
                    model_version TEXT,
                    device TEXT,
                    
                    actual_duration REAL,
                    actual_resolution TEXT,
                    actual_fps REAL,
                    num_frames INTEGER,
                    generation_time REAL,
                    
                    created_at TEXT,
                    completed_at TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    
                    evaluation_id TEXT,
                    overall_score REAL,
                    confidence_level TEXT,
                    requires_review INTEGER,
                    
                    tags TEXT,
                    notes TEXT,
                    metadata_version TEXT
                )
            ''')
            
            # Create indexes for common queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_video_path ON video_metadata(video_path)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_prompt ON video_metadata(prompt)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON video_metadata(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_quality ON video_metadata(quality)')
            
            conn.commit()
    
    async def store_metadata(self, metadata: VideoGenerationMetadata) -> bool:
        """Store metadata in SQLite database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Convert dataclass to dict and handle special types
                data = asdict(metadata)
                data['file_created_at'] = metadata.file_created_at.isoformat()
                data['created_at'] = metadata.created_at.isoformat()
                data['completed_at'] = metadata.completed_at.isoformat() if metadata.completed_at else None
                data['tags'] = json.dumps(metadata.tags) if metadata.tags else None
                data['requires_review'] = int(metadata.requires_review) if metadata.requires_review is not None else None
                
                # Insert or replace
                placeholders = ', '.join(['?' for _ in data.keys()])
                columns = ', '.join(data.keys())
                
                conn.execute(
                    f'INSERT OR REPLACE INTO video_metadata ({columns}) VALUES ({placeholders})',
                    list(data.values())
                )
                conn.commit()
                
            logger.info(f"✅ Stored metadata for video: {metadata.filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store metadata: {e}")
            return False
    
    async def get_metadata(self, video_path: str) -> Optional[VideoGenerationMetadata]:
        """Get metadata by video path"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM video_metadata WHERE video_path = ?',
                    (str(video_path),)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_metadata(row)
                
        except Exception as e:
            logger.error(f"❌ Failed to get metadata for {video_path}: {e}")
        
        return None
    
    async def get_metadata_by_id(self, generation_id: str) -> Optional[VideoGenerationMetadata]:
        """Get metadata by generation ID"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM video_metadata WHERE generation_id = ?',
                    (generation_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_metadata(row)
                
        except Exception as e:
            logger.error(f"❌ Failed to get metadata for ID {generation_id}: {e}")
        
        return None
    
    async def list_all_metadata(self) -> List[VideoGenerationMetadata]:
        """List all metadata records"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM video_metadata ORDER BY created_at DESC'
                )
                rows = cursor.fetchall()
                
                return [self._row_to_metadata(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to list metadata: {e}")
            return []
    
    async def update_evaluation_results(self, generation_id: str, evaluation_data: Dict[str, Any]) -> bool:
        """Update metadata with evaluation results"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    UPDATE video_metadata 
                    SET evaluation_id = ?, overall_score = ?, confidence_level = ?, requires_review = ?
                    WHERE generation_id = ?
                ''', (
                    evaluation_data.get('evaluation_id'),
                    evaluation_data.get('overall_score'),
                    evaluation_data.get('confidence_level'),
                    int(evaluation_data.get('requires_review', False)),
                    generation_id
                ))
                conn.commit()
                
            logger.info(f"✅ Updated evaluation results for {generation_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update evaluation results: {e}")
            return False
    
    def _row_to_metadata(self, row: sqlite3.Row) -> VideoGenerationMetadata:
        """Convert SQLite row to VideoGenerationMetadata"""
        return VideoGenerationMetadata(
            generation_id=row['generation_id'],
            video_id=row['video_id'],
            video_path=row['video_path'],
            filename=row['filename'],
            file_size_bytes=row['file_size_bytes'],
            file_created_at=datetime.fromisoformat(row['file_created_at']),
            
            prompt=row['prompt'],
            negative_prompt=row['negative_prompt'],
            duration=row['duration'],
            fps=row['fps'],
            resolution=row['resolution'],
            quality=row['quality'],
            seed=row['seed'],
            guidance_scale=row['guidance_scale'],
            num_inference_steps=row['num_inference_steps'],
            
            model_name=row['model_name'],
            model_version=row['model_version'],
            device=row['device'],
            
            actual_duration=row['actual_duration'],
            actual_resolution=row['actual_resolution'],
            actual_fps=row['actual_fps'],
            num_frames=row['num_frames'],
            generation_time=row['generation_time'],
            
            created_at=datetime.fromisoformat(row['created_at']),
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            user_id=row['user_id'],
            session_id=row['session_id'],
            
            evaluation_id=row['evaluation_id'],
            overall_score=row['overall_score'],
            confidence_level=row['confidence_level'],
            requires_review=bool(row['requires_review']) if row['requires_review'] is not None else None,
            
            tags=json.loads(row['tags']) if row['tags'] else None,
            notes=row['notes'],
            metadata_version=row['metadata_version']
        )

class JSONSidecarStorage(VideoMetadataStorage):
    """JSON sidecar file storage for portable metadata"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else None
    
    def _get_sidecar_path(self, video_path: str) -> Path:
        """Get the JSON sidecar file path for a video"""
        video_path = Path(video_path)
        if self.base_dir:
            # Store all sidecars in a specific directory
            sidecar_dir = self.base_dir / "metadata"
            sidecar_dir.mkdir(exist_ok=True)
            return sidecar_dir / f"{video_path.stem}.metadata.json"
        else:
            # Store sidecar next to video file
            return video_path.parent / f"{video_path.stem}.metadata.json"
    
    async def store_metadata(self, metadata: VideoGenerationMetadata) -> bool:
        """Store metadata as JSON sidecar file"""
        try:
            sidecar_path = self._get_sidecar_path(metadata.video_path)
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to JSON-serializable format
            data = asdict(metadata)
            data['file_created_at'] = metadata.file_created_at.isoformat()
            data['created_at'] = metadata.created_at.isoformat()
            data['completed_at'] = metadata.completed_at.isoformat() if metadata.completed_at else None
            
            with open(sidecar_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"✅ Stored sidecar metadata: {sidecar_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store sidecar metadata: {e}")
            return False
    
    async def get_metadata(self, video_path: str) -> Optional[VideoGenerationMetadata]:
        """Get metadata from JSON sidecar file"""
        try:
            sidecar_path = self._get_sidecar_path(video_path)
            
            if sidecar_path.exists():
                with open(sidecar_path, 'r') as f:
                    data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                data['file_created_at'] = datetime.fromisoformat(data['file_created_at'])
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['completed_at'] = datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None
                
                return VideoGenerationMetadata(**data)
                
        except Exception as e:
            logger.error(f"❌ Failed to get sidecar metadata for {video_path}: {e}")
        
        return None
    
    async def get_metadata_by_id(self, generation_id: str) -> Optional[VideoGenerationMetadata]:
        """Get metadata by generation ID (requires scanning all files)"""
        # This is inefficient for large datasets - database is preferred for ID lookups
        all_metadata = await self.list_all_metadata()
        for metadata in all_metadata:
            if metadata.generation_id == generation_id:
                return metadata
        return None
    
    async def list_all_metadata(self) -> List[VideoGenerationMetadata]:
        """List all metadata from sidecar files"""
        metadata_list = []
        
        if self.base_dir:
            # Scan metadata directory
            metadata_dir = self.base_dir / "metadata"
            if metadata_dir.exists():
                for json_file in metadata_dir.glob("*.metadata.json"):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Convert datetime strings
                        data['file_created_at'] = datetime.fromisoformat(data['file_created_at'])
                        data['created_at'] = datetime.fromisoformat(data['created_at'])
                        data['completed_at'] = datetime.fromisoformat(data['completed_at']) if data['completed_at'] else None
                        
                        metadata_list.append(VideoGenerationMetadata(**data))
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load metadata from {json_file}: {e}")
        
        return sorted(metadata_list, key=lambda x: x.created_at, reverse=True)
    
    async def update_evaluation_results(self, generation_id: str, evaluation_data: Dict[str, Any]) -> bool:
        """Update sidecar file with evaluation results"""
        # Find the metadata first
        metadata = await self.get_metadata_by_id(generation_id)
        if metadata:
            # Update evaluation fields
            metadata.evaluation_id = evaluation_data.get('evaluation_id')
            metadata.overall_score = evaluation_data.get('overall_score')
            metadata.confidence_level = evaluation_data.get('confidence_level')
            metadata.requires_review = evaluation_data.get('requires_review')
            
            # Store updated metadata
            return await self.store_metadata(metadata)
        
        return False

class VideoMetadataTracker:
    """
    Main interface for video generation metadata tracking
    Supports multiple storage backends for redundancy and flexibility
    """
    
    def __init__(self, 
                 storage_backends: Optional[List[VideoMetadataStorage]] = None,
                 generated_videos_dir: str = "generated"):
        
        self.generated_videos_dir = Path(generated_videos_dir)
        
        # Initialize default storage backends if none provided
        if storage_backends is None:
            self.storage_backends = [
                SQLiteMetadataStorage(str(self.generated_videos_dir / "metadata.db")),
                JSONSidecarStorage(str(self.generated_videos_dir))
            ]
        else:
            self.storage_backends = storage_backends
        
        logger.info(f"🗃️ Video Metadata Tracker initialized")
        logger.info(f"   Storage backends: {len(self.storage_backends)}")
        logger.info(f"   Generated videos dir: {self.generated_videos_dir}")
    
    async def track_video_generation(self, 
                                   video_path: str,
                                   request: VideoGenerationRequest,
                                   generation_result: Dict[str, Any],
                                   generation_time: float,
                                   model_info: Dict[str, Any]) -> str:
        """
        Track a video generation with full metadata
        Returns the generation_id for later reference
        """
        
        video_path = Path(video_path)
        generation_id = str(uuid.uuid4())
        video_id = str(uuid.uuid4())
        
        # Get file information
        file_stats = video_path.stat()
        
        # Create metadata object
        metadata = VideoGenerationMetadata(
            generation_id=generation_id,
            video_id=video_id,
            video_path=str(video_path.absolute()),
            filename=video_path.name,
            file_size_bytes=file_stats.st_size,
            file_created_at=datetime.fromtimestamp(file_stats.st_mtime),
            
            # Generation parameters from request
            prompt=request.prompt,
            negative_prompt=getattr(request, 'negative_prompt', None),
            duration=request.duration,
            fps=request.fps,
            resolution=request.resolution,
            quality=request.quality.value if isinstance(request.quality, VideoQuality) else str(request.quality),
            seed=request.seed,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            
            # Model information
            model_name=model_info.get('model_name', 'unknown'),
            model_version=model_info.get('model_version'),
            device=model_info.get('device', 'unknown'),
            
            # Generation results
            actual_duration=generation_result.get('duration'),
            actual_resolution=generation_result.get('resolution'),
            actual_fps=generation_result.get('fps'),
            num_frames=generation_result.get('num_frames'),
            generation_time=generation_time,
            
            # System metadata
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            user_id=None,  # Could be added later
            session_id=None,  # Could be added later
            
            # Tags for organization
            tags=["wan_generated", f"quality_{request.quality}"],
            notes=f"Generated with {model_info.get('model_name', 'unknown')} model"
        )
        
        # Store in all backends
        success_count = 0
        for backend in self.storage_backends:
            try:
                if await backend.store_metadata(metadata):
                    success_count += 1
            except Exception as e:
                logger.error(f"❌ Storage backend failed: {type(backend).__name__}: {e}")
        
        if success_count > 0:
            logger.info(f"✅ Tracked video generation: {video_path.name}")
            logger.info(f"   Generation ID: {generation_id}")
            logger.info(f"   Stored in {success_count}/{len(self.storage_backends)} backends")
        else:
            logger.error(f"❌ Failed to store metadata in any backend")
        
        return generation_id
    
    async def get_video_metadata(self, video_path: str) -> Optional[VideoGenerationMetadata]:
        """Get metadata for a video file"""
        # Try each storage backend
        for backend in self.storage_backends:
            try:
                metadata = await backend.get_metadata(video_path)
                if metadata:
                    return metadata
            except Exception as e:
                logger.debug(f"Backend {type(backend).__name__} failed: {e}")
        
        logger.warning(f"⚠️ No metadata found for video: {video_path}")
        return None
    
    async def discover_existing_videos(self) -> List[Dict[str, Any]]:
        """
        Discover existing videos in the generated directory
        Returns list of videos with available metadata
        """
        discovered = []
        
        if not self.generated_videos_dir.exists():
            logger.warning(f"⚠️ Generated videos directory not found: {self.generated_videos_dir}")
            return discovered
        
        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.generated_videos_dir.glob(f"*{ext}"))
        
        logger.info(f"🔍 Discovered {len(video_files)} video files")
        
        # Check metadata for each video
        for video_file in video_files:
            metadata = await self.get_video_metadata(str(video_file))
            
            discovered.append({
                'video_path': str(video_file),
                'filename': video_file.name,
                'has_metadata': metadata is not None,
                'metadata': metadata,
                'file_size_mb': video_file.stat().st_size / (1024 * 1024),
                'modified_at': datetime.fromtimestamp(video_file.stat().st_mtime)
            })
        
        return sorted(discovered, key=lambda x: x['modified_at'], reverse=True)
    
    async def update_evaluation_results(self, 
                                      generation_id: str, 
                                      evaluation_result: Any) -> bool:
        """Update metadata with evaluation results"""
        
        evaluation_data = {
            'evaluation_id': getattr(evaluation_result, 'evaluation_id', None),
            'overall_score': getattr(evaluation_result, 'overall_score', None),
            'confidence_level': getattr(evaluation_result, 'confidence_level', None),
            'requires_review': getattr(evaluation_result, 'requires_human_review', None)
        }
        
        # Update in all backends
        success_count = 0
        for backend in self.storage_backends:
            try:
                if await backend.update_evaluation_results(generation_id, evaluation_data):
                    success_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to update evaluation in {type(backend).__name__}: {e}")
        
        return success_count > 0

# Global metadata tracker instance
metadata_tracker = VideoMetadataTracker()
