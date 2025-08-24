"""
PostgreSQL Production Metadata Storage
Enterprise-grade database backend for video generation metadata tracking.

Features:
- Connection pooling for high-performance
- Async operations for scalability
- Proper indexing for query optimization
- Migration support for schema evolution
- Cloud-ready configuration
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

import asyncpg
from asyncpg import Pool, Connection

from ...shared.monitoring import get_logger
from .video_metadata_tracker import VideoMetadataStorage, VideoGenerationMetadata

logger = get_logger(__name__)

class PostgreSQLMetadataStorage(VideoMetadataStorage):
    """Production PostgreSQL backend for video metadata storage"""
    
    def __init__(self, 
                 database_url: Optional[str] = None,
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "visionflow",
                 username: str = "visionflow_user",
                 password: Optional[str] = None,
                 min_pool_size: int = 5,
                 max_pool_size: int = 20):
        
        # Database connection configuration
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password or os.getenv('POSTGRES_PASSWORD', 'visionflow_pass')
        
        # Connection pool settings
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool: Optional[Pool] = None
        
        # Schema version for migrations
        self.schema_version = "1.0.0"
        
        logger.info(f"🐘 PostgreSQL Metadata Storage configured")
        logger.info(f"   Host: {self.host}:{self.port}")
        logger.info(f"   Database: {self.database}")
        logger.info(f"   Pool size: {self.min_pool_size}-{self.max_pool_size}")
    
    async def initialize(self):
        """Initialize database connection pool and schema"""
        try:
            if self.database_url:
                # Use database URL (for cloud deployments)
                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=self.min_pool_size,
                    max_size=self.max_pool_size,
                    command_timeout=60
                )
            else:
                # Use individual connection parameters
                self.pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.username,
                    password=self.password,
                    min_size=self.min_pool_size,
                    max_size=self.max_pool_size,
                    command_timeout=60
                )
            
            # Run schema migrations
            await self._run_migrations()
            
            logger.info("✅ PostgreSQL connection pool initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("🔐 PostgreSQL connection pool closed")
    
    async def _run_migrations(self):
        """Run database schema migrations"""
        async with self.pool.acquire() as conn:
            # Create schema version table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(50) PRIMARY KEY,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            ''')
            
            # Check current schema version
            current_version = await conn.fetchval(
                'SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1'
            )
            
            if current_version != self.schema_version:
                await self._apply_migration_v1_0_0(conn)
                
                # Record migration
                await conn.execute('''
                    INSERT INTO schema_migrations (version, description)
                    VALUES ($1, $2)
                    ON CONFLICT (version) DO NOTHING
                ''', self.schema_version, "Initial video metadata schema")
                
                logger.info(f"✅ Applied schema migration: {self.schema_version}")
    
    async def _apply_migration_v1_0_0(self, conn: Connection):
        """Apply initial schema migration"""
        
        # Create main video metadata table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS video_metadata (
                generation_id UUID PRIMARY KEY,
                video_id UUID UNIQUE NOT NULL,
                video_path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_size_bytes BIGINT,
                file_created_at TIMESTAMP WITH TIME ZONE,
                
                -- Generation parameters
                prompt TEXT NOT NULL,
                negative_prompt TEXT,
                duration REAL,
                fps INTEGER,
                resolution TEXT,
                quality TEXT,
                seed INTEGER,
                guidance_scale REAL,
                num_inference_steps INTEGER,
                
                -- Model information
                model_name TEXT,
                model_version TEXT,
                device TEXT,
                
                -- Generation results
                actual_duration REAL,
                actual_resolution TEXT,
                actual_fps REAL,
                num_frames INTEGER,
                generation_time REAL,
                
                -- System metadata
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                completed_at TIMESTAMP WITH TIME ZONE,
                user_id TEXT,
                session_id TEXT,
                
                -- Evaluation metadata
                evaluation_id UUID,
                overall_score REAL,
                confidence_level TEXT,
                requires_review BOOLEAN,
                
                -- Additional metadata
                tags JSONB,
                notes TEXT,
                metadata_version TEXT DEFAULT '1.0'
            )
        ''')
        
        # Create indexes for performance
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_video_metadata_video_path ON video_metadata(video_path)',
            'CREATE INDEX IF NOT EXISTS idx_video_metadata_prompt ON video_metadata USING gin(to_tsvector(\'english\', prompt))',
            'CREATE INDEX IF NOT EXISTS idx_video_metadata_created_at ON video_metadata(created_at)',
            'CREATE INDEX IF NOT EXISTS idx_video_metadata_quality ON video_metadata(quality)',
            'CREATE INDEX IF NOT EXISTS idx_video_metadata_model_name ON video_metadata(model_name)',
            'CREATE INDEX IF NOT EXISTS idx_video_metadata_evaluation ON video_metadata(evaluation_id) WHERE evaluation_id IS NOT NULL',
            'CREATE INDEX IF NOT EXISTS idx_video_metadata_tags ON video_metadata USING gin(tags)',
            'CREATE INDEX IF NOT EXISTS idx_video_metadata_user_session ON video_metadata(user_id, session_id)',
            'CREATE INDEX IF NOT EXISTS idx_video_metadata_confidence ON video_metadata(confidence_level, requires_review)'
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        
        # Create evaluation results table for detailed dimension scores
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_results (
                evaluation_id UUID PRIMARY KEY,
                generation_id UUID REFERENCES video_metadata(generation_id) ON DELETE CASCADE,
                
                -- Overall results
                overall_score REAL NOT NULL,
                overall_confidence REAL NOT NULL,
                confidence_level TEXT NOT NULL,
                decision TEXT NOT NULL,
                requires_human_review BOOLEAN NOT NULL,
                
                -- Dimension scores (JSONB for flexibility)
                dimension_scores JSONB NOT NULL,
                
                -- Evaluation metadata
                evaluation_strategy TEXT,
                sampling_strategy TEXT,
                frames_evaluated INTEGER,
                processing_time REAL,
                
                -- Timestamps
                evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                evaluator_version TEXT
            )
        ''')
        
        # Indexes for evaluation results
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_evaluation_results_generation ON evaluation_results(generation_id)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_evaluation_results_score ON evaluation_results(overall_score)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_evaluation_results_confidence ON evaluation_results(confidence_level)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_evaluation_results_date ON evaluation_results(evaluated_at)')
        
        # Create review queue table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS review_queue (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                evaluation_id UUID REFERENCES evaluation_results(evaluation_id) ON DELETE CASCADE,
                generation_id UUID REFERENCES video_metadata(generation_id) ON DELETE CASCADE,
                
                priority TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                assigned_to TEXT,
                
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                assigned_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE,
                
                review_notes TEXT,
                review_decision TEXT
            )
        ''')
        
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue(status, priority)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_review_queue_assigned ON review_queue(assigned_to, status)')
        
        logger.info("✅ Created PostgreSQL schema tables and indexes")
    
    async def store_metadata(self, metadata: VideoGenerationMetadata) -> bool:
        """Store metadata in PostgreSQL database"""
        try:
            async with self.pool.acquire() as conn:
                # Convert tags to JSONB
                tags_json = json.dumps(metadata.tags) if metadata.tags else None
                
                await conn.execute('''
                    INSERT INTO video_metadata (
                        generation_id, video_id, video_path, filename, file_size_bytes, file_created_at,
                        prompt, negative_prompt, duration, fps, resolution, quality, seed, 
                        guidance_scale, num_inference_steps,
                        model_name, model_version, device,
                        actual_duration, actual_resolution, actual_fps, num_frames, generation_time,
                        created_at, completed_at, user_id, session_id,
                        evaluation_id, overall_score, confidence_level, requires_review,
                        tags, notes, metadata_version
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34
                    )
                    ON CONFLICT (generation_id) DO UPDATE SET
                        video_path = EXCLUDED.video_path,
                        filename = EXCLUDED.filename,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        prompt = EXCLUDED.prompt,
                        negative_prompt = EXCLUDED.negative_prompt,
                        duration = EXCLUDED.duration,
                        fps = EXCLUDED.fps,
                        resolution = EXCLUDED.resolution,
                        quality = EXCLUDED.quality,
                        seed = EXCLUDED.seed,
                        guidance_scale = EXCLUDED.guidance_scale,
                        num_inference_steps = EXCLUDED.num_inference_steps,
                        model_name = EXCLUDED.model_name,
                        model_version = EXCLUDED.model_version,
                        device = EXCLUDED.device,
                        actual_duration = EXCLUDED.actual_duration,
                        actual_resolution = EXCLUDED.actual_resolution,
                        actual_fps = EXCLUDED.actual_fps,
                        num_frames = EXCLUDED.num_frames,
                        generation_time = EXCLUDED.generation_time,
                        completed_at = EXCLUDED.completed_at,
                        user_id = EXCLUDED.user_id,
                        session_id = EXCLUDED.session_id,
                        evaluation_id = EXCLUDED.evaluation_id,
                        overall_score = EXCLUDED.overall_score,
                        confidence_level = EXCLUDED.confidence_level,
                        requires_review = EXCLUDED.requires_review,
                        tags = EXCLUDED.tags,
                        notes = EXCLUDED.notes
                ''', 
                uuid.UUID(metadata.generation_id),
                uuid.UUID(metadata.video_id),
                metadata.video_path,
                metadata.filename,
                metadata.file_size_bytes,
                metadata.file_created_at,
                metadata.prompt,
                metadata.negative_prompt,
                metadata.duration,
                metadata.fps,
                metadata.resolution,
                metadata.quality,
                metadata.seed,
                metadata.guidance_scale,
                metadata.num_inference_steps,
                metadata.model_name,
                metadata.model_version,
                metadata.device,
                metadata.actual_duration,
                metadata.actual_resolution,
                metadata.actual_fps,
                metadata.num_frames,
                metadata.generation_time,
                metadata.created_at,
                metadata.completed_at,
                metadata.user_id,
                metadata.session_id,
                uuid.UUID(metadata.evaluation_id) if metadata.evaluation_id else None,
                metadata.overall_score,
                metadata.confidence_level,
                metadata.requires_review,
                tags_json,
                metadata.notes,
                metadata.metadata_version
                )
                
            logger.debug(f"✅ Stored metadata in PostgreSQL: {metadata.filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store metadata in PostgreSQL: {e}")
            return False
    
    async def get_metadata(self, video_path: str) -> Optional[VideoGenerationMetadata]:
        """Get metadata by video path"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    'SELECT * FROM video_metadata WHERE video_path = $1',
                    video_path
                )
                
                if row:
                    return self._row_to_metadata(row)
                    
        except Exception as e:
            logger.error(f"❌ Failed to get metadata from PostgreSQL: {e}")
        
        return None
    
    async def get_metadata_by_id(self, generation_id: str) -> Optional[VideoGenerationMetadata]:
        """Get metadata by generation ID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    'SELECT * FROM video_metadata WHERE generation_id = $1',
                    uuid.UUID(generation_id)
                )
                
                if row:
                    return self._row_to_metadata(row)
                    
        except Exception as e:
            logger.error(f"❌ Failed to get metadata by ID from PostgreSQL: {e}")
        
        return None
    
    async def list_all_metadata(self, 
                              limit: int = 100, 
                              offset: int = 0,
                              order_by: str = "created_at",
                              order_desc: bool = True) -> List[VideoGenerationMetadata]:
        """List metadata with pagination and ordering"""
        try:
            order_direction = "DESC" if order_desc else "ASC"
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(f'''
                    SELECT * FROM video_metadata 
                    ORDER BY {order_by} {order_direction}
                    LIMIT $1 OFFSET $2
                ''', limit, offset)
                
                return [self._row_to_metadata(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to list metadata from PostgreSQL: {e}")
            return []
    
    async def search_metadata(self, 
                            query: str,
                            filters: Optional[Dict[str, Any]] = None,
                            limit: int = 50) -> List[VideoGenerationMetadata]:
        """Search metadata with filters"""
        try:
            where_clauses = []
            params = []
            param_count = 0
            
            # Text search in prompt
            if query:
                param_count += 1
                where_clauses.append(f"to_tsvector('english', prompt) @@ plainto_tsquery('english', ${param_count})")
                params.append(query)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if key in ['quality', 'model_name', 'device', 'confidence_level']:
                        param_count += 1
                        where_clauses.append(f"{key} = ${param_count}")
                        params.append(value)
                    elif key == 'created_after':
                        param_count += 1
                        where_clauses.append(f"created_at >= ${param_count}")
                        params.append(value)
                    elif key == 'score_min':
                        param_count += 1
                        where_clauses.append(f"overall_score >= ${param_count}")
                        params.append(value)
            
            where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            params.append(limit)
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(f'''
                    SELECT * FROM video_metadata 
                    {where_sql}
                    ORDER BY created_at DESC
                    LIMIT ${param_count + 1}
                ''', *params)
                
                return [self._row_to_metadata(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Failed to search metadata in PostgreSQL: {e}")
            return []
    
    async def update_evaluation_results(self, generation_id: str, evaluation_data: Dict[str, Any]) -> bool:
        """Update metadata with evaluation results"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    UPDATE video_metadata 
                    SET evaluation_id = $1, overall_score = $2, confidence_level = $3, requires_review = $4
                    WHERE generation_id = $5
                ''',
                uuid.UUID(evaluation_data.get('evaluation_id')) if evaluation_data.get('evaluation_id') else None,
                evaluation_data.get('overall_score'),
                evaluation_data.get('confidence_level'),
                evaluation_data.get('requires_review'),
                uuid.UUID(generation_id)
                )
                
            logger.debug(f"✅ Updated evaluation results in PostgreSQL: {generation_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update evaluation results in PostgreSQL: {e}")
            return False
    
    async def store_evaluation_result(self, evaluation_result: Any) -> bool:
        """Store detailed evaluation results"""
        try:
            # Convert dimension scores to JSONB
            dimension_scores = []
            for dim_score in evaluation_result.dimension_scores:
                dimension_scores.append({
                    'dimension': dim_score.dimension.value,
                    'score': dim_score.score,
                    'confidence': dim_score.confidence,
                    'details': dim_score.details
                })
            
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO evaluation_results (
                        evaluation_id, generation_id, overall_score, overall_confidence,
                        confidence_level, decision, requires_human_review, dimension_scores,
                        evaluation_strategy, sampling_strategy, frames_evaluated,
                        processing_time, evaluator_version
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (evaluation_id) DO UPDATE SET
                        overall_score = EXCLUDED.overall_score,
                        overall_confidence = EXCLUDED.overall_confidence,
                        confidence_level = EXCLUDED.confidence_level,
                        decision = EXCLUDED.decision,
                        requires_human_review = EXCLUDED.requires_human_review,
                        dimension_scores = EXCLUDED.dimension_scores
                ''',
                uuid.UUID(evaluation_result.evaluation_id),
                uuid.UUID(evaluation_result.generation_id) if hasattr(evaluation_result, 'generation_id') else None,
                evaluation_result.overall_score,
                evaluation_result.overall_confidence,
                evaluation_result.confidence_level.value,
                evaluation_result.decision,
                evaluation_result.requires_human_review,
                json.dumps(dimension_scores),
                getattr(evaluation_result, 'evaluation_strategy', 'multi_dimensional'),
                getattr(evaluation_result, 'sampling_strategy', 'adaptive'),
                getattr(evaluation_result, 'frames_evaluated', 0),
                getattr(evaluation_result, 'processing_time', 0.0),
                getattr(evaluation_result, 'evaluator_version', '1.0.0')
                )
                
            logger.debug(f"✅ Stored evaluation result in PostgreSQL: {evaluation_result.evaluation_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store evaluation result in PostgreSQL: {e}")
            return False
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get analytics and statistics"""
        try:
            async with self.pool.acquire() as conn:
                # Basic statistics
                total_videos = await conn.fetchval('SELECT COUNT(*) FROM video_metadata')
                total_evaluated = await conn.fetchval('SELECT COUNT(*) FROM video_metadata WHERE evaluation_id IS NOT NULL')
                
                # Quality distribution
                quality_dist = await conn.fetch('''
                    SELECT quality, COUNT(*) as count 
                    FROM video_metadata 
                    GROUP BY quality
                ''')
                
                # Model usage
                model_usage = await conn.fetch('''
                    SELECT model_name, COUNT(*) as count 
                    FROM video_metadata 
                    GROUP BY model_name 
                    ORDER BY count DESC
                ''')
                
                # Average scores
                avg_scores = await conn.fetchrow('''
                    SELECT 
                        AVG(overall_score) as avg_score,
                        MIN(overall_score) as min_score,
                        MAX(overall_score) as max_score
                    FROM video_metadata 
                    WHERE overall_score IS NOT NULL
                ''')
                
                # Confidence distribution
                confidence_dist = await conn.fetch('''
                    SELECT confidence_level, COUNT(*) as count 
                    FROM video_metadata 
                    WHERE confidence_level IS NOT NULL
                    GROUP BY confidence_level
                ''')
                
                return {
                    'total_videos': total_videos,
                    'total_evaluated': total_evaluated,
                    'evaluation_coverage': total_evaluated / total_videos if total_videos > 0 else 0,
                    'quality_distribution': {row['quality']: row['count'] for row in quality_dist},
                    'model_usage': {row['model_name']: row['count'] for row in model_usage},
                    'score_statistics': dict(avg_scores) if avg_scores else {},
                    'confidence_distribution': {row['confidence_level']: row['count'] for row in confidence_dist}
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get analytics from PostgreSQL: {e}")
            return {}
    
    def _row_to_metadata(self, row) -> VideoGenerationMetadata:
        """Convert PostgreSQL row to VideoGenerationMetadata"""
        # Parse tags from JSONB
        tags = json.loads(row['tags']) if row['tags'] else None
        
        return VideoGenerationMetadata(
            generation_id=str(row['generation_id']),
            video_id=str(row['video_id']),
            video_path=row['video_path'],
            filename=row['filename'],
            file_size_bytes=row['file_size_bytes'],
            file_created_at=row['file_created_at'],
            
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
            
            created_at=row['created_at'],
            completed_at=row['completed_at'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            
            evaluation_id=str(row['evaluation_id']) if row['evaluation_id'] else None,
            overall_score=row['overall_score'],
            confidence_level=row['confidence_level'],
            requires_review=row['requires_review'],
            
            tags=tags,
            notes=row['notes'],
            metadata_version=row['metadata_version']
        )

# Production configuration helper
def get_postgresql_storage() -> PostgreSQLMetadataStorage:
    """Get configured PostgreSQL storage backend"""
    
    # Production/Cloud configuration
    if os.getenv('DATABASE_URL'):
        # Cloud deployment (Heroku, Railway, etc.)
        return PostgreSQLMetadataStorage(database_url=os.getenv('DATABASE_URL'))
    
    # Local development configuration
    return PostgreSQLMetadataStorage(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'visionflow'),
        username=os.getenv('POSTGRES_USER', 'visionflow_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'visionflow_pass'),
        min_pool_size=int(os.getenv('POSTGRES_MIN_POOL', '5')),
        max_pool_size=int(os.getenv('POSTGRES_MAX_POOL', '20'))
    )
