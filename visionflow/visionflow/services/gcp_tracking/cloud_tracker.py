"""
GCP Cloud Logging and Monitoring based tracking
Replaces MLFlow experiment tracking with native GCP services
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from google.cloud import logging as cloud_logging
from google.cloud import monitoring_v3
from google.cloud import storage
from google.cloud.monitoring_v3 import TimeSeries, Point, TimeInterval
from google.cloud.monitoring_v3 import MetricDescriptor

from ...shared.config import get_settings
from ...shared.monitoring import get_logger

logger = get_logger("cloud_tracker")
settings = get_settings()


class CloudTracker:
    """GCP-based tracking for video generation runs"""
    
    def __init__(self):
        self.settings = get_settings()
        self.current_run = None
        self.project_id = None
        
        # Initialize GCP clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize GCP service clients"""
        try:
            # Get project ID from environment or Vertex AI config
            self.project_id = (
                self.settings.monitoring.vertex_ai_project 
                if hasattr(self.settings.monitoring, 'vertex_ai_project') 
                else "visionflow-gcp-project"
            )
            
            # Initialize Cloud Logging
            self.logging_client = cloud_logging.Client(project=self.project_id)
            self.cloud_logger = self.logging_client.logger("visionflow-experiments")
            
            # Initialize Cloud Monitoring
            self.monitoring_client = monitoring_v3.MetricServiceClient()
            
            # Initialize Cloud Storage for artifacts
            self.storage_client = storage.Client(project=self.project_id)
            self.artifacts_bucket = self.settings.storage.bucket_name
            
            logger.info("GCP tracking clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            # Continue without GCP tracking if initialization fails
            self.logging_client = None
            self.monitoring_client = None
            self.storage_client = None
    
    def start_run(
        self, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False
    ) -> Optional[str]:
        """Start a new tracking run"""
        if not self.logging_client:
            logger.warning("GCP tracking not initialized, skipping run start")
            return None
        
        try:
            run_id = str(uuid.uuid4())
            
            if not run_name:
                run_name = f"video_generation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create run metadata
            run_metadata = {
                "run_id": run_id,
                "run_name": run_name,
                "start_time": datetime.utcnow().isoformat(),
                "status": "RUNNING",
                "tags": tags or {},
                "service": "visionflow",
                "component": "video_generation"
            }
            
            # Store current run
            self.current_run = run_metadata
            
            # Log run start to Cloud Logging
            self.cloud_logger.log_struct(
                {
                    "event_type": "run_start",
                    "run_metadata": run_metadata,
                    "timestamp": datetime.utcnow().isoformat()
                },
                severity="INFO"
            )
            
            logger.info(f"Started tracking run: {run_id} ({run_name})")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start tracking run: {e}")
            return None
    
    def log_parameters(self, params: Dict[str, Any]) -> bool:
        """Log parameters to current run"""
        if not self.current_run:
            logger.warning("No active tracking run")
            return False
        
        try:
            # Add parameters to run metadata
            self.current_run["parameters"] = params
            
            # Log to Cloud Logging
            self.cloud_logger.log_struct(
                {
                    "event_type": "parameters_logged",
                    "run_id": self.current_run["run_id"],
                    "parameters": params,
                    "timestamp": datetime.utcnow().isoformat()
                },
                severity="INFO"
            )
            
            logger.debug(f"Logged parameters: {list(params.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            return False
    
    def log_metrics(
        self, 
        metrics: Dict[str, Union[float, int]], 
        step: Optional[int] = None
    ) -> bool:
        """Log metrics to Cloud Monitoring and Cloud Logging"""
        if not self.current_run:
            logger.warning("No active tracking run")
            return False
        
        try:
            # Log to Cloud Logging for historical tracking
            self.cloud_logger.log_struct(
                {
                    "event_type": "metrics_logged",
                    "run_id": self.current_run["run_id"],
                    "metrics": metrics,
                    "step": step,
                    "timestamp": datetime.utcnow().isoformat()
                },
                severity="INFO"
            )
            
            # Send to Cloud Monitoring for real-time monitoring
            if self.monitoring_client:
                try:
                    self._send_metrics_to_monitoring(metrics, step)
                except Exception as e:
                    logger.warning(f"Failed to send metrics to Cloud Monitoring: {e}")
            
            # Store in run metadata
            if "metrics" not in self.current_run:
                self.current_run["metrics"] = {}
            self.current_run["metrics"].update(metrics)
            
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            return False
    
    def _send_metrics_to_monitoring(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Send metrics to Cloud Monitoring"""
        project_name = f"projects/{self.project_id}"
        now = datetime.utcnow()
        
        for metric_name, value in metrics.items():
            # Create time series
            series = TimeSeries()
            series.metric.type = f"custom.googleapis.com/visionflow/{metric_name}"
            series.resource.type = "global"
            
            # Add run metadata as labels
            series.metric.labels["run_id"] = self.current_run["run_id"]
            series.metric.labels["run_name"] = self.current_run["run_name"]
            
            # Create data point
            point = Point()
            point.value.double_value = float(value)
            point.interval.end_time.FromDatetime(now)
            series.points = [point]
            
            # Send to Cloud Monitoring
            try:
                self.monitoring_client.create_time_series(
                    name=project_name, 
                    time_series=[series]
                )
            except Exception as e:
                logger.debug(f"Could not send metric {metric_name} to Cloud Monitoring: {e}")
    
    def log_artifact(
        self, 
        local_path: str, 
        artifact_path: Optional[str] = None
    ) -> bool:
        """Log artifact to Cloud Storage"""
        if not self.current_run or not self.storage_client:
            logger.warning("No active tracking run or storage client")
            return False
        
        try:
            # Get bucket
            bucket = self.storage_client.bucket(self.artifacts_bucket)
            
            # Create blob path
            run_id = self.current_run["run_id"]
            if artifact_path:
                blob_name = f"experiments/{run_id}/{artifact_path}/{Path(local_path).name}"
            else:
                blob_name = f"experiments/{run_id}/artifacts/{Path(local_path).name}"
            
            # Upload file
            blob = bucket.blob(blob_name)
            
            if Path(local_path).is_file():
                blob.upload_from_filename(local_path)
            else:
                logger.error(f"Artifact path does not exist: {local_path}")
                return False
            
            # Log artifact info
            self.cloud_logger.log_struct(
                {
                    "event_type": "artifact_logged",
                    "run_id": run_id,
                    "local_path": local_path,
                    "gcs_path": f"gs://{self.artifacts_bucket}/{blob_name}",
                    "artifact_path": artifact_path,
                    "timestamp": datetime.utcnow().isoformat()
                },
                severity="INFO"
            )
            
            logger.debug(f"Logged artifact: {local_path} → gs://{self.artifacts_bucket}/{blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
            return False
    
    def log_video_generation_run(
        self,
        user_request: str,
        generation_params: Dict[str, Any],
        output_video_path: str,
        metadata: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        run_name: Optional[str] = None
    ) -> Optional[str]:
        """Log a complete video generation run"""
        
        # Start run
        run_id = self.start_run(
            run_name=run_name,
            tags={
                "task": "video_generation",
                "user_request": user_request[:100],  # Truncate for tag
                "model": generation_params.get("model", "wan2-1-fast")
            }
        )
        
        if not run_id:
            return None
        
        try:
            # Log parameters
            self.log_parameters({
                "user_request": user_request,
                "prompt": generation_params.get("prompt", ""),
                "duration": generation_params.get("duration", 5),
                "resolution": generation_params.get("resolution", "512x512"),
                "fps": generation_params.get("fps", 24),
                "model": generation_params.get("model", "wan2-1-fast"),
                "seed": generation_params.get("seed", None)
            })
            
            # Log metrics if provided
            if metrics:
                self.log_metrics(metrics)
            
            # Log metadata as metrics where applicable
            if metadata:
                numeric_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        numeric_metadata[f"metadata_{key}"] = value
                
                if numeric_metadata:
                    self.log_metrics(numeric_metadata)
            
            # Log output video
            if output_video_path and Path(output_video_path).exists():
                self.log_artifact(output_video_path, "generated_videos")
            
            # Log generation metadata as JSON to Cloud Storage
            metadata_blob_name = f"experiments/{run_id}/metadata.json"
            bucket = self.storage_client.bucket(self.artifacts_bucket)
            metadata_blob = bucket.blob(metadata_blob_name)
            metadata_blob.upload_from_string(
                json.dumps(metadata, indent=2, default=str),
                content_type='application/json'
            )
            
            logger.info(f"Logged video generation run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to log video generation run: {e}")
            return None
        
        finally:
            self.end_run()
    
    def search_runs(
        self,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search runs from Cloud Logging"""
        if not self.logging_client:
            return []
        
        try:
            # Build Cloud Logging filter
            base_filter = 'resource.type="global" AND jsonPayload.event_type="run_start"'
            
            if filter_conditions:
                for key, value in filter_conditions.items():
                    base_filter += f' AND jsonPayload.run_metadata.{key}="{value}"'
            
            # Query Cloud Logging
            entries = self.logging_client.list_entries(
                filter_=base_filter,
                max_results=limit,
                order_by=cloud_logging.DESCENDING
            )
            
            runs = []
            for entry in entries:
                if hasattr(entry, 'payload') and 'run_metadata' in entry.payload:
                    runs.append(entry.payload['run_metadata'])
            
            return runs
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def end_run(self, status: str = "FINISHED") -> bool:
        """End current run"""
        if not self.current_run:
            return True
        
        try:
            # Update run status
            self.current_run["status"] = status
            self.current_run["end_time"] = datetime.utcnow().isoformat()
            
            # Log run completion
            self.cloud_logger.log_struct(
                {
                    "event_type": "run_end",
                    "run_id": self.current_run["run_id"],
                    "status": status,
                    "duration_seconds": (
                        datetime.utcnow() - datetime.fromisoformat(self.current_run["start_time"])
                    ).total_seconds(),
                    "final_metadata": self.current_run,
                    "timestamp": datetime.utcnow().isoformat()
                },
                severity="INFO"
            )
            
            logger.debug(f"Ended tracking run: {self.current_run['run_id']}")
            self.current_run = None
            return True
            
        except Exception as e:
            logger.error(f"Failed to end tracking run: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check GCP services connectivity and health"""
        health = {
            "cloud_logging_available": False,
            "cloud_monitoring_available": False,
            "cloud_storage_available": False,
            "project_id": self.project_id
        }
        
        try:
            # Check Cloud Logging
            if self.logging_client:
                # Try to log a test message
                self.cloud_logger.log_text("Health check", severity="DEBUG")
                health["cloud_logging_available"] = True
            
            # Check Cloud Monitoring
            if self.monitoring_client:
                # Try to list metric descriptors (limited call)
                project_name = f"projects/{self.project_id}"
                list(self.monitoring_client.list_metric_descriptors(
                    name=project_name, 
                    page_size=1
                ))
                health["cloud_monitoring_available"] = True
            
            # Check Cloud Storage
            if self.storage_client:
                bucket = self.storage_client.bucket(self.artifacts_bucket)
                bucket.exists()  # Check if bucket is accessible
                health["cloud_storage_available"] = True
            
        except Exception as e:
            logger.error(f"GCP health check failed: {e}")
        
        return health


# Global instance
_cloud_tracker_instance = None

def get_cloud_tracker() -> CloudTracker:
    """Get or create cloud tracker instance"""
    global _cloud_tracker_instance
    if _cloud_tracker_instance is None:
        _cloud_tracker_instance = CloudTracker()
    return _cloud_tracker_instance
