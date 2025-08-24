"""
MLFlow model tracking integration for VisionFlow
Tracks experiments, model performance, and artifacts
"""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.transformers
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import torch

from ...shared.config import get_settings
from ...shared.monitoring import get_logger

logger = get_logger("mlflow_tracker")
settings = get_settings()


class ModelTracker:
    """MLFlow integration for tracking video generation models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.current_run = None
        self._initialize_mlflow()
    
    def _initialize_mlflow(self):
        """Initialize MLFlow tracking"""
        try:
            # Set MLFlow tracking URI
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
            mlflow.set_tracking_uri(tracking_uri)
            
            # Initialize MLFlow client
            self.client = MlflowClient(tracking_uri)
            
            # Set default experiment
            experiment_name = "visionflow-video-generation"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(
                        experiment_name,
                        artifact_location=f"gs://visionflow-mlflow-artifacts/{experiment_name}"
                    )
                else:
                    experiment_id = experiment.experiment_id
                
                mlflow.set_experiment(experiment_name)
                
                logger.info(f"MLFlow initialized with experiment: {experiment_name}")
                
            except Exception as e:
                logger.warning(f"Could not set MLFlow experiment: {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLFlow: {e}")
            # Continue without MLFlow if initialization fails
            self.client = None
    
    def start_run(
        self, 
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False
    ) -> Optional[str]:
        """Start a new MLFlow run"""
        if not self.client:
            logger.warning("MLFlow not initialized, skipping run start")
            return None
        
        try:
            if not run_name:
                run_name = f"video_generation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Default tags
            default_tags = {
                "service": "visionflow",
                "component": "video_generation",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if tags:
                default_tags.update(tags)
            
            self.current_run = mlflow.start_run(
                run_name=run_name,
                tags=default_tags,
                nested=nested
            )
            
            run_id = self.current_run.info.run_id
            logger.info(f"Started MLFlow run: {run_id} ({run_name})")
            
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLFlow run: {e}")
            return None
    
    def log_parameters(self, params: Dict[str, Any]) -> bool:
        """Log parameters to current run"""
        if not self.current_run:
            logger.warning("No active MLFlow run")
            return False
        
        try:
            # Convert all values to strings for MLFlow
            str_params = {}
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    str_params[key] = str(value)
                else:
                    str_params[key] = value
            
            mlflow.log_params(str_params)
            logger.debug(f"Logged parameters: {list(str_params.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            return False
    
    def log_metrics(
        self, 
        metrics: Dict[str, Union[float, int]], 
        step: Optional[int] = None
    ) -> bool:
        """Log metrics to current run"""
        if not self.current_run:
            logger.warning("No active MLFlow run")
            return False
        
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            return False
    
    def log_artifact(
        self, 
        local_path: str, 
        artifact_path: Optional[str] = None
    ) -> bool:
        """Log artifact to current run"""
        if not self.current_run:
            logger.warning("No active MLFlow run")
            return False
        
        try:
            if os.path.isfile(local_path):
                mlflow.log_artifact(local_path, artifact_path)
            elif os.path.isdir(local_path):
                mlflow.log_artifacts(local_path, artifact_path)
            else:
                logger.error(f"Artifact path does not exist: {local_path}")
                return False
            
            logger.debug(f"Logged artifact: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
            return False
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        model_name: Optional[str] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        conda_env: Optional[str] = None,
        pip_requirements: Optional[List[str]] = None
    ) -> bool:
        """Log model to current run"""
        if not self.current_run:
            logger.warning("No active MLFlow run")
            return False
        
        try:
            if hasattr(model, 'state_dict'):  # PyTorch model
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements
                )
            elif hasattr(model, 'save_pretrained'):  # HuggingFace model
                mlflow.transformers.log_model(
                    transformers_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements
                )
            else:
                # Generic model logging
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env,
                    pip_requirements=pip_requirements
                )
            
            logger.info(f"Logged model to artifact path: {artifact_path}")
            
            # Register model if name provided
            if model_name:
                model_uri = f"runs:/{self.current_run.info.run_id}/{artifact_path}"
                self.register_model(model_uri, model_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
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
            if output_video_path and os.path.exists(output_video_path):
                self.log_artifact(output_video_path, "generated_videos")
            
            # Log generation metadata as JSON
            import json
            metadata_path = f"/tmp/metadata_{run_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            self.log_artifact(metadata_path, "metadata")
            os.remove(metadata_path)  # Cleanup
            
            logger.info(f"Logged video generation run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to log video generation run: {e}")
            return None
        
        finally:
            self.end_run()
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Register model in MLFlow Model Registry"""
        if not self.client:
            logger.warning("MLFlow not initialized, skipping model registration")
            return None
        
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[Any]:
        """Get specific model version from registry"""
        if not self.client:
            return None
        
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            elif stage:
                model_versions = self.client.get_latest_versions(model_name, stages=[stage])
                model_version = model_versions[0] if model_versions else None
            else:
                model_versions = self.client.get_latest_versions(model_name)
                model_version = model_versions[0] if model_versions else None
            
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[Any]:
        """Load model from MLFlow registry"""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            # Try different model flavors
            try:
                return mlflow.pytorch.load_model(model_uri)
            except:
                try:
                    return mlflow.transformers.load_model(model_uri)
                except:
                    return mlflow.sklearn.load_model(model_uri)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = True
    ) -> bool:
        """Transition model to new stage"""
        if not self.client:
            return False
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False
    
    def search_runs(
        self,
        experiment_ids: Optional[List[str]] = None,
        filter_string: Optional[str] = None,
        run_view_type: ViewType = ViewType.ACTIVE_ONLY,
        max_results: int = 100
    ) -> List[Any]:
        """Search runs with filters"""
        if not self.client:
            return []
        
        try:
            runs = self.client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                run_view_type=run_view_type,
                max_results=max_results
            )
            
            return runs
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def end_run(self, status: str = "FINISHED") -> bool:
        """End current run"""
        if not self.current_run:
            return True
        
        try:
            mlflow.end_run(status=status)
            logger.debug(f"Ended MLFlow run: {self.current_run.info.run_id}")
            self.current_run = None
            return True
            
        except Exception as e:
            logger.error(f"Failed to end MLFlow run: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check MLFlow connectivity and health"""
        health = {
            "mlflow_initialized": False,
            "tracking_server_accessible": False,
            "experiment_accessible": False,
            "model_registry_accessible": False
        }
        
        try:
            # Check initialization
            health["mlflow_initialized"] = self.client is not None
            
            # Check tracking server
            if self.client:
                experiments = self.client.list_experiments()
                health["tracking_server_accessible"] = True
                health["experiment_accessible"] = len(experiments) > 0
                
                # Check model registry
                try:
                    models = self.client.list_registered_models(max_results=1)
                    health["model_registry_accessible"] = True
                except:
                    health["model_registry_accessible"] = False
            
        except Exception as e:
            logger.error(f"MLFlow health check failed: {e}")
        
        return health


# Global instance
_model_tracker_instance = None

def get_model_tracker() -> ModelTracker:
    """Get or create model tracker instance"""
    global _model_tracker_instance
    if _model_tracker_instance is None:
        _model_tracker_instance = ModelTracker()
    return _model_tracker_instance
