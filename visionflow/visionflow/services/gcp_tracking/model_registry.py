"""
GCP-based model registry using Cloud Storage and Kubernetes ConfigMaps
Replaces MLFlow Model Registry with native GCP services
"""

import json
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from enum import Enum

from google.cloud import storage
from kubernetes import client, config

from ...shared.config import get_settings
from ...shared.monitoring import get_logger

logger = get_logger("cloud_model_registry")
settings = get_settings()


class ModelStage(Enum):
    """Model deployment stages"""
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    NONE = "none"


class CloudModelRegistry:
    """GCP-based model registry for video generation models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.project_id = (
            self.settings.monitoring.vertex_ai_project 
            if hasattr(self.settings.monitoring, 'vertex_ai_project') 
            else "visionflow-gcp-project"
        )
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize GCP and Kubernetes clients"""
        try:
            # Initialize Cloud Storage
            self.storage_client = storage.Client(project=self.project_id)
            self.models_bucket = self.settings.storage.bucket_name
            
            # Initialize Kubernetes client
            try:
                config.load_incluster_config()  # Try in-cluster config first
            except:
                try:
                    config.load_kube_config()  # Fallback to local config
                except:
                    logger.warning("Could not load Kubernetes config")
                    self.k8s_client = None
                    return
            
            self.k8s_client = client.CoreV1Api()
            self.k8s_namespace = "visionflow"
            
            logger.info("Model registry clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model registry clients: {e}")
            self.storage_client = None
            self.k8s_client = None
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        stage: ModelStage = ModelStage.NONE
    ) -> Optional[str]:
        """Register a new model version"""
        if not self.storage_client:
            logger.warning("Model registry not initialized")
            return None
        
        try:
            # Generate version if not provided
            if not version:
                version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Create model metadata
            model_metadata = {
                "name": model_name,
                "version": version,
                "description": description or "",
                "tags": tags or {},
                "stage": stage.value,
                "model_path": model_path,
                "registration_time": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Store metadata in Cloud Storage
            bucket = self.storage_client.bucket(self.models_bucket)
            metadata_blob_name = f"models/{model_name}/versions/{version}/metadata.json"
            metadata_blob = bucket.blob(metadata_blob_name)
            metadata_blob.upload_from_string(
                json.dumps(model_metadata, indent=2),
                content_type='application/json'
            )
            
            # Update model index
            self._update_model_index(model_name, version, model_metadata)
            
            # Update Kubernetes ConfigMap if model is for staging/production
            if stage in [ModelStage.STAGING, ModelStage.PRODUCTION]:
                self._update_k8s_config(model_name, version, stage)
            
            logger.info(f"Registered model: {model_name} version {version}")
            return version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def _update_model_index(self, model_name: str, version: str, metadata: Dict[str, Any]):
        """Update the model index with new version"""
        try:
            bucket = self.storage_client.bucket(self.models_bucket)
            index_blob_name = f"models/{model_name}/index.json"
            index_blob = bucket.blob(index_blob_name)
            
            # Get existing index or create new one
            try:
                existing_index = json.loads(index_blob.download_as_text())
            except:
                existing_index = {
                    "model_name": model_name,
                    "versions": {},
                    "latest_version": None,
                    "created_time": datetime.utcnow().isoformat()
                }
            
            # Add new version
            existing_index["versions"][version] = metadata
            existing_index["latest_version"] = version
            existing_index["last_updated"] = datetime.utcnow().isoformat()
            
            # Upload updated index
            index_blob.upload_from_string(
                json.dumps(existing_index, indent=2),
                content_type='application/json'
            )
            
        except Exception as e:
            logger.error(f"Failed to update model index: {e}")
    
    def _update_k8s_config(self, model_name: str, version: str, stage: ModelStage):
        """Update Kubernetes ConfigMap with model deployment info"""
        if not self.k8s_client:
            logger.warning("Kubernetes client not available")
            return
        
        try:
            config_name = f"model-{stage.value}-config"
            
            # Get existing ConfigMap or create new one
            try:
                config_map = self.k8s_client.read_namespaced_config_map(
                    name=config_name,
                    namespace=self.k8s_namespace
                )
                config_data = config_map.data or {}
            except:
                config_data = {}
            
            # Update model configuration
            config_data[f"{model_name}_version"] = version
            config_data[f"{model_name}_path"] = f"gs://{self.models_bucket}/models/{model_name}/versions/{version}"
            config_data[f"{model_name}_updated"] = datetime.utcnow().isoformat()
            
            # Create or update ConfigMap
            config_map_body = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(
                    name=config_name,
                    namespace=self.k8s_namespace
                ),
                data=config_data
            )
            
            try:
                self.k8s_client.patch_namespaced_config_map(
                    name=config_name,
                    namespace=self.k8s_namespace,
                    body=config_map_body
                )
            except:
                self.k8s_client.create_namespaced_config_map(
                    namespace=self.k8s_namespace,
                    body=config_map_body
                )
            
            logger.info(f"Updated Kubernetes config for {model_name} {stage.value}")
            
        except Exception as e:
            logger.error(f"Failed to update Kubernetes config: {e}")
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[Dict[str, Any]]:
        """Get specific model version metadata"""
        if not self.storage_client:
            return None
        
        try:
            bucket = self.storage_client.bucket(self.models_bucket)
            
            if version:
                # Get specific version
                metadata_blob_name = f"models/{model_name}/versions/{version}/metadata.json"
                metadata_blob = bucket.blob(metadata_blob_name)
                return json.loads(metadata_blob.download_as_text())
            
            elif stage:
                # Get version by stage - find from index
                index_blob_name = f"models/{model_name}/index.json"
                index_blob = bucket.blob(index_blob_name)
                index_data = json.loads(index_blob.download_as_text())
                
                # Find version with matching stage
                for ver, metadata in index_data["versions"].items():
                    if metadata.get("stage") == stage.value:
                        return metadata
                
                return None
            
            else:
                # Get latest version
                index_blob_name = f"models/{model_name}/index.json"
                index_blob = bucket.blob(index_blob_name)
                index_data = json.loads(index_blob.download_as_text())
                
                latest_version = index_data.get("latest_version")
                if latest_version:
                    return index_data["versions"][latest_version]
                
                return None
            
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
    
    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a model"""
        if not self.storage_client:
            return []
        
        try:
            bucket = self.storage_client.bucket(self.models_bucket)
            index_blob_name = f"models/{model_name}/index.json"
            index_blob = bucket.blob(index_blob_name)
            
            index_data = json.loads(index_blob.download_as_text())
            return list(index_data["versions"].values())
            
        except Exception as e:
            logger.error(f"Failed to list model versions: {e}")
            return []
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        if not self.storage_client:
            return []
        
        try:
            bucket = self.storage_client.bucket(self.models_bucket)
            blobs = bucket.list_blobs(prefix="models/")
            
            model_names = set()
            for blob in blobs:
                # Extract model name from path: models/{model_name}/...
                path_parts = blob.name.split('/')
                if len(path_parts) >= 2 and path_parts[0] == "models":
                    model_names.add(path_parts[1])
            
            return list(model_names)
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        new_stage: ModelStage,
        archive_existing: bool = True
    ) -> bool:
        """Transition model to new stage"""
        try:
            # Get current model metadata
            current_metadata = self.get_model_version(model_name, version)
            if not current_metadata:
                logger.error(f"Model version not found: {model_name}:{version}")
                return False
            
            # Archive existing models in target stage if requested
            if archive_existing and new_stage != ModelStage.NONE:
                self._archive_existing_in_stage(model_name, new_stage)
            
            # Update stage in metadata
            current_metadata["stage"] = new_stage.value
            current_metadata["last_updated"] = datetime.utcnow().isoformat()
            
            # Save updated metadata
            bucket = self.storage_client.bucket(self.models_bucket)
            metadata_blob_name = f"models/{model_name}/versions/{version}/metadata.json"
            metadata_blob = bucket.blob(metadata_blob_name)
            metadata_blob.upload_from_string(
                json.dumps(current_metadata, indent=2),
                content_type='application/json'
            )
            
            # Update model index
            self._update_model_index(model_name, version, current_metadata)
            
            # Update Kubernetes config
            if new_stage in [ModelStage.STAGING, ModelStage.PRODUCTION]:
                self._update_k8s_config(model_name, version, new_stage)
            
            logger.info(f"Transitioned {model_name}:{version} to {new_stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False
    
    def _archive_existing_in_stage(self, model_name: str, stage: ModelStage):
        """Archive existing models in the target stage"""
        try:
            versions = self.list_model_versions(model_name)
            for version_metadata in versions:
                if version_metadata.get("stage") == stage.value:
                    version_metadata["stage"] = ModelStage.ARCHIVED.value
                    version_metadata["archived_time"] = datetime.utcnow().isoformat()
                    
                    # Save updated metadata
                    bucket = self.storage_client.bucket(self.models_bucket)
                    metadata_blob_name = f"models/{model_name}/versions/{version_metadata['version']}/metadata.json"
                    metadata_blob = bucket.blob(metadata_blob_name)
                    metadata_blob.upload_from_string(
                        json.dumps(version_metadata, indent=2),
                        content_type='application/json'
                    )
            
        except Exception as e:
            logger.error(f"Failed to archive existing models: {e}")
    
    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get production model version"""
        return self.get_model_version(model_name, stage=ModelStage.PRODUCTION)
    
    def get_staging_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get staging model version"""
        return self.get_model_version(model_name, stage=ModelStage.STAGING)
    
    def deploy_to_production(self, model_name: str, version: str) -> bool:
        """Deploy model version to production"""
        return self.transition_model_stage(
            model_name, version, ModelStage.PRODUCTION, archive_existing=True
        )
    
    def deploy_to_staging(self, model_name: str, version: str) -> bool:
        """Deploy model version to staging"""
        return self.transition_model_stage(
            model_name, version, ModelStage.STAGING, archive_existing=True
        )
    
    def rollback_production(
        self, 
        model_name: str, 
        target_version: Optional[str] = None
    ) -> bool:
        """Rollback production to previous version"""
        try:
            if target_version:
                # Rollback to specific version
                return self.deploy_to_production(model_name, target_version)
            else:
                # Rollback to previous production version
                versions = self.list_model_versions(model_name)
                
                # Sort by registration time, find previous production version
                production_versions = [
                    v for v in versions 
                    if v.get("stage") == ModelStage.ARCHIVED.value 
                    and v.get("archived_time")
                ]
                
                if not production_versions:
                    logger.error("No previous production version found for rollback")
                    return False
                
                # Get most recent archived version
                latest_archived = max(
                    production_versions, 
                    key=lambda x: x.get("archived_time", "")
                )
                
                return self.deploy_to_production(model_name, latest_archived["version"])
            
        except Exception as e:
            logger.error(f"Failed to rollback production model: {e}")
            return False
    
    def compare_model_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
        comparison_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        try:
            # Get metadata for both versions
            metadata_a = self.get_model_version(model_name, version_a)
            metadata_b = self.get_model_version(model_name, version_b)
            
            if not metadata_a or not metadata_b:
                return {"error": "One or both model versions not found"}
            
            comparison = {
                "model_name": model_name,
                "version_a": version_a,
                "version_b": version_b,
                "comparison_time": datetime.utcnow().isoformat(),
                "metadata_comparison": {
                    "version_a": metadata_a,
                    "version_b": metadata_b
                }
            }
            
            # Add metrics comparison if available
            if comparison_metrics:
                comparison["metrics_comparison"] = {}
                for metric in comparison_metrics:
                    val_a = metadata_a.get("metrics", {}).get(metric)
                    val_b = metadata_b.get("metrics", {}).get(metric)
                    
                    if val_a is not None and val_b is not None:
                        comparison["metrics_comparison"][metric] = {
                            "version_a": val_a,
                            "version_b": val_b,
                            "difference": val_b - val_a,
                            "improvement_percentage": ((val_b - val_a) / val_a * 100) if val_a != 0 else None
                        }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare model versions: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check model registry health"""
        health = {
            "cloud_storage_available": False,
            "kubernetes_available": False,
            "models_bucket_accessible": False
        }
        
        try:
            # Check Cloud Storage
            if self.storage_client:
                bucket = self.storage_client.bucket(self.models_bucket)
                bucket.exists()
                health["cloud_storage_available"] = True
                health["models_bucket_accessible"] = True
            
            # Check Kubernetes
            if self.k8s_client:
                self.k8s_client.list_namespaced_config_map(
                    namespace=self.k8s_namespace,
                    limit=1
                )
                health["kubernetes_available"] = True
            
        except Exception as e:
            logger.error(f"Model registry health check failed: {e}")
        
        return health


# Global instance
_model_registry_instance = None

def get_model_registry() -> CloudModelRegistry:
    """Get or create model registry instance"""
    global _model_registry_instance
    if _model_registry_instance is None:
        _model_registry_instance = CloudModelRegistry()
    return _model_registry_instance
