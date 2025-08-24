"""
MLFlow Model Registry integration for VisionFlow
Manages model versions, staging, and deployment
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from .model_tracker import get_model_tracker
from ...shared.monitoring import get_logger

logger = get_logger("model_registry")


class ModelStage(Enum):
    """Model stages in MLFlow registry"""
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    NONE = "None"


class ModelRegistry:
    """Manages MLFlow Model Registry for video generation models"""
    
    def __init__(self):
        self.tracker = get_model_tracker()
        self.client = self.tracker.client
    
    def register_wan_model(
        self,
        model_path: str,
        model_name: str = "wan2-video-generation",
        description: str = "WAN 2.1 Fast video generation model",
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[ModelVersion]:
        """Register WAN model in MLFlow registry"""
        
        if not self.client:
            logger.warning("MLFlow not initialized")
            return None
        
        try:
            # Default tags for WAN models
            default_tags = {
                "model_type": "video_generation",
                "architecture": "wan2",
                "framework": "pytorch",
                "task": "text_to_video",
                "created_date": datetime.utcnow().isoformat()
            }
            
            if tags:
                default_tags.update(tags)
            
            # Register model
            model_version = self.tracker.register_model(
                model_uri=model_path,
                model_name=model_name,
                description=description,
                tags=default_tags
            )
            
            if model_version:
                logger.info(f"Registered WAN model: {model_name} version {model_version}")
                
                # Add model version metadata
                self.client.update_model_version(
                    name=model_name,
                    version=model_version,
                    description=f"{description}\n\nRegistered: {datetime.utcnow().isoformat()}"
                )
            
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register WAN model: {e}")
            return None
    
    def create_model_alias(
        self,
        model_name: str,
        version: str,
        alias: str
    ) -> bool:
        """Create alias for model version"""
        
        if not self.client:
            return False
        
        try:
            self.client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=version
            )
            
            logger.info(f"Created alias '{alias}' for {model_name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model alias: {e}")
            return False
    
    def deploy_to_staging(
        self,
        model_name: str,
        version: str,
        validation_results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Deploy model version to staging"""
        
        try:
            # Validate model before staging
            if validation_results:
                self._log_validation_results(model_name, version, validation_results)
            
            # Transition to staging
            success = self.tracker.transition_model_stage(
                model_name=model_name,
                version=version,
                stage=ModelStage.STAGING.value,
                archive_existing_versions=True
            )
            
            if success:
                # Update model version with staging metadata
                self.client.update_model_version(
                    name=model_name,
                    version=version,
                    description=f"Deployed to staging: {datetime.utcnow().isoformat()}"
                )
                
                # Create staging alias
                self.create_model_alias(model_name, version, "staging")
                
                logger.info(f"Deployed {model_name} v{version} to staging")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to deploy to staging: {e}")
            return False
    
    def deploy_to_production(
        self,
        model_name: str,
        version: str,
        approval_metadata: Optional[Dict[str, Any]] = None,
        rollback_version: Optional[str] = None
    ) -> bool:
        """Deploy model version to production with safety checks"""
        
        try:
            # Check if model is in staging
            model_version = self.tracker.get_model_version(model_name, version)
            if not model_version or model_version.current_stage != ModelStage.STAGING.value:
                logger.error(f"Model {model_name} v{version} must be in staging before production")
                return False
            
            # Store current production version for rollback
            current_prod_versions = self.client.get_latest_versions(
                model_name, 
                stages=[ModelStage.PRODUCTION.value]
            )
            
            if current_prod_versions and rollback_version is None:
                rollback_version = current_prod_versions[0].version
            
            # Transition to production
            success = self.tracker.transition_model_stage(
                model_name=model_name,
                version=version,
                stage=ModelStage.PRODUCTION.value,
                archive_existing_versions=True
            )
            
            if success:
                # Update model version with production metadata
                prod_metadata = {
                    "deployed_to_production": datetime.utcnow().isoformat(),
                    "previous_production_version": rollback_version
                }
                
                if approval_metadata:
                    prod_metadata.update(approval_metadata)
                
                self.client.update_model_version(
                    name=model_name,
                    version=version,
                    description=f"Production deployment: {json.dumps(prod_metadata, default=str)}"
                )
                
                # Create production alias
                self.create_model_alias(model_name, version, "production")
                
                # Log deployment event
                self._log_deployment_event(model_name, version, "production", prod_metadata)
                
                logger.info(f"Deployed {model_name} v{version} to production")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to deploy to production: {e}")
            return False
    
    def rollback_production(
        self,
        model_name: str,
        target_version: Optional[str] = None
    ) -> bool:
        """Rollback production to previous or specified version"""
        
        try:
            if target_version is None:
                # Find previous production version
                current_prod = self.get_production_model(model_name)
                if not current_prod:
                    logger.error("No current production model found")
                    return False
                
                # Extract previous version from metadata
                try:
                    description = current_prod.description or "{}"
                    metadata = json.loads(description)
                    target_version = metadata.get("previous_production_version")
                except:
                    # Fallback: get second latest version
                    all_versions = self.list_model_versions(model_name)
                    if len(all_versions) >= 2:
                        target_version = all_versions[1].version
                
                if not target_version:
                    logger.error("No previous version found for rollback")
                    return False
            
            # Validate target version exists
            target_model = self.tracker.get_model_version(model_name, target_version)
            if not target_model:
                logger.error(f"Target version {target_version} not found")
                return False
            
            # Transition target version to production
            success = self.tracker.transition_model_stage(
                model_name=model_name,
                version=target_version,
                stage=ModelStage.PRODUCTION.value,
                archive_existing_versions=True
            )
            
            if success:
                # Update metadata
                rollback_metadata = {
                    "rollback_performed": datetime.utcnow().isoformat(),
                    "rollback_reason": "Manual rollback"
                }
                
                self.client.update_model_version(
                    name=model_name,
                    version=target_version,
                    description=f"Rollback deployment: {json.dumps(rollback_metadata, default=str)}"
                )
                
                # Update production alias
                self.create_model_alias(model_name, target_version, "production")
                
                # Log rollback event
                self._log_deployment_event(model_name, target_version, "rollback", rollback_metadata)
                
                logger.info(f"Rolled back {model_name} to version {target_version}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rollback production: {e}")
            return False
    
    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get current production model version"""
        return self.tracker.get_model_version(model_name, stage=ModelStage.PRODUCTION.value)
    
    def get_staging_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get current staging model version"""
        return self.tracker.get_model_version(model_name, stage=ModelStage.STAGING.value)
    
    def list_model_versions(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> List[ModelVersion]:
        """List all versions of a model"""
        
        if not self.client:
            return []
        
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                model_details = self.client.get_registered_model(model_name)
                versions = model_details.latest_versions
            
            # Sort by version number (descending)
            versions.sort(key=lambda v: int(v.version), reverse=True)
            return versions
            
        except Exception as e:
            logger.error(f"Failed to list model versions: {e}")
            return []
    
    def compare_model_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
        comparison_metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        
        try:
            comparison = {
                "version_a": version_a,
                "version_b": version_b,
                "metrics_comparison": {},
                "metadata_comparison": {}
            }
            
            # Get model versions
            model_a = self.tracker.get_model_version(model_name, version_a)
            model_b = self.tracker.get_model_version(model_name, version_b)
            
            if not model_a or not model_b:
                comparison["error"] = "One or both model versions not found"
                return comparison
            
            # Compare metadata
            comparison["metadata_comparison"] = {
                "version_a_stage": model_a.current_stage,
                "version_b_stage": model_b.current_stage,
                "version_a_creation_time": model_a.creation_timestamp,
                "version_b_creation_time": model_b.creation_timestamp
            }
            
            # Get runs associated with these versions
            run_a = self.client.get_run(model_a.run_id) if model_a.run_id else None
            run_b = self.client.get_run(model_b.run_id) if model_b.run_id else None
            
            # Compare metrics
            if run_a and run_b:
                for metric in comparison_metrics:
                    metric_a = run_a.data.metrics.get(metric)
                    metric_b = run_b.data.metrics.get(metric)
                    
                    comparison["metrics_comparison"][metric] = {
                        "version_a": metric_a,
                        "version_b": metric_b,
                        "difference": metric_a - metric_b if metric_a and metric_b else None,
                        "better_version": "a" if metric_a and metric_b and metric_a > metric_b else "b" if metric_a and metric_b else None
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare model versions: {e}")
            return {"error": str(e)}
    
    def archive_old_versions(
        self,
        model_name: str,
        keep_latest: int = 5,
        exclude_stages: List[str] = None
    ) -> int:
        """Archive old model versions to save space"""
        
        if exclude_stages is None:
            exclude_stages = [ModelStage.PRODUCTION.value, ModelStage.STAGING.value]
        
        try:
            versions = self.list_model_versions(model_name)
            archived_count = 0
            
            # Skip the latest N versions and versions in excluded stages
            for version in versions[keep_latest:]:
                if version.current_stage not in exclude_stages:
                    success = self.tracker.transition_model_stage(
                        model_name=model_name,
                        version=version.version,
                        stage=ModelStage.ARCHIVED.value,
                        archive_existing_versions=False
                    )
                    
                    if success:
                        archived_count += 1
            
            logger.info(f"Archived {archived_count} old versions of {model_name}")
            return archived_count
            
        except Exception as e:
            logger.error(f"Failed to archive old versions: {e}")
            return 0
    
    def _log_validation_results(
        self,
        model_name: str,
        version: str,
        validation_results: Dict[str, Any]
    ):
        """Log model validation results"""
        
        try:
            # Start a validation run
            run_id = self.tracker.start_run(
                run_name=f"validation_{model_name}_v{version}",
                tags={
                    "validation": "true",
                    "model_name": model_name,
                    "model_version": version
                }
            )
            
            if run_id:
                # Log validation metrics
                metrics = validation_results.get("metrics", {})
                self.tracker.log_metrics(metrics)
                
                # Log validation parameters
                params = validation_results.get("parameters", {})
                self.tracker.log_parameters(params)
                
                self.tracker.end_run()
                
        except Exception as e:
            logger.error(f"Failed to log validation results: {e}")
    
    def _log_deployment_event(
        self,
        model_name: str,
        version: str,
        event_type: str,
        metadata: Dict[str, Any]
    ):
        """Log deployment event for audit trail"""
        
        try:
            run_id = self.tracker.start_run(
                run_name=f"deployment_{event_type}_{model_name}_v{version}",
                tags={
                    "deployment_event": event_type,
                    "model_name": model_name,
                    "model_version": version
                }
            )
            
            if run_id:
                self.tracker.log_parameters(metadata)
                self.tracker.end_run()
                
        except Exception as e:
            logger.error(f"Failed to log deployment event: {e}")
    
    def get_model_lineage(self, model_name: str) -> Dict[str, Any]:
        """Get model lineage and deployment history"""
        
        try:
            lineage = {
                "model_name": model_name,
                "versions": [],
                "deployment_history": []
            }
            
            # Get all versions
            versions = self.list_model_versions(model_name)
            
            for version in versions:
                version_info = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "creation_time": version.creation_timestamp,
                    "description": version.description
                }
                
                # Get associated run info
                if version.run_id:
                    try:
                        run = self.client.get_run(version.run_id)
                        version_info["run_metrics"] = run.data.metrics
                        version_info["run_params"] = run.data.params
                    except:
                        pass
                
                lineage["versions"].append(version_info)
            
            # Get deployment events
            deployment_runs = self.tracker.search_runs(
                filter_string=f"tags.model_name = '{model_name}' and tags.deployment_event != ''",
                max_results=100
            )
            
            for run in deployment_runs:
                event = {
                    "event_type": run.data.tags.get("deployment_event"),
                    "model_version": run.data.tags.get("model_version"),
                    "timestamp": run.info.start_time,
                    "parameters": run.data.params
                }
                lineage["deployment_history"].append(event)
            
            # Sort deployment history by timestamp
            lineage["deployment_history"].sort(key=lambda x: x["timestamp"], reverse=True)
            
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return {"error": str(e)}


# Global instance
_model_registry_instance = None

def get_model_registry() -> ModelRegistry:
    """Get or create model registry instance"""
    global _model_registry_instance
    if _model_registry_instance is None:
        _model_registry_instance = ModelRegistry()
    return _model_registry_instance
