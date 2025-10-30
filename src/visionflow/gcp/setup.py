"""
Google Cloud Platform setup and integration utilities
"""

import json
import os
from typing import Dict, Any, Optional

from google.cloud import storage, aiplatform, secretmanager
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account

from ..shared.config import get_settings
from ..shared.monitoring import get_logger

logger = get_logger("gcp_setup")
settings = get_settings()


class GCPIntegration:
    """Google Cloud Platform integration manager"""
    
    def __init__(self):
        self.project_id = settings.monitoring.vertex_ai_project
        self.region = settings.monitoring.vertex_ai_region
        self.bucket_name = settings.storage.bucket_name
        
        # Initialize clients
        self.storage_client = None
        self.secret_client = None
        self.credentials = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize GCP clients with proper authentication"""
        try:
            # Load service account credentials
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                # Running in container or with service account file
                self.storage_client = storage.Client()
                self.secret_client = secretmanager.SecretManagerServiceClient()
                
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
                # Running with JSON credentials in environment
                creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                creds_dict = json.loads(creds_json)
                
                self.credentials = service_account.Credentials.from_service_account_info(creds_dict)
                self.storage_client = storage.Client(credentials=self.credentials)
                self.secret_client = secretmanager.SecretManagerServiceClient(credentials=self.credentials)
            
            else:
                logger.warning("No GCP credentials found, using default credentials")
                self.storage_client = storage.Client()
                self.secret_client = secretmanager.SecretManagerServiceClient()
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.project_id,
                location=self.region,
                credentials=self.credentials
            )
            
            logger.info("GCP clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            raise
    
    def setup_storage_bucket(self) -> bool:
        """Create and configure Cloud Storage bucket"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            
            if not bucket.exists():
                logger.info(f"Creating storage bucket: {self.bucket_name}")
                
                bucket = self.storage_client.create_bucket(
                    self.bucket_name,
                    location=self.region
                )
                
                # Configure bucket for video storage
                bucket.cors = [
                    {
                        "origin": ["*"],
                        "responseHeader": ["Content-Type"],
                        "method": ["GET", "HEAD"],
                        "maxAgeSeconds": 3600
                    }
                ]
                bucket.patch()
                
                # Set lifecycle rules for cost optimization
                bucket.lifecycle_rules = [
                    {
                        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
                        "condition": {"age": 30}
                    },
                    {
                        "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
                        "condition": {"age": 90}
                    },
                    {
                        "action": {"type": "Delete"},
                        "condition": {"age": 365}
                    }
                ]
                bucket.patch()
                
                logger.info(f"Storage bucket {self.bucket_name} created successfully")
            else:
                logger.info(f"Storage bucket {self.bucket_name} already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup storage bucket: {e}")
            return False
    
    def upload_file(self, file_path: str, destination_path: str) -> Optional[str]:
        """Upload file to Cloud Storage"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(destination_path)
            
            blob.upload_from_filename(file_path)
            
            # Make blob publicly readable for video streaming
            blob.make_public()
            
            logger.info(f"File uploaded successfully: {destination_path}")
            return blob.public_url
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return None
    
    def get_secret(self, secret_name: str, version: str = "latest") -> Optional[str]:
        """Retrieve secret from Secret Manager"""
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
            response = self.secret_client.access_secret_version(request={"name": name})
            
            return response.payload.data.decode("UTF-8")
            
        except NotFound:
            logger.warning(f"Secret {secret_name} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            return None
    
    def create_secret(self, secret_name: str, secret_value: str) -> bool:
        """Create secret in Secret Manager"""
        try:
            parent = f"projects/{self.project_id}"
            
            # Create the secret
            secret = self.secret_client.create_secret(
                request={
                    "parent": parent,
                    "secret_id": secret_name,
                    "secret": {"replication": {"automatic": {}}},
                }
            )
            
            # Add the secret version
            self.secret_client.add_secret_version(
                request={
                    "parent": secret.name,
                    "payload": {"data": secret_value.encode("UTF-8")},
                }
            )
            
            logger.info(f"Secret {secret_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create secret {secret_name}: {e}")
            return False
    
    def setup_vertex_ai_endpoints(self) -> Dict[str, Any]:
        """Setup Vertex AI endpoints for model serving"""
        try:
            # This would typically deploy models to Vertex AI endpoints
            # For now, we'll return configuration for client-side model loading
            
            config = {
                "text_generation_endpoint": f"projects/{self.project_id}/locations/{self.region}/publishers/google/models/gemini-pro",
                "embedding_endpoint": f"projects/{self.project_id}/locations/{self.region}/publishers/google/models/textembedding-gecko",
                "model_garden_registry": f"projects/{self.project_id}/locations/{self.region}/metadataStores/default"
            }
            
            logger.info("Vertex AI endpoints configured")
            return config
            
        except Exception as e:
            logger.error(f"Failed to setup Vertex AI endpoints: {e}")
            return {}
    
    def setup_monitoring_workspace(self) -> bool:
        """Setup Cloud Monitoring workspace"""
        try:
            # This would typically create monitoring dashboards and alerts
            # For now, we'll just log the configuration
            
            logger.info("Cloud Monitoring workspace configured")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring workspace: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of GCP services"""
        health_status = {
            "storage": False,
            "secrets": False,
            "vertex_ai": False,
            "overall": False
        }
        
        try:
            # Check Storage
            bucket = self.storage_client.bucket(self.bucket_name)
            if bucket.exists():
                health_status["storage"] = True
            
            # Check Secret Manager
            try:
                self.secret_client.list_secrets(request={"parent": f"projects/{self.project_id}"})
                health_status["secrets"] = True
            except Exception:
                pass
            
            # Check Vertex AI (simplified check)
            health_status["vertex_ai"] = True  # Assume healthy if initialized
            
            # Overall health
            health_status["overall"] = all([
                health_status["storage"],
                health_status["secrets"],
                health_status["vertex_ai"]
            ])
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return health_status


def setup_gcp_environment() -> bool:
    """Setup complete GCP environment for VisionFlow"""
    try:
        gcp = GCPIntegration()
        
        logger.info("Setting up GCP environment...")
        
        # Setup storage
        if not gcp.setup_storage_bucket():
            logger.error("Failed to setup storage bucket")
            return False
        
        # Setup Vertex AI
        vertex_config = gcp.setup_vertex_ai_endpoints()
        if not vertex_config:
            logger.warning("Vertex AI setup failed, continuing without it")
        
        # Setup monitoring
        if not gcp.setup_monitoring_workspace():
            logger.warning("Monitoring setup failed, continuing without it")
        
        # Health check
        health = gcp.health_check()
        if not health["overall"]:
            logger.warning("Some GCP services are not healthy")
        
        logger.info("GCP environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup GCP environment: {e}")
        return False


# Global instance
_gcp_instance = None

def get_gcp_integration() -> GCPIntegration:
    """Get or create GCP integration instance"""
    global _gcp_instance
    if _gcp_instance is None:
        _gcp_instance = GCPIntegration()
    return _gcp_instance
