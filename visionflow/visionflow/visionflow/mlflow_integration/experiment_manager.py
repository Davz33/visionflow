"""
MLFlow experiment management for VisionFlow
Manages experiments, hyperparameter tuning, and A/B testing
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from .model_tracker import get_model_tracker
from ...shared.monitoring import get_logger

logger = get_logger("experiment_manager")


@dataclass
class ExperimentConfig:
    """Configuration for video generation experiments"""
    name: str
    description: str
    parameters: Dict[str, Any]
    metrics: List[str]
    tags: Optional[Dict[str, str]] = None


class ExperimentManager:
    """Manages MLFlow experiments for video generation model evaluation"""
    
    def __init__(self):
        self.tracker = get_model_tracker()
        self.client = self.tracker.client
    
    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Create a new experiment"""
        if not self.client:
            logger.warning("MLFlow not initialized")
            return None
        
        try:
            # Check if experiment already exists
            existing_experiment = mlflow.get_experiment_by_name(name)
            if existing_experiment:
                logger.warning(f"Experiment {name} already exists")
                return existing_experiment.experiment_id
            
            # Set artifact location if not provided
            if not artifact_location:
                artifact_location = f"gs://visionflow-mlflow-artifacts/{name}"
            
            experiment_id = mlflow.create_experiment(
                name=name,
                artifact_location=artifact_location,
                tags=tags
            )
            
            # Update description if provided
            if description:
                self.client.update_experiment(experiment_id, description=description)
            
            logger.info(f"Created experiment: {name} (ID: {experiment_id})")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return None
    
    def run_hyperparameter_sweep(
        self,
        experiment_name: str,
        parameter_grid: Dict[str, List[Any]],
        objective_metric: str,
        generation_function: callable,
        test_prompts: List[str],
        max_runs: int = 50
    ) -> List[Dict[str, Any]]:
        """Run hyperparameter sweep for video generation models"""
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        results = []
        run_count = 0
        
        try:
            # Generate parameter combinations
            import itertools
            
            keys = list(parameter_grid.keys())
            values = list(parameter_grid.values())
            
            for combination in itertools.product(*values):
                if run_count >= max_runs:
                    break
                
                params = dict(zip(keys, combination))
                
                # Start run for this parameter combination
                run_name = f"hyperparam_sweep_{run_count:03d}"
                run_id = self.tracker.start_run(
                    run_name=run_name,
                    tags={
                        "sweep": "hyperparameter",
                        "objective_metric": objective_metric
                    }
                )
                
                if not run_id:
                    continue
                
                try:
                    # Log parameters
                    self.tracker.log_parameters(params)
                    
                    # Run generation with these parameters
                    metrics = {}
                    generated_videos = []
                    
                    for i, prompt in enumerate(test_prompts):
                        try:
                            result = generation_function(prompt, **params)
                            
                            if result and 'metrics' in result:
                                # Aggregate metrics across prompts
                                for metric_name, value in result['metrics'].items():
                                    if metric_name not in metrics:
                                        metrics[metric_name] = []
                                    metrics[metric_name].append(value)
                            
                            if result and 'video_path' in result:
                                generated_videos.append(result['video_path'])
                                
                        except Exception as e:
                            logger.error(f"Generation failed for prompt {i}: {e}")
                    
                    # Calculate average metrics
                    avg_metrics = {}
                    for metric_name, values in metrics.items():
                        if values:
                            avg_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
                            avg_metrics[f"std_{metric_name}"] = (
                                sum((x - avg_metrics[f"avg_{metric_name}"]) ** 2 for x in values) / len(values)
                            ) ** 0.5
                    
                    # Log metrics
                    self.tracker.log_metrics(avg_metrics)
                    
                    # Log sample videos
                    for i, video_path in enumerate(generated_videos[:3]):  # Log first 3 videos
                        self.tracker.log_artifact(video_path, f"sample_videos/video_{i}")
                    
                    # Store result
                    result_data = {
                        "run_id": run_id,
                        "parameters": params,
                        "metrics": avg_metrics,
                        "objective_value": avg_metrics.get(f"avg_{objective_metric}", 0)
                    }
                    results.append(result_data)
                    
                    logger.info(f"Completed sweep run {run_count}: {objective_metric}="
                               f"{result_data['objective_value']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Hyperparameter sweep run {run_count} failed: {e}")
                
                finally:
                    self.tracker.end_run()
                    run_count += 1
            
            # Sort results by objective metric
            results.sort(key=lambda x: x['objective_value'], reverse=True)
            
            logger.info(f"Completed hyperparameter sweep with {len(results)} runs")
            return results
            
        except Exception as e:
            logger.error(f"Hyperparameter sweep failed: {e}")
            return []
    
    def run_ab_test(
        self,
        experiment_name: str,
        model_a_config: Dict[str, Any],
        model_b_config: Dict[str, Any],
        test_prompts: List[str],
        evaluation_metrics: List[str],
        generation_function: callable,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """Run A/B test between two model configurations"""
        
        mlflow.set_experiment(experiment_name)
        
        results = {
            "model_a": {"metrics": {}, "runs": []},
            "model_b": {"metrics": {}, "runs": []},
            "statistical_test": {}
        }
        
        try:
            # Test Model A
            logger.info("Testing Model A...")
            for i in range(sample_size):
                run_name = f"ab_test_model_a_{i:03d}"
                run_id = self.tracker.start_run(
                    run_name=run_name,
                    tags={
                        "ab_test": "model_a",
                        "test_group": "A"
                    }
                )
                
                if run_id:
                    try:
                        self.tracker.log_parameters(model_a_config)
                        
                        # Random prompt selection
                        import random
                        prompt = random.choice(test_prompts)
                        
                        result = generation_function(prompt, **model_a_config)
                        
                        if result and 'metrics' in result:
                            self.tracker.log_metrics(result['metrics'])
                            results["model_a"]["runs"].append(result['metrics'])
                        
                    except Exception as e:
                        logger.error(f"Model A test run {i} failed: {e}")
                    finally:
                        self.tracker.end_run()
            
            # Test Model B
            logger.info("Testing Model B...")
            for i in range(sample_size):
                run_name = f"ab_test_model_b_{i:03d}"
                run_id = self.tracker.start_run(
                    run_name=run_name,
                    tags={
                        "ab_test": "model_b",
                        "test_group": "B"
                    }
                )
                
                if run_id:
                    try:
                        self.tracker.log_parameters(model_b_config)
                        
                        # Random prompt selection
                        import random
                        prompt = random.choice(test_prompts)
                        
                        result = generation_function(prompt, **model_b_config)
                        
                        if result and 'metrics' in result:
                            self.tracker.log_metrics(result['metrics'])
                            results["model_b"]["runs"].append(result['metrics'])
                        
                    except Exception as e:
                        logger.error(f"Model B test run {i} failed: {e}")
                    finally:
                        self.tracker.end_run()
            
            # Calculate aggregate metrics
            for model_key in ["model_a", "model_b"]:
                runs = results[model_key]["runs"]
                if runs:
                    for metric in evaluation_metrics:
                        values = [run.get(metric, 0) for run in runs if metric in run]
                        if values:
                            results[model_key]["metrics"][metric] = {
                                "mean": sum(values) / len(values),
                                "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5,
                                "count": len(values)
                            }
            
            # Perform statistical significance test
            results["statistical_test"] = self._perform_statistical_test(
                results["model_a"]["runs"],
                results["model_b"]["runs"],
                evaluation_metrics
            )
            
            # Log A/B test summary
            summary_run_id = self.tracker.start_run(
                run_name="ab_test_summary",
                tags={"ab_test": "summary"}
            )
            
            if summary_run_id:
                # Log summary metrics
                summary_metrics = {}
                for model_key in ["model_a", "model_b"]:
                    for metric, stats in results[model_key]["metrics"].items():
                        summary_metrics[f"{model_key}_{metric}_mean"] = stats["mean"]
                        summary_metrics[f"{model_key}_{metric}_std"] = stats["std"]
                
                self.tracker.log_metrics(summary_metrics)
                
                # Log statistical test results
                self.tracker.log_parameters({
                    "statistical_test_results": json.dumps(results["statistical_test"], default=str)
                })
                
                self.tracker.end_run()
            
            logger.info("A/B test completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"A/B test failed: {e}")
            return results
    
    def _perform_statistical_test(
        self,
        group_a_runs: List[Dict[str, Any]],
        group_b_runs: List[Dict[str, Any]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform statistical significance test between two groups"""
        
        try:
            from scipy import stats
            
            test_results = {}
            
            for metric in metrics:
                # Extract metric values
                values_a = [run.get(metric, 0) for run in group_a_runs if metric in run]
                values_b = [run.get(metric, 0) for run in group_b_runs if metric in run]
                
                if len(values_a) < 3 or len(values_b) < 3:
                    test_results[metric] = {"error": "Insufficient data for statistical test"}
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(values_a, values_b)
                
                # Calculate effect size (Cohen's d)
                pooled_std = ((len(values_a) - 1) * (sum((x - sum(values_a) / len(values_a)) ** 2 for x in values_a) / (len(values_a) - 1)) +
                             (len(values_b) - 1) * (sum((x - sum(values_b) / len(values_b)) ** 2 for x in values_b) / (len(values_b) - 1))) / (len(values_a) + len(values_b) - 2)
                
                if pooled_std > 0:
                    cohens_d = (sum(values_a) / len(values_a) - sum(values_b) / len(values_b)) / (pooled_std ** 0.5)
                else:
                    cohens_d = 0
                
                test_results[metric] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "significant": p_value < 0.05,
                    "sample_size_a": len(values_a),
                    "sample_size_b": len(values_b)
                }
            
            return test_results
            
        except ImportError:
            logger.warning("scipy not available for statistical tests")
            return {"error": "scipy not available"}
        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return {"error": str(e)}
    
    def get_best_run(
        self,
        experiment_name: str,
        metric_name: str,
        maximize: bool = True
    ) -> Optional[Any]:
        """Get best run from experiment based on metric"""
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                logger.error(f"Experiment {experiment_name} not found")
                return None
            
            # Search runs
            runs = self.tracker.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"metrics.{metric_name} IS NOT NULL"
            )
            
            if not runs:
                logger.warning(f"No runs found with metric {metric_name}")
                return None
            
            # Sort by metric
            runs.sort(
                key=lambda run: run.data.metrics.get(metric_name, float('-inf') if maximize else float('inf')),
                reverse=maximize
            )
            
            best_run = runs[0]
            logger.info(f"Best run: {best_run.info.run_id} with {metric_name}="
                       f"{best_run.data.metrics.get(metric_name)}")
            
            return best_run
            
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None
    
    def compare_experiments(
        self,
        experiment_names: List[str],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple experiments"""
        
        comparison = {}
        
        try:
            for exp_name in experiment_names:
                experiment = mlflow.get_experiment_by_name(exp_name)
                if not experiment:
                    comparison[exp_name] = {"error": "Experiment not found"}
                    continue
                
                runs = self.tracker.search_runs(
                    experiment_ids=[experiment.experiment_id]
                )
                
                exp_metrics = {}
                for metric in metrics:
                    values = [run.data.metrics.get(metric) for run in runs 
                             if run.data.metrics.get(metric) is not None]
                    
                    if values:
                        exp_metrics[metric] = {
                            "mean": sum(values) / len(values),
                            "std": (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)) ** 0.5,
                            "min": min(values),
                            "max": max(values),
                            "count": len(values)
                        }
                    else:
                        exp_metrics[metric] = {"error": "No data"}
                
                comparison[exp_name] = {
                    "metrics": exp_metrics,
                    "total_runs": len(runs)
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare experiments: {e}")
            return {}


# Global instance
_experiment_manager_instance = None

def get_experiment_manager() -> ExperimentManager:
    """Get or create experiment manager instance"""
    global _experiment_manager_instance
    if _experiment_manager_instance is None:
        _experiment_manager_instance = ExperimentManager()
    return _experiment_manager_instance
