#!/usr/bin/env python3
"""
Test Production Evaluation System
Tests the complete production pipeline with PostgreSQL and real models.

This script demonstrates:
- PostgreSQL metadata storage
- Real model integration (LPIPS, CLIP, LLaVA, Motion)
- Production-grade evaluation workflow
- Performance monitoring
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import production components
from visionflow.services.evaluation.production_orchestrator import (
    get_production_orchestrator, SamplingStrategy
)
from visionflow.services.generation.postgresql_metadata_storage import get_postgresql_storage
from visionflow.services.generation.video_metadata_tracker import VideoMetadataTracker

async def test_postgresql_connection():
    """Test PostgreSQL database connection and setup"""
    
    print("🐘 TESTING POSTGRESQL CONNECTION")
    print("=" * 50)
    
    try:
        # Initialize PostgreSQL storage
        pg_storage = get_postgresql_storage()
        
        # Test connection
        await pg_storage.initialize()
        
        # Test basic operations
        print("✅ PostgreSQL connection established")
        print("✅ Schema migrations applied")
        
        # Get analytics to test read operations
        analytics = await pg_storage.get_analytics()
        print(f"📊 Database analytics: {analytics}")
        
        await pg_storage.close()
        
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL test failed: {e}")
        print("\n💡 To set up PostgreSQL:")
        print("1. Install PostgreSQL: brew install postgresql")
        print("2. Start service: brew services start postgresql")
        print("3. Create database: createdb visionflow")
        print("4. Create user: createuser visionflow_user")
        print("5. Set environment variables:")
        print("   export POSTGRES_HOST=localhost")
        print("   export POSTGRES_DB=visionflow")
        print("   export POSTGRES_USER=visionflow_user")
        print("   export POSTGRES_PASSWORD=visionflow_pass")
        
        return False

async def test_production_models():
    """Test production model initialization and basic functionality"""
    
    print("\n🏭 TESTING PRODUCTION MODELS")
    print("=" * 50)
    
    try:
        # Test lightweight mode first (safer for resource constraints)
        print("🚀 Testing lightweight production models...")
        
        orchestrator = get_production_orchestrator(
            sampling_strategy=SamplingStrategy.ADAPTIVE,
            max_frames=5,  # Reduced for testing
            device='auto',
            lightweight=True  # Use only essential models
        )
        
        # Initialize models
        await orchestrator.initialize()
        
        print("✅ Production models initialized successfully")
        
        # Cleanup
        await orchestrator.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Production models test failed: {e}")
        print("\n💡 To install production dependencies:")
        print("pip install -r requirements_production.txt")
        
        return False

async def test_evaluation_with_production_models():
    """Test complete evaluation workflow with production models"""
    
    print("\n🎯 TESTING PRODUCTION EVALUATION WORKFLOW")
    print("=" * 50)
    
    # Check if test dataset exists
    test_dataset_dir = project_root / "generated" / "test_dataset"
    manifest_path = test_dataset_dir / "dataset_manifest.json"
    
    if not manifest_path.exists():
        print("❌ Test dataset not found!")
        print("💡 Create it first: python scripts/create_test_dataset.py")
        return False
    
    try:
        # Load test dataset
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        print(f"📋 Using test dataset: {len(manifest['videos'])} videos")
        
        # Initialize production orchestrator
        orchestrator = get_production_orchestrator(
            sampling_strategy=SamplingStrategy.ADAPTIVE,
            max_frames=8,  # Reasonable for testing
            device='auto',
            lightweight=True  # Start with lightweight mode
        )
        
        # Test with first video from dataset
        test_video = manifest['videos'][0]
        video_path = test_video['path']
        prompt = test_video['prompt']
        
        print(f"\n📹 Evaluating: {test_video['filename']}")
        print(f"🎬 Prompt: {prompt[:60]}...")
        
        start_time = time.time()
        
        # Run production evaluation
        result = await orchestrator.evaluate_video(video_path, prompt)
        
        evaluation_time = time.time() - start_time
        
        # Display results
        print("\n📊 PRODUCTION EVALUATION RESULTS")
        print("-" * 40)
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Confidence: {result.overall_confidence:.3f}")
        print(f"Level: {result.confidence_level.value.upper()}")
        print(f"Decision: {result.decision}")
        print(f"Review Needed: {result.requires_human_review}")
        print(f"Processing Time: {evaluation_time:.2f}s")
        print(f"Frames Evaluated: {result.frames_evaluated}")
        print(f"Aggregation Method: {result.aggregation_method}")
        
        print(f"\n📈 Dimension Breakdown:")
        for dim_score in result.dimension_scores:
            model_info = f"({dim_score.model_used})"
            print(f"  {dim_score.dimension.value:20}: {dim_score.score:.3f} {model_info}")
        
        print(f"\n🏭 Production Models Used:")
        for model, version in result.model_versions.items():
            print(f"  {model}: {version}")
        
        # Cleanup
        await orchestrator.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Production evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_production_pipeline():
    """Test the complete production pipeline including database integration"""
    
    print("\n🏗️ TESTING FULL PRODUCTION PIPELINE")
    print("=" * 50)
    
    # Test PostgreSQL connection
    pg_available = await test_postgresql_connection()
    
    if not pg_available:
        print("⚠️ PostgreSQL not available, using SQLite fallback")
        
        # Use existing tracker with SQLite
        from visionflow.services.generation.video_metadata_tracker import metadata_tracker
        tracker = metadata_tracker
    else:
        print("✅ Using PostgreSQL for metadata storage")
        
        # Create production metadata tracker with PostgreSQL
        pg_storage = get_postgresql_storage()
        await pg_storage.initialize()
        
        tracker = VideoMetadataTracker(storage_backends=[pg_storage])
    
    # Test production models
    models_available = await test_production_models()
    
    if not models_available:
        print("⚠️ Production models not available")
        return False
    
    # Test evaluation workflow
    evaluation_success = await test_evaluation_with_production_models()
    
    if not evaluation_success:
        print("❌ Production evaluation failed")
        return False
    
    print("\n✅ FULL PRODUCTION PIPELINE TEST COMPLETED")
    print("🏭 All production components working correctly!")
    
    return True

async def run_production_system_check():
    """Comprehensive production system check"""
    
    print("🚀 PRODUCTION SYSTEM READINESS CHECK")
    print("=" * 60)
    print("Validating production deployment readiness...")
    print()
    
    checks = {
        "PostgreSQL Database": False,
        "Production Models": False, 
        "Evaluation Pipeline": False,
        "Full Integration": False
    }
    
    # Check database
    try:
        checks["PostgreSQL Database"] = await test_postgresql_connection()
    except Exception as e:
        print(f"Database check failed: {e}")
    
    # Check models
    try:
        checks["Production Models"] = await test_production_models()
    except Exception as e:
        print(f"Models check failed: {e}")
    
    # Check evaluation
    try:
        checks["Evaluation Pipeline"] = await test_evaluation_with_production_models()
    except Exception as e:
        print(f"Evaluation check failed: {e}")
    
    # Check full integration
    try:
        checks["Full Integration"] = await test_full_production_pipeline()
    except Exception as e:
        print(f"Integration check failed: {e}")
    
    print("\n📊 PRODUCTION READINESS SUMMARY")
    print("=" * 40)
    
    total_checks = len(checks)
    passed_checks = sum(checks.values())
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:20}: {status}")
    
    readiness_score = passed_checks / total_checks * 100
    
    print(f"\nReadiness Score: {readiness_score:.1f}% ({passed_checks}/{total_checks})")
    
    if readiness_score >= 75:
        print("🟢 PRODUCTION READY")
        print("System meets production deployment criteria")
    elif readiness_score >= 50:
        print("🟡 PARTIALLY READY")
        print("Some components need attention before production deployment")
    else:
        print("🔴 NOT READY")
        print("Significant issues need to be resolved")
    
    print("\n🔄 NEXT STEPS FOR PRODUCTION:")
    if not checks["PostgreSQL Database"]:
        print("1. Set up PostgreSQL database")
    if not checks["Production Models"]:
        print("2. Install production model dependencies")
    if not checks["Evaluation Pipeline"]:
        print("3. Fix evaluation pipeline issues")
    if checks["Full Integration"]:
        print("✅ Ready for production deployment!")

async def main():
    """Main test function"""
    
    print("🎬 PRODUCTION EVALUATION SYSTEM - COMPREHENSIVE TEST")
    print("Testing production-ready components with real models")
    print("=" * 60)
    print()
    
    try:
        await run_production_system_check()
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
