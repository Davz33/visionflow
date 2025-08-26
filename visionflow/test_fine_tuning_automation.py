#!/usr/bin/env python3
"""
Test script for automated fine-tuning trigger system.
Tests both manual invocation and Celery task execution.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from visionflow.services.evaluation.confidence_manager import ConfidenceManager
from visionflow.tasks import check_fine_tuning_triggers_task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_confidence_manager_direct():
    """Test the confidence manager directly (not through Celery)"""
    logger.info("🧪 Testing ConfidenceManager directly...")
    
    try:
        confidence_manager = ConfidenceManager()
        triggers = await confidence_manager.check_fine_tuning_triggers()
        
        logger.info("✅ Direct test completed successfully")
        logger.info(f"   Low confidence trigger: {triggers['low_confidence_trigger']}")
        logger.info(f"   High automation trigger: {triggers['high_automation_trigger']}")
        logger.info(f"   Recommendations: {len(triggers['recommendations'])}")
        
        return triggers
        
    except Exception as e:
        logger.error(f"❌ Direct test failed: {e}")
        raise


def test_celery_task():
    """Test the Celery task (synchronous version)"""
    logger.info("🧪 Testing Celery task execution...")
    
    try:
        # This tests the task function directly (not through the broker)
        result = check_fine_tuning_triggers_task()
        
        logger.info("✅ Celery task test completed successfully")
        if isinstance(result, dict):
            if 'error' in result:
                logger.warning(f"⚠️ Task returned error: {result['error']}")
            else:
                logger.info(f"   Low confidence trigger: {result.get('low_confidence_trigger', 'N/A')}")
                logger.info(f"   High automation trigger: {result.get('high_automation_trigger', 'N/A')}")
                logger.info(f"   Alert sent: {result.get('alert_sent', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Celery task test failed: {e}")
        raise


async def test_system_integration():
    """Test the complete system integration"""
    logger.info("🚀 Starting complete fine-tuning automation test...")
    logger.info("=" * 60)
    
    # Test 1: Direct confidence manager
    logger.info("TEST 1: Direct ConfidenceManager")
    logger.info("-" * 40)
    direct_result = await test_confidence_manager_direct()
    
    print()
    
    # Test 2: Celery task
    logger.info("TEST 2: Celery Task Function")
    logger.info("-" * 40)
    celery_result = test_celery_task()
    
    print()
    
    # Summary
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    if direct_result and celery_result:
        logger.info("✅ All tests passed! Fine-tuning automation is working correctly.")
        
        # Show metrics if available
        if 'metrics' in direct_result:
            metrics = direct_result['metrics']
            logger.info("📈 Current System Metrics:")
            logger.info(f"   Total Evaluations: {metrics.total_evaluations}")
            logger.info(f"   Auto Approved: {metrics.auto_approved}")
            logger.info(f"   Low Confidence Rate: {metrics.low_confidence_rate:.1%}")
            logger.info(f"   High Automation Rate: {metrics.high_automation_rate:.1%}")
            logger.info(f"   Average Confidence: {metrics.avg_confidence:.3f}")
        
        # Show recommendations if any
        recommendations = direct_result.get('recommendations', [])
        if recommendations:
            logger.info("🔧 Active Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec.get('type', 'Unknown')} ({rec.get('priority', 'medium')} priority)")
                logger.info(f"      Reason: {rec.get('reason', 'No reason provided')}")
        else:
            logger.info("✅ No fine-tuning recommendations at this time")
            
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        
    logger.info("=" * 60)


def print_integration_instructions():
    """Print instructions for integrating with production"""
    logger.info("🔧 PRODUCTION INTEGRATION INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("1. Start the system with Docker Compose:")
    logger.info("   cd visionflow && docker-compose up -d")
    logger.info("")
    logger.info("2. The celery-beat service will automatically start and run:")
    logger.info("   - Fine-tuning triggers check: Every hour")
    logger.info("   - System health check: Daily at midnight")
    logger.info("")
    logger.info("3. Monitor logs for fine-tuning alerts:")
    logger.info("   docker logs -f visionflow-beat")
    logger.info("   docker logs -f visionflow-worker")
    logger.info("")
    logger.info("4. Look for these log patterns:")
    logger.info("   - 'FINE_TUNING_ALERT:' for triggered alerts")
    logger.info("   - '🚨 URGENT:' for critical confidence issues")
    logger.info("   - '⚠️ High automation rate' for quality gate warnings")
    logger.info("")
    logger.info("5. Customize alerting in visionflow/tasks.py:")
    logger.info("   - Add Slack/email integrations to send_fine_tuning_alerts()")
    logger.info("   - Adjust thresholds in confidence_manager.py")
    logger.info("=" * 60)


async def main():
    """Main test function"""
    try:
        # Run the integration test
        await test_system_integration()
        
        print()
        
        # Print integration instructions
        print_integration_instructions()
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
