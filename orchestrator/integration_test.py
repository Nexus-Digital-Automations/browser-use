#!/usr/bin/env python3
"""
Browser Orchestration Integration Validation Test

Validates end-to-end browser orchestration workflow with real browser-use agents.
Tests the complete integration from task creation to execution.

Author: Claude Code - Integration Validation
Version: 1.0.0
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import structlog

# Add orchestrator modules to path
sys.path.insert(0, str(Path(__file__).parent))

from coordinator import BrowserAgentCoordinator, OrchestrationRequest, TaskType, TaskPriority


def setup_logging():
    """Setup structured logging for test"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


async def test_browser_orchestration_integration():
    """
    Test complete browser orchestration integration workflow
    
    Tests:
    1. Coordinator initialization
    2. Agent pool creation with real browsers
    3. Session management lifecycle
    4. Task queue processing
    5. End-to-end task execution with real browser-use agents
    """
    logger = structlog.get_logger("integration_test")
    
    logger.info("Starting Browser Orchestration Integration Test")
    
    # Create temporary storage directory
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        
        # Test configuration for local-only deployment
        test_config = {
            "agent_pool": {
                "min_agents": 1,
                "max_agents": 2, 
                "headless": True,  # Use headless for testing
                "viewport_width": 1280,
                "viewport_height": 720,
                "max_tasks_per_agent": 1
            },
            "task_queue": {
                "max_queue_size": 10,
                "max_priority_levels": 5
            },
            "session_manager": {
                "max_sessions_per_agent": 2,
                "max_total_sessions": 5,
                "session_idle_timeout": 300,  # 5 minutes for testing
                "session_max_lifetime": 1800  # 30 minutes for testing
            }
        }
        
        try:
            # Test 1: Initialize Coordinator
            logger.info("Test 1: Initializing Browser Agent Coordinator")
            coordinator = BrowserAgentCoordinator(test_config, storage_path)
            
            init_result = await coordinator.initialize()
            if not init_result["success"]:
                raise Exception(f"Coordinator initialization failed: {init_result}")
                
            logger.info("✅ Coordinator initialized successfully", 
                       agents_created=init_result.get("agents_created", 0))
            
            # Test 2: Create and submit test task
            logger.info("Test 2: Creating browser automation test task")
            
            test_request = OrchestrationRequest(
                request_id="test_integration_001",
                task_type=TaskType.NAVIGATION,
                priority=TaskPriority.HIGH,
                action="Navigate to test page and extract title",
                target_url="https://httpbin.org/html",
                parameters={
                    "extract_data": True,
                    "take_screenshot": True,
                    "max_steps": 5
                },
                requirements={
                    "browser_type": "chrome",
                    "headless": True,
                    "javascript": True
                },
                timeout_ms=60000,  # 1 minute timeout
                retry_count=1,
                metadata={
                    "test_case": "integration_validation",
                    "expected_outcome": "page_title_extracted"
                }
            )
            
            # Test 3: Execute orchestration workflow
            logger.info("Test 3: Executing orchestration workflow")
            
            execution_result = await coordinator.execute_workflow(test_request)
            
            if not execution_result["success"]:
                logger.error("❌ Workflow execution failed", 
                           error=execution_result.get("error", "Unknown error"))
                return False
                
            logger.info("✅ Workflow executed successfully",
                       workflow_id=execution_result.get("workflow_id"),
                       execution_time_ms=execution_result.get("execution_time_ms", 0))
            
            # Test 4: Validate results
            logger.info("Test 4: Validating execution results")
            
            result_data = execution_result.get("result_data", {})
            
            validation_checks = {
                "task_completed": result_data.get("success", False),
                "url_accessed": result_data.get("url") == test_request.target_url,
                "agent_execution": result_data.get("performance", {}).get("agent_execution_successful", False),
                "steps_executed": result_data.get("performance", {}).get("steps_executed", 0) > 0
            }
            
            all_checks_passed = all(validation_checks.values())
            
            if all_checks_passed:
                logger.info("✅ All validation checks passed", checks=validation_checks)
            else:
                logger.warning("⚠️ Some validation checks failed", checks=validation_checks)
            
            # Test 5: Cleanup and metrics collection
            logger.info("Test 5: Collecting final metrics and cleanup")
            
            metrics = await coordinator.get_metrics()
            logger.info("📊 Final orchestration metrics", metrics=metrics)
            
            # Graceful shutdown
            await coordinator.shutdown()
            logger.info("✅ Coordinator shutdown completed")
            
            return all_checks_passed
            
        except Exception as error:
            logger.error("❌ Integration test failed with error", 
                        error=str(error), 
                        error_type=type(error).__name__)
            return False


async def run_comprehensive_validation():
    """Run comprehensive validation suite"""
    logger = structlog.get_logger("validation_suite")
    
    logger.info("🚀 Starting Comprehensive Browser Orchestration Validation")
    
    # Check system prerequisites
    logger.info("Checking system prerequisites...")
    
    try:
        # Check if browser-use is available
        import browser_use
        logger.info("✅ browser-use library available", version=getattr(browser_use, '__version__', 'unknown'))
        
        # Check if required orchestration modules exist
        orchestrator_files = [
            "coordinator.py",
            "agent_pool.py", 
            "queue_manager.py",
            "session_manager.py"
        ]
        
        for file_name in orchestrator_files:
            file_path = Path(__file__).parent / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required orchestrator file missing: {file_name}")
        
        logger.info("✅ All orchestrator modules available")
        
        # Run integration test
        test_passed = await test_browser_orchestration_integration()
        
        if test_passed:
            logger.info("🎉 COMPREHENSIVE VALIDATION PASSED - Browser orchestration integration is fully functional!")
            return True
        else:
            logger.error("❌ VALIDATION FAILED - Integration issues detected")
            return False
            
    except ImportError as error:
        logger.error("❌ Import error - missing dependencies", error=str(error))
        return False
    except Exception as error:
        logger.error("❌ Validation error", error=str(error))
        return False


if __name__ == "__main__":
    setup_logging()
    
    # Set environment variables for testing
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")
    os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder") 
    
    # Run validation
    result = asyncio.run(run_comprehensive_validation())
    
    # Exit with appropriate code
    sys.exit(0 if result else 1)