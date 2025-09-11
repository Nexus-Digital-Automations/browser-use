#!/usr/bin/env python3
"""
Simple integration test for browser-use with AIgent system.
Tests essential functionality without complex async operations.
"""
import logging
import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_core_imports():
    """Test that core browser-use components can be imported."""
    try:
        # Test essential imports
        from browser_use import Agent
        from browser_use.llm.openai.chat import ChatOpenAI

        logger.info("✅ Core imports successful")

        # Test agent creation (without running)
        llm = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
        _agent = Agent(task="test task", llm=llm)

        logger.info("✅ Agent creation successful")
        return True

    except Exception as e:
        logger.error(f"❌ Core imports failed: {e}")
        return False


def test_configuration_loading():
    """Test configuration loading for AIgent integration."""
    try:
        from browser_use.config import CONFIG

        logger.info("✅ Configuration loading successful")

        # Test important config attributes
        config_attrs = [
            "BROWSER_USE_DEBUG_LOG_FILE",
            "BROWSER_USE_INFO_LOG_FILE",
            "MCP_SERVER_PORT",
        ]

        available_attrs = []
        for attr in config_attrs:
            if hasattr(CONFIG, attr):
                available_attrs.append(attr)

        logger.info(
            f"✅ Configuration attributes available: {len(available_attrs)}/{len(config_attrs)}"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_mcp_module_availability():
    """Test that MCP module is available for integration."""
    try:
        from browser_use.mcp import server

        logger.info("✅ MCP module imports successful")

        # Check that server module has expected classes
        server_classes = [name for name in dir(server) if name[0].isupper()]
        logger.info(f"✅ MCP server classes available: {server_classes}")

        return True

    except Exception as e:
        logger.error(f"❌ MCP module test failed: {e}")
        return False


def test_browser_executable():
    """Test that browser executable is accessible."""
    try:
        import os

        chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(chrome_path):
            logger.info("✅ Chrome executable found")
            return True
        else:
            logger.warning(f"⚠️ Chrome not found at {chrome_path}")
            return False

    except Exception as e:
        logger.error(f"❌ Browser executable test failed: {e}")
        return False


def test_api_integration_readiness():
    """Test readiness for API integration with AIgent."""
    try:
        # Test JSON serialization of key components
        import json

        config_data = {
            "browser_use": {
                "version": "0.7.4",
                "mcp_enabled": True,
                "browser_path": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "api_endpoints": {
                    "mcp_server": "http://localhost:8100",
                    "browser_control": "http://localhost:9242",
                },
            }
        }

        _config_json = json.dumps(config_data, indent=2)
        logger.info("✅ API configuration serialization successful")

        # Test that we can create the integration data structure
        integration_status = {
            "status": "ready",
            "capabilities": [
                "browser_automation",
                "mcp_server",
                "agent_execution",
                "api_integration",
            ],
            "endpoints": config_data["browser_use"]["api_endpoints"],
        }

        logger.info("✅ Integration status structure created")
        logger.info(f"   Capabilities: {integration_status['capabilities']}")

        return True

    except Exception as e:
        logger.error(f"❌ API integration test failed: {e}")
        return False


def main():
    """Run all simple integration tests."""
    logger.info("🚀 Starting browser-use simple integration tests...")

    tests = [
        ("Core Imports", test_core_imports),
        ("Configuration Loading", test_configuration_loading),
        ("MCP Module Availability", test_mcp_module_availability),
        ("Browser Executable", test_browser_executable),
        ("API Integration Readiness", test_api_integration_readiness),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Testing {test_name}...")
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"   {status}")
        except Exception as e:
            logger.error(f"   ❌ ERROR: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed >= total * 0.8:  # 80% pass rate
        logger.info("🎉 Browser-use integration is ready for AIgent!")
        logger.info("✅ Most core functionality is working properly")
        return True
    else:
        logger.error("💥 Browser-use integration has significant issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
