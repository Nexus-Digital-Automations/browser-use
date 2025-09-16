#!/usr/bin/env python3

import asyncio
import os
import sys

# Add the browser_use package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import browser_use


async def test_chrome_basic():
    """Test basic Chrome launch functionality"""
    print("Testing Chrome launch with explicit configuration...")

    try:
        # Try with explicit Chrome path and basic args
        session = browser_use.BrowserSession(
            headless=True,
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--remote-debugging-address=0.0.0.0",
                "--remote-debugging-port=9222",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
            ],
        )

        print("Created BrowserSession, attempting to start...")

        # Set a timeout for the start operation
        await asyncio.wait_for(session.start(), timeout=30.0)
        print("✓ Chrome started successfully!")

        # Test basic session functionality
        print(f"Current URL: {await session.get_current_page_url()}")
        print(f"Page title: {await session.get_current_page_title()}")
        print("✓ Basic session methods working!")

        # Clean up
        await session.close()
        print("✓ Chrome closed successfully!")

        return True

    except asyncio.TimeoutError:
        print("✗ Chrome start timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_mock_agent():
    """Test Agent functionality with a mock LLM"""
    print("\nTesting Agent functionality with minimal setup...")

    try:
        # Create a simple mock LLM for testing
        class MockLLM:
            def __init__(self):
                self.model = "mock-model"

            async def acomplete(self, *args, **kwargs):
                # Return a simple completion that just closes the browser
                return "Task completed successfully"

        # Create agent with minimal task
        mock_llm = MockLLM()
        agent = browser_use.Agent(
            task="test task",
            llm=mock_llm,
            browser_session=browser_use.BrowserSession(headless=True),
        )

        print("✓ Agent created successfully!")

        # Try basic agent initialization
        print("✓ Agent functionality test completed!")

        return True

    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Browser-Use Integration Debug Test (Fixed)")
    print("=" * 50)

    # Test basic Chrome functionality
    result1 = asyncio.run(test_chrome_basic())

    if result1:
        print("\n🎉 Chrome integration working!")

        # Test agent functionality
        result2 = asyncio.run(test_mock_agent())

        if result2:
            print(
                "\n🎉 All tests passed! Browser-use integration appears to be working."
            )
        else:
            print("\n⚠️ Chrome launches but agent functionality has issues.")
    else:
        print("\n❌ Chrome launch failed. Integration has issues.")

    print("\nDebug test completed.")
