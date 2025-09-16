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

        # Try to navigate to a simple page
        await session.goto("data:text/html,<h1>Test</h1>")
        print("✓ Navigation successful!")

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


async def test_minimal_agent():
    """Test minimal agent functionality without LLM"""
    print("\nTesting minimal browser functionality...")

    try:
        session = browser_use.BrowserSession(headless=True)
        await asyncio.wait_for(session.start(), timeout=20.0)

        # Test basic DOM operations
        await session.goto("data:text/html,<h1>Hello World</h1><p>Test page</p>")
        print("✓ Basic page loaded")

        # Test if we can get page content
        page_content = await session.get_page_content()
        if "Hello World" in page_content:
            print("✓ Page content retrieval working")
        else:
            print("✗ Page content retrieval failed")

        await session.close()
        return True

    except Exception as e:
        print(f"✗ Minimal test failed: {e}")
        return False


if __name__ == "__main__":
    print("Browser-Use Integration Debug Test")
    print("=" * 50)

    # Test basic Chrome functionality
    result1 = asyncio.run(test_chrome_basic())

    if result1:
        # If basic Chrome works, test minimal agent functionality
        result2 = asyncio.run(test_minimal_agent())

        if result2:
            print(
                "\n🎉 All tests passed! Browser-use integration appears to be working."
            )
        else:
            print("\n⚠️ Chrome launches but agent functionality has issues.")
    else:
        print("\n❌ Chrome launch failed. Integration has issues.")

    print("\nDebug test completed.")
