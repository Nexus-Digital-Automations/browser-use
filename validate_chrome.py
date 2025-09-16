#!/usr/bin/env python3

import asyncio

import browser_use


async def validate_chrome():
    """Simple validation that Chrome starts and stops correctly"""
    print("🔍 Validating browser-use Chrome integration...")

    try:
        # Create session with explicit Chrome path
        session = browser_use.BrowserSession(
            headless=True,
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        )

        print("⏳ Starting Chrome...")
        await asyncio.wait_for(session.start(), timeout=20.0)
        print("✅ Chrome started successfully!")

        # Test basic functionality
        url = await session.get_current_page_url()
        title = await session.get_current_page_title()
        print(f"📄 Current URL: {url}")
        print(f"📄 Current Title: {title}")

        # Cleanup
        print("⏳ Stopping Chrome...")
        await session.stop()
        print("✅ Chrome stopped successfully!")

        print("\n🎉 Browser-use Chrome integration is WORKING!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(validate_chrome())
    if result:
        print("\n✅ Chrome/Chromium drivers are working properly!")
    else:
        print("\n❌ Chrome integration has issues.")
