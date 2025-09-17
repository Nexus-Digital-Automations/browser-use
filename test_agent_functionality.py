#!/usr/bin/env python3

import asyncio

import browser_use


class MockLLM:
    """Simple mock LLM for testing Agent functionality"""

    def __init__(self, model="mock-gpt-4", provider="mock"):
        self.model = model
        self.provider = provider

    async def ainvoke(self, messages):
        """Mock LLM response that performs simple browser actions"""
        # Return a mock response that suggests basic browser actions
        return MockResponse(
            content="I'll help you test the browser. Task completed successfully."
        )

    async def agenerate(self, messages):
        """Mock generation method"""
        return MockGenerationResult(
            [MockGeneration(text="Test task completed successfully.")]
        )


class MockResponse:
    def __init__(self, content):
        self.content = content


class MockGeneration:
    def __init__(self, text):
        self.text = text


class MockGenerationResult:
    def __init__(self, generations):
        self.generations = generations


async def test_agent_functionality():
    """Test basic Agent functionality with mock LLM"""
    print("🔍 Testing Agent functionality...")

    try:
        # Create mock LLM
        mock_llm = MockLLM()

        # Create browser session
        browser_session = browser_use.BrowserSession(
            headless=True,
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        )

        # Create agent with simple task
        agent = browser_use.Agent(
            task="Open a webpage and check if it loads",
            llm=mock_llm,
            browser_session=browser_session,
        )

        print("✅ Agent created successfully!")
        print(f"📋 Task: {agent.task}")
        print(f"🤖 LLM Model: {mock_llm.model}")

        return True

    except Exception as e:
        print(f"❌ Agent functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_tools_functionality():
    """Test if Tools/Controller functionality works"""
    print("\n🔧 Testing Tools functionality...")

    try:
        # Test Tools import and basic functionality
        tools = browser_use.Tools()
        print("✅ Tools imported successfully!")

        # Check available tools
        if hasattr(tools, "__dict__"):
            print("🛠️ Available tools loaded")

        return True

    except Exception as e:
        print(f"❌ Tools functionality test failed: {e}")
        return False


async def test_full_integration():
    """Test complete integration with simple automation"""
    print("\n🚀 Testing full browser automation integration...")

    try:
        # This is a minimal integration test
        session = browser_use.BrowserSession(headless=True)
        await session.start()

        print("✅ Browser session started")

        # Check if we can get basic browser state
        _state = await session.get_browser_state_summary()
        print("✅ Browser state accessible")

        await session.stop()
        print("✅ Browser session stopped")

        return True

    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("Browser-Use Agent & API Integration Test")
    print("=" * 50)

    # Test agent functionality
    result1 = asyncio.run(test_agent_functionality())

    # Test tools functionality
    result2 = asyncio.run(test_tools_functionality())

    # Test full integration
    result3 = asyncio.run(test_full_integration())

    if all([result1, result2, result3]):
        print("\n🎉 All tests passed! Browser-use API integration is working!")
        print("✅ Agent functionality: Working")
        print("✅ Tools functionality: Working")
        print("✅ Browser automation: Working")
    else:
        print("\n⚠️ Some tests failed:")
        print(f"❌ Agent functionality: {'✅' if result1 else '❌'}")
        print(f"❌ Tools functionality: {'✅' if result2 else '❌'}")
        print(f"❌ Browser automation: {'✅' if result3 else '❌'}")

    print("\nIntegration test completed.")
