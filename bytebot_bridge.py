#!/usr/bin/env python3
"""
Bytebot Bridge for Browser-Use Integration

This bridge script provides a communication interface between the Bytebot TypeScript
service and the browser-use Python library. It handles browser session management,
action execution, data extraction, and provides structured responses for all
browser automation operations.

Key Features:
- JSON-based command/response protocol
- Browser session lifecycle management
- Comprehensive action execution (click, type, navigate, etc.)
- Data extraction with multiple strategies
- Screenshot capture and visual feedback
- Error handling and recovery mechanisms
- Resource cleanup and memory management

Protocol:
- Commands are received via stdin as JSON objects
- Responses are sent via stdout as JSON objects
- Each command has a unique ID for correlation
- Supports timeout handling and cancellation

Dependencies: browser-use, asyncio, json, logging
"""

import asyncio
import json
import logging
import sys
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
import signal
import atexit

# Import browser-use components
try:
    from browser_use import Agent, Browser, BrowserSession
    from browser_use.agent.views import ActionResult
    from browser_use.browser.views import BrowserStateSummary
    from browser_use.dom.service import DomService
    from browser_use.tools.service import Tools
    from browser_use.llm.openai import ChatOpenAI
    from browser_use.browser.profile import BrowserProfile, ProxySettings
except ImportError as e:
    print(f"Error importing browser-use: {e}", file=sys.stderr)
    sys.exit(1)

class BytebotBrowserBridge:
    """
    Bridge between Bytebot TypeScript service and browser-use Python library.

    Manages browser sessions, executes automation commands, and handles
    communication protocol with the TypeScript parent process.
    """

    def __init__(self, config_file: str):
        """Initialize the bridge with configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        self.session_id = os.environ.get('BROWSER_USE_SESSION_ID', str(uuid.uuid4()))
        self.browser_session: Optional[BrowserSession] = None
        self.agent: Optional[Agent] = None
        self.running = True

        # Set up logging
        self._setup_logging()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self._cleanup)

        self.logger.info(f"Bytebot Browser Bridge initialized for session: {self.session_id}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.INFO
        if os.environ.get('BROWSER_USE_DEBUG') == 'true':
            log_level = logging.DEBUG

        # Create temp directory for logs if it doesn't exist
        temp_dir = Path(self.config.get('tempDirectory', '/tmp'))
        temp_dir.mkdir(exist_ok=True)

        log_file = temp_dir / f"browser_bridge_{self.session_id}.log"

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stderr)
            ]
        )

        self.logger = logging.getLogger(f'BytebotBridge.{self.session_id[-8:]}')
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _cleanup(self):
        """Clean up resources."""
        try:
            if self.browser_session:
                asyncio.run(self._close_browser_session())
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def _close_browser_session(self):
        """Close the browser session gracefully."""
        try:
            if self.browser_session:
                await self.browser_session.close()
                self.browser_session = None
                self.logger.info("Browser session closed")
        except Exception as e:
            self.logger.error(f"Error closing browser session: {e}")

    async def run(self):
        """Main execution loop - process commands from stdin."""
        self.logger.info("Starting command processing loop")

        try:
            while self.running:
                # Read command from stdin
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    # Parse command
                    command = json.loads(line)
                    await self._process_command(command)

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON command: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing command: {e}")
                    continue

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
        finally:
            await self._cleanup_async()

    async def _cleanup_async(self):
        """Async cleanup operations."""
        await self._close_browser_session()

    async def _process_command(self, command: Dict[str, Any]):
        """Process a single command and send response."""
        command_id = command.get('id', str(uuid.uuid4()))
        action = command.get('action', '')
        session_id = command.get('sessionId', '')
        parameters = command.get('parameters', {})
        timeout = command.get('timeout', 30000)

        start_time = time.time()

        self.logger.debug(f"Processing command: {action} (ID: {command_id})")

        try:
            # Route command to appropriate handler
            if action == 'initialize_session':
                result = await self._initialize_session(parameters)
            elif action == 'close_session':
                result = await self._close_session(parameters)
            elif action == 'navigate':
                result = await self._navigate(parameters)
            elif action == 'execute_action':
                result = await self._execute_action(parameters)
            elif action == 'capture_screenshot':
                result = await self._capture_screenshot(parameters)
            elif action == 'extract_data':
                result = await self._extract_data(parameters)
            elif action == 'get_page_info':
                result = await self._get_page_info(parameters)
            else:
                raise ValueError(f"Unknown action: {action}")

            execution_time = int((time.time() - start_time) * 1000)

            # Send success response
            response = {
                'id': command_id,
                'success': True,
                'data': result,
                'executionTimeMs': execution_time
            }

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            error_message = str(e)

            self.logger.error(f"Command failed: {action} - {error_message}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")

            # Send error response
            response = {
                'id': command_id,
                'success': False,
                'error': error_message,
                'executionTimeMs': execution_time
            }

        # Send response to stdout
        response_json = json.dumps(response)
        print(response_json, flush=True)

        self.logger.debug(f"Command completed: {action} (ID: {command_id}) in {execution_time}ms")

    async def _initialize_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a new browser session."""
        self.logger.info("Initializing browser session")

        # Create browser profile
        profile = BrowserProfile(
            headless=parameters.get('headless', True),
            width=parameters.get('viewport', {}).get('width', 1920),
            height=parameters.get('viewport', {}).get('height', 1080),
            user_agent=parameters.get('userAgent'),
            proxy=self._create_proxy_settings(parameters.get('proxy')),
            args=parameters.get('extraArgs', [])
        )

        # Create browser session
        self.browser_session = Browser(profile=profile)

        # Initialize agent (optional - for AI-powered actions)
        if parameters.get('enableAI', False):
            # You would need to provide an LLM instance here
            # For now, we'll skip AI features to avoid requiring API keys
            pass

        await self.browser_session.start()

        self.logger.info("Browser session initialized successfully")

        return {
            'sessionId': self.session_id,
            'status': 'active',
            'configuration': {
                'headless': profile.headless,
                'viewport': {'width': profile.width, 'height': profile.height},
                'userAgent': profile.user_agent
            }
        }

    async def _close_session(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Close the browser session."""
        self.logger.info("Closing browser session")

        await self._close_browser_session()

        return {
            'sessionId': self.session_id,
            'status': 'closed'
        }

    async def _navigate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to a URL."""
        url = parameters.get('url')
        wait_for_load = parameters.get('waitForLoad', True)
        wait_for_selector = parameters.get('waitForSelector')
        timeout = parameters.get('timeout', 30000)

        if not url:
            raise ValueError("URL is required for navigation")

        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")

        self.logger.info(f"Navigating to: {url}")

        # Navigate to URL
        await self.browser_session.go_to(url)

        # Wait for specific selector if provided
        if wait_for_selector:
            try:
                await self.browser_session.wait_for_selector(wait_for_selector, timeout=timeout/1000)
            except Exception as e:
                self.logger.warning(f"Failed to wait for selector '{wait_for_selector}': {e}")

        # Get current page info
        current_url = await self.browser_session.get_url()
        title = await self.browser_session.get_title()

        self.logger.info(f"Navigation completed. Final URL: {current_url}")

        return {
            'finalUrl': current_url,
            'title': title,
            'statusCode': 200  # Browser-use doesn't expose HTTP status codes directly
        }

    async def _execute_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a browser action."""
        action_type = parameters.get('action')
        target = parameters.get('target', {})

        if not action_type:
            raise ValueError("Action type is required")

        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")

        self.logger.info(f"Executing action: {action_type}")

        # Get the current page
        page = self.browser_session.page

        # Execute action based on type
        if action_type == 'click':
            await self._execute_click_action(page, parameters)
        elif action_type == 'type':
            await self._execute_type_action(page, parameters)
        elif action_type == 'scroll':
            await self._execute_scroll_action(page, parameters)
        elif action_type == 'hover':
            await self._execute_hover_action(page, parameters)
        elif action_type == 'wait':
            await self._execute_wait_action(page, parameters)
        elif action_type == 'press_key':
            await self._execute_key_press_action(page, parameters)
        else:
            raise ValueError(f"Unsupported action type: {action_type}")

        # Get element info if target was specified
        element_info = None
        if target.get('selector'):
            try:
                element_info = await self._get_element_info(page, target['selector'])
            except Exception as e:
                self.logger.debug(f"Could not get element info: {e}")

        return {
            'actionType': action_type,
            'elementInfo': element_info
        }

    async def _execute_click_action(self, page, parameters: Dict[str, Any]):
        """Execute click action."""
        target = parameters.get('target', {})

        if target.get('coordinates'):
            # Click at coordinates
            x = target['coordinates']['x']
            y = target['coordinates']['y']
            await page.mouse.click(x, y)
        elif target.get('selector'):
            # Click element by selector
            await page.click(target['selector'])
        elif target.get('xpath'):
            # Click element by XPath
            element = await page.wait_for_selector(f"xpath={target['xpath']}")
            await element.click()
        else:
            raise ValueError("Click target must specify coordinates, selector, or xpath")

    async def _execute_type_action(self, page, parameters: Dict[str, Any]):
        """Execute type action."""
        text = parameters.get('text', '')
        target = parameters.get('target', {})

        if target.get('selector'):
            # Type in element by selector
            await page.fill(target['selector'], text)
        elif target.get('xpath'):
            # Type in element by XPath
            element = await page.wait_for_selector(f"xpath={target['xpath']}")
            await element.fill(text)
        else:
            # Type at current focus
            await page.keyboard.type(text)

    async def _execute_scroll_action(self, page, parameters: Dict[str, Any]):
        """Execute scroll action."""
        scroll_distance = parameters.get('scrollDistance', 500)
        target = parameters.get('target', {})

        if target.get('selector'):
            # Scroll element into view
            await page.locator(target['selector']).scroll_into_view_if_needed()
        else:
            # Scroll page
            await page.mouse.wheel(0, scroll_distance)

    async def _execute_hover_action(self, page, parameters: Dict[str, Any]):
        """Execute hover action."""
        target = parameters.get('target', {})

        if target.get('selector'):
            await page.hover(target['selector'])
        elif target.get('xpath'):
            element = await page.wait_for_selector(f"xpath={target['xpath']}")
            await element.hover()
        else:
            raise ValueError("Hover target must specify selector or xpath")

    async def _execute_wait_action(self, page, parameters: Dict[str, Any]):
        """Execute wait action."""
        wait_duration = parameters.get('waitDuration', 1000)
        await asyncio.sleep(wait_duration / 1000)  # Convert to seconds

    async def _execute_key_press_action(self, page, parameters: Dict[str, Any]):
        """Execute key press action."""
        keys = parameters.get('keys', '')
        await page.keyboard.press(keys)

    async def _capture_screenshot(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Capture screenshot of the current page."""
        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")

        self.logger.info("Capturing screenshot")

        # Capture screenshot
        screenshot_bytes = await self.browser_session.page.screenshot(
            type='png',
            full_page=parameters.get('fullPage', False)
        )

        # Convert to base64
        import base64
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        # Get page info
        current_url = await self.browser_session.get_url()
        title = await self.browser_session.get_title()

        # Get viewport size
        viewport = await self.browser_session.page.viewport_size()

        return {
            'image': screenshot_b64,
            'format': 'png',
            'width': viewport['width'] if viewport else 0,
            'height': viewport['height'] if viewport else 0,
            'fileSizeBytes': len(screenshot_bytes),
            'pageUrl': current_url,
            'pageTitle': title
        }

    async def _extract_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from the current page."""
        extraction_type = parameters.get('extractionType')

        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")

        self.logger.info(f"Extracting data: {extraction_type}")

        page = self.browser_session.page

        if extraction_type == 'css_selector':
            return await self._extract_by_css_selector(page, parameters)
        elif extraction_type == 'xpath':
            return await self._extract_by_xpath(page, parameters)
        elif extraction_type == 'text_content':
            return await self._extract_text_content(page, parameters)
        elif extraction_type == 'attributes':
            return await self._extract_attributes(page, parameters)
        else:
            raise ValueError(f"Unsupported extraction type: {extraction_type}")

    async def _extract_by_css_selector(self, page, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data using CSS selector."""
        selector = parameters.get('selector')
        extract_all = parameters.get('extractAll', False)
        include_metadata = parameters.get('includeMetadata', False)

        if not selector:
            raise ValueError("CSS selector is required")

        if extract_all:
            elements = await page.query_selector_all(selector)
        else:
            element = await page.query_selector(selector)
            elements = [element] if element else []

        extracted_data = []
        element_details = []

        for element in elements:
            if element:
                # Extract text content
                text_content = await element.text_content()
                extracted_data.append(text_content)

                if include_metadata:
                    # Get element details
                    tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                    attributes = await element.evaluate('el => Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value]))')
                    bounding_box = await element.bounding_box()

                    element_details.append({
                        'tagName': tag_name,
                        'textContent': text_content,
                        'attributes': attributes,
                        'boundingRect': bounding_box
                    })

        return {
            'extractedData': extracted_data,
            'elementCount': len(extracted_data),
            'elements': element_details if include_metadata else None
        }

    async def _extract_by_xpath(self, page, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data using XPath."""
        xpath = parameters.get('xpath')
        extract_all = parameters.get('extractAll', False)

        if not xpath:
            raise ValueError("XPath expression is required")

        if extract_all:
            elements = await page.query_selector_all(f"xpath={xpath}")
        else:
            element = await page.query_selector(f"xpath={xpath}")
            elements = [element] if element else []

        extracted_data = []
        for element in elements:
            if element:
                text_content = await element.text_content()
                extracted_data.append(text_content)

        return {
            'extractedData': extracted_data,
            'elementCount': len(extracted_data)
        }

    async def _extract_text_content(self, page, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all text content from page."""
        text_content = await page.text_content('body')

        return {
            'extractedData': text_content,
            'elementCount': 1
        }

    async def _extract_attributes(self, page, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attributes from elements."""
        selector = parameters.get('selector')
        attributes = parameters.get('attributes', [])

        if not selector:
            raise ValueError("CSS selector is required for attribute extraction")

        elements = await page.query_selector_all(selector)
        extracted_data = []

        for element in elements:
            if element:
                element_attrs = {}
                for attr in attributes:
                    value = await element.get_attribute(attr)
                    element_attrs[attr] = value
                extracted_data.append(element_attrs)

        return {
            'extractedData': extracted_data,
            'elementCount': len(extracted_data)
        }

    async def _get_page_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get current page information."""
        if not self.browser_session:
            raise RuntimeError("Browser session not initialized")

        current_url = await self.browser_session.get_url()
        title = await self.browser_session.get_title()

        return {
            'url': current_url,
            'title': title
        }

    async def _get_element_info(self, page, selector: str) -> Dict[str, Any]:
        """Get information about an element."""
        element = await page.query_selector(selector)
        if not element:
            return None

        tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
        text_content = await element.text_content()
        attributes = await element.evaluate('el => Object.fromEntries(Array.from(el.attributes).map(attr => [attr.name, attr.value]))')
        bounding_box = await element.bounding_box()

        return {
            'tagName': tag_name,
            'textContent': text_content,
            'attributes': attributes,
            'boundingRect': bounding_box
        }

    def _create_proxy_settings(self, proxy_config: Optional[Dict[str, Any]]) -> Optional[ProxySettings]:
        """Create proxy settings from configuration."""
        if not proxy_config:
            return None

        return ProxySettings(
            server=f"{proxy_config['host']}:{proxy_config['port']}",
            username=proxy_config.get('username'),
            password=proxy_config.get('password')
        )

async def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python bytebot_bridge.py <config_file>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]

    # Create and run the bridge
    bridge = BytebotBrowserBridge(config_file)
    await bridge.run()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bridge terminated by user", file=sys.stderr)
    except Exception as e:
        print(f"Bridge failed: {e}", file=sys.stderr)
        sys.exit(1)