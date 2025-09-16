"""
Browser Session Management System

Comprehensive browser session lifecycle management for concurrent browser automation.
Handles session creation, state management, resource tracking, and cleanup.

Key Features:
- Browser session lifecycle management
- Session isolation and state management
- Resource tracking and optimization
- Cookie and storage management
- Screenshot and artifact coordination
- Headless/headful mode switching
- Local-only architecture compliance

Author: Claude Code
Version: 1.0.0
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import structlog


class SessionStatus(Enum):
    """Browser session status"""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


class BrowserType(Enum):
    """Supported browser types"""

    CHROME = "chrome"
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


@dataclass
class SessionConfiguration:
    """Browser session configuration"""

    browser_type: BrowserType = BrowserType.CHROME
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent: Optional[str] = None
    proxy_config: Optional[Dict[str, str]] = None  # Local-only, no external proxies
    disable_javascript: bool = False
    disable_images: bool = False
    disable_web_security: bool = False
    enable_devtools: bool = False
    custom_chrome_args: List[str] = field(default_factory=list)
    download_path: Optional[str] = None
    timeout_ms: int = 30000
    page_load_timeout_ms: int = 30000

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "browser_type": self.browser_type.value,
            "headless": self.headless,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "user_agent": self.user_agent,
            "proxy_config": self.proxy_config,
            "disable_javascript": self.disable_javascript,
            "disable_images": self.disable_images,
            "disable_web_security": self.disable_web_security,
            "enable_devtools": self.enable_devtools,
            "custom_chrome_args": self.custom_chrome_args,
            "download_path": self.download_path,
            "timeout_ms": self.timeout_ms,
            "page_load_timeout_ms": self.page_load_timeout_ms,
        }


@dataclass
class SessionState:
    """Browser session state tracking"""

    current_url: Optional[str] = None
    page_title: Optional[str] = None
    cookies: List[Dict] = field(default_factory=list)
    local_storage: Dict[str, str] = field(default_factory=dict)
    session_storage: Dict[str, str] = field(default_factory=dict)
    tabs_count: int = 1
    active_tab_index: int = 0
    navigation_history: List[str] = field(default_factory=list)
    javascript_enabled: bool = True
    images_enabled: bool = True
    last_activity: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionResources:
    """Browser session resource tracking"""

    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_requests_count: int = 0
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    cache_size_mb: float = 0.0
    screenshots_taken: int = 0
    files_downloaded: int = 0
    dom_nodes_count: int = 0
    javascript_errors: int = 0
    console_messages: List[Dict] = field(default_factory=list)


@dataclass
class BrowserSession:
    """Browser session instance representation"""

    session_id: str
    agent_id: str
    configuration: SessionConfiguration
    status: SessionStatus = SessionStatus.INITIALIZING
    state: SessionState = field(default_factory=SessionState)
    resources: SessionResources = field(default_factory=SessionResources)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    browser_process_id: Optional[int] = None
    browser_context_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Task tracking
    current_task_id: Optional[str] = None
    assigned_tasks: Set[str] = field(default_factory=set)
    completed_tasks: Set[str] = field(default_factory=set)

    def is_active(self) -> bool:
        """Check if session is currently active"""
        return self.status in [SessionStatus.ACTIVE, SessionStatus.IDLE]

    def get_idle_duration(self) -> timedelta:
        """Get duration since last activity"""
        return datetime.now() - self.last_accessed

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_accessed = datetime.now()
        self.state.last_activity = datetime.now()

    def add_task(self, task_id: str) -> None:
        """Add task to session"""
        self.assigned_tasks.add(task_id)
        self.current_task_id = task_id
        self.update_activity()

    def complete_task(self, task_id: str) -> None:
        """Mark task as completed"""
        if task_id in self.assigned_tasks:
            self.assigned_tasks.remove(task_id)
            self.completed_tasks.add(task_id)

        if self.current_task_id == task_id:
            self.current_task_id = None

        self.update_activity()


class SessionManager:
    """
    Comprehensive browser session manager for automation orchestration.

    Manages the complete lifecycle of browser sessions including creation,
    state management, resource optimization, and cleanup coordination.
    """

    def __init__(self, config: Dict[str, Any], storage_path: Optional[Path] = None):
        """
        Initialize session manager.

        Args:
            config: Session management configuration
            storage_path: Local storage path for session data
        """
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.config = config
        self.storage_path = storage_path or Path("./browser_sessions")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Session management
        self.sessions: Dict[str, BrowserSession] = {}
        self.agent_sessions: Dict[str, Set[str]] = {}  # agent_id -> session_ids

        # Configuration
        self.max_sessions_per_agent = config.get("max_sessions_per_agent", 3)
        self.max_total_sessions = config.get("max_total_sessions", 50)
        self.session_idle_timeout = config.get(
            "session_idle_timeout", 1800
        )  # 30 minutes
        self.session_max_lifetime = config.get("session_max_lifetime", 14400)  # 4 hours
        self.cleanup_interval = config.get("cleanup_interval", 300)  # 5 minutes

        # Resource limits (local-only architecture)
        self.max_memory_per_session_mb = config.get("max_memory_per_session_mb", 512)
        self.max_total_memory_mb = config.get("max_total_memory_mb", 8192)  # 8GB
        self.max_concurrent_downloads = config.get("max_concurrent_downloads", 5)

        # Default session configuration
        self.default_session_config = SessionConfiguration(
            browser_type=BrowserType(config.get("default_browser", "chrome")),
            headless=config.get("default_headless", True),
            viewport_width=config.get("default_viewport_width", 1920),
            viewport_height=config.get("default_viewport_height", 1080),
            timeout_ms=config.get("default_timeout_ms", 30000),
        )

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None

        # Statistics
        self.session_stats = {
            "sessions_created": 0,
            "sessions_terminated": 0,
            "sessions_timeout": 0,
            "sessions_error": 0,
            "total_tasks_completed": 0,
            "total_screenshots": 0,
            "total_downloads": 0,
        }

        self.logger.info(
            "Session Manager initialized",
            max_sessions_per_agent=self.max_sessions_per_agent,
            max_total_sessions=self.max_total_sessions,
            storage_path=str(self.storage_path),
        )

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize session manager and start background tasks.

        Returns:
            Initialization result
        """
        operation_id = self._generate_operation_id()
        self.logger.info(f"[{operation_id}] Initializing session manager")

        try:
            # Load persisted sessions if available
            await self._load_persisted_sessions()

            # Start background tasks
            await self._start_background_tasks()

            # Initialize metrics
            await self._initialize_metrics()

            self.logger.info(
                f"[{operation_id}] Session manager initialized successfully",
                persisted_sessions=len(self.sessions),
            )

            return {
                "success": True,
                "operation_id": operation_id,
                "persisted_sessions": len(self.sessions),
                "max_total_sessions": self.max_total_sessions,
            }

        except Exception as error:
            self.logger.error(
                f"[{operation_id}] Session manager initialization failed",
                error=str(error),
                exc_info=True,
            )
            raise

    async def create_session(
        self,
        agent_id: str,
        config: Optional[SessionConfiguration] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create new browser session for agent.

        Args:
            agent_id: Agent identifier
            config: Optional session configuration
            metadata: Optional session metadata

        Returns:
            Session creation result
        """
        # Check capacity limits
        if len(self.sessions) >= self.max_total_sessions:
            return {
                "success": False,
                "reason": "Maximum total sessions reached",
                "current_count": len(self.sessions),
                "max_count": self.max_total_sessions,
            }

        agent_session_count = len(self.agent_sessions.get(agent_id, set()))
        if agent_session_count >= self.max_sessions_per_agent:
            return {
                "success": False,
                "reason": "Maximum sessions per agent reached",
                "agent_sessions": agent_session_count,
                "max_per_agent": self.max_sessions_per_agent,
            }

        # Generate session ID
        session_id = f"session_{int(time.time())}_{str(uuid4())[:8]}"

        # Use provided config or default
        session_config = config or self.default_session_config

        # Create session instance
        session = BrowserSession(
            session_id=session_id,
            agent_id=agent_id,
            configuration=session_config,
            metadata=metadata or {},
        )

        try:
            # Actually create browser instance
            from browser_use import Browser, BrowserProfile

            # Create browser profile from session configuration
            browser_profile = BrowserProfile(
                headless=session_config.headless,
                viewport_size=(
                    session_config.viewport_width,
                    session_config.viewport_height,
                ),
                user_data_dir=f"/tmp/browser_session_{session_id}",
                downloads_folder=f"/tmp/browser_session_{session_id}/downloads",
            )

            # Create and start browser instance
            browser = Browser(profile=browser_profile)
            await browser.start()

            # Get browser process information
            session.browser_process_id = (
                browser.process_id if hasattr(browser, "process_id") else None
            )
            session.browser_context_id = (
                browser.context_id
                if hasattr(browser, "context_id")
                else f"context_{session_id}"
            )
            session.browser_instance = browser  # Store browser reference for later use
            session.status = SessionStatus.ACTIVE

            # Update session resources with actual browser info
            if browser.page and hasattr(browser.page, "url"):
                session.resources.current_url = browser.page.url

            self.logger.info(
                f"Browser instance created for session {session_id} with process {session.browser_process_id}"
            )

            # Register session
            self.sessions[session_id] = session

            # Track agent sessions
            if agent_id not in self.agent_sessions:
                self.agent_sessions[agent_id] = set()
            self.agent_sessions[agent_id].add(session_id)

            # Update statistics
            self.session_stats["sessions_created"] += 1

            # Persist session
            await self._persist_session(session)

            self.logger.info(
                f"Created browser session {session_id}",
                agent_id=agent_id,
                browser_type=session_config.browser_type.value,
                headless=session_config.headless,
            )

            return {
                "success": True,
                "session_id": session_id,
                "configuration": session_config.to_dict(),
                "created_at": session.created_at.isoformat(),
            }

        except Exception as error:
            # Cleanup on failure
            if session_id in self.sessions:
                del self.sessions[session_id]

            if (
                agent_id in self.agent_sessions
                and session_id in self.agent_sessions[agent_id]
            ):
                self.agent_sessions[agent_id].remove(session_id)

            self.logger.error(
                f"Failed to create session for agent {agent_id}",
                error=str(error),
                exc_info=True,
            )

            return {
                "success": False,
                "reason": f"Session creation failed: {str(error)}",
                "error": str(error),
            }

    async def get_session(self, session_id: str) -> Optional[BrowserSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session instance or None if not found
        """
        session = self.sessions.get(session_id)
        if session:
            session.update_activity()
        return session

    async def get_agent_sessions(self, agent_id: str) -> List[BrowserSession]:
        """
        Get all active sessions for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of agent's sessions
        """
        session_ids = self.agent_sessions.get(agent_id, set())
        return [
            self.sessions[session_id]
            for session_id in session_ids
            if session_id in self.sessions and self.sessions[session_id].is_active()
        ]

    async def get_available_session(
        self, agent_id: str, task_requirements: Optional[Dict] = None
    ) -> Optional[BrowserSession]:
        """
        Get available session for agent that can handle task requirements.

        Args:
            agent_id: Agent identifier
            task_requirements: Optional task requirements

        Returns:
            Available session or None
        """
        agent_sessions = await self.get_agent_sessions(agent_id)

        for session in agent_sessions:
            # Check if session is idle and available
            if session.status == SessionStatus.IDLE or (
                session.status == SessionStatus.ACTIVE and not session.current_task_id
            ):

                # Check task requirements if provided
                if task_requirements and not self._session_meets_requirements(
                    session, task_requirements
                ):
                    continue

                return session

        return None

    async def assign_task_to_session(
        self, session_id: str, task_id: str
    ) -> Dict[str, Any]:
        """
        Assign task to browser session.

        Args:
            session_id: Session identifier
            task_id: Task identifier

        Returns:
            Assignment result
        """
        session = self.sessions.get(session_id)
        if not session:
            return {
                "success": False,
                "reason": "Session not found",
                "session_id": session_id,
            }

        if not session.is_active():
            return {
                "success": False,
                "reason": "Session not active",
                "session_id": session_id,
                "status": session.status.value,
            }

        # Assign task
        session.add_task(task_id)
        session.status = SessionStatus.ACTIVE

        self.logger.debug(
            f"Assigned task {task_id} to session {session_id}",
            agent_id=session.agent_id,
        )

        return {
            "success": True,
            "session_id": session_id,
            "task_id": task_id,
            "assigned_at": datetime.now().isoformat(),
        }

    async def complete_task_on_session(
        self, session_id: str, task_id: str
    ) -> Dict[str, Any]:
        """
        Mark task as completed on session.

        Args:
            session_id: Session identifier
            task_id: Task identifier

        Returns:
            Completion result
        """
        session = self.sessions.get(session_id)
        if not session:
            return {
                "success": False,
                "reason": "Session not found",
                "session_id": session_id,
            }

        session.complete_task(task_id)

        # Update session status
        if not session.current_task_id and not session.assigned_tasks:
            session.status = SessionStatus.IDLE

        # Update statistics
        self.session_stats["total_tasks_completed"] += 1

        self.logger.debug(
            f"Completed task {task_id} on session {session_id}",
            remaining_tasks=len(session.assigned_tasks),
        )

        return {
            "success": True,
            "session_id": session_id,
            "task_id": task_id,
            "completed_at": datetime.now().isoformat(),
        }

    async def update_session_state(
        self, session_id: str, state_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update session state information.

        Args:
            session_id: Session identifier
            state_update: State update data

        Returns:
            Update result
        """
        session = self.sessions.get(session_id)
        if not session:
            return {
                "success": False,
                "reason": "Session not found",
                "session_id": session_id,
            }

        # Update state fields
        state = session.state

        if "current_url" in state_update:
            state.current_url = state_update["current_url"]
            if state_update["current_url"] not in state.navigation_history:
                state.navigation_history.append(state_update["current_url"])

        if "page_title" in state_update:
            state.page_title = state_update["page_title"]

        if "cookies" in state_update:
            state.cookies = state_update["cookies"]

        if "local_storage" in state_update:
            state.local_storage.update(state_update["local_storage"])

        if "session_storage" in state_update:
            state.session_storage.update(state_update["session_storage"])

        if "tabs_count" in state_update:
            state.tabs_count = state_update["tabs_count"]

        if "active_tab_index" in state_update:
            state.active_tab_index = state_update["active_tab_index"]

        # Update activity
        session.update_activity()

        return {
            "success": True,
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
        }

    async def update_session_resources(
        self, session_id: str, resource_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update session resource usage information.

        Args:
            session_id: Session identifier
            resource_update: Resource update data

        Returns:
            Update result
        """
        session = self.sessions.get(session_id)
        if not session:
            return {
                "success": False,
                "reason": "Session not found",
                "session_id": session_id,
            }

        # Update resource fields
        resources = session.resources

        if "memory_usage_mb" in resource_update:
            resources.memory_usage_mb = resource_update["memory_usage_mb"]

        if "cpu_usage_percent" in resource_update:
            resources.cpu_usage_percent = resource_update["cpu_usage_percent"]

        if "network_requests_count" in resource_update:
            resources.network_requests_count = resource_update["network_requests_count"]

        if "screenshots_taken" in resource_update:
            resources.screenshots_taken = resource_update["screenshots_taken"]
            self.session_stats["total_screenshots"] += 1

        if "files_downloaded" in resource_update:
            resources.files_downloaded = resource_update["files_downloaded"]
            self.session_stats["total_downloads"] += 1

        if "console_message" in resource_update:
            resources.console_messages.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    **resource_update["console_message"],
                }
            )

        # Check resource limits
        if resources.memory_usage_mb > self.max_memory_per_session_mb:
            self.logger.warning(
                f"Session {session_id} exceeds memory limit",
                usage_mb=resources.memory_usage_mb,
                limit_mb=self.max_memory_per_session_mb,
            )

        session.update_activity()

        return {
            "success": True,
            "session_id": session_id,
            "resource_updated_at": datetime.now().isoformat(),
        }

    async def suspend_session(
        self, session_id: str, reason: str = "Manual"
    ) -> Dict[str, Any]:
        """
        Suspend browser session temporarily.

        Args:
            session_id: Session identifier
            reason: Suspension reason

        Returns:
            Suspension result
        """
        session = self.sessions.get(session_id)
        if not session:
            return {
                "success": False,
                "reason": "Session not found",
                "session_id": session_id,
            }

        if not session.is_active():
            return {
                "success": False,
                "reason": "Session not active",
                "session_id": session_id,
                "status": session.status.value,
            }

        session.status = SessionStatus.SUSPENDED
        session.metadata["suspended_at"] = datetime.now().isoformat()
        session.metadata["suspension_reason"] = reason

        self.logger.info(f"Suspended session {session_id}", reason=reason)

        return {
            "success": True,
            "session_id": session_id,
            "suspended_at": datetime.now().isoformat(),
            "reason": reason,
        }

    async def resume_session(self, session_id: str) -> Dict[str, Any]:
        """
        Resume suspended browser session.

        Args:
            session_id: Session identifier

        Returns:
            Resume result
        """
        session = self.sessions.get(session_id)
        if not session:
            return {
                "success": False,
                "reason": "Session not found",
                "session_id": session_id,
            }

        if session.status != SessionStatus.SUSPENDED:
            return {
                "success": False,
                "reason": "Session not suspended",
                "session_id": session_id,
                "status": session.status.value,
            }

        session.status = (
            SessionStatus.IDLE if not session.current_task_id else SessionStatus.ACTIVE
        )
        session.metadata["resumed_at"] = datetime.now().isoformat()
        session.update_activity()

        self.logger.info(f"Resumed session {session_id}")

        return {
            "success": True,
            "session_id": session_id,
            "resumed_at": datetime.now().isoformat(),
            "status": session.status.value,
        }

    async def terminate_session(
        self, session_id: str, reason: str = "Manual"
    ) -> Dict[str, Any]:
        """
        Terminate browser session and cleanup resources.

        Args:
            session_id: Session identifier
            reason: Termination reason

        Returns:
            Termination result
        """
        session = self.sessions.get(session_id)
        if not session:
            return {
                "success": False,
                "reason": "Session not found",
                "session_id": session_id,
            }

        session.status = SessionStatus.TERMINATING

        try:
            # TODO: Actually terminate browser process
            # if session.browser_process_id:
            #     await browser.close()

            # Clean up session data
            session.status = SessionStatus.TERMINATED
            session.metadata["terminated_at"] = datetime.now().isoformat()
            session.metadata["termination_reason"] = reason

            # Update agent sessions tracking
            if session.agent_id in self.agent_sessions:
                self.agent_sessions[session.agent_id].discard(session_id)

            # Update statistics
            self.session_stats["sessions_terminated"] += 1

            # Persist termination
            await self._persist_session_termination(session)

            self.logger.info(
                f"Terminated session {session_id}",
                reason=reason,
                lifetime_minutes=(datetime.now() - session.created_at).total_seconds()
                / 60,
            )

            return {
                "success": True,
                "session_id": session_id,
                "terminated_at": datetime.now().isoformat(),
                "reason": reason,
            }

        except Exception as error:
            session.status = SessionStatus.ERROR
            self.session_stats["sessions_error"] += 1

            self.logger.error(
                f"Failed to terminate session {session_id}",
                error=str(error),
                exc_info=True,
            )

            return {
                "success": False,
                "session_id": session_id,
                "reason": f"Termination failed: {str(error)}",
                "error": str(error),
            }

    async def get_session_status(self) -> Dict[str, Any]:
        """
        Get comprehensive session manager status.

        Returns:
            Session status information
        """
        # Calculate session statistics by status
        status_counts = {}
        for status in SessionStatus:
            status_counts[status.value] = sum(
                1 for session in self.sessions.values() if session.status == status
            )

        # Calculate resource usage
        total_memory_mb = sum(
            session.resources.memory_usage_mb for session in self.sessions.values()
        )

        avg_cpu_percent = sum(
            session.resources.cpu_usage_percent for session in self.sessions.values()
        ) / max(1, len(self.sessions))

        # Agent distribution
        agent_distribution = {
            agent_id: len(session_ids)
            for agent_id, session_ids in self.agent_sessions.items()
        }

        return {
            "total_sessions": len(self.sessions),
            "max_sessions": self.max_total_sessions,
            "utilization_percent": (len(self.sessions) / self.max_total_sessions) * 100,
            "session_status": status_counts,
            "agent_distribution": agent_distribution,
            "resource_usage": {
                "total_memory_mb": total_memory_mb,
                "memory_limit_mb": self.max_total_memory_mb,
                "average_cpu_percent": avg_cpu_percent,
                "memory_utilization_percent": (
                    total_memory_mb / self.max_total_memory_mb
                )
                * 100,
            },
            "statistics": self.session_stats.copy(),
            "timestamp": datetime.now().isoformat(),
        }

    async def cleanup_idle_sessions(self) -> Dict[str, Any]:
        """
        Cleanup idle and expired sessions.

        Returns:
            Cleanup operation result
        """
        current_time = datetime.now()
        idle_sessions = []
        expired_sessions = []

        for session_id, session in list(self.sessions.items()):
            # Check for idle timeout
            if (
                session.status == SessionStatus.IDLE
                and session.get_idle_duration().total_seconds()
                > self.session_idle_timeout
            ):
                idle_sessions.append(session_id)

            # Check for maximum lifetime
            lifetime = (current_time - session.created_at).total_seconds()
            if lifetime > self.session_max_lifetime:
                expired_sessions.append(session_id)

        # Terminate idle sessions
        for session_id in idle_sessions:
            await self.terminate_session(session_id, "Idle timeout")
            self.session_stats["sessions_timeout"] += 1

        # Terminate expired sessions
        for session_id in expired_sessions:
            await self.terminate_session(session_id, "Lifetime exceeded")

        # Clean up terminated sessions
        terminated_sessions = [
            session_id
            for session_id, session in list(self.sessions.items())
            if session.status == SessionStatus.TERMINATED
        ]

        for session_id in terminated_sessions:
            del self.sessions[session_id]

        self.logger.info(
            "Session cleanup completed",
            idle_terminated=len(idle_sessions),
            expired_terminated=len(expired_sessions),
            total_cleaned=len(terminated_sessions),
        )

        return {
            "idle_sessions_terminated": len(idle_sessions),
            "expired_sessions_terminated": len(expired_sessions),
            "total_sessions_cleaned": len(terminated_sessions),
            "remaining_sessions": len(self.sessions),
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown session manager"""
        self.logger.info("Shutting down session manager")

        # Cancel background tasks
        for task in [self.cleanup_task, self.monitor_task, self.metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Terminate all active sessions
        termination_tasks = []
        for session_id in list(self.sessions.keys()):
            if self.sessions[session_id].is_active():
                task = asyncio.create_task(
                    self.terminate_session(session_id, "System shutdown")
                )
                termination_tasks.append(task)

        if termination_tasks:
            await asyncio.gather(*termination_tasks, return_exceptions=True)

        # Save final statistics
        await self._save_session_statistics()

        self.logger.info("Session manager shutdown complete")

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    async def _start_background_tasks(self) -> None:
        """Start background monitoring and cleanup tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())

        self.logger.debug("Session manager background tasks started")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_idle_sessions()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Session cleanup loop error", error=str(error))

    async def _monitoring_loop(self) -> None:
        """Background session monitoring loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                await self._monitor_session_health()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Session monitoring loop error", error=str(error))

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop"""
        while True:
            try:
                await asyncio.sleep(120)  # Collect every 2 minutes
                await self._collect_session_metrics()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Metrics collection loop error", error=str(error))

    async def _monitor_session_health(self) -> None:
        """Monitor session health and handle issues"""
        for session_id, session in list(self.sessions.items()):
            # Check for high resource usage
            if session.resources.memory_usage_mb > self.max_memory_per_session_mb:
                self.logger.warning(
                    f"Session {session_id} high memory usage",
                    usage_mb=session.resources.memory_usage_mb,
                    limit_mb=self.max_memory_per_session_mb,
                )

                # Consider suspending or terminating session
                if (
                    session.resources.memory_usage_mb
                    > self.max_memory_per_session_mb * 1.5
                ):
                    await self.terminate_session(session_id, "Excessive memory usage")

            # Check for error accumulation
            if session.state.error_count > 10:
                self.logger.warning(
                    f"Session {session_id} has high error count",
                    error_count=session.state.error_count,
                )

    async def _collect_session_metrics(self) -> None:
        """Collect session metrics for monitoring"""
        for session in self.sessions.values():
            # TODO: Collect real metrics from browser instances
            # This would integrate with actual browser monitoring
            pass

    def _session_meets_requirements(
        self, session: BrowserSession, requirements: Dict
    ) -> bool:
        """Check if session meets task requirements"""
        # Check browser type requirement
        if "browser_type" in requirements:
            if session.configuration.browser_type.value != requirements["browser_type"]:
                return False

        # Check headless requirement
        if "headless" in requirements:
            if session.configuration.headless != requirements["headless"]:
                return False

        # Check capability requirements
        if "javascript" in requirements:
            if requirements["javascript"] and session.configuration.disable_javascript:
                return False

        if "images" in requirements:
            if requirements["images"] and session.configuration.disable_images:
                return False

        return True

    async def _load_persisted_sessions(self) -> None:
        """Load previously persisted sessions"""
        sessions_file = self.storage_path / "sessions.json"
        if not sessions_file.exists():
            return

        try:
            # Use asyncio.to_thread to run blocking I/O in thread executor
            session_data = await asyncio.to_thread(self._load_json_file, sessions_file)

            for session_dict in session_data:
                # Only restore sessions that were active
                if session_dict["status"] in ["active", "idle", "suspended"]:
                    session = self._deserialize_session(session_dict)
                    self.sessions[session.session_id] = session

                    # Restore agent tracking
                    if session.agent_id not in self.agent_sessions:
                        self.agent_sessions[session.agent_id] = set()
                    self.agent_sessions[session.agent_id].add(session.session_id)

            self.logger.info(f"Loaded {len(session_data)} persisted sessions")

        except Exception as error:
            self.logger.error(f"Failed to load persisted sessions: {error}")

    async def _persist_session(self, session: BrowserSession) -> None:
        """Persist single session"""
        # TODO: Implement efficient session persistence
        pass

    async def _persist_session_termination(self, session: BrowserSession) -> None:
        """Persist session termination"""
        # TODO: Implement session termination persistence
        pass

    async def _save_session_statistics(self) -> None:
        """Save session statistics"""
        stats_file = self.storage_path / "session_statistics.json"

        try:
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "statistics": self.session_stats,
                "final_session_count": len(self.sessions),
                "configuration": {
                    "max_sessions_per_agent": self.max_sessions_per_agent,
                    "max_total_sessions": self.max_total_sessions,
                    "session_idle_timeout": self.session_idle_timeout,
                    "session_max_lifetime": self.session_max_lifetime,
                },
            }

            # Use asyncio.to_thread to run blocking I/O in thread executor
            await asyncio.to_thread(self._save_json_file, stats_file, stats_data)

        except Exception as error:
            self.logger.error(f"Failed to save session statistics: {error}")

    async def _initialize_metrics(self) -> None:
        """Initialize metrics collection"""
        # Initialize metrics based on existing sessions
        pass

    def _serialize_session(self, session: BrowserSession) -> Dict[str, Any]:
        """Serialize session for persistence"""
        return {
            "session_id": session.session_id,
            "agent_id": session.agent_id,
            "status": session.status.value,
            "configuration": session.configuration.to_dict(),
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "browser_process_id": session.browser_process_id,
            "browser_context_id": session.browser_context_id,
            "current_task_id": session.current_task_id,
            "assigned_tasks": list(session.assigned_tasks),
            "completed_tasks": list(session.completed_tasks),
            "state": {
                "current_url": session.state.current_url,
                "page_title": session.state.page_title,
                "cookies": session.state.cookies,
                "local_storage": session.state.local_storage,
                "session_storage": session.state.session_storage,
                "tabs_count": session.state.tabs_count,
                "active_tab_index": session.state.active_tab_index,
                "navigation_history": session.state.navigation_history,
                "error_count": session.state.error_count,
            },
            "metadata": session.metadata,
        }

    def _deserialize_session(self, session_data: Dict[str, Any]) -> BrowserSession:
        """Deserialize session from persistence data"""
        # Reconstruct configuration
        config_data = session_data["configuration"]
        configuration = SessionConfiguration(
            browser_type=BrowserType(config_data["browser_type"]),
            headless=config_data["headless"],
            viewport_width=config_data["viewport_width"],
            viewport_height=config_data["viewport_height"],
            user_agent=config_data.get("user_agent"),
            disable_javascript=config_data.get("disable_javascript", False),
            disable_images=config_data.get("disable_images", False),
            timeout_ms=config_data.get("timeout_ms", 30000),
        )

        # Reconstruct state
        state_data = session_data["state"]
        state = SessionState(
            current_url=state_data.get("current_url"),
            page_title=state_data.get("page_title"),
            cookies=state_data.get("cookies", []),
            local_storage=state_data.get("local_storage", {}),
            session_storage=state_data.get("session_storage", {}),
            tabs_count=state_data.get("tabs_count", 1),
            active_tab_index=state_data.get("active_tab_index", 0),
            navigation_history=state_data.get("navigation_history", []),
            error_count=state_data.get("error_count", 0),
        )

        return BrowserSession(
            session_id=session_data["session_id"],
            agent_id=session_data["agent_id"],
            configuration=configuration,
            status=SessionStatus(session_data["status"]),
            state=state,
            created_at=datetime.fromisoformat(session_data["created_at"]),
            last_accessed=datetime.fromisoformat(session_data["last_accessed"]),
            browser_process_id=session_data.get("browser_process_id"),
            browser_context_id=session_data.get("browser_context_id"),
            current_task_id=session_data.get("current_task_id"),
            assigned_tasks=set(session_data.get("assigned_tasks", [])),
            completed_tasks=set(session_data.get("completed_tasks", [])),
            metadata=session_data.get("metadata", {}),
        )

    def _load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Helper method to load JSON data from file (runs in thread executor)"""
        import json

        with open(file_path, "r") as f:
            return json.load(f)

    def _save_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Helper method to save JSON data to file (runs in thread executor)"""
        import json

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_operation_id(self) -> str:
        """Generate unique operation ID for tracking"""
        return f"session_{int(time.time())}_{str(uuid4())[:8]}"


# Export main classes
__all__ = [
    "SessionManager",
    "BrowserSession",
    "SessionConfiguration",
    "SessionStatus",
    "BrowserType",
    "SessionState",
    "SessionResources",
]
