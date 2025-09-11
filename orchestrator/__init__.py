"""
Browser-Use Agent Orchestration System

Comprehensive orchestration system for managing browser automation agents,
task distribution, session coordination, and results aggregation.

Author: Claude Code
Version: 1.0.0
Date: September 8, 2025
"""

from .agent_pool import BrowserAgentPool
from .coordinator import BrowserAgentCoordinator
from .queue_manager import TaskQueueManager
from .resource_manager import ResourceManager
from .session_manager import SessionManager

__all__ = [
    "BrowserAgentPool",
    "BrowserAgentCoordinator",
    "TaskQueueManager",
    "ResourceManager",
    "SessionManager",
]
