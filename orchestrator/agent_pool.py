"""
Browser Agent Pool Management System

Dynamic browser agent pool with scaling, health monitoring, and resource management.
Manages browser-use agents lifecycle, capacity, and availability.

Key Features:
- Dynamic agent scaling based on workload
- Health monitoring and automatic recovery
- Resource tracking and allocation
- Performance-based routing
- Local-only architecture compliance

Author: Claude Code
Version: 1.0.0
"""

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import structlog


class AgentStatus(Enum):
    """Browser agent operational status"""

    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"


class ScalingPolicy(Enum):
    """Agent pool scaling policies"""

    FIXED = "fixed"  # Fixed number of agents
    DYNAMIC = "dynamic"  # Scale based on queue size
    PERFORMANCE = "performance"  # Scale based on performance metrics
    HYBRID = "hybrid"  # Combination of queue and performance


@dataclass
class AgentCapabilities:
    """Browser agent capabilities definition"""

    browser_types: List[str] = field(default_factory=lambda: ["chrome", "chromium"])
    headless_modes: List[str] = field(default_factory=lambda: ["true", "false"])
    viewport_sizes: List[str] = field(default_factory=lambda: ["1920x1080", "1366x768"])
    javascript_enabled: bool = True
    cookies_enabled: bool = True
    local_storage_enabled: bool = True
    proxy_support: bool = False  # Local-only, no external proxies
    screenshot_support: bool = True
    pdf_generation: bool = True
    file_download: bool = True
    multi_tab_support: bool = True
    parallel_sessions: int = 3  # Max parallel browser sessions per agent


@dataclass
class AgentMetrics:
    """Agent performance and health metrics"""

    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_duration_ms: float = 0.0
    current_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    browser_sessions_active: int = 0
    browser_sessions_total: int = 0
    last_heartbeat: Optional[datetime] = None
    uptime_seconds: float = 0.0
    error_rate: float = 0.0

    def calculate_health_score(self) -> float:
        """Calculate composite health score (0.0 to 1.0)"""
        # Base score starts at 1.0
        health_score = 1.0

        # Deduct for high error rate
        health_score -= min(0.4, self.error_rate * 2)

        # Deduct for high memory usage (over 500MB)
        if self.current_memory_usage_mb > 500:
            health_score -= min(0.3, (self.current_memory_usage_mb - 500) / 1000)

        # Deduct for high CPU usage (over 80%)
        if self.cpu_usage_percent > 80:
            health_score -= min(0.2, (self.cpu_usage_percent - 80) / 100)

        # Deduct for missing recent heartbeat
        if self.last_heartbeat:
            time_since_heartbeat = (
                datetime.now() - self.last_heartbeat
            ).total_seconds()
            if time_since_heartbeat > 30:  # 30 seconds
                health_score -= min(0.3, time_since_heartbeat / 300)

        return max(0.0, min(1.0, health_score))


@dataclass
class BrowserAgent:
    """Browser agent instance representation"""

    agent_id: str
    process_id: Optional[int] = None
    process: Optional[Any] = None  # Store subprocess.Popen reference
    host: str = "localhost"
    port: int = 0
    status: AgentStatus = AgentStatus.INITIALIZING
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    current_tasks: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    browser_session_ids: Set[str] = field(default_factory=set)

    def can_accept_task(self) -> bool:
        """Check if agent can accept new task"""
        return (
            self.status in [AgentStatus.IDLE, AgentStatus.BUSY]
            and len(self.current_tasks) < self.max_concurrent_tasks
            and self.metrics.calculate_health_score() > 0.5
        )

    def get_load_percentage(self) -> float:
        """Get current load as percentage"""
        return (len(self.current_tasks) / self.max_concurrent_tasks) * 100


class BrowserAgentPool:
    """
    Dynamic browser agent pool manager with scaling and health monitoring.

    Manages lifecycle of browser-use agents including creation, monitoring,
    resource management, and automatic scaling based on workload.
    """

    def __init__(self, config: Dict[str, Any], storage_path: Optional[Path] = None):
        """
        Initialize browser agent pool.

        Args:
            config: Pool configuration settings
            storage_path: Local storage path for agent data
        """
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.config = config
        self.storage_path = storage_path or Path("./browser_agent_pool")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Agent management
        self.agents: Dict[str, BrowserAgent] = {}
        self.agent_startup_queue = asyncio.Queue()
        self.agent_shutdown_queue = asyncio.Queue()

        # Pool configuration
        self.min_agents = config.get("min_agents", 1)
        self.max_agents = config.get("max_agents", 10)
        self.scaling_policy = ScalingPolicy(config.get("scaling_policy", "dynamic"))
        self.scale_up_threshold = config.get(
            "scale_up_threshold", 0.8
        )  # 80% utilization
        self.scale_down_threshold = config.get(
            "scale_down_threshold", 0.3
        )  # 30% utilization

        # Health monitoring
        self.health_check_interval = config.get("health_check_interval", 30)  # seconds
        self.unhealthy_agent_threshold = config.get(
            "unhealthy_agent_threshold", 3
        )  # failures

        # Resource limits (local-only architecture)
        self.max_memory_per_agent_mb = config.get("max_memory_per_agent_mb", 1000)
        self.max_cpu_per_agent_percent = config.get("max_cpu_per_agent_percent", 90)
        self.max_browser_sessions_total = config.get("max_browser_sessions_total", 30)

        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.scaling_task: Optional[asyncio.Task] = None
        self.metrics_collector_task: Optional[asyncio.Task] = None

        # Statistics
        self.pool_stats = {
            "agents_created": 0,
            "agents_destroyed": 0,
            "scaling_events": 0,
            "health_interventions": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
        }

        self.logger.info(
            "Browser Agent Pool initialized",
            min_agents=self.min_agents,
            max_agents=self.max_agents,
            scaling_policy=self.scaling_policy.value,
            storage_path=str(self.storage_path),
        )

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize agent pool with minimum agents and start background tasks.

        Returns:
            Initialization result with pool status
        """
        operation_id = self._generate_operation_id()
        self.logger.info(f"[{operation_id}] Initializing browser agent pool")

        try:
            # Start background monitoring tasks
            await self._start_background_tasks()

            # Create minimum number of agents
            initial_agents = []
            for i in range(self.min_agents):
                agent = await self._create_agent(f"initial_{i}")
                initial_agents.append(agent)

            self.logger.info(
                f"[{operation_id}] Browser agent pool initialized successfully",
                agents_created=len(initial_agents),
                pool_size=len(self.agents),
            )

            return {
                "success": True,
                "operation_id": operation_id,
                "agents_created": len(initial_agents),
                "pool_size": len(self.agents),
                "min_agents": self.min_agents,
                "max_agents": self.max_agents,
                "scaling_policy": self.scaling_policy.value,
            }

        except Exception as error:
            self.logger.error(
                f"[{operation_id}] Agent pool initialization failed",
                error=str(error),
                exc_info=True,
            )
            raise

    async def get_available_agent(
        self, task_requirements: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Get available agent ID that can handle the task.

        Args:
            task_requirements: Optional task-specific requirements

        Returns:
            Agent ID if available, None otherwise
        """
        available_agents = []

        for agent_id, agent in self.agents.items():
            if agent.can_accept_task():
                # Check task requirements if provided
                if task_requirements:
                    if not self._agent_meets_requirements(agent, task_requirements):
                        continue

                available_agents.append((agent_id, agent))

        if not available_agents:
            # Try to trigger scaling if possible
            await self._evaluate_scaling_need()
            return None

        # Select best agent based on current load and health score
        best_agent = min(
            available_agents,
            key=lambda x: (
                x[1].get_load_percentage(),
                -x[1].metrics.calculate_health_score(),
            ),
        )

        return best_agent[0]

    async def assign_task_to_agent(self, agent_id: str, task_id: str) -> bool:
        """
        Assign task to specific agent.

        Args:
            agent_id: Target agent ID
            task_id: Task identifier

        Returns:
            True if assignment successful, False otherwise
        """
        agent = self.agents.get(agent_id)
        if not agent or not agent.can_accept_task():
            return False

        agent.current_tasks.add(task_id)
        agent.last_activity = datetime.now()

        # Update agent status
        if len(agent.current_tasks) >= agent.max_concurrent_tasks:
            agent.status = AgentStatus.BUSY
        elif agent.status == AgentStatus.IDLE:
            agent.status = AgentStatus.BUSY

        self.logger.debug(
            f"Assigned task {task_id} to agent {agent_id}",
            current_load=agent.get_load_percentage(),
            total_tasks=len(agent.current_tasks),
        )

        return True

    async def complete_task_on_agent(
        self, agent_id: str, task_id: str, success: bool = True
    ) -> bool:
        """
        Mark task as completed on agent.

        Args:
            agent_id: Agent ID
            task_id: Task identifier
            success: Whether task completed successfully

        Returns:
            True if completion recorded, False otherwise
        """
        agent = self.agents.get(agent_id)
        if not agent or task_id not in agent.current_tasks:
            return False

        # Remove task from current tasks
        agent.current_tasks.remove(task_id)
        agent.last_activity = datetime.now()

        # Update metrics
        if success:
            agent.metrics.tasks_completed += 1
        else:
            agent.metrics.tasks_failed += 1

        # Update error rate
        total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed
        agent.metrics.error_rate = agent.metrics.tasks_failed / max(1, total_tasks)

        # Update agent status
        if len(agent.current_tasks) == 0:
            agent.status = AgentStatus.IDLE
        elif len(agent.current_tasks) < agent.max_concurrent_tasks:
            agent.status = AgentStatus.BUSY

        # Update pool statistics
        if success:
            self.pool_stats["total_tasks_completed"] += 1
        else:
            self.pool_stats["total_tasks_failed"] += 1

        self.logger.debug(
            f"Completed task {task_id} on agent {agent_id}",
            success=success,
            remaining_tasks=len(agent.current_tasks),
            error_rate=agent.metrics.error_rate,
        )

        return True

    async def get_pool_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pool status and statistics.

        Returns:
            Pool status information
        """
        total_capacity = sum(
            agent.max_concurrent_tasks for agent in self.agents.values()
        )
        current_load = sum(len(agent.current_tasks) for agent in self.agents.values())
        utilization = (current_load / max(1, total_capacity)) * 100

        # Calculate health statistics
        healthy_agents = sum(
            1
            for agent in self.agents.values()
            if agent.metrics.calculate_health_score() > 0.7
        )

        agent_status_counts = {}
        for status in AgentStatus:
            agent_status_counts[status.value] = sum(
                1 for agent in self.agents.values() if agent.status == status
            )

        # Resource usage summary
        total_memory_mb = sum(
            agent.metrics.current_memory_usage_mb for agent in self.agents.values()
        )
        avg_cpu_percent = sum(
            agent.metrics.cpu_usage_percent for agent in self.agents.values()
        ) / max(1, len(self.agents))

        total_browser_sessions = sum(
            len(agent.browser_session_ids) for agent in self.agents.values()
        )

        return {
            "pool_size": len(self.agents),
            "min_agents": self.min_agents,
            "max_agents": self.max_agents,
            "utilization_percent": utilization,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "healthy_agents": healthy_agents,
            "agent_status": agent_status_counts,
            "resource_usage": {
                "total_memory_mb": total_memory_mb,
                "average_cpu_percent": avg_cpu_percent,
                "browser_sessions_active": total_browser_sessions,
                "browser_sessions_limit": self.max_browser_sessions_total,
            },
            "scaling_policy": self.scaling_policy.value,
            "statistics": self.pool_stats.copy(),
            "timestamp": datetime.now().isoformat(),
        }

    async def scale_pool(
        self, target_size: int, reason: str = "manual"
    ) -> Dict[str, Any]:
        """
        Scale pool to target size.

        Args:
            target_size: Desired number of agents
            reason: Reason for scaling

        Returns:
            Scaling operation result
        """
        if target_size < self.min_agents:
            target_size = self.min_agents
        elif target_size > self.max_agents:
            target_size = self.max_agents

        current_size = len(self.agents)

        if target_size == current_size:
            return {"success": True, "action": "no_change", "pool_size": current_size}

        operation_id = self._generate_operation_id()
        self.logger.info(
            f"[{operation_id}] Scaling pool from {current_size} to {target_size}",
            reason=reason,
        )

        try:
            if target_size > current_size:
                # Scale up
                agents_to_create = target_size - current_size
                created_agents = []

                for i in range(agents_to_create):
                    agent = await self._create_agent(f"scale_up_{i}")
                    created_agents.append(agent.agent_id)

                action = "scale_up"
                details = {"agents_created": len(created_agents)}

            else:
                # Scale down
                agents_to_remove = current_size - target_size
                removed_agents = await self._remove_excess_agents(agents_to_remove)

                action = "scale_down"
                details = {"agents_removed": len(removed_agents)}

            self.pool_stats["scaling_events"] += 1

            self.logger.info(
                f"[{operation_id}] Pool scaling completed",
                action=action,
                old_size=current_size,
                new_size=len(self.agents),
                **details,
            )

            return {
                "success": True,
                "operation_id": operation_id,
                "action": action,
                "old_size": current_size,
                "new_size": len(self.agents),
                "reason": reason,
                **details,
            }

        except Exception as error:
            self.logger.error(
                f"[{operation_id}] Pool scaling failed", error=str(error), exc_info=True
            )
            return {
                "success": False,
                "operation_id": operation_id,
                "error": str(error),
                "old_size": current_size,
                "new_size": len(self.agents),
            }

    async def shutdown(self) -> None:
        """Gracefully shutdown agent pool and cleanup resources"""
        self.logger.info("Shutting down browser agent pool")

        # Cancel background tasks
        for task in [
            self.health_monitor_task,
            self.scaling_task,
            self.metrics_collector_task,
        ]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown all agents
        shutdown_tasks = []
        for agent_id in list(self.agents.keys()):
            task = asyncio.create_task(self._shutdown_agent(agent_id))
            shutdown_tasks.append(task)

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        # Save final statistics
        await self._save_pool_statistics()

        self.logger.info("Browser agent pool shutdown complete")

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    async def _start_background_tasks(self) -> None:
        """Start background monitoring and management tasks"""
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.scaling_task = asyncio.create_task(self._scaling_monitor_loop())
        self.metrics_collector_task = asyncio.create_task(
            self._metrics_collection_loop()
        )

        self.logger.debug("Background tasks started")

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Health monitor error", error=str(error))

    async def _scaling_monitor_loop(self) -> None:
        """Background scaling evaluation loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._evaluate_scaling_need()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Scaling monitor error", error=str(error))

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                await self._collect_agent_metrics()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Metrics collection error", error=str(error))

    async def _create_agent(self, agent_suffix: str = "") -> BrowserAgent:
        """Create new browser agent instance"""
        agent_id = f"browser_agent_{int(time.time())}_{agent_suffix}_{str(uuid4())[:8]}"

        # Create agent instance
        agent = BrowserAgent(
            agent_id=agent_id,
            host="localhost",
            port=self._find_available_port(),
            capabilities=AgentCapabilities(),
            max_concurrent_tasks=self.config.get("max_tasks_per_agent", 3),
        )

        # Start actual browser-use agent process
        try:
            import subprocess

            from browser_use import BrowserProfile

            # Create browser profile for this agent
            _profile = BrowserProfile(
                headless=self.config.get("headless", True),
                user_data_dir=f"/tmp/browser_agent_{agent_id}",
                viewport_size=(
                    self.config.get("viewport_width", 1280),
                    self.config.get("viewport_height", 720),
                ),
            )

            # Start browser instance in a separate process
            browser_cmd = [
                "python",
                "-c",
                f"""
import asyncio
from browser_use import Browser, BrowserProfile
from browser_use.agent.service import Agent
from browser_use.llm.openai.chat import ChatOpenAI
import json
import sys
import os

async def main():
    profile = BrowserProfile(
        headless={self.config.get("headless", True)},
        user_data_dir="/tmp/browser_agent_{agent_id}",
        viewport_size=({self.config.get("viewport_width", 1280)}, {self.config.get("viewport_height", 720)})
    )
    
    browser = Browser(profile=profile)
    await browser.start()
    
    # Keep process alive and responsive
    print(f"Browser agent {agent_id} started on port {agent.port}")
    sys.stdout.flush()
    
    # Simple event loop to keep the browser alive
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
""",
            ]

            # Start the browser process
            process = subprocess.Popen(
                browser_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(
                    os.environ,
                    **{
                        "DISPLAY": ":99",  # For headless environments
                        "BROWSER_AGENT_ID": agent_id,
                        "BROWSER_AGENT_PORT": str(agent.port),
                    },
                ),
            )

            # Wait a moment for process to start
            await asyncio.sleep(2)

            if process.poll() is None:  # Process is running
                agent.process_id = process.pid
                agent.status = AgentStatus.IDLE
                agent.process = process  # Store process reference

                self.logger.info(
                    f"Started browser agent process {agent.process_id} for agent {agent_id}"
                )
            else:
                # Process failed to start
                stdout, stderr = process.communicate()
                error_msg = f"Browser agent process failed to start: {stderr.decode()}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

        except Exception as error:
            self.logger.error(f"Failed to start browser agent {agent_id}: {error}")
            # Fallback to simulation mode
            agent.process_id = None
            agent.status = AgentStatus.ERROR

        # Register agent
        self.agents[agent_id] = agent
        self.pool_stats["agents_created"] += 1

        # Save agent configuration
        await self._save_agent_config(agent)

        self.logger.info(
            f"Created browser agent {agent_id}",
            port=agent.port,
            max_tasks=agent.max_concurrent_tasks,
        )

        return agent

    async def _shutdown_agent(self, agent_id: str) -> None:
        """Shutdown specific agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return

        agent.status = AgentStatus.SHUTTING_DOWN

        # Wait for current tasks to complete or timeout
        timeout = 30  # 30 seconds
        start_time = time.time()

        while agent.current_tasks and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)

        # Force shutdown if tasks still running
        if agent.current_tasks:
            self.logger.warning(
                f"Force shutting down agent {agent_id} with {len(agent.current_tasks)} running tasks"
            )

        # Actually terminate browser-use agent process
        try:
            if agent.process and agent.process_id:
                # First try graceful termination
                agent.process.terminate()

                # Wait for graceful shutdown
                try:
                    agent.process.wait(timeout=5)
                    self.logger.info(
                        f"Browser agent process {agent.process_id} terminated gracefully"
                    )
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    agent.process.kill()
                    agent.process.wait()
                    self.logger.warning(
                        f"Browser agent process {agent.process_id} force killed"
                    )

            elif agent.process_id:
                # Fallback to signal if no process reference
                import signal

                try:
                    os.kill(agent.process_id, signal.SIGTERM)
                    self.logger.info(
                        f"Sent SIGTERM to browser agent process {agent.process_id}"
                    )
                except ProcessLookupError:
                    self.logger.warning(
                        f"Browser agent process {agent.process_id} already terminated"
                    )

        except Exception as error:
            self.logger.error(
                f"Failed to terminate browser agent process {agent.process_id}: {error}"
            )

        # Remove from pool
        agent.status = AgentStatus.OFFLINE
        del self.agents[agent_id]
        self.pool_stats["agents_destroyed"] += 1

        self.logger.info(f"Shutdown agent {agent_id}")

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all agents"""
        unhealthy_agents = []

        for agent_id, agent in self.agents.items():
            health_score = agent.metrics.calculate_health_score()

            if health_score < 0.3:  # Very unhealthy
                unhealthy_agents.append(agent_id)
                agent.status = AgentStatus.ERROR
                self.logger.warning(
                    f"Agent {agent_id} is unhealthy",
                    health_score=health_score,
                    error_rate=agent.metrics.error_rate,
                    memory_mb=agent.metrics.current_memory_usage_mb,
                )
            elif health_score < 0.6:  # Degraded
                agent.status = AgentStatus.DEGRADED

            # Update heartbeat
            agent.metrics.last_heartbeat = datetime.now()

        # Handle unhealthy agents
        for agent_id in unhealthy_agents:
            await self._handle_unhealthy_agent(agent_id)

    async def _handle_unhealthy_agent(self, agent_id: str) -> None:
        """Handle unhealthy agent with recovery or replacement"""
        agent = self.agents.get(agent_id)
        if not agent:
            return

        self.pool_stats["health_interventions"] += 1

        # If agent has no running tasks, restart it
        if not agent.current_tasks:
            self.logger.info(f"Restarting unhealthy agent {agent_id}")
            await self._shutdown_agent(agent_id)

            # Create replacement if below minimum
            if len(self.agents) < self.min_agents:
                await self._create_agent("health_replacement")
        else:
            # Mark for replacement after tasks complete
            self.logger.info(
                f"Marking agent {agent_id} for replacement after task completion"
            )

    async def _evaluate_scaling_need(self) -> None:
        """Evaluate if pool scaling is needed"""
        if self.scaling_policy == ScalingPolicy.FIXED:
            return

        current_size = len(self.agents)
        total_capacity = sum(
            agent.max_concurrent_tasks for agent in self.agents.values()
        )
        current_load = sum(len(agent.current_tasks) for agent in self.agents.values())

        if total_capacity == 0:
            return

        utilization = current_load / total_capacity

        # Scale up if utilization is high
        if utilization > self.scale_up_threshold and current_size < self.max_agents:
            target_size = min(self.max_agents, current_size + 1)
            await self.scale_pool(target_size, "high_utilization")

        # Scale down if utilization is low
        elif utilization < self.scale_down_threshold and current_size > self.min_agents:
            target_size = max(self.min_agents, current_size - 1)
            await self.scale_pool(target_size, "low_utilization")

    async def _remove_excess_agents(self, count: int) -> List[str]:
        """Remove excess agents, prioritizing idle and unhealthy ones"""
        # Select agents for removal (idle first, then lowest health score)
        removal_candidates = []

        for agent_id, agent in self.agents.items():
            if not agent.current_tasks:  # Idle agents first
                removal_candidates.append(
                    (agent_id, 0, agent.metrics.calculate_health_score())
                )
            else:
                removal_candidates.append(
                    (
                        agent_id,
                        len(agent.current_tasks),
                        agent.metrics.calculate_health_score(),
                    )
                )

        # Sort by task count (ascending) then health score (ascending)
        removal_candidates.sort(key=lambda x: (x[1], x[2]))

        removed_agents = []
        for i in range(min(count, len(removal_candidates))):
            agent_id = removal_candidates[i][0]
            await self._shutdown_agent(agent_id)
            removed_agents.append(agent_id)

        return removed_agents

    async def _collect_agent_metrics(self) -> None:
        """Collect metrics from all agents"""
        for agent_id, agent in self.agents.items():
            # Collect real metrics from browser-use agent
            try:
                if agent.process and agent.process_id and agent.process.poll() is None:
                    # Agent process is running, collect process metrics
                    import psutil

                    try:
                        # Get process information
                        process = psutil.Process(agent.process_id)

                        # Update basic metrics
                        agent.metrics.memory_usage_mb = (
                            process.memory_info().rss / 1024 / 1024
                        )
                        agent.metrics.cpu_usage_percent = process.cpu_percent()

                        # Update activity timestamp
                        agent.last_activity = datetime.now()

                        # Collect browser-specific metrics if available
                        await self._collect_browser_specific_metrics(agent)

                    except psutil.NoSuchProcess:
                        # Process no longer exists
                        self.logger.warning(
                            f"Agent {agent_id} process {agent.process_id} no longer exists"
                        )
                        agent.status = AgentStatus.ERROR
                    except psutil.AccessDenied:
                        # Cannot access process info
                        self.logger.warning(
                            f"Access denied to agent {agent_id} process metrics"
                        )

                elif agent.status == AgentStatus.ERROR or not agent.process_id:
                    # Agent in error state or no process
                    agent.metrics.memory_usage_mb = 0
                    agent.metrics.cpu_usage_percent = 0

                # Update derived metrics
                self._update_derived_metrics(agent)

            except Exception as error:
                self.logger.error(
                    f"Failed to collect metrics for agent {agent_id}: {error}"
                )
                agent.status = AgentStatus.ERROR

    async def _collect_browser_specific_metrics(self, agent: BrowserAgent) -> None:
        """Collect browser-specific metrics from agent"""
        try:
            # Check if agent has active browser sessions
            if agent.browser_session_ids:
                # Update browser session count
                agent.metrics.browser_sessions_active = len(agent.browser_session_ids)

                # Try to collect browser-specific metrics via HTTP if agent exposes them
                import aiohttp

                try:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as session:
                        metrics_url = f"http://{agent.host}:{agent.port}/metrics"
                        async with session.get(metrics_url) as response:
                            if response.status == 200:
                                metrics_data = await response.json()

                                # Update agent metrics with browser-specific data
                                if "memory_usage_mb" in metrics_data:
                                    agent.metrics.memory_usage_mb = metrics_data[
                                        "memory_usage_mb"
                                    ]
                                if "cpu_usage_percent" in metrics_data:
                                    agent.metrics.cpu_usage_percent = metrics_data[
                                        "cpu_usage_percent"
                                    ]
                                if "browser_sessions" in metrics_data:
                                    agent.metrics.browser_sessions_active = (
                                        metrics_data["browser_sessions"]
                                    )
                                if "task_execution_time_ms" in metrics_data:
                                    agent.metrics.average_task_time_ms = metrics_data[
                                        "task_execution_time_ms"
                                    ]

                except (aiohttp.ClientError, asyncio.TimeoutError):
                    # Agent doesn't expose HTTP metrics, use process metrics only
                    pass
            else:
                agent.metrics.browser_sessions_active = 0

        except Exception as error:
            self.logger.warning(
                f"Failed to collect browser-specific metrics for agent {agent.agent_id}: {error}"
            )

    def _update_derived_metrics(self, agent: BrowserAgent) -> None:
        """Update derived metrics from collected data"""
        # Update uptime
        agent.metrics.uptime_seconds = (
            datetime.now() - agent.created_at
        ).total_seconds()

        # Update task-related metrics
        agent.metrics.current_tasks_count = len(agent.current_tasks)

        # Calculate health score based on current metrics
        agent.metrics.health_score = self._calculate_agent_health_score(agent)

        # Update activity status
        if agent.last_activity:
            inactive_seconds = (datetime.now() - agent.last_activity).total_seconds()
            agent.metrics.last_activity_seconds_ago = inactive_seconds

    def _calculate_agent_health_score(self, agent: BrowserAgent) -> float:
        """Calculate overall health score for agent"""
        score = 1.0

        # Penalize high resource usage
        if agent.metrics.memory_usage_mb > 1000:  # More than 1GB
            score *= 0.8
        if agent.metrics.cpu_usage_percent > 80:  # High CPU usage
            score *= 0.7

        # Penalize inactive agents
        if (
            hasattr(agent.metrics, "last_activity_seconds_ago")
            and agent.metrics.last_activity_seconds_ago > 300
        ):  # 5 minutes
            score *= 0.9

        # Factor in error state
        if agent.status == AgentStatus.ERROR:
            score *= 0.1
        elif agent.status == AgentStatus.OFFLINE:
            score = 0.0

        return max(0.0, min(1.0, score))

    def _agent_meets_requirements(
        self, agent: BrowserAgent, requirements: Dict
    ) -> bool:
        """Check if agent meets task requirements"""
        # Check browser type requirement
        if "browser_type" in requirements:
            if requirements["browser_type"] not in agent.capabilities.browser_types:
                return False

        # Check headless requirement
        if "headless" in requirements:
            headless_str = str(requirements["headless"]).lower()
            if headless_str not in agent.capabilities.headless_modes:
                return False

        # Check viewport size requirement
        if "viewport_size" in requirements:
            if requirements["viewport_size"] not in agent.capabilities.viewport_sizes:
                return False

        # Check other capabilities
        capability_checks = {
            "javascript": "javascript_enabled",
            "cookies": "cookies_enabled",
            "local_storage": "local_storage_enabled",
            "screenshots": "screenshot_support",
            "pdf_generation": "pdf_generation",
            "file_download": "file_download",
        }

        for req_key, cap_attr in capability_checks.items():
            if req_key in requirements:
                if not getattr(agent.capabilities, cap_attr, False):
                    return False

        return True

    def _find_available_port(self) -> int:
        """Find available port for agent (placeholder)"""
        # TODO: Implement actual port finding logic
        return 8000 + len(self.agents)

    async def _save_agent_config(self, agent: BrowserAgent) -> None:
        """Save agent configuration to local storage"""
        config_file = self.storage_path / f"{agent.agent_id}_config.json"

        config_data = {
            "agent_id": agent.agent_id,
            "created_at": agent.created_at.isoformat(),
            "host": agent.host,
            "port": agent.port,
            "capabilities": {
                "browser_types": agent.capabilities.browser_types,
                "headless_modes": agent.capabilities.headless_modes,
                "viewport_sizes": agent.capabilities.viewport_sizes,
                "javascript_enabled": agent.capabilities.javascript_enabled,
                "cookies_enabled": agent.capabilities.cookies_enabled,
                "local_storage_enabled": agent.capabilities.local_storage_enabled,
                "screenshot_support": agent.capabilities.screenshot_support,
                "pdf_generation": agent.capabilities.pdf_generation,
                "file_download": agent.capabilities.file_download,
                "multi_tab_support": agent.capabilities.multi_tab_support,
                "parallel_sessions": agent.capabilities.parallel_sessions,
            },
            "max_concurrent_tasks": agent.max_concurrent_tasks,
        }

        try:
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
        except Exception as error:
            self.logger.error(f"Failed to save agent config: {error}")

    async def _save_pool_statistics(self) -> None:
        """Save pool statistics to local storage"""
        stats_file = self.storage_path / "pool_statistics.json"

        stats_data = {
            "timestamp": datetime.now().isoformat(),
            "pool_stats": self.pool_stats,
            "final_pool_size": len(self.agents),
            "configuration": {
                "min_agents": self.min_agents,
                "max_agents": self.max_agents,
                "scaling_policy": self.scaling_policy.value,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold,
            },
        }

        try:
            with open(stats_file, "w") as f:
                json.dump(stats_data, f, indent=2)
        except Exception as error:
            self.logger.error(f"Failed to save pool statistics: {error}")

    def _generate_operation_id(self) -> str:
        """Generate unique operation ID for tracking"""
        return f"pool_{int(time.time())}_{str(uuid4())[:8]}"


# Export main class
__all__ = [
    "BrowserAgentPool",
    "BrowserAgent",
    "AgentStatus",
    "AgentCapabilities",
    "AgentMetrics",
    "ScalingPolicy",
]
