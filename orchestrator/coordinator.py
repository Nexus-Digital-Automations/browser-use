"""
Browser-Use Agent Orchestration Coordinator

Main coordination system that integrates agent pool, task queue, session management,
and results aggregation for comprehensive browser automation orchestration.

Key Features:
- Complete workflow orchestration from task creation to completion
- Intelligent agent assignment and resource allocation
- Browser session lifecycle coordination
- Results aggregation and error recovery
- Performance monitoring and optimization
- Integration with existing AIgent multi-agent architecture

Author: Claude Code
Version: 1.0.0
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

from .agent_pool import BrowserAgentPool
from .queue_manager import BrowserTask, TaskPriority, TaskQueueManager, TaskType
from .session_manager import BrowserSession, SessionManager


class OrchestrationStatus(Enum):
    """Orchestration workflow status"""

    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class WorkflowStatus(Enum):
    """Individual workflow execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class OrchestrationMetrics:
    """Orchestration performance metrics"""

    workflows_total: int = 0
    workflows_completed: int = 0
    workflows_failed: int = 0
    workflows_cancelled: int = 0

    average_workflow_duration_ms: float = 0.0
    average_queue_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0

    agent_utilization_percent: float = 0.0
    session_utilization_percent: float = 0.0
    queue_utilization_percent: float = 0.0

    tasks_per_minute: float = 0.0
    error_rate: float = 0.0
    retry_rate: float = 0.0


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""

    workflow_id: str
    task_id: str
    agent_id: str
    session_id: str
    status: WorkflowStatus

    # Execution details
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None

    # Results and data
    result_data: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    screenshots: List[str] = field(default_factory=list)
    downloads: List[str] = field(default_factory=list)

    # Performance metrics
    queue_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    network_requests: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationRequest:
    """Browser automation orchestration request"""

    request_id: str
    task_type: TaskType
    priority: TaskPriority
    action: str
    target_url: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 300000
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class BrowserAgentCoordinator:
    """
    Main orchestration coordinator for browser-use agents.

    Integrates agent pool management, task queuing, session coordination,
    and results aggregation to provide comprehensive browser automation
    orchestration capabilities.
    """

    def __init__(self, config: Dict[str, Any], storage_path: Optional[Path] = None):
        """
        Initialize browser agent coordinator.

        Args:
            config: Coordinator configuration
            storage_path: Local storage path for orchestration data
        """
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.config = config
        self.storage_path = storage_path or Path("./browser_orchestration")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Core components
        self.agent_pool: Optional[BrowserAgentPool] = None
        self.task_queue: Optional[TaskQueueManager] = None
        self.session_manager: Optional[SessionManager] = None

        # Orchestration state
        self.status = OrchestrationStatus.INITIALIZING
        self.active_workflows: Dict[str, WorkflowResult] = {}
        self.completed_workflows: Dict[str, WorkflowResult] = {}
        self.workflow_history: List[WorkflowResult] = []

        # Configuration
        self.max_concurrent_workflows = config.get("max_concurrent_workflows", 20)
        self.workflow_timeout_ms = config.get(
            "workflow_timeout_ms", 600000
        )  # 10 minutes
        self.results_retention_hours = config.get("results_retention_hours", 24)
        self.auto_scale_agents = config.get("auto_scale_agents", True)
        self.auto_cleanup_sessions = config.get("auto_cleanup_sessions", True)

        # Performance monitoring
        self.metrics = OrchestrationMetrics()
        self.performance_samples: List[Dict] = []

        # Background tasks
        self.orchestration_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None

        # Event handlers
        self.workflow_event_handlers: Dict[str, List] = {
            "workflow_started": [],
            "workflow_completed": [],
            "workflow_failed": [],
            "workflow_cancelled": [],
        }

        self.logger.info(
            "Browser Agent Coordinator initialized",
            max_concurrent_workflows=self.max_concurrent_workflows,
            storage_path=str(self.storage_path),
        )

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize orchestration coordinator and all sub-components.

        Returns:
            Initialization result with component status
        """
        operation_id = self._generate_operation_id()
        self.logger.info(
            f"[{operation_id}] Initializing browser orchestration coordinator"
        )

        try:
            # Initialize agent pool
            agent_pool_config = self.config.get("agent_pool", {})
            self.agent_pool = BrowserAgentPool(
                agent_pool_config, self.storage_path / "agent_pool"
            )
            await self.agent_pool.initialize()

            # Initialize task queue
            queue_config = self.config.get("task_queue", {})
            self.task_queue = TaskQueueManager(
                queue_config, self.storage_path / "task_queue"
            )
            await self.task_queue.initialize()

            # Initialize session manager
            session_config = self.config.get("session_manager", {})
            self.session_manager = SessionManager(
                session_config, self.storage_path / "sessions"
            )
            await self.session_manager.initialize()

            # Start background orchestration
            await self._start_background_tasks()

            # Load persisted workflows if available
            await self._load_persisted_workflows()

            # Update status
            self.status = OrchestrationStatus.READY

            self.logger.info(
                f"[{operation_id}] Browser orchestration coordinator initialized successfully",
                agent_pool_size=len(self.agent_pool.agents),
                queue_size=len(self.task_queue.tasks),
                sessions=len(self.session_manager.sessions),
            )

            return {
                "success": True,
                "operation_id": operation_id,
                "status": self.status.value,
                "components": {
                    "agent_pool": {
                        "initialized": True,
                        "agents": len(self.agent_pool.agents),
                    },
                    "task_queue": {
                        "initialized": True,
                        "queued_tasks": sum(
                            len(q) for q in self.task_queue.priority_queues.values()
                        ),
                    },
                    "session_manager": {
                        "initialized": True,
                        "active_sessions": len(self.session_manager.sessions),
                    },
                },
            }

        except Exception as error:
            self.status = OrchestrationStatus.ERROR
            self.logger.error(
                f"[{operation_id}] Orchestration coordinator initialization failed",
                error=str(error),
                exc_info=True,
            )
            raise

    async def submit_workflow(self, request: OrchestrationRequest) -> Dict[str, Any]:
        """
        Submit browser automation workflow for orchestration.

        Args:
            request: Orchestration request

        Returns:
            Workflow submission result
        """
        if self.status != OrchestrationStatus.READY:
            return {
                "success": False,
                "reason": f"Orchestrator not ready: {self.status.value}",
                "status": self.status.value,
            }

        # Check capacity limits
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            return {
                "success": False,
                "reason": "Maximum concurrent workflows reached",
                "active_workflows": len(self.active_workflows),
                "max_workflows": self.max_concurrent_workflows,
            }

        workflow_id = f"workflow_{int(time.time())}_{str(uuid4())[:8]}"

        # Create browser task
        task = BrowserTask(
            task_id=f"task_{workflow_id}",
            task_type=request.task_type,
            priority=request.priority,
            action=request.action,
            target_url=request.target_url,
            parameters=request.parameters,
            requirements=request.requirements,
            timeout_ms=request.timeout_ms,
            metadata=TaskMetadata(max_retries=request.retry_count),
        )

        # Create workflow result tracking
        workflow_result = WorkflowResult(
            workflow_id=workflow_id,
            task_id=task.task_id,
            agent_id="",  # Will be assigned later
            session_id="",  # Will be assigned later
            status=WorkflowStatus.PENDING,
            created_at=datetime.now(),
            metadata=request.metadata,
        )

        try:
            # Submit task to queue
            enqueue_result = await self.task_queue.enqueue_task(task)
            if not enqueue_result["success"]:
                return {
                    "success": False,
                    "reason": f"Task enqueue failed: {enqueue_result['reason']}",
                    "workflow_id": workflow_id,
                }

            # Track active workflow
            self.active_workflows[workflow_id] = workflow_result

            # Update metrics
            self.metrics.workflows_total += 1

            # Persist workflow
            await self._persist_workflow(workflow_result)

            self.logger.info(
                f"Submitted workflow {workflow_id}",
                task_type=request.task_type.value,
                priority=request.priority.name,
                queue_position=enqueue_result.get("queue_position"),
            )

            return {
                "success": True,
                "workflow_id": workflow_id,
                "task_id": task.task_id,
                "queue_position": enqueue_result.get("queue_position"),
                "estimated_wait_time_ms": enqueue_result.get("estimated_wait_time_ms"),
                "created_at": workflow_result.created_at.isoformat(),
            }

        except Exception as error:
            # Cleanup on failure
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

            self.logger.error(
                f"Failed to submit workflow {workflow_id}",
                error=str(error),
                exc_info=True,
            )

            return {
                "success": False,
                "reason": f"Workflow submission failed: {str(error)}",
                "workflow_id": workflow_id,
                "error": str(error),
            }

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow status information or None if not found
        """
        # Check active workflows
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]

            # Get task status
            task = self.task_queue.tasks.get(workflow.task_id)
            task_status = task.status.value if task else "unknown"

            return {
                "workflow_id": workflow_id,
                "status": workflow.status.value,
                "task_status": task_status,
                "agent_id": workflow.agent_id,
                "session_id": workflow.session_id,
                "created_at": workflow.created_at.isoformat(),
                "started_at": (
                    workflow.started_at.isoformat() if workflow.started_at else None
                ),
                "completed_at": (
                    workflow.completed_at.isoformat() if workflow.completed_at else None
                ),
                "duration_ms": workflow.duration_ms,
                "queue_time_ms": workflow.queue_time_ms,
                "execution_time_ms": workflow.execution_time_ms,
                "result_available": workflow.result_data is not None,
                "error_info": workflow.error_info,
                "metadata": workflow.metadata,
            }

        # Check completed workflows
        if workflow_id in self.completed_workflows:
            workflow = self.completed_workflows[workflow_id]

            return {
                "workflow_id": workflow_id,
                "status": workflow.status.value,
                "agent_id": workflow.agent_id,
                "session_id": workflow.session_id,
                "created_at": workflow.created_at.isoformat(),
                "started_at": (
                    workflow.started_at.isoformat() if workflow.started_at else None
                ),
                "completed_at": (
                    workflow.completed_at.isoformat() if workflow.completed_at else None
                ),
                "duration_ms": workflow.duration_ms,
                "queue_time_ms": workflow.queue_time_ms,
                "execution_time_ms": workflow.execution_time_ms,
                "result_available": workflow.result_data is not None,
                "screenshots": workflow.screenshots,
                "downloads": workflow.downloads,
                "memory_usage_mb": workflow.memory_usage_mb,
                "network_requests": workflow.network_requests,
                "error_info": workflow.error_info,
                "metadata": workflow.metadata,
            }

        return None

    async def get_workflow_result(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete workflow execution result.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Complete workflow result or None if not found/not completed
        """
        workflow = self.completed_workflows.get(workflow_id)
        if not workflow or workflow.status != WorkflowStatus.COMPLETED:
            return None

        return {
            "workflow_id": workflow_id,
            "success": True,
            "result_data": workflow.result_data,
            "execution_details": {
                "agent_id": workflow.agent_id,
                "session_id": workflow.session_id,
                "duration_ms": workflow.duration_ms,
                "queue_time_ms": workflow.queue_time_ms,
                "execution_time_ms": workflow.execution_time_ms,
                "memory_usage_mb": workflow.memory_usage_mb,
                "network_requests": workflow.network_requests,
            },
            "artifacts": {
                "screenshots": workflow.screenshots,
                "downloads": workflow.downloads,
            },
            "metadata": workflow.metadata,
            "completed_at": workflow.completed_at.isoformat(),
        }

    async def cancel_workflow(
        self, workflow_id: str, reason: str = "User requested"
    ) -> Dict[str, Any]:
        """
        Cancel active workflow.

        Args:
            workflow_id: Workflow identifier
            reason: Cancellation reason

        Returns:
            Cancellation result
        """
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return {
                "success": False,
                "reason": "Workflow not found or not active",
                "workflow_id": workflow_id,
            }

        try:
            # Cancel task in queue
            if workflow.task_id:
                await self.task_queue.cancel_task(workflow.task_id, reason)

            # Update workflow status
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.now()
            workflow.error_info = {
                "reason": reason,
                "cancelled_at": datetime.now().isoformat(),
            }

            if workflow.started_at:
                workflow.duration_ms = (
                    workflow.completed_at - workflow.started_at
                ).total_seconds() * 1000

            # Move to completed workflows
            self.completed_workflows[workflow_id] = workflow
            del self.active_workflows[workflow_id]

            # Update metrics
            self.metrics.workflows_cancelled += 1

            # Notify event handlers
            await self._notify_workflow_event("workflow_cancelled", workflow)

            self.logger.info(f"Cancelled workflow {workflow_id}", reason=reason)

            return {
                "success": True,
                "workflow_id": workflow_id,
                "cancelled_at": datetime.now().isoformat(),
                "reason": reason,
            }

        except Exception as error:
            self.logger.error(
                f"Failed to cancel workflow {workflow_id}",
                error=str(error),
                exc_info=True,
            )

            return {
                "success": False,
                "reason": f"Cancellation failed: {str(error)}",
                "workflow_id": workflow_id,
                "error": str(error),
            }

    async def get_orchestration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive orchestration status and metrics.

        Returns:
            Complete orchestration status
        """
        # Get component status
        agent_pool_status = (
            await self.agent_pool.get_pool_status() if self.agent_pool else {}
        )
        queue_status = (
            await self.task_queue.get_queue_status() if self.task_queue else {}
        )
        session_status = (
            await self.session_manager.get_session_status()
            if self.session_manager
            else {}
        )

        # Calculate utilization metrics
        self._update_utilization_metrics(
            agent_pool_status, queue_status, session_status
        )

        # Workflow statistics
        workflow_stats = {
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "total_workflows": self.metrics.workflows_total,
            "success_rate": (
                self.metrics.workflows_completed / max(1, self.metrics.workflows_total)
            ),
            "error_rate": self.metrics.error_rate,
            "retry_rate": self.metrics.retry_rate,
        }

        return {
            "orchestration_status": self.status.value,
            "capacity": {
                "max_concurrent_workflows": self.max_concurrent_workflows,
                "active_workflows": len(self.active_workflows),
                "utilization_percent": (
                    len(self.active_workflows) / self.max_concurrent_workflows
                )
                * 100,
            },
            "workflow_statistics": workflow_stats,
            "performance_metrics": {
                "average_workflow_duration_ms": self.metrics.average_workflow_duration_ms,
                "average_queue_time_ms": self.metrics.average_queue_time_ms,
                "average_execution_time_ms": self.metrics.average_execution_time_ms,
                "tasks_per_minute": self.metrics.tasks_per_minute,
                "agent_utilization_percent": self.metrics.agent_utilization_percent,
                "session_utilization_percent": self.metrics.session_utilization_percent,
                "queue_utilization_percent": self.metrics.queue_utilization_percent,
            },
            "component_status": {
                "agent_pool": agent_pool_status,
                "task_queue": queue_status,
                "session_manager": session_status,
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown orchestration coordinator"""
        self.logger.info("Shutting down browser orchestration coordinator")
        self.status = OrchestrationStatus.SHUTDOWN

        # Cancel background tasks
        for task in [self.orchestration_task, self.monitoring_task, self.cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Cancel active workflows
        cancellation_tasks = []
        for workflow_id in list(self.active_workflows.keys()):
            task = asyncio.create_task(
                self.cancel_workflow(workflow_id, "System shutdown")
            )
            cancellation_tasks.append(task)

        if cancellation_tasks:
            await asyncio.gather(*cancellation_tasks, return_exceptions=True)

        # Shutdown components
        if self.session_manager:
            await self.session_manager.shutdown()

        if self.task_queue:
            await self.task_queue.shutdown()

        if self.agent_pool:
            await self.agent_pool.shutdown()

        # Save final state
        await self._persist_orchestration_state()

        self.logger.info("Browser orchestration coordinator shutdown complete")

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    async def _start_background_tasks(self) -> None:
        """Start background orchestration tasks"""
        self.orchestration_task = asyncio.create_task(self._orchestration_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.debug("Background orchestration tasks started")

    async def _orchestration_loop(self) -> None:
        """Main orchestration coordination loop"""
        while True:
            try:
                await asyncio.sleep(1)  # Process every second

                if self.status == OrchestrationStatus.READY:
                    await self._process_pending_workflows()

            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Orchestration loop error", error=str(error))
                self.status = OrchestrationStatus.DEGRADED

    async def _monitoring_loop(self) -> None:
        """Background monitoring and metrics collection loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                await self._collect_performance_metrics()
                await self._monitor_workflow_health()

            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Monitoring loop error", error=str(error))

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_completed_workflows()

                if self.auto_cleanup_sessions:
                    await self.session_manager.cleanup_idle_sessions()

            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Cleanup loop error", error=str(error))

    async def _process_pending_workflows(self) -> None:
        """Process pending workflows and coordinate execution"""
        # Get next available task from queue
        task = await self.task_queue.get_next_task()
        if not task:
            return

        # Find workflow for this task
        workflow = None
        for wf in self.active_workflows.values():
            if wf.task_id == task.task_id:
                workflow = wf
                break

        if not workflow:
            self.logger.warning(f"No workflow found for task {task.task_id}")
            return

        try:
            # Get available agent
            agent_id = await self.agent_pool.get_available_agent(task.requirements)
            if not agent_id:
                # No agents available, put task back
                return

            # Get or create browser session
            session = await self._get_or_create_session(agent_id, task.requirements)
            if not session:
                self.logger.warning(f"Failed to get session for agent {agent_id}")
                return

            # Assign task to agent and session
            await self.task_queue.assign_task(task.task_id, agent_id)
            await self.agent_pool.assign_task_to_agent(agent_id, task.task_id)
            await self.session_manager.assign_task_to_session(
                session.session_id, task.task_id
            )

            # Update workflow
            workflow.agent_id = agent_id
            workflow.session_id = session.session_id
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            workflow.queue_time_ms = (
                workflow.started_at - workflow.created_at
            ).total_seconds() * 1000

            # Start task execution
            await self.task_queue.start_task_execution(task.task_id)

            # Execute task (this would integrate with actual browser-use agent)
            await self._execute_workflow_task(workflow, task, session)

        except Exception as error:
            # Handle execution error
            await self._handle_workflow_error(workflow, error)

    async def _get_or_create_session(
        self, agent_id: str, requirements: Dict[str, Any]
    ) -> Optional[BrowserSession]:
        """Get existing or create new browser session for agent"""
        # Try to get available session
        session = await self.session_manager.get_available_session(
            agent_id, requirements
        )

        if session:
            return session

        # Create new session if needed
        session_config = None
        if requirements:
            from .session_manager import BrowserType, SessionConfiguration

            session_config = SessionConfiguration(
                browser_type=BrowserType(requirements.get("browser_type", "chrome")),
                headless=requirements.get("headless", True),
                viewport_width=requirements.get("viewport_width", 1920),
                viewport_height=requirements.get("viewport_height", 1080),
                disable_javascript=not requirements.get("javascript", True),
                disable_images=not requirements.get("images", True),
            )

        create_result = await self.session_manager.create_session(
            agent_id, session_config
        )

        if create_result["success"]:
            return await self.session_manager.get_session(create_result["session_id"])

        return None

    async def _execute_workflow_task(
        self, workflow: WorkflowResult, task: BrowserTask, session: BrowserSession
    ) -> None:
        """Execute browser automation task within workflow"""
        execution_start = time.time()

        try:
            # Import browser-use agent components
            from browser_use import BrowserSession as BUBrowserSession
            from browser_use.agent.service import Agent
            from browser_use.llm.openai.chat import ChatOpenAI

            # Convert our session to browser-use BrowserSession
            browser_session = BUBrowserSession(
                profile=session.configuration.profile, session_id=session.session_id
            )

            # Create browser-use agent with task details
            llm = ChatOpenAI(model=session.configuration.llm_model)
            agent = Agent(
                task=self._format_task_for_agent(task, workflow),
                llm=llm,
                browser_session=browser_session,
                context={"workflow_id": workflow.workflow_id, "task_id": task.task_id},
            )

            # Execute the browser automation task
            try:
                result_history = await agent.run(
                    max_steps=task.max_steps or 10,
                    on_step_start=lambda step: self._on_agent_step_start(
                        workflow, task, step
                    ),
                    on_step_end=lambda step: self._on_agent_step_end(
                        workflow, task, step
                    ),
                )

                # Extract results from agent history
                result_data = {
                    "success": True,
                    "action": task.action,
                    "url": task.target_url,
                    "screenshots": self._extract_screenshots_from_history(
                        result_history
                    ),
                    "data_extracted": self._extract_data_from_history(result_history),
                    "history_steps": (
                        len(result_history.history)
                        if hasattr(result_history, "history")
                        else 0
                    ),
                    "performance": {
                        "steps_executed": (
                            len(result_history.history)
                            if hasattr(result_history, "history")
                            else 0
                        ),
                        "agent_execution_successful": True,
                    },
                }

            except Exception as agent_error:
                # Handle browser-use agent execution errors
                self.logger.error(f"Browser-use agent execution failed: {agent_error}")
                result_data = {
                    "success": False,
                    "action": task.action,
                    "url": task.target_url,
                    "error": str(agent_error),
                    "error_type": type(agent_error).__name__,
                }

            # Complete workflow successfully
            await self._complete_workflow(workflow, result_data, execution_start)

        except Exception as error:
            await self._handle_workflow_error(workflow, error, execution_start)

    async def _complete_workflow(
        self,
        workflow: WorkflowResult,
        result_data: Dict[str, Any],
        execution_start: float,
    ) -> None:
        """Complete workflow successfully"""
        current_time = datetime.now()
        execution_time_ms = (time.time() - execution_start) * 1000

        # Update workflow result
        workflow.status = WorkflowStatus.COMPLETED
        workflow.completed_at = current_time
        workflow.duration_ms = (
            current_time - workflow.created_at
        ).total_seconds() * 1000
        workflow.execution_time_ms = execution_time_ms
        workflow.result_data = result_data

        # Extract artifacts
        if "screenshots" in result_data:
            workflow.screenshots = result_data["screenshots"]
        if "downloads" in result_data:
            workflow.downloads = result_data["downloads"]

        # Update resource usage
        if workflow.session_id:
            session = await self.session_manager.get_session(workflow.session_id)
            if session:
                workflow.memory_usage_mb = session.resources.memory_usage_mb
                workflow.network_requests = session.resources.network_requests_count

        # Complete task in queue
        if workflow.task_id:
            await self.task_queue.complete_task(workflow.task_id, result_data)

        # Complete task on agent and session
        if workflow.agent_id:
            await self.agent_pool.complete_task_on_agent(
                workflow.agent_id, workflow.task_id, True
            )

        if workflow.session_id:
            await self.session_manager.complete_task_on_session(
                workflow.session_id, workflow.task_id
            )

        # Move to completed workflows
        self.completed_workflows[workflow.workflow_id] = workflow
        del self.active_workflows[workflow.workflow_id]

        # Update metrics
        self.metrics.workflows_completed += 1
        self._update_average_workflow_duration(workflow.duration_ms)

        # Notify event handlers
        await self._notify_workflow_event("workflow_completed", workflow)

        # Persist result
        await self._persist_workflow(workflow)

        self.logger.info(
            f"Completed workflow {workflow.workflow_id}",
            duration_ms=workflow.duration_ms,
            execution_time_ms=execution_time_ms,
        )

    async def _handle_workflow_error(
        self,
        workflow: WorkflowResult,
        error: Exception,
        execution_start: Optional[float] = None,
    ) -> None:
        """Handle workflow execution error"""
        current_time = datetime.now()
        execution_time_ms = (
            (time.time() - execution_start) * 1000 if execution_start else 0
        )

        # Update workflow result
        workflow.status = WorkflowStatus.FAILED
        workflow.completed_at = current_time
        workflow.duration_ms = (
            current_time - workflow.created_at
        ).total_seconds() * 1000
        workflow.execution_time_ms = execution_time_ms
        workflow.error_info = {
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": current_time.isoformat(),
        }

        # Fail task in queue
        if workflow.task_id:
            await self.task_queue.fail_task(workflow.task_id, workflow.error_info)

        # Complete task on agent and session
        if workflow.agent_id:
            await self.agent_pool.complete_task_on_agent(
                workflow.agent_id, workflow.task_id, False
            )

        if workflow.session_id:
            await self.session_manager.complete_task_on_session(
                workflow.session_id, workflow.task_id
            )

        # Move to completed workflows
        self.completed_workflows[workflow.workflow_id] = workflow
        del self.active_workflows[workflow.workflow_id]

        # Update metrics
        self.metrics.workflows_failed += 1
        self.metrics.error_rate = self.metrics.workflows_failed / max(
            1, self.metrics.workflows_total
        )

        # Notify event handlers
        await self._notify_workflow_event("workflow_failed", workflow)

        # Persist result
        await self._persist_workflow(workflow)

        self.logger.error(
            f"Workflow {workflow.workflow_id} failed",
            error=str(error),
            duration_ms=workflow.duration_ms,
        )

    def _format_task_for_agent(
        self, task: BrowserTask, workflow: WorkflowResult
    ) -> str:
        """Format browser task for browser-use agent"""
        task_description = f"Browser automation task: {task.action}"

        if task.target_url:
            task_description += f"\nNavigate to: {task.target_url}"

        if task.parameters:
            if "query" in task.parameters:
                task_description += f"\nQuery/Search: {task.parameters['query']}"
            if "form_data" in task.parameters:
                task_description += (
                    f"\nFill form with data: {task.parameters['form_data']}"
                )
            if "element_selector" in task.parameters:
                task_description += (
                    f"\nFind element: {task.parameters['element_selector']}"
                )
            if "text_to_type" in task.parameters:
                task_description += f"\nType text: {task.parameters['text_to_type']}"

        task_description += f"\nWorkflow ID: {workflow.workflow_id}"
        task_description += f"\nTask ID: {task.task_id}"

        return task_description

    async def _on_agent_step_start(
        self, workflow: WorkflowResult, task: BrowserTask, step_data: Any
    ) -> None:
        """Handle agent step start event"""
        self.logger.debug(f"Agent step started for workflow {workflow.workflow_id}")

        # Update workflow metadata with step information
        if "steps" not in workflow.metadata:
            workflow.metadata["steps"] = []

        workflow.metadata["steps"].append(
            {
                "step_start": datetime.now().isoformat(),
                "step_data": str(step_data) if step_data else None,
            }
        )

    async def _on_agent_step_end(
        self, workflow: WorkflowResult, task: BrowserTask, step_data: Any
    ) -> None:
        """Handle agent step end event"""
        self.logger.debug(f"Agent step completed for workflow {workflow.workflow_id}")

        # Update the last step with completion info
        if "steps" in workflow.metadata and workflow.metadata["steps"]:
            last_step = workflow.metadata["steps"][-1]
            last_step["step_end"] = datetime.now().isoformat()
            last_step["step_result"] = str(step_data) if step_data else None

    def _extract_screenshots_from_history(self, result_history) -> List[str]:
        """Extract screenshot paths from browser-use agent history"""
        screenshots = []

        try:
            if hasattr(result_history, "history") and result_history.history:
                for step in result_history.history:
                    if hasattr(step, "screenshot") and step.screenshot:
                        screenshots.append(step.screenshot)
        except Exception as error:
            self.logger.warning(f"Failed to extract screenshots: {error}")

        return screenshots

    def _extract_data_from_history(self, result_history) -> Dict[str, Any]:
        """Extract data from browser-use agent history"""
        extracted_data = {
            "steps_completed": 0,
            "pages_visited": [],
            "actions_performed": [],
            "content_extracted": {},
        }

        try:
            if hasattr(result_history, "history") and result_history.history:
                extracted_data["steps_completed"] = len(result_history.history)

                for step in result_history.history:
                    if hasattr(step, "action") and step.action:
                        extracted_data["actions_performed"].append(str(step.action))

                    if hasattr(step, "url") and step.url:
                        if step.url not in extracted_data["pages_visited"]:
                            extracted_data["pages_visited"].append(step.url)

                    # Extract any structured data from the step
                    if hasattr(step, "result") and step.result:
                        step_content = str(step.result)
                        if step_content not in extracted_data.get(
                            "content_extracted", {}
                        ):
                            extracted_data["content_extracted"][
                                f"step_{len(extracted_data['content_extracted'])}"
                            ] = step_content

        except Exception as error:
            self.logger.warning(f"Failed to extract data from history: {error}")

        return extracted_data

    async def _monitor_workflow_health(self) -> None:
        """Monitor health of active workflows"""
        current_time = datetime.now()
        timeout_workflows = []

        for workflow_id, workflow in list(self.active_workflows.items()):
            # Check for workflow timeout
            if workflow.started_at:
                execution_time = (
                    current_time - workflow.started_at
                ).total_seconds() * 1000
                if execution_time > self.workflow_timeout_ms:
                    timeout_workflows.append(workflow_id)

        # Handle timeout workflows
        for workflow_id in timeout_workflows:
            workflow = self.active_workflows[workflow_id]
            await self._handle_workflow_timeout(workflow)

    async def _handle_workflow_timeout(self, workflow: WorkflowResult) -> None:
        """Handle workflow timeout"""
        self.logger.warning(f"Workflow {workflow.workflow_id} timed out")

        # Cancel task
        if workflow.task_id:
            await self.task_queue.cancel_task(workflow.task_id, "Workflow timeout")

        # Update workflow
        workflow.status = WorkflowStatus.TIMEOUT
        workflow.completed_at = datetime.now()
        workflow.duration_ms = (
            workflow.completed_at - workflow.created_at
        ).total_seconds() * 1000
        workflow.error_info = {
            "reason": "Workflow execution timeout",
            "timeout_ms": self.workflow_timeout_ms,
            "timestamp": datetime.now().isoformat(),
        }

        # Cleanup resources
        if workflow.agent_id:
            await self.agent_pool.complete_task_on_agent(
                workflow.agent_id, workflow.task_id, False
            )

        if workflow.session_id:
            await self.session_manager.complete_task_on_session(
                workflow.session_id, workflow.task_id
            )

        # Move to completed
        self.completed_workflows[workflow.workflow_id] = workflow
        del self.active_workflows[workflow.workflow_id]

        # Update metrics
        self.metrics.workflows_failed += 1

        await self._notify_workflow_event("workflow_failed", workflow)

    async def _collect_performance_metrics(self) -> None:
        """Collect and update performance metrics"""
        # Collect current performance data
        sample = {
            "timestamp": datetime.now().isoformat(),
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "agent_pool_utilization": 0,
            "session_utilization": 0,
            "queue_utilization": 0,
        }

        # Get component utilizations
        if self.agent_pool:
            pool_status = await self.agent_pool.get_pool_status()
            sample["agent_pool_utilization"] = pool_status.get("utilization_percent", 0)

        if self.session_manager:
            session_status = await self.session_manager.get_session_status()
            sample["session_utilization"] = session_status.get("utilization_percent", 0)

        if self.task_queue:
            queue_status = await self.task_queue.get_queue_status()
            capacity = queue_status.get("capacity", {})
            sample["queue_utilization"] = capacity.get("utilization_percent", 0)

        # Store sample
        self.performance_samples.append(sample)

        # Keep only recent samples (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.performance_samples = [
            s
            for s in self.performance_samples
            if datetime.fromisoformat(s["timestamp"]) > cutoff
        ]

        # Calculate tasks per minute
        if len(self.performance_samples) > 1:
            time_span_minutes = (
                datetime.fromisoformat(self.performance_samples[-1]["timestamp"])
                - datetime.fromisoformat(self.performance_samples[0]["timestamp"])
            ).total_seconds() / 60

            if time_span_minutes > 0:
                tasks_completed = self.performance_samples[-1]["completed_workflows"]
                tasks_started = self.performance_samples[0]["completed_workflows"]
                self.metrics.tasks_per_minute = (
                    tasks_completed - tasks_started
                ) / time_span_minutes

    def _update_utilization_metrics(
        self, agent_status: Dict, queue_status: Dict, session_status: Dict
    ) -> None:
        """Update utilization metrics from component status"""
        self.metrics.agent_utilization_percent = agent_status.get(
            "utilization_percent", 0
        )
        self.metrics.session_utilization_percent = session_status.get(
            "utilization_percent", 0
        )

        capacity = queue_status.get("capacity", {})
        self.metrics.queue_utilization_percent = capacity.get("utilization_percent", 0)

    def _update_average_workflow_duration(self, duration_ms: float) -> None:
        """Update average workflow duration metric"""
        if self.metrics.workflows_completed == 1:
            self.metrics.average_workflow_duration_ms = duration_ms
        else:
            # Running average
            total = self.metrics.workflows_completed
            current_avg = self.metrics.average_workflow_duration_ms
            self.metrics.average_workflow_duration_ms = (
                (current_avg * (total - 1)) + duration_ms
            ) / total

    async def _cleanup_completed_workflows(self) -> None:
        """Cleanup old completed workflows"""
        cutoff_time = datetime.now() - timedelta(hours=self.results_retention_hours)

        old_workflows = [
            workflow_id
            for workflow_id, workflow in self.completed_workflows.items()
            if workflow.completed_at and workflow.completed_at < cutoff_time
        ]

        for workflow_id in old_workflows:
            workflow = self.completed_workflows[workflow_id]
            self.workflow_history.append(workflow)
            del self.completed_workflows[workflow_id]

        if old_workflows:
            self.logger.info(f"Cleaned up {len(old_workflows)} old completed workflows")

    async def _notify_workflow_event(
        self, event_type: str, workflow: WorkflowResult
    ) -> None:
        """Notify registered workflow event handlers"""
        handlers = self.workflow_event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(workflow)
                else:
                    handler(workflow)
            except Exception as error:
                self.logger.error(f"Workflow event handler error: {error}")

    async def _load_persisted_workflows(self) -> None:
        """Load persisted workflows from storage"""
        workflows_file = self.storage_path / "workflows.json"
        if not workflows_file.exists():
            return

        try:
            with open(workflows_file, "r") as f:
                workflow_data = json.load(f)

            for workflow_dict in workflow_data:
                workflow = self._deserialize_workflow(workflow_dict)

                if workflow.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                    self.active_workflows[workflow.workflow_id] = workflow
                else:
                    self.completed_workflows[workflow.workflow_id] = workflow

            self.logger.info(f"Loaded {len(workflow_data)} persisted workflows")

        except Exception as error:
            self.logger.error(f"Failed to load persisted workflows: {error}")

    async def _persist_workflow(self, workflow: WorkflowResult) -> None:
        """Persist workflow to storage"""
        # TODO: Implement efficient workflow persistence
        pass

    async def _persist_orchestration_state(self) -> None:
        """Persist complete orchestration state"""
        state_file = self.storage_path / "orchestration_state.json"

        try:
            all_workflows = list(self.active_workflows.values()) + list(
                self.completed_workflows.values()
            )
            workflow_data = [self._serialize_workflow(wf) for wf in all_workflows]

            state_data = {
                "timestamp": datetime.now().isoformat(),
                "status": self.status.value,
                "metrics": {
                    "workflows_total": self.metrics.workflows_total,
                    "workflows_completed": self.metrics.workflows_completed,
                    "workflows_failed": self.metrics.workflows_failed,
                    "workflows_cancelled": self.metrics.workflows_cancelled,
                    "average_workflow_duration_ms": self.metrics.average_workflow_duration_ms,
                    "error_rate": self.metrics.error_rate,
                },
                "workflows": workflow_data,
            }

            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)

            self.logger.debug(
                f"Persisted orchestration state with {len(workflow_data)} workflows"
            )

        except Exception as error:
            self.logger.error(f"Failed to persist orchestration state: {error}")

    def _serialize_workflow(self, workflow: WorkflowResult) -> Dict[str, Any]:
        """Serialize workflow for persistence"""
        return {
            "workflow_id": workflow.workflow_id,
            "task_id": workflow.task_id,
            "agent_id": workflow.agent_id,
            "session_id": workflow.session_id,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "started_at": (
                workflow.started_at.isoformat() if workflow.started_at else None
            ),
            "completed_at": (
                workflow.completed_at.isoformat() if workflow.completed_at else None
            ),
            "duration_ms": workflow.duration_ms,
            "queue_time_ms": workflow.queue_time_ms,
            "execution_time_ms": workflow.execution_time_ms,
            "memory_usage_mb": workflow.memory_usage_mb,
            "network_requests": workflow.network_requests,
            "result_data": workflow.result_data,
            "error_info": workflow.error_info,
            "screenshots": workflow.screenshots,
            "downloads": workflow.downloads,
            "metadata": workflow.metadata,
        }

    def _deserialize_workflow(self, workflow_data: Dict[str, Any]) -> WorkflowResult:
        """Deserialize workflow from persistence data"""
        return WorkflowResult(
            workflow_id=workflow_data["workflow_id"],
            task_id=workflow_data["task_id"],
            agent_id=workflow_data["agent_id"],
            session_id=workflow_data["session_id"],
            status=WorkflowStatus(workflow_data["status"]),
            created_at=datetime.fromisoformat(workflow_data["created_at"]),
            started_at=(
                datetime.fromisoformat(workflow_data["started_at"])
                if workflow_data["started_at"]
                else None
            ),
            completed_at=(
                datetime.fromisoformat(workflow_data["completed_at"])
                if workflow_data["completed_at"]
                else None
            ),
            duration_ms=workflow_data.get("duration_ms"),
            queue_time_ms=workflow_data.get("queue_time_ms", 0),
            execution_time_ms=workflow_data.get("execution_time_ms", 0),
            memory_usage_mb=workflow_data.get("memory_usage_mb", 0),
            network_requests=workflow_data.get("network_requests", 0),
            result_data=workflow_data.get("result_data"),
            error_info=workflow_data.get("error_info"),
            screenshots=workflow_data.get("screenshots", []),
            downloads=workflow_data.get("downloads", []),
            metadata=workflow_data.get("metadata", {}),
        )

    def _generate_operation_id(self) -> str:
        """Generate unique operation ID for tracking"""
        return f"coord_{int(time.time())}_{str(uuid4())[:8]}"


# Export main classes
__all__ = [
    "BrowserAgentCoordinator",
    "OrchestrationRequest",
    "WorkflowResult",
    "OrchestrationStatus",
    "WorkflowStatus",
    "OrchestrationMetrics",
]
