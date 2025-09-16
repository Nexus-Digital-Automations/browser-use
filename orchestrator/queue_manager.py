"""
Task Queue Management System for Browser Automation

Priority-based task queue with advanced scheduling, dependency management,
and distributed processing capabilities for browser automation tasks.

Key Features:
- Priority-based task queuing with dynamic priority adjustment
- Task dependency management and scheduling
- Distributed task processing with load balancing
- Dead letter queue for failed tasks
- Task retry mechanisms with exponential backoff
- Comprehensive task lifecycle tracking
- Local-only architecture compliance

Author: Claude Code
Version: 1.0.0
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import structlog


class TaskPriority(IntEnum):
    """Task priority levels (higher values = higher priority)"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class TaskStatus(Enum):
    """Task execution status"""

    QUEUED = "queued"
    SCHEDULED = "scheduled"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


class TaskType(Enum):
    """Browser automation task types"""

    NAVIGATION = "navigation"
    INTERACTION = "interaction"
    EXTRACTION = "extraction"
    FORM_SUBMISSION = "form_submission"
    FILE_DOWNLOAD = "file_download"
    SCREENSHOT = "screenshot"
    PDF_GENERATION = "pdf_generation"
    MULTI_TAB = "multi_tab"
    AUTOMATION_FLOW = "automation_flow"
    CUSTOM = "custom"


@dataclass
class TaskDependency:
    """Task dependency specification"""

    depends_on: str  # Task ID
    dependency_type: str = "completion"  # completion, data, condition
    wait_timeout_ms: int = 300000  # 5 minutes default
    pass_data: bool = False  # Pass dependent task data to this task


@dataclass
class TaskMetadata:
    """Task execution metadata and tracking"""

    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_retry_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_ms: int = 300000  # 5 minutes default
    execution_history: List[Dict] = field(default_factory=list)
    error_history: List[Dict] = field(default_factory=list)


@dataclass
class BrowserTask:
    """Browser automation task specification"""

    task_id: str
    task_type: TaskType
    priority: TaskPriority
    status: TaskStatus = TaskStatus.QUEUED

    # Task execution details
    action: str  # Specific action to perform
    target_url: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    context_data: Dict[str, Any] = field(default_factory=dict)

    # Requirements and constraints
    requirements: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 300000
    retry_policy: Dict[str, Any] = field(default_factory=dict)

    # Dependency management
    dependencies: List[TaskDependency] = field(default_factory=list)
    dependents: Set[str] = field(default_factory=set)

    # Assignment and tracking
    assigned_agent: Optional[str] = None
    session_id: Optional[str] = None
    metadata: TaskMetadata = field(default_factory=TaskMetadata)

    # Results
    result_data: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None

    def can_be_scheduled(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied"""
        for dep in self.dependencies:
            if dep.depends_on not in completed_tasks:
                return False
        return True

    def calculate_effective_priority(self, current_time: datetime) -> float:
        """Calculate effective priority considering age and other factors"""
        base_priority = float(self.priority.value)

        # Age bonus (tasks get higher priority over time)
        age_hours = (current_time - self.metadata.created_at).total_seconds() / 3600
        age_bonus = min(1.0, age_hours / 24)  # Max 1.0 bonus after 24 hours

        # Retry penalty (failed tasks get lower priority initially)
        retry_penalty = self.metadata.retry_count * 0.1

        # Critical task boost
        critical_boost = 2.0 if self.priority == TaskPriority.CRITICAL else 0.0
        emergency_boost = 5.0 if self.priority == TaskPriority.EMERGENCY else 0.0

        return (
            base_priority + age_bonus + critical_boost + emergency_boost - retry_penalty
        )


@dataclass
class QueueStatistics:
    """Queue performance and usage statistics"""

    tasks_queued: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0
    tasks_dead_letter: int = 0

    average_queue_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0
    current_queue_size: int = 0
    peak_queue_size: int = 0

    throughput_per_minute: float = 0.0
    error_rate: float = 0.0
    retry_rate: float = 0.0


class TaskQueueManager:
    """
    Advanced task queue manager with priority scheduling and dependency management.

    Provides comprehensive task lifecycle management including queueing, scheduling,
    dependency resolution, retry mechanisms, and distributed processing coordination.
    """

    def __init__(self, config: Dict[str, Any], storage_path: Optional[Path] = None):
        """
        Initialize task queue manager.

        Args:
            config: Queue configuration settings
            storage_path: Local storage path for queue persistence
        """
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.config = config
        self.storage_path = storage_path or Path("./task_queue_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Task storage and indexing
        self.tasks: Dict[str, BrowserTask] = {}  # All tasks
        self.priority_queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.scheduled_tasks: Dict[str, BrowserTask] = {}  # Tasks ready for assignment
        self.running_tasks: Dict[str, BrowserTask] = {}  # Currently executing tasks
        self.completed_tasks: Set[str] = set()  # Completed task IDs
        self.dead_letter_queue: Dict[str, BrowserTask] = {}  # Failed tasks

        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(
            set
        )  # task_id -> dependents
        self.waiting_for_deps: Dict[str, BrowserTask] = (
            {}
        )  # Tasks waiting for dependencies

        # Queue configuration
        self.max_queue_size = config.get("max_queue_size", 10000)
        self.max_priority_boost = config.get("max_priority_boost", 2.0)
        self.default_retry_count = config.get("default_retry_count", 3)
        self.default_timeout_ms = config.get("default_timeout_ms", 300000)
        self.cleanup_interval = config.get("cleanup_interval", 3600)  # 1 hour

        # Retry configuration
        self.retry_base_delay = config.get("retry_base_delay", 5)  # seconds
        self.retry_max_delay = config.get("retry_max_delay", 300)  # 5 minutes
        self.retry_multiplier = config.get("retry_multiplier", 2.0)

        # Background tasks
        self.scheduler_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None

        # Statistics and monitoring
        self.stats = QueueStatistics()
        self.throughput_samples: deque = deque(maxlen=60)  # 1 minute samples
        self.task_completion_times: deque = deque(
            maxlen=1000
        )  # Recent completion times

        # Event notification
        self.task_event_handlers: Dict[str, List] = defaultdict(list)

        self.logger.info(
            "Task Queue Manager initialized",
            max_queue_size=self.max_queue_size,
            default_timeout_ms=self.default_timeout_ms,
            storage_path=str(self.storage_path),
        )

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize queue manager and start background tasks.

        Returns:
            Initialization result with queue status
        """
        operation_id = self._generate_operation_id()
        self.logger.info(f"[{operation_id}] Initializing task queue manager")

        try:
            # Load persisted tasks if available
            await self._load_persisted_tasks()

            # Start background tasks
            await self._start_background_tasks()

            # Initialize metrics
            await self._initialize_metrics()

            self.logger.info(
                f"[{operation_id}] Task queue manager initialized successfully",
                persisted_tasks=len(self.tasks),
                queue_size=sum(len(q) for q in self.priority_queues.values()),
            )

            return {
                "success": True,
                "operation_id": operation_id,
                "persisted_tasks": len(self.tasks),
                "queue_size": sum(len(q) for q in self.priority_queues.values()),
                "max_queue_size": self.max_queue_size,
            }

        except Exception as error:
            self.logger.error(
                f"[{operation_id}] Queue manager initialization failed",
                error=str(error),
                exc_info=True,
            )
            raise

    async def enqueue_task(self, task: BrowserTask) -> Dict[str, Any]:
        """
        Add task to appropriate priority queue.

        Args:
            task: Browser task to enqueue

        Returns:
            Enqueue operation result
        """
        if len(self.tasks) >= self.max_queue_size:
            return {
                "success": False,
                "reason": "Queue at maximum capacity",
                "queue_size": len(self.tasks),
                "max_size": self.max_queue_size,
            }

        # Validate task
        validation_result = self._validate_task(task)
        if not validation_result["valid"]:
            return {
                "success": False,
                "reason": f"Task validation failed: {validation_result['error']}",
                "task_id": task.task_id,
            }

        # Set default metadata if not provided
        if not task.metadata:
            task.metadata = TaskMetadata()

        # Configure retry policy
        if not task.retry_policy:
            task.retry_policy = {
                "max_retries": self.default_retry_count,
                "base_delay": self.retry_base_delay,
                "max_delay": self.retry_max_delay,
                "multiplier": self.retry_multiplier,
            }

        # Store task
        self.tasks[task.task_id] = task

        # Handle dependencies
        if task.dependencies:
            # Add to dependency waiting queue
            self.waiting_for_deps[task.task_id] = task

            # Update dependency graph
            for dep in task.dependencies:
                self.dependency_graph[dep.depends_on].add(task.task_id)
        else:
            # Add directly to priority queue
            self.priority_queues[task.priority].append(task.task_id)

        # Update statistics
        self.stats.tasks_queued += 1
        self.stats.current_queue_size = len(self.tasks)
        self.stats.peak_queue_size = max(
            self.stats.peak_queue_size, self.stats.current_queue_size
        )

        # Persist task
        await self._persist_task(task)

        # Notify event handlers
        await self._notify_task_event("task_queued", task)

        self.logger.debug(
            f"Enqueued task {task.task_id}",
            task_type=task.task_type.value,
            priority=task.priority.name,
            has_dependencies=bool(task.dependencies),
        )

        return {
            "success": True,
            "task_id": task.task_id,
            "queue_position": self._get_queue_position(task.task_id),
            "estimated_wait_time_ms": self._estimate_wait_time(task),
        }

    async def get_next_task(
        self, agent_requirements: Optional[Dict] = None
    ) -> Optional[BrowserTask]:
        """
        Get next highest priority task ready for execution.

        Args:
            agent_requirements: Agent capability requirements for task matching

        Returns:
            Next task to execute, or None if no suitable task available
        """
        current_time = datetime.now()

        # Check all priority levels from highest to lowest
        for priority in reversed(list(TaskPriority)):
            queue = self.priority_queues[priority]

            # Find first task that meets requirements
            for _ in range(len(queue)):
                task_id = queue.popleft()
                task = self.tasks.get(task_id)

                if not task or task.status != TaskStatus.QUEUED:
                    continue

                # Check agent requirements
                if agent_requirements and not self._task_meets_agent_requirements(
                    task, agent_requirements
                ):
                    queue.append(task_id)  # Put back at end
                    continue

                # Check if task can be scheduled (dependencies satisfied)
                if not task.can_be_scheduled(self.completed_tasks):
                    queue.append(task_id)  # Put back at end
                    continue

                # Task is ready for execution
                task.status = TaskStatus.SCHEDULED
                task.metadata.scheduled_at = current_time
                self.scheduled_tasks[task_id] = task

                await self._notify_task_event("task_scheduled", task)

                self.logger.debug(
                    f"Scheduled task {task_id} for execution",
                    priority=priority.name,
                    task_type=task.task_type.value,
                )

                return task

        return None

    async def assign_task(self, task_id: str, agent_id: str) -> Dict[str, Any]:
        """
        Assign scheduled task to specific agent.

        Args:
            task_id: Task identifier
            agent_id: Agent identifier

        Returns:
            Assignment operation result
        """
        task = self.scheduled_tasks.get(task_id)
        if not task:
            return {
                "success": False,
                "reason": "Task not found or not scheduled",
                "task_id": task_id,
            }

        # Assign task to agent
        task.assigned_agent = agent_id
        task.status = TaskStatus.ASSIGNED
        task.metadata.started_at = datetime.now()

        # Move to running tasks
        self.running_tasks[task_id] = task
        del self.scheduled_tasks[task_id]

        await self._notify_task_event("task_assigned", task)

        self.logger.debug(
            f"Assigned task {task_id} to agent {agent_id}",
            task_type=task.task_type.value,
        )

        return {
            "success": True,
            "task_id": task_id,
            "agent_id": agent_id,
            "assigned_at": task.metadata.started_at.isoformat(),
        }

    async def start_task_execution(self, task_id: str) -> Dict[str, Any]:
        """
        Mark task as running and start execution tracking.

        Args:
            task_id: Task identifier

        Returns:
            Execution start result
        """
        task = self.running_tasks.get(task_id)
        if not task:
            return {
                "success": False,
                "reason": "Task not found in running tasks",
                "task_id": task_id,
            }

        task.status = TaskStatus.RUNNING
        execution_record = {
            "started_at": datetime.now().isoformat(),
            "agent_id": task.assigned_agent,
            "attempt": task.metadata.retry_count + 1,
        }
        task.metadata.execution_history.append(execution_record)

        await self._notify_task_event("task_started", task)

        return {"success": True, "task_id": task_id, "execution_started": True}

    async def complete_task(
        self, task_id: str, result_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Mark task as completed with results.

        Args:
            task_id: Task identifier
            result_data: Task execution results

        Returns:
            Completion operation result
        """
        task = self.running_tasks.get(task_id)
        if not task:
            return {
                "success": False,
                "reason": "Task not found in running tasks",
                "task_id": task_id,
            }

        # Complete task
        task.status = TaskStatus.COMPLETED
        task.result_data = result_data
        task.metadata.completed_at = datetime.now()

        # Calculate execution time
        if task.metadata.started_at:
            execution_time_ms = (
                task.metadata.completed_at - task.metadata.started_at
            ).total_seconds() * 1000
            self.task_completion_times.append(execution_time_ms)

        # Move to completed set
        self.completed_tasks.add(task_id)
        del self.running_tasks[task_id]

        # Update statistics
        self.stats.tasks_completed += 1
        self._update_throughput_metrics()

        # Check for dependent tasks that can now be scheduled
        await self._check_and_schedule_dependents(task_id)

        # Persist completion
        await self._persist_task_completion(task)

        await self._notify_task_event("task_completed", task)

        self.logger.info(
            f"Completed task {task_id}",
            execution_time_ms=execution_time_ms if task.metadata.started_at else None,
            agent_id=task.assigned_agent,
        )

        return {
            "success": True,
            "task_id": task_id,
            "completed_at": task.metadata.completed_at.isoformat(),
            "execution_time_ms": (
                execution_time_ms if task.metadata.started_at else None
            ),
        }

    async def fail_task(
        self, task_id: str, error_info: Dict[str, Any], retry: bool = True
    ) -> Dict[str, Any]:
        """
        Mark task as failed and potentially retry.

        Args:
            task_id: Task identifier
            error_info: Error information
            retry: Whether to attempt retry

        Returns:
            Failure handling result
        """
        task = self.running_tasks.get(task_id)
        if not task:
            return {
                "success": False,
                "reason": "Task not found in running tasks",
                "task_id": task_id,
            }

        # Record error
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error": error_info,
            "attempt": task.metadata.retry_count + 1,
            "agent_id": task.assigned_agent,
        }
        task.metadata.error_history.append(error_record)

        # Determine if retry should be attempted
        max_retries = task.retry_policy.get("max_retries", self.default_retry_count)
        should_retry = retry and task.metadata.retry_count < max_retries

        if should_retry:
            # Prepare for retry
            task.status = TaskStatus.RETRYING
            task.metadata.retry_count += 1
            task.metadata.last_retry_at = datetime.now()
            task.assigned_agent = None

            # Calculate retry delay
            delay = self._calculate_retry_delay(task)

            # Schedule retry
            asyncio.create_task(self._schedule_retry(task_id, delay))

            # Move back to scheduled
            self.scheduled_tasks[task_id] = task
            del self.running_tasks[task_id]

            self.stats.retry_rate = (self.stats.retry_rate * 0.9) + (
                0.1
            )  # Exponential moving average

            await self._notify_task_event("task_retry_scheduled", task)

            self.logger.warning(
                f"Task {task_id} failed, retry {task.metadata.retry_count}/{max_retries} scheduled",
                delay_seconds=delay,
                error=error_info.get("message", "Unknown error"),
            )

            return {
                "success": True,
                "task_id": task_id,
                "action": "retry_scheduled",
                "retry_count": task.metadata.retry_count,
                "retry_delay_seconds": delay,
            }
        else:
            # Send to dead letter queue
            task.status = (
                TaskStatus.DEAD_LETTER
                if task.metadata.retry_count >= max_retries
                else TaskStatus.FAILED
            )
            task.error_info = error_info

            self.dead_letter_queue[task_id] = task
            del self.running_tasks[task_id]

            self.stats.tasks_failed += 1
            self.stats.tasks_dead_letter += (
                1 if task.status == TaskStatus.DEAD_LETTER else 0
            )

            await self._persist_failed_task(task)
            await self._notify_task_event("task_failed", task)

            self.logger.error(
                f"Task {task_id} failed permanently",
                retry_count=task.metadata.retry_count,
                max_retries=max_retries,
                error=error_info.get("message", "Unknown error"),
            )

            return {
                "success": True,
                "task_id": task_id,
                "action": "failed_permanently",
                "retry_count": task.metadata.retry_count,
                "final_status": task.status.value,
            }

    async def cancel_task(
        self, task_id: str, reason: str = "User requested"
    ) -> Dict[str, Any]:
        """
        Cancel task regardless of current status.

        Args:
            task_id: Task identifier
            reason: Cancellation reason

        Returns:
            Cancellation operation result
        """
        task = self.tasks.get(task_id)
        if not task:
            return {"success": False, "reason": "Task not found", "task_id": task_id}

        if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
            return {
                "success": False,
                "reason": f"Task already {task.status.value}",
                "task_id": task_id,
            }

        # Remove from appropriate queue/collection
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
        elif task_id in self.running_tasks:
            del self.running_tasks[task_id]
        elif task_id in self.waiting_for_deps:
            del self.waiting_for_deps[task_id]
        else:
            # Remove from priority queue
            for queue in self.priority_queues.values():
                try:
                    queue.remove(task_id)
                    break
                except ValueError:
                    continue

        # Mark as cancelled
        task.status = TaskStatus.CANCELLED
        task.error_info = {"reason": reason, "cancelled_at": datetime.now().isoformat()}

        self.stats.tasks_cancelled += 1

        await self._notify_task_event("task_cancelled", task)

        self.logger.info(f"Cancelled task {task_id}", reason=reason)

        return {
            "success": True,
            "task_id": task_id,
            "cancelled_at": datetime.now().isoformat(),
            "reason": reason,
        }

    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get comprehensive queue status and statistics.

        Returns:
            Queue status information
        """
        # Calculate queue sizes by priority
        priority_sizes = {
            priority.name: len(queue)
            for priority, queue in self.priority_queues.items()
        }

        # Calculate average metrics
        avg_queue_time = (
            sum(self.task_completion_times) / len(self.task_completion_times)
            if self.task_completion_times
            else 0.0
        )

        # Calculate error rate
        total_processed = self.stats.tasks_completed + self.stats.tasks_failed
        error_rate = (
            self.stats.tasks_failed / total_processed if total_processed > 0 else 0.0
        )

        # Get throughput
        current_throughput = (
            sum(self.throughput_samples) / len(self.throughput_samples)
            if self.throughput_samples
            else 0.0
        )

        return {
            "queue_sizes": priority_sizes,
            "total_queued": sum(priority_sizes.values()),
            "scheduled_tasks": len(self.scheduled_tasks),
            "running_tasks": len(self.running_tasks),
            "waiting_for_dependencies": len(self.waiting_for_deps),
            "completed_tasks": len(self.completed_tasks),
            "dead_letter_queue": len(self.dead_letter_queue),
            "statistics": {
                "tasks_queued": self.stats.tasks_queued,
                "tasks_completed": self.stats.tasks_completed,
                "tasks_failed": self.stats.tasks_failed,
                "tasks_cancelled": self.stats.tasks_cancelled,
                "average_execution_time_ms": avg_queue_time,
                "throughput_per_minute": current_throughput,
                "error_rate": error_rate,
                "retry_rate": self.stats.retry_rate,
            },
            "capacity": {
                "current_size": len(self.tasks),
                "max_size": self.max_queue_size,
                "utilization_percent": (len(self.tasks) / self.max_queue_size) * 100,
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown queue manager and persist state"""
        self.logger.info("Shutting down task queue manager")

        # Cancel background tasks
        for task in [self.scheduler_task, self.cleanup_task, self.metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Persist all current tasks
        await self._persist_all_tasks()

        # Save final statistics
        await self._save_queue_statistics()

        self.logger.info("Task queue manager shutdown complete")

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    async def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())

        self.logger.debug("Background tasks started")

    async def _scheduler_loop(self) -> None:
        """Background task scheduler loop"""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                await self._process_dependency_updates()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Scheduler loop error", error=str(error))

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_tasks()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Cleanup loop error", error=str(error))

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                self._update_throughput_metrics()
            except asyncio.CancelledError:
                break
            except Exception as error:
                self.logger.error("Metrics collection error", error=str(error))

    def _validate_task(self, task: BrowserTask) -> Dict[str, Any]:
        """Validate task before queueing"""
        if not task.task_id:
            return {"valid": False, "error": "Task ID is required"}

        if task.task_id in self.tasks:
            return {"valid": False, "error": "Task ID already exists"}

        if not task.action:
            return {"valid": False, "error": "Task action is required"}

        if task.timeout_ms <= 0:
            return {"valid": False, "error": "Task timeout must be positive"}

        # Validate dependencies
        for dep in task.dependencies:
            if dep.depends_on == task.task_id:
                return {"valid": False, "error": "Task cannot depend on itself"}

        return {"valid": True}

    def _task_meets_agent_requirements(
        self, task: BrowserTask, requirements: Dict
    ) -> bool:
        """Check if task can be handled by agent with given requirements"""
        # Check browser type requirement
        if "browser_types" in requirements:
            required_browser = task.requirements.get("browser_type")
            if (
                required_browser
                and required_browser not in requirements["browser_types"]
            ):
                return False

        # Check capability requirements
        for capability in ["javascript", "cookies", "screenshots", "pdf_generation"]:
            if task.requirements.get(capability) and not requirements.get(capability):
                return False

        return True

    def _get_queue_position(self, task_id: str) -> int:
        """Get approximate position of task in queue"""
        task = self.tasks.get(task_id)
        if not task:
            return -1

        position = 1
        for priority in reversed(list(TaskPriority)):
            if priority == task.priority:
                # Count tasks in same priority queue before this task
                queue = self.priority_queues[priority]
                try:
                    return position + list(queue).index(task_id)
                except ValueError:
                    return position
            else:
                position += len(self.priority_queues[priority])

        return position

    def _estimate_wait_time(self, task: BrowserTask) -> int:
        """Estimate wait time for task in milliseconds"""
        position = self._get_queue_position(task.task_id)

        # Use average execution time from recent completions
        avg_execution_time = (
            sum(self.task_completion_times) / len(self.task_completion_times)
            if self.task_completion_times
            else 60000  # Default 1 minute
        )

        return int(position * avg_execution_time)

    async def _check_and_schedule_dependents(self, completed_task_id: str) -> None:
        """Check and schedule tasks that were waiting for this task"""
        dependent_task_ids = self.dependency_graph.get(completed_task_id, set())

        for task_id in dependent_task_ids:
            task = self.waiting_for_deps.get(task_id)
            if not task:
                continue

            # Check if all dependencies are now satisfied
            if task.can_be_scheduled(self.completed_tasks):
                # Move task to appropriate priority queue
                self.priority_queues[task.priority].append(task_id)
                del self.waiting_for_deps[task_id]

                self.logger.debug(
                    f"Task {task_id} dependencies satisfied, moved to queue",
                    priority=task.priority.name,
                )

    async def _process_dependency_updates(self) -> None:
        """Process and update dependency relationships"""
        # Check for tasks with satisfied dependencies
        ready_tasks = []

        for task_id, task in list(self.waiting_for_deps.items()):
            if task.can_be_scheduled(self.completed_tasks):
                ready_tasks.append(task_id)

        # Move ready tasks to priority queues
        for task_id in ready_tasks:
            task = self.waiting_for_deps[task_id]
            self.priority_queues[task.priority].append(task_id)
            del self.waiting_for_deps[task_id]

    async def _schedule_retry(self, task_id: str, delay_seconds: float) -> None:
        """Schedule task retry after delay"""
        await asyncio.sleep(delay_seconds)

        task = self.scheduled_tasks.get(task_id)
        if task and task.status == TaskStatus.RETRYING:
            # Move back to priority queue
            self.priority_queues[task.priority].appendleft(task_id)  # High priority
            task.status = TaskStatus.QUEUED
            del self.scheduled_tasks[task_id]

            self.logger.debug(f"Task {task_id} retry ready for scheduling")

    def _calculate_retry_delay(self, task: BrowserTask) -> float:
        """Calculate exponential backoff retry delay"""
        base_delay = task.retry_policy.get("base_delay", self.retry_base_delay)
        max_delay = task.retry_policy.get("max_delay", self.retry_max_delay)
        multiplier = task.retry_policy.get("multiplier", self.retry_multiplier)

        delay = base_delay * (multiplier ** (task.metadata.retry_count - 1))
        return min(delay, max_delay)

    def _update_throughput_metrics(self) -> None:
        """Update throughput metrics"""
        # Simple throughput calculation based on completed tasks
        recent_completions = sum(1 for _ in self.task_completion_times)
        self.throughput_samples.append(recent_completions)

        # Update average throughput
        if self.throughput_samples:
            self.stats.throughput_per_minute = sum(self.throughput_samples) / len(
                self.throughput_samples
            )

    async def _cleanup_old_tasks(self) -> None:
        """Cleanup old completed and failed tasks"""
        cutoff_time = datetime.now() - timedelta(days=7)  # Keep tasks for 7 days

        # Clean completed tasks
        old_completed = [
            task_id
            for task_id in self.completed_tasks
            if (
                self.tasks.get(task_id)
                and self.tasks[task_id].metadata.completed_at
                and self.tasks[task_id].metadata.completed_at < cutoff_time
            )
        ]

        for task_id in old_completed:
            self.completed_tasks.remove(task_id)
            if task_id in self.tasks:
                del self.tasks[task_id]

        # Clean dead letter queue
        old_failed = [
            task_id
            for task_id, task in self.dead_letter_queue.items()
            if task.metadata.completed_at and task.metadata.completed_at < cutoff_time
        ]

        for task_id in old_failed:
            del self.dead_letter_queue[task_id]
            if task_id in self.tasks:
                del self.tasks[task_id]

        if old_completed or old_failed:
            self.logger.info(
                f"Cleaned up {len(old_completed)} completed and {len(old_failed)} failed tasks"
            )

    async def _notify_task_event(self, event_type: str, task: BrowserTask) -> None:
        """Notify registered event handlers"""
        handlers = self.task_event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(task)
                else:
                    handler(task)
            except Exception as error:
                self.logger.error(f"Event handler error: {error}")

    async def _load_persisted_tasks(self) -> None:
        """Load previously persisted tasks"""
        tasks_file = self.storage_path / "tasks.json"
        if not tasks_file.exists():
            return

        try:
            # Use asyncio.to_thread to run blocking I/O in thread executor
            task_data = await asyncio.to_thread(self._load_json_file, tasks_file)

            for task_dict in task_data:
                # Reconstruct task object
                task = self._deserialize_task(task_dict)
                self.tasks[task.task_id] = task

                # Restore to appropriate queue based on status
                if task.status == TaskStatus.QUEUED:
                    if task.dependencies and not task.can_be_scheduled(
                        self.completed_tasks
                    ):
                        self.waiting_for_deps[task.task_id] = task
                    else:
                        self.priority_queues[task.priority].append(task.task_id)
                elif task.status == TaskStatus.COMPLETED:
                    self.completed_tasks.add(task.task_id)
                elif task.status in [TaskStatus.FAILED, TaskStatus.DEAD_LETTER]:
                    self.dead_letter_queue[task.task_id] = task

            self.logger.info(f"Loaded {len(task_data)} persisted tasks")

        except Exception as error:
            self.logger.error(f"Failed to load persisted tasks: {error}")

    async def _persist_task(self, task: BrowserTask) -> None:
        """Persist single task to storage"""
        # TODO: Implement efficient task persistence
        pass

    async def _persist_task_completion(self, task: BrowserTask) -> None:
        """Persist task completion"""
        # TODO: Implement task completion persistence
        pass

    async def _persist_failed_task(self, task: BrowserTask) -> None:
        """Persist failed task"""
        # TODO: Implement failed task persistence
        pass

    async def _persist_all_tasks(self) -> None:
        """Persist all current tasks"""
        tasks_file = self.storage_path / "tasks.json"

        try:
            task_data = [self._serialize_task(task) for task in self.tasks.values()]

            # Use asyncio.to_thread to run blocking I/O in thread executor
            await asyncio.to_thread(self._save_json_file, tasks_file, task_data)

            self.logger.debug(f"Persisted {len(task_data)} tasks")

        except Exception as error:
            self.logger.error(f"Failed to persist tasks: {error}")

    async def _save_queue_statistics(self) -> None:
        """Save queue statistics"""
        stats_file = self.storage_path / "queue_statistics.json"

        try:
            stats_data = {
                "timestamp": datetime.now().isoformat(),
                "statistics": {
                    "tasks_queued": self.stats.tasks_queued,
                    "tasks_completed": self.stats.tasks_completed,
                    "tasks_failed": self.stats.tasks_failed,
                    "tasks_cancelled": self.stats.tasks_cancelled,
                    "tasks_dead_letter": self.stats.tasks_dead_letter,
                    "throughput_per_minute": self.stats.throughput_per_minute,
                    "error_rate": self.stats.error_rate,
                    "retry_rate": self.stats.retry_rate,
                },
                "final_queue_size": len(self.tasks),
            }

            # Use asyncio.to_thread to run blocking I/O in thread executor
            await asyncio.to_thread(self._save_json_file, stats_file, stats_data)

        except Exception as error:
            self.logger.error(f"Failed to save statistics: {error}")

    async def _initialize_metrics(self) -> None:
        """Initialize metrics collection"""
        # Initialize metrics based on persisted data
        self.stats.current_queue_size = len(self.tasks)
        self.stats.peak_queue_size = self.stats.current_queue_size

    def _serialize_task(self, task: BrowserTask) -> Dict[str, Any]:
        """Serialize task for persistence"""
        return {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "priority": task.priority.value,
            "status": task.status.value,
            "action": task.action,
            "target_url": task.target_url,
            "parameters": task.parameters,
            "context_data": task.context_data,
            "requirements": task.requirements,
            "timeout_ms": task.timeout_ms,
            "retry_policy": task.retry_policy,
            "dependencies": [
                {
                    "depends_on": dep.depends_on,
                    "dependency_type": dep.dependency_type,
                    "wait_timeout_ms": dep.wait_timeout_ms,
                    "pass_data": dep.pass_data,
                }
                for dep in task.dependencies
            ],
            "assigned_agent": task.assigned_agent,
            "session_id": task.session_id,
            "metadata": {
                "created_at": task.metadata.created_at.isoformat(),
                "scheduled_at": (
                    task.metadata.scheduled_at.isoformat()
                    if task.metadata.scheduled_at
                    else None
                ),
                "started_at": (
                    task.metadata.started_at.isoformat()
                    if task.metadata.started_at
                    else None
                ),
                "completed_at": (
                    task.metadata.completed_at.isoformat()
                    if task.metadata.completed_at
                    else None
                ),
                "retry_count": task.metadata.retry_count,
                "max_retries": task.metadata.max_retries,
                "timeout_ms": task.metadata.timeout_ms,
                "execution_history": task.metadata.execution_history,
                "error_history": task.metadata.error_history,
            },
            "result_data": task.result_data,
            "error_info": task.error_info,
        }

    def _deserialize_task(self, task_data: Dict[str, Any]) -> BrowserTask:
        """Deserialize task from persistence data"""
        metadata = TaskMetadata(
            created_at=datetime.fromisoformat(task_data["metadata"]["created_at"]),
            scheduled_at=(
                datetime.fromisoformat(task_data["metadata"]["scheduled_at"])
                if task_data["metadata"]["scheduled_at"]
                else None
            ),
            started_at=(
                datetime.fromisoformat(task_data["metadata"]["started_at"])
                if task_data["metadata"]["started_at"]
                else None
            ),
            completed_at=(
                datetime.fromisoformat(task_data["metadata"]["completed_at"])
                if task_data["metadata"]["completed_at"]
                else None
            ),
            retry_count=task_data["metadata"]["retry_count"],
            max_retries=task_data["metadata"]["max_retries"],
            timeout_ms=task_data["metadata"]["timeout_ms"],
            execution_history=task_data["metadata"]["execution_history"],
            error_history=task_data["metadata"]["error_history"],
        )

        dependencies = [
            TaskDependency(
                depends_on=dep_data["depends_on"],
                dependency_type=dep_data["dependency_type"],
                wait_timeout_ms=dep_data["wait_timeout_ms"],
                pass_data=dep_data["pass_data"],
            )
            for dep_data in task_data["dependencies"]
        ]

        return BrowserTask(
            task_id=task_data["task_id"],
            task_type=TaskType(task_data["task_type"]),
            priority=TaskPriority(task_data["priority"]),
            status=TaskStatus(task_data["status"]),
            action=task_data["action"],
            target_url=task_data["target_url"],
            parameters=task_data["parameters"],
            context_data=task_data["context_data"],
            requirements=task_data["requirements"],
            timeout_ms=task_data["timeout_ms"],
            retry_policy=task_data["retry_policy"],
            dependencies=dependencies,
            assigned_agent=task_data["assigned_agent"],
            session_id=task_data["session_id"],
            metadata=metadata,
            result_data=task_data["result_data"],
            error_info=task_data["error_info"],
        )

    def _load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Helper method to load JSON data from file (runs in thread executor)"""
        import json

        with open(file_path, "r") as f:
            return json.load(f)

    def _save_json_file(self, file_path: Path, data: List[Dict[str, Any]]) -> None:
        """Helper method to save JSON data to file (runs in thread executor)"""
        import json

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_operation_id(self) -> str:
        """Generate unique operation ID for tracking"""
        return f"queue_{int(time.time())}_{str(uuid4())[:8]}"


# Export main classes
__all__ = [
    "TaskQueueManager",
    "BrowserTask",
    "TaskPriority",
    "TaskStatus",
    "TaskType",
    "TaskDependency",
    "TaskMetadata",
    "QueueStatistics",
]
