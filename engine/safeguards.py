"""Computational circuit breakers, divergence monitoring, and state-rollback checkpointing logic."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from config import settings

from .state import SimulationTickPacket

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Tripped, blocking operations
    HALF_OPEN = "half_open"  # Testing if recovered


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyAlert:
    """A safety alert/event."""

    alert_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: AlertSeverity = AlertSeverity.INFO
    source: str = ""  # Component that generated the alert
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class Checkpoint:
    """A simulation checkpoint for rollback."""

    checkpoint_id: str = field(default_factory=lambda: str(uuid4()))
    epoch_id: str = ""
    tick_number: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tick_packet: SimulationTickPacket | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 30.0,
        excluded_exceptions: set[type] | None = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.excluded_exceptions = excluded_exceptions or set()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._lock = RLock()
        self._state_change_callbacks: list[
            Callable[[CircuitState, CircuitState], None]
        ] = []

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if timeout has elapsed to transition to half-open
                if self._last_failure_time:
                    elapsed = (
                        datetime.utcnow() - self._last_failure_time
                    ).total_seconds()
                    if elapsed >= self.timeout_seconds:
                        self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception:
                pass  # Don't let callbacks break the circuit breaker

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._success_count = 0

    def record_failure(self, exception: Exception) -> None:
        with self._lock:
            # Don't count excluded exceptions
            if type(exception) in self.excluded_exceptions:
                return

            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = datetime.utcnow()

            if (
                self._state == CircuitState.HALF_OPEN
                or self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._transition_to(CircuitState.OPEN)

    @contextmanager
    def protect(self):
        """Context manager for protecting operations."""
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")

        try:
            yield
            self.record_success()
        except Exception as e:
            self.record_failure(e)
            raise

    def add_state_change_callback(
        self, callback: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        with self._lock:
            self._state_change_callbacks.append(callback)

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = RLock()

    def get_or_create(self, name: str, **kwargs) -> CircuitBreaker:
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, **kwargs)
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        with self._lock:
            return self._breakers.get(name)

    def remove(self, name: str) -> bool:
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False

    def get_all_states(self) -> dict[str, CircuitState]:
        with self._lock:
            return {name: breaker.state for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry
class DivergenceMonitor:
    """Monitors agent and swarm divergence from expected behavior."""

    def __init__(
        self,
        max_divergence_score: float = 10.0,
        window_size: int = 100,
        alert_threshold: float = 7.0,
    ):
        self.max_divergence_score = max_divergence_score
        self.window_size = window_size
        self.alert_threshold = alert_threshold

        self._agent_scores: dict[str, list[float]] = {}
        self._swarm_scores: list[float] = []
        self._alerts: list[SafetyAlert] = []
        self._lock = RLock()

    def update_agent_score(self, agent_id: str, score: float) -> SafetyAlert | None:
        """Update an agent's divergence score and check for alerts."""
        with self._lock:
            if agent_id not in self._agent_scores:
                self._agent_scores[agent_id] = []

            self._agent_scores[agent_id].append(score)

            # Trim window
            if len(self._agent_scores[agent_id]) > self.window_size:
                self._agent_scores[agent_id] = self._agent_scores[agent_id][
                    -self.window_size :
                ]

            # Check for alert
            if score >= self.alert_threshold:
                alert = SafetyAlert(
                    severity=(
                        AlertSeverity.CRITICAL
                        if score >= self.max_divergence_score
                        else AlertSeverity.WARNING
                    ),
                    source="divergence_monitor",
                    message=f"Agent {agent_id} divergence score {score:.2f} exceeds threshold",
                    details={
                        "agent_id": agent_id,
                        "score": score,
                        "threshold": self.alert_threshold,
                    },
                )
                self._alerts.append(alert)
                return alert

            return None

    def update_swarm_score(self, score: float) -> SafetyAlert | None:
        """Update swarm-level divergence score."""
        with self._lock:
            self._swarm_scores.append(score)

            if len(self._swarm_scores) > self.window_size:
                self._swarm_scores = self._swarm_scores[-self.window_size :]

            if score >= self.alert_threshold:
                alert = SafetyAlert(
                    severity=(
                        AlertSeverity.CRITICAL
                        if score >= self.max_divergence_score
                        else AlertSeverity.WARNING
                    ),
                    source="divergence_monitor",
                    message=f"Swarm divergence score {score:.2f} exceeds threshold",
                    details={"score": score, "threshold": self.alert_threshold},
                )
                self._alerts.append(alert)
                return alert

            return None

    def get_agent_score(self, agent_id: str) -> float:
        """Get current divergence score for an agent."""
        with self._lock:
            scores = self._agent_scores.get(agent_id, [])
            return scores[-1] if scores else 0.0

    def get_swarm_score(self) -> float:
        """Get current swarm divergence score."""
        with self._lock:
            return self._swarm_scores[-1] if self._swarm_scores else 0.0

    def get_recent_alerts(
        self, since: datetime | None = None, limit: int = 100
    ) -> list[SafetyAlert]:
        """Get recent alerts."""
        with self._lock:
            alerts = self._alerts
            if since:
                alerts = [a for a in alerts if a.timestamp >= since]
            return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
            return False

    def is_agent_diverged(self, agent_id: str) -> bool:
        """Check if agent has exceeded max divergence."""
        return self.get_agent_score(agent_id) >= self.max_divergence_score

    def is_swarm_diverged(self) -> bool:
        """Check if swarm has exceeded max divergence."""
        return self.get_swarm_score() >= self.max_divergence_score


circuit_breaker_registry = CircuitBreakerRegistry()


class CheckpointManager:
    """Manages simulation checkpoints for rollback capability."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 100,
        max_size_mb: int = 500,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.max_size_bytes = max_size_mb * 1024 * 1024

        self._checkpoints: dict[str, Checkpoint] = {}
        self._epoch_checkpoints: dict[str, list[str]] = {}
        self._lock = RLock()

    def create_checkpoint(
        self,
        epoch_id: str,
        tick_number: int,
        tick_packet: SimulationTickPacket,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint."""
        with self._lock:
            checkpoint = Checkpoint(
                epoch_id=epoch_id,
                tick_number=tick_number,
                tick_packet=tick_packet,
                metadata=metadata or {},
            )

            serialized = tick_packet.model_dump_json()
            checkpoint.size_bytes = len(serialized.encode("utf-8"))

            checkpoint_file = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
            checkpoint_file.write_text(serialized)

            meta_file = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.meta.json"
            meta_file.write_text(
                json.dumps(
                    {
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "epoch_id": checkpoint.epoch_id,
                        "tick_number": checkpoint.tick_number,
                        "timestamp": checkpoint.timestamp.isoformat(),
                        "size_bytes": checkpoint.size_bytes,
                        "metadata": checkpoint.metadata,
                    }
                )
            )

            self._checkpoints[checkpoint.checkpoint_id] = checkpoint

            if epoch_id not in self._epoch_checkpoints:
                self._epoch_checkpoints[epoch_id] = []
            self._epoch_checkpoints[epoch_id].append(checkpoint.checkpoint_id)

            self._enforce_limits()

            return checkpoint

    def _enforce_limits(self) -> None:
        """Enforce checkpoint count and size limits."""
        all_checkpoints = list(self._checkpoints.values())
        all_checkpoints.sort(key=lambda c: c.timestamp)

        while len(all_checkpoints) > self.max_checkpoints:
            oldest = all_checkpoints.pop(0)
            self._delete_checkpoint(oldest.checkpoint_id)

        total_size = sum(c.size_bytes for c in all_checkpoints)
        while total_size > self.max_size_bytes and all_checkpoints:
            oldest = all_checkpoints.pop(0)
            total_size -= oldest.size_bytes
            self._delete_checkpoint(oldest.checkpoint_id)

    def _delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint from disk and memory."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return

        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        meta_file = self.checkpoint_dir / f"{checkpoint_id}.meta.json"
        checkpoint_file.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)

        if checkpoint.epoch_id in self._epoch_checkpoints:
            self._epoch_checkpoints[checkpoint.epoch_id] = [
                cid
                for cid in self._epoch_checkpoints[checkpoint.epoch_id]
                if cid != checkpoint_id
            ]

        del self._checkpoints[checkpoint_id]

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a checkpoint by ID."""
        with self._lock:
            return self._checkpoints.get(checkpoint_id)

    def load_checkpoint(self, checkpoint_id: str) -> SimulationTickPacket | None:
        """Load a checkpoint's tick packet from disk."""
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return None

            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            if not checkpoint_file.exists():
                return None

            try:
                data = checkpoint_file.read_text()
                return SimulationTickPacket.model_validate_json(data)
            except Exception:
                return None

    def get_epoch_checkpoints(self, epoch_id: str) -> list[Checkpoint]:
        """Get all checkpoints for an epoch."""
        with self._lock:
            checkpoint_ids = self._epoch_checkpoints.get(epoch_id, [])
            return [
                self._checkpoints[cid]
                for cid in checkpoint_ids
                if cid in self._checkpoints
            ]

    def get_latest_checkpoint(self, epoch_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for an epoch."""
        checkpoints = self.get_epoch_checkpoints(epoch_id)
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda c: c.tick_number)

    def rollback_to_checkpoint(self, checkpoint_id: str) -> SimulationTickPacket | None:
        """Rollback to a specific checkpoint."""
        return self.load_checkpoint(checkpoint_id)

    def cleanup_epoch(self, epoch_id: str) -> int:
        """Remove all checkpoints for an epoch."""
        with self._lock:
            checkpoint_ids = self._epoch_checkpoints.get(epoch_id, [])
            count = 0
            for cid in checkpoint_ids:
                self._delete_checkpoint(cid)
                count += 1
            if epoch_id in self._epoch_checkpoints:
                del self._epoch_checkpoints[epoch_id]
            return count


class SafeguardSystem:
    """Integrated safeguard system combining all safety mechanisms."""

    def __init__(
        self,
        epoch_id: str,
        max_divergence_score: float | None = None,
        checkpoint_interval: int = 10,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.epoch_id = epoch_id
        self.max_divergence_score = (
            max_divergence_score or settings.max_divergence_score
        )
        self.checkpoint_interval = checkpoint_interval

        self.circuit_breakers = CircuitBreakerRegistry()
        self.divergence_monitor = DivergenceMonitor(
            max_divergence_score=self.max_divergence_score
        )
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        self._tick_count = 0
        self._alerts: list[SafetyAlert] = []
        self._lock = RLock()

        self._init_default_breakers()

        self._circuit_break_count = 0
        self._rollback_count = 0

        self._register_breaker_callbacks()

    def _init_default_breakers(self) -> None:
        """Initialize default circuit breakers."""
        self.circuit_breakers.get_or_create(
            "agent_execution", failure_threshold=3, timeout_seconds=60.0
        )
        self.circuit_breakers.get_or_create(
            "llm_call", failure_threshold=5, timeout_seconds=30.0
        )
        self.circuit_breakers.get_or_create(
            "skill_execution", failure_threshold=3, timeout_seconds=60.0
        )
        self.circuit_breakers.get_or_create(
            "graphrag_query", failure_threshold=10, timeout_seconds=10.0
        )

    def _register_breaker_callbacks(self) -> None:
        """Register callbacks for circuit breaker state changes."""

        for name, breaker in self.circuit_breakers._breakers.items():

            def callback(old: CircuitState, new: CircuitState, n: str = name) -> None:
                self._on_breaker_state_change(n, old, new)

            breaker.add_state_change_callback(callback)  # type: ignore[return-value]

    def _on_breaker_state_change(
        self, name: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            self._circuit_break_count += 1
            logger.warning(
                f"Circuit breaker '{name}' opened (total breaks: {self._circuit_break_count})"
            )

    def on_tick_start(self, tick_number: int) -> list[SafetyAlert]:
        """Called at the start of each tick."""
        with self._lock:
            self._tick_count = tick_number
            alerts = []

            for name, breaker in self.circuit_breakers._breakers.items():
                if breaker.state == CircuitState.OPEN:
                    alerts.append(
                        SafetyAlert(
                            severity=AlertSeverity.CRITICAL,
                            source="safeguard_system",
                            message=f"Circuit breaker '{name}' is OPEN",
                            details={"breaker": name, "state": breaker.state.value},
                        )
                    )

            return alerts

    def on_tick_end(self, tick_packet: SimulationTickPacket) -> list[SafetyAlert]:
        """Called at the end of each tick."""
        with self._lock:
            alerts = []

            for agent_id, agent_state in tick_packet.agent_states.items():
                alert = self.divergence_monitor.update_agent_score(
                    agent_id, agent_state.divergence_score
                )
                if alert:
                    alerts.append(alert)

            swarm_alert = self.divergence_monitor.update_swarm_score(
                tick_packet.global_loss
            )
            if swarm_alert:
                alerts.append(swarm_alert)

            if self._tick_count % self.checkpoint_interval == 0:
                checkpoint = self.checkpoint_manager.create_checkpoint(
                    epoch_id=self.epoch_id,
                    tick_number=self._tick_count,
                    tick_packet=tick_packet,
                )
                alerts.append(
                    SafetyAlert(
                        severity=AlertSeverity.INFO,
                        source="safeguard_system",
                        message=f"Checkpoint created at tick {self._tick_count}",
                        details={"checkpoint_id": checkpoint.checkpoint_id},
                    )
                )

            if self.divergence_monitor.is_swarm_diverged():
                alerts.append(
                    SafetyAlert(
                        severity=AlertSeverity.EMERGENCY,
                        source="safeguard_system",
                        message="Swarm divergence exceeds maximum threshold - emergency stop recommended",
                        details={
                            "swarm_score": self.divergence_monitor.get_swarm_score()
                        },
                    )
                )

            # Auto-rollback when swarm divergence exceeds max threshold
            if self.divergence_monitor.is_swarm_diverged():
                rollback_packet = self.emergency_stop()
                if rollback_packet:
                    self._rollback_count += 1
                    alerts.append(
                        SafetyAlert(
                            severity=AlertSeverity.CRITICAL,
                            source="safeguard_system",
                            message=f"Auto-rollback performed to tick {rollback_packet.tick_number}",
                            details={
                                "rollback_tick": rollback_packet.tick_number,
                                "rollback_count": self._rollback_count,
                            },
                        )
                    )

            # Also check if any agent has exceeded max divergence
            for agent_id in tick_packet.agent_states:
                if self.divergence_monitor.is_agent_diverged(agent_id):
                    rollback_packet = self.emergency_stop()
                    if rollback_packet:
                        self._rollback_count += 1
                        alerts.append(
                            SafetyAlert(
                                severity=AlertSeverity.CRITICAL,
                                source="safeguard_system",
                                message=f"Auto-rollback performed due to agent {agent_id} divergence",
                                details={
                                    "agent_id": agent_id,
                                    "rollback_tick": rollback_packet.tick_number,
                                    "rollback_count": self._rollback_count,
                                },
                            )
                        )
                    break  # Only rollback once per tick

            self._alerts.extend(alerts)
            return alerts

    def on_agent_error(self, agent_id: str, error: Exception) -> None:
        """Record an agent error."""
        self.circuit_breakers.get_or_create("agent_execution").record_failure(error)

        alert = SafetyAlert(
            severity=AlertSeverity.WARNING,
            source="safeguard_system",
            message=f"Agent {agent_id} error: {type(error).__name__}",
            details={"agent_id": agent_id, "error": str(error)},
        )
        self._alerts.append(alert)

    def on_agent_success(self, agent_id: str) -> None:
        """Record an agent success."""
        self.circuit_breakers.get_or_create("agent_execution").record_success()

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            return {
                "epoch_id": self.epoch_id,
                "tick_count": self._tick_count,
                "circuit_breakers": self.circuit_breakers.get_all_states(),
                "divergence": {
                    "swarm_score": self.divergence_monitor.get_swarm_score(),
                    "max_threshold": self.max_divergence_score,
                    "agent_scores": {
                        aid: self.divergence_monitor.get_agent_score(aid)
                        for aid in self.divergence_monitor._agent_scores
                    },
                },
                "checkpoints": {
                    "total": len(self.checkpoint_manager._checkpoints),
                    "epoch_checkpoints": len(
                        self.checkpoint_manager.get_epoch_checkpoints(self.epoch_id)
                    ),
                },
                "alerts": {
                    "total": len(self._alerts),
                    "unacknowledged": sum(
                        1 for a in self._alerts if not a.acknowledged
                    ),
                    "by_severity": {
                        sev.value: sum(1 for a in self._alerts if a.severity == sev)
                        for sev in AlertSeverity
                    },
                },
                "circuit_break_count": self._circuit_break_count,
                "rollback_count": self._rollback_count,
            }

    def emergency_stop(self) -> SimulationTickPacket | None:
        """Perform emergency stop and return latest checkpoint for rollback."""
        with self._lock:
            for breaker in self.circuit_breakers._breakers.values():
                breaker._transition_to(CircuitState.OPEN)

            latest = self.checkpoint_manager.get_latest_checkpoint(self.epoch_id)
            if latest:
                return self.checkpoint_manager.load_checkpoint(latest.checkpoint_id)
            return None


circuit_breaker_registry = CircuitBreakerRegistry()
