"""Compressed state vector schemas for FNSE.

Defines Pydantic models for agent state, simulation packets, and messages
to prevent context bloat and ensure type-safe serialization.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentRole(str, Enum):
    """Predefined agent roles in the fractal swarm."""

    EXPLORER = "explorer"
    OPTIMIZER = "optimizer"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    COORDINATOR = "coordinator"


class AgentStatus(str, Enum):
    """Current execution status of an agent."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


class MessageType(str, Enum):
    """Types of inter-agent messages."""

    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    PROPOSAL = "proposal"
    VOTE = "vote"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"


class CompressedVector(BaseModel):
    """A compressed semantic vector representation.

    Used to minimize context window usage by storing dense embeddings
    instead of raw text history.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimensions: int = Field(gt=0, description="Vector dimensionality")
    values: list[float] = Field(min_length=1, description="Dense vector values")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """Compressed state vector for a single agent.

    Contains only essential information for decision making,
    with full history offloaded to Redis/graph storage.
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: AgentRole
    status: AgentStatus = AgentStatus.IDLE

    # Compressed semantic state
    working_memory: CompressedVector
    long_term_memory_ref: str | None = Field(
        None, description="Redis key for full history"
    )
    knowledge_graph_ref: str | None = Field(
        None, description="Graph node ID in GraphRAG"
    )

    # Current task context
    current_objective: str | None = None
    active_skill_id: str | None = None
    skill_stack: list[str] = Field(
        default_factory=list, description="Call stack of active skills"
    )

    # Performance metrics
    tick_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_tokens_used: int = 0
    avg_latency_ms: float = 0.0

    # Divergence tracking
    divergence_score: float = 0.0
    last_checkpoint_tick: int = 0

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: dict[str, str] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """Inter-agent message with compressed payload."""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    recipient_id: str | None = Field(None, description="None for broadcast")
    msg_type: MessageType

    # Compressed payload
    payload_vector: CompressedVector
    payload_summary: str = Field(max_length=500, description="Human-readable summary")

    # Routing
    thread_id: str | None = None
    reply_to: str | None = None
    priority: int = Field(default=0, ge=0, le=10)

    # Timing
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    ttl_ticks: int = Field(default=10, gt=0)


class SimulationTickPacket(BaseModel):
    """Complete state snapshot for a single simulation tick.

    This is the primary unit of serialization for checkpointing,
    rollback, and inter-process communication.
    """

    model_config = ConfigDict(extra="forbid")

    epoch_id: str
    tick_number: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Agent states (compressed)
    agent_states: dict[str, AgentState] = Field(default_factory=dict)

    # Message queue for this tick
    message_queue: list[AgentMessage] = Field(default_factory=list)

    # Global metrics
    global_loss: float = 0.0
    global_loss_history: list[float] = Field(default_factory=list)
    convergence_rate: float = 0.0

    # Swarm topology
    adjacency_matrix: dict[str, list[str]] = Field(default_factory=dict)
    active_clusters: list[list[str]] = Field(default_factory=list)

    # Skill execution
    skills_executed: dict[str, int] = Field(default_factory=dict)
    skills_failed: dict[str, int] = Field(default_factory=dict)
    new_skills_compiled: list[str] = Field(default_factory=list)

    # Safety
    circuit_breaker_triggered: bool = False
    divergence_alerts: list[str] = Field(default_factory=list)
    rollback_point: int | None = None

    # Resource usage
    total_tokens_this_tick: int = 0
    total_latency_ms: float = 0.0
    redis_ops_count: int = 0


class EpochResult(BaseModel):
    """Final result of a simulation epoch."""

    model_config = ConfigDict(extra="forbid")

    epoch_id: str
    started_at: datetime
    completed_at: datetime
    total_ticks: int

    # Outcome
    converged: bool
    final_global_loss: float
    loss_trajectory: list[float]

    # Agent outcomes
    agent_final_states: dict[str, AgentState]
    top_performers: list[str] = Field(default_factory=list)
    terminated_agents: list[str] = Field(default_factory=list)

    # Skills
    skills_compiled: list[str] = Field(default_factory=list)
    skills_deprecated: list[str] = Field(default_factory=list)

    # Safety
    circuit_breaks: int = 0
    rollbacks_performed: int = 0

    # Resources
    total_tokens: int = 0
    total_duration_seconds: float = 0.0
    peak_memory_mb: float = 0.0


class SkillManifest(BaseModel):
    """Manifest for a compiled skill."""

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    name: str
    description: str
    version: int = 1

    # Code
    source_code: str
    entry_point: str = "execute"
    signature: str = Field(description="Type signature as string")

    # Metadata
    author_agent_id: str
    parent_skill_ids: list[str] = Field(default_factory=list)
    compilation_tick: int
    compilation_epoch: str

    # Validation
    test_cases: list[dict[str, Any]] = Field(default_factory=list)
    passed_tests: int = 0
    failed_tests: int = 0

    # Usage
    invocation_count: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0

    # Safety
    is_sandboxed: bool = True
    allowed_imports: list[str] = Field(default_factory=list)
    max_execution_time_ms: int = 5000

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class GraphNode(BaseModel):
    """Node in the knowledge graph."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding_ref: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class GraphEdge(BaseModel):
    """Edge in the knowledge graph."""

    model_config = ConfigDict(extra="forbid")

    edge_id: str
    source_id: str
    target_id: str
    relationship: str
    weight: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
