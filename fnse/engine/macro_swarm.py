"""LangGraph-based macro swarm orchestration.

Implements the top-level simulation graph that spawns heterogeneous agents,
handles tick-based message passing, and evaluates the global loss function.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from fnse.config import settings

from .graph_rag import GraphRAG, graph_rag_manager
from .state import (
    AgentMessage,
    AgentRole,
    AgentState,
    AgentStatus,
    CompressedVector,
    EpochResult,
    MessageType,
    SimulationTickPacket,
)

# LiteLLM integration
try:
    import litellm
    from litellm import acompletion, completion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    litellm = None  # type: ignore[assignment]
    completion = None  # type: ignore[assignment]
    acompletion = None  # type: ignore[assignment]


class AgentStateDict(TypedDict, total=False):
    agent_id: str
    contribution: float


class SwarmConfig(BaseModel):
    """Configuration for a swarm simulation."""

    model_config = {"extra": "forbid"}

    epoch_id: str
    num_agents: int = Field(default=10, ge=1, le=100)
    agent_roles: list[AgentRole] = Field(
        default_factory=lambda: [
            AgentRole.EXPLORER,
            AgentRole.OPTIMIZER,
            AgentRole.CRITIC,
            AgentRole.SYNTHESIZER,
            AgentRole.COORDINATOR,
        ]
    )
    max_ticks: int = Field(default=100, ge=1, le=1000)
    global_objective: str = "minimize_loss"
    loss_function: str = "mse"
    convergence_threshold: float = 0.01
    checkpoint_interval: int = 10


@dataclass
class AgentNode:
    """Represents an agent node in the LangGraph."""

    agent_id: str
    role: AgentRole
    state: AgentState
    compiled_graph: Any = None  # Compiled LangGraph for this agent


class MacroSwarm:
    """Top-level orchestration of the fractal neural swarm.

    Manages:
    - Agent lifecycle (spawn, tick, terminate)
    - Message passing between agents
    - Global loss computation
    - Checkpointing and rollback
    - GraphRAG integration for shared knowledge
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.epoch_id = config.epoch_id
        self.agents: dict[str, AgentNode] = {}
        self.message_bus: list[AgentMessage] = []
        self.tick_number = 0
        self.global_loss_history: list[float] = []
        self.checkpoints: dict[int, SimulationTickPacket] = {}
        self.graph_rag: GraphRAG = graph_rag_manager.get_or_create(config.epoch_id)
        self._lock = RLock()
        self._running = False
        self._start_time = datetime.now(timezone.utc)

        # LiteLLM configuration
        self._model = settings.default_model
        self._fallback_models = settings.litellm_fallback_models
        self._temperature = settings.model_temperature
        self._max_tokens = settings.max_tokens
        self._litellm_timeout = settings.litellm_request_timeout
        self._litellm_max_retries = settings.litellm_max_retries

        # Configure LiteLLM if available
        if LITELLM_AVAILABLE and litellm:
            litellm.drop_params = True
            litellm.set_verbose = False
        # Loss function registry
        from collections.abc import Callable

        LossFn = Callable[[dict[str, float]], float]
        self._loss_functions: dict[str, LossFn] = {
            "mse": self._mse_loss,
            "mae": self._mae_loss,
            "cosine": self._cosine_loss,
            "custom": self._custom_loss,
        }

    def spawn_agents(self) -> dict[str, AgentState]:
        """Spawn heterogeneous agents based on configuration."""
        with self._lock:
            agents_created = {}

            for i in range(self.config.num_agents):
                role = self.config.agent_roles[i % len(self.config.agent_roles)]
                agent_id = f"{self.epoch_id}_agent_{i}_{role.value}"

                # Create initial compressed state
                working_memory = CompressedVector(
                    dimensions=384,
                    values=[0.0] * 384,  # Zero vector initially
                    metadata={"role": role.value, "tick": 0},
                )

                agent_state = AgentState(
                    agent_id=agent_id,
                    role=role,
                    status=AgentStatus.IDLE,
                    working_memory=working_memory,
                    long_term_memory_ref=None,
                    knowledge_graph_ref=None,
                    current_objective=self.config.global_objective,
                )

                # Register in GraphRAG
                from .state import GraphNode

                graph_node = GraphNode(
                    node_id=agent_id,
                    node_type="agent",
                    label=f"{role.value}_{i}",
                    properties={"role": role.value, "epoch": self.epoch_id},
                )
                self.graph_rag.add_node(graph_node)

                # Create agent node
                agent_node = AgentNode(agent_id=agent_id, role=role, state=agent_state)

                self.agents[agent_id] = agent_node
                agents_created[agent_id] = agent_state

            return agents_created

    def build_agent_graph(
        self, agent_node: AgentNode
    ) -> CompiledStateGraph[AgentStateDict, None, AgentStateDict, AgentStateDict]:
        """Build a LangGraph for a single agent's tick execution."""

        def think_node(state: AgentStateDict) -> AgentStateDict:
            """Agent thinking phase - query LLM/GraphRAG, update working memory."""
            agent = self.agents[state["agent_id"]]
            agent.state.status = AgentStatus.THINKING

            # Query GraphRAG for relevant context
            context = self._get_agent_context(agent.agent_id)

            # In production: call LLM via litellm with context
            # For now, simulate thinking
            new_memory = self._simulate_thinking(agent, context)
            agent.state.working_memory = new_memory
            agent.state.status = AgentStatus.ACTING

            return {"agent_id": agent.agent_id, "contribution": 0.0}

        def act_node(state: AgentStateDict) -> AgentStateDict:
            """Agent acting phase - execute skills, send messages."""
            agent = self.agents[state["agent_id"]]

            # Execute active skill if any
            if agent.state.active_skill_id:
                result = self._execute_skill(agent.state.active_skill_id, agent)
                if result.get("success"):
                    agent.state.success_count += 1
                else:
                    agent.state.failure_count += 1

            # Generate messages for other agents
            messages = self._generate_messages(agent)
            for msg in messages:
                self.message_bus.append(msg)

            agent.state.status = AgentStatus.IDLE
            agent.state.tick_count += 1
            agent.state.updated_at = datetime.now(timezone.utc)

            return {"agent_id": agent.agent_id, "contribution": 0.0}

        def evaluate_node(state: AgentStateDict) -> AgentStateDict:
            """Evaluate agent's contribution to global loss."""
            agent = self.agents[state["agent_id"]]
            contribution = self._evaluate_agent_contribution(agent)
            return {"agent_id": agent.agent_id, "contribution": contribution}

        # Build the graph
        graph: StateGraph[AgentStateDict] = StateGraph(AgentStateDict)
        graph.add_node("think", think_node)
        graph.add_node("act", act_node)
        graph.add_node("evaluate", evaluate_node)

        graph.set_entry_point("think")
        graph.add_edge("think", "act")
        graph.add_edge("act", "evaluate")
        graph.add_edge("evaluate", END)

        return graph.compile(checkpointer=MemorySaver())

    def tick(self) -> SimulationTickPacket:
        """Execute one simulation tick for all agents."""
        with self._lock:
            if not self._running:
                self._running = True
                self.spawn_agents()

            self.tick_number += 1

            # Build/compile agent graphs if needed
            for agent_node in self.agents.values():
                if agent_node.compiled_graph is None:
                    agent_node.compiled_graph = self.build_agent_graph(agent_node)

            # Execute each agent's tick
            contributions = {}
            for agent_id, agent_node in self.agents.items():
                try:
                    result = agent_node.compiled_graph.invoke(
                        {"agent_id": agent_id},
                        config={"configurable": {"thread_id": agent_id}},
                    )
                    contributions[agent_id] = result.get("contribution", 0.0)
                except (RuntimeError, ValueError, KeyError, TypeError):
                    agent_node.state.status = AgentStatus.ERROR
                    contributions[agent_id] = 0.0

            # Process message bus
            self._process_messages()

            # Compute global loss
            global_loss = self._compute_global_loss(contributions)
            self.global_loss_history.append(global_loss)

            # Check convergence
            converged = global_loss < self.config.convergence_threshold

            # Create tick packet
            packet = SimulationTickPacket(
                epoch_id=self.epoch_id,
                tick_number=self.tick_number,
                timestamp=datetime.now(timezone.utc),
                agent_states={aid: an.state for aid, an in self.agents.items()},
                message_queue=self.message_bus.copy(),
                global_loss=global_loss,
                global_loss_history=self.global_loss_history.copy(),
                convergence_rate=self._compute_convergence_rate(),
                adjacency_matrix=self._get_adjacency_matrix(),
                skills_executed=self._get_skills_executed(),
            )

            # Clear message bus for next tick
            self.message_bus.clear()

            # Checkpoint if needed
            if self.tick_number % self.config.checkpoint_interval == 0:
                self.checkpoints[self.tick_number] = packet

            # Check termination conditions
            if converged or self.tick_number >= self.config.max_ticks:
                self._running = False

            return packet

    def _get_agent_context(self, agent_id: str) -> dict[str, Any]:
        """Retrieve relevant context from GraphRAG for an agent."""
        # Extract subgraph around agent
        subgraph = self.graph_rag.extract_subgraph(agent_id, radius=2, max_nodes=20)

        # Get recent messages for this agent
        recent_messages = [
            msg
            for msg in self.message_bus
            if msg.recipient_id == agent_id or msg.recipient_id is None
        ][
            -5:
        ]  # Last 5 messages

        return {
            "subgraph_nodes": len(subgraph.nodes),
            "subgraph_edges": len(subgraph.edges),
            "recent_messages": [m.payload_summary for m in recent_messages],
            "global_loss": (
                self.global_loss_history[-1] if self.global_loss_history else 0.0
            ),
            "tick": self.tick_number,
        }

    def _simulate_thinking(
        self, agent: AgentNode, context: dict[str, Any]
    ) -> CompressedVector:
        """Agent thinking phase - calls LLM via LiteLLM with context."""
        import numpy as np

        # Build prompt from context
        system_prompt = f"You are a {agent.role.value} agent in a fractal neural swarm. Current tick: {self.tick_number}. Global loss: {context.get('global_loss', 0):.4f}. Objective: {self.config.global_objective}."
        user_prompt = f"Context: {json.dumps(context, default=str)}. Provide your reasoning and next action as a JSON object with keys: reasoning, action, confidence (0-1)."

        # Try primary model with fallbacks
        models_to_try = [self._model] + self._fallback_models
        response_text = None

        for model in models_to_try:
            try:
                if LITELLM_AVAILABLE and litellm:
                    response = completion(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        timeout=self._litellm_timeout,
                        num_retries=self._litellm_max_retries,
                    )
                    response_text = response.choices[0].message.content
                    # Track token usage
                    if hasattr(response, "usage") and response.usage:
                        agent.state.total_tokens_used += response.usage.total_tokens
                    break
            except (
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                ConnectionError,
                TimeoutError,
            ) as e:
                print(f"Model {model} failed: {e}")
                continue

        # Fallback to simulated thinking if all models fail
        if response_text is None:
            seed = hash(f"{agent.agent_id}_{self.tick_number}") % (2**32)
            np.random.seed(seed)
            values = np.random.randn(384).tolist()
            np.random.seed()

            return CompressedVector(
                dimensions=384,
                values=values,
                metadata={
                    "role": agent.role.value,
                    "tick": self.tick_number,
                    "context_hash": hash(str(context)) % (2**32),
                    "llm_failed": True,
                },
            )

        # Parse LLM response and create vector from it
        try:
            parsed = json.loads(response_text)
            reasoning = parsed.get("reasoning", "")
            confidence = parsed.get("confidence", 0.5)
        except (json.JSONDecodeError, TypeError, ValueError):
            reasoning = response_text[:500]
            confidence = 0.5

        # Create deterministic vector from reasoning hash
        seed = hash(f"{agent.agent_id}_{self.tick_number}_{reasoning}") % (2**32)
        np.random.seed(seed)
        values = np.random.randn(384).tolist()
        np.random.seed()

        return CompressedVector(
            dimensions=384,
            values=values,
            metadata={
                "role": agent.role.value,
                "tick": self.tick_number,
                "context_hash": hash(str(context)) % (2**32),
                "reasoning": reasoning[:200],
                "confidence": confidence,
                "model_used": models_to_try[0],
            },
        )

    def _execute_skill(self, skill_id: str, agent: AgentNode) -> dict[str, Any]:
        """Execute a skill (placeholder for skill compiler integration)."""
        # In production: load and execute compiled skill
        return {"success": True, "result": None}

    def _generate_messages(self, agent: AgentNode) -> list[AgentMessage]:
        """Generate messages to send to other agents."""
        messages = []

        # Coordinator broadcasts periodically
        if agent.role == AgentRole.COORDINATOR and self.tick_number % 5 == 0:
            payload = CompressedVector(
                dimensions=384,
                values=[0.1] * 384,
                metadata={"type": "coordination", "tick": self.tick_number},
            )
            messages.append(
                AgentMessage(
                    sender_id=agent.agent_id,
                    recipient_id=None,  # Broadcast
                    msg_type=MessageType.BROADCAST,
                    payload_vector=payload,
                    payload_summary=f"Coordination signal at tick {self.tick_number}",
                    priority=5,
                )
            )

        return messages

    def _process_messages(self) -> None:
        """Process messages in the bus, update recipient states."""
        for msg in self.message_bus:
            if msg.recipient_id and msg.recipient_id in self.agents:
                recipient = self.agents[msg.recipient_id]
                _ = recipient  # Acknowledge recipient exists
                # Update recipient's working memory based on message
                # In production: merge vectors, update context

    def _evaluate_agent_contribution(self, agent: AgentNode) -> float:
        """Evaluate how much this agent contributed to reducing global loss."""
        # Placeholder: use success rate and activity as proxy
        if agent.state.tick_count == 0:
            return 0.0
        success_rate = agent.state.success_count / max(1, agent.state.tick_count)
        return success_rate * (1.0 - agent.state.divergence_score / 10.0)

    def _compute_global_loss(self, contributions: dict[str, float]) -> float:
        """Compute global loss from agent contributions."""
        loss_fn = self._loss_functions.get(self.config.loss_function, self._mse_loss)
        return loss_fn(contributions)

    def _mse_loss(self, contributions: dict[str, float]) -> float:
        """Mean squared error from target (1.0 = perfect)."""
        if not contributions:
            return 1.0
        target = 1.0
        mse = sum((target - c) ** 2 for c in contributions.values()) / len(
            contributions
        )
        return min(1.0, mse)

    def _mae_loss(self, contributions: dict[str, float]) -> float:
        """Mean absolute error from target."""
        if not contributions:
            return 1.0
        target = 1.0
        mae = sum(abs(target - c) for c in contributions.values()) / len(contributions)
        return min(1.0, mae)

    def _cosine_loss(self, contributions: dict[str, float]) -> float:
        """Cosine similarity based loss."""
        import numpy as np
        from numpy.typing import NDArray

        if not contributions:
            return 1.0
        values: NDArray[np.float64] = np.array(
            list(contributions.values()), dtype=np.float64
        )
        target: NDArray[np.float64] = np.ones_like(values)
        if np.linalg.norm(values) == 0:
            return 1.0
        cos_sim = np.dot(values, target) / (
            np.linalg.norm(values) * np.linalg.norm(target)
        )
        return float(1.0 - cos_sim)

    def _custom_loss(self, contributions: dict[str, float]) -> float:
        """Custom loss function placeholder."""
        return self._mse_loss(contributions)

    def _compute_convergence_rate(self) -> float:
        """Compute rate of loss convergence."""
        if len(self.global_loss_history) < 2:
            return 0.0
        recent = self.global_loss_history[-10:]
        if len(recent) < 2:
            return 0.0
        return (recent[0] - recent[-1]) / len(recent)

    def _get_adjacency_matrix(self) -> dict[str, list[str]]:
        """Get current agent communication topology."""
        adj = {}
        for agent_id in self.agents:
            adj[agent_id] = [aid for aid in self.agents if aid != agent_id]
        return adj

    def _get_skills_executed(self) -> dict[str, int]:
        """Get skill execution counts."""
        return {}

    def get_epoch_result(self) -> EpochResult:
        """Get final epoch result."""
        now = datetime.now(timezone.utc)
        return EpochResult(
            epoch_id=self.epoch_id,
            started_at=self._start_time,
            completed_at=now,
            total_ticks=self.tick_number,
            converged=(
                self.global_loss_history[-1] < self.config.convergence_threshold
                if self.global_loss_history
                else False
            ),
            final_global_loss=(
                self.global_loss_history[-1] if self.global_loss_history else 1.0
            ),
            loss_trajectory=self.global_loss_history,
            agent_final_states={aid: an.state for aid, an in self.agents.items()},
            total_tokens=sum(a.state.total_tokens_used for a in self.agents.values()),
            total_duration_seconds=(now - self._start_time).total_seconds(),
        )

    def rollback_to_checkpoint(self, tick: int) -> bool:
        """Rollback simulation to a checkpoint."""
        with self._lock:
            if tick not in self.checkpoints:
                return False

            checkpoint = self.checkpoints[tick]
            self.tick_number = tick
            self.global_loss_history = checkpoint.global_loss_history.copy()

            # Restore agent states
            for agent_id, state in checkpoint.agent_states.items():
                if agent_id in self.agents:
                    self.agents[agent_id].state = state

            # Remove later checkpoints
            to_remove = [t for t in self.checkpoints if t > tick]
            for t in to_remove:
                del self.checkpoints[t]

            return True


class SwarmManager:
    """Manages multiple swarm simulations."""

    def __init__(self):
        self.swarms: dict[str, MacroSwarm] = {}
        self._lock = RLock()

    def create_swarm(self, config: SwarmConfig) -> MacroSwarm:
        """Create a new swarm simulation."""
        with self._lock:
            swarm = MacroSwarm(config)
            self.swarms[config.epoch_id] = swarm
            return swarm

    def get_swarm(self, epoch_id: str) -> MacroSwarm | None:
        """Get an existing swarm."""
        with self._lock:
            return self.swarms.get(epoch_id)

    def remove_swarm(self, epoch_id: str) -> bool:
        """Remove a swarm and cleanup resources."""
        with self._lock:
            if epoch_id in self.swarms:
                graph_rag_manager.remove(epoch_id)
                del self.swarms[epoch_id]
                return True
            return False

    def list_swarms(self) -> list[str]:
        """List all active swarm epochs."""
        with self._lock:
            return list(self.swarms.keys())


# Global instance
swarm_manager = SwarmManager()
