"""FastAPI application for FNSE - Fractal Neural Simulation Engine.

Exposes endpoints to initialize, run, and monitor simulation epochs.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from fnse.config import settings
from fnse.engine.graph_rag import graph_rag_manager
from fnse.engine.macro_swarm import SwarmConfig, swarm_manager
from fnse.engine.safeguards import AlertSeverity, SafeguardSystem
from fnse.engine.skill_compiler import skill_registry
from fnse.engine.state import AgentRole

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


class CreateEpochRequest(BaseModel):
    num_agents: int = Field(default=10, ge=1, le=100)
    max_ticks: int = Field(default=100, ge=1, le=1000)
    global_objective: str = "minimize_loss"
    loss_function: str = Field(default="mse", pattern="^(mse|mae|cosine|custom)$")
    convergence_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    checkpoint_interval: int = Field(default=10, ge=1, le=100)
    agent_roles: list[str] | None = None
    seed_entities: list[dict[str, Any]] | None = None
    model: str | None = None
    model_temperature: float | None = None


class CreateEpochResponse(BaseModel):
    epoch_id: str
    status: str
    config: dict[str, Any]


class TickResponse(BaseModel):
    epoch_id: str
    tick_number: int
    global_loss: float
    convergence_rate: float
    agent_count: int
    alerts: list[dict[str, Any]]
    timestamp: str


class EpochStatusResponse(BaseModel):
    epoch_id: str
    running: bool
    tick_number: int
    global_loss: float
    convergence_rate: float
    converged: bool
    agent_states: dict[str, dict[str, Any]]
    circuit_breakers: dict[str, str]
    alerts_summary: dict[str, int]
    checkpoints: int
    uptime_seconds: float


class EpochResultResponse(BaseModel):
    epoch_id: str
    converged: bool
    final_global_loss: float
    total_ticks: int
    loss_trajectory: list[float]
    total_duration_seconds: float
    total_tokens: int
    top_performers: list[str]
    skills_compiled: int
    circuit_breaks: int
    rollbacks_performed: int


class SkillCompileRequest(BaseModel):
    name: str
    description: str
    source_code: str
    author_agent_id: str
    test_cases: list[dict[str, Any]] | None = None


class GraphQueryRequest(BaseModel):
    query_vector: list[float] | None = None
    node_types: list[str] | None = None
    top_k: int = 10
    center_node: str | None = None
    radius: int = 2


class GraphSeedRequest(BaseModel):
    entities: list[dict[str, Any]]


_running_epochs: dict[str, asyncio.Task] = {}
_epoch_safeguards: dict[str, SafeguardSystem] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FNSE API server")
    yield
    logger.info("Shutting down FNSE API server")
    for task in _running_epochs.values():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    for safeguard in _epoch_safeguards.values():
        safeguard.emergency_stop()


app = FastAPI(
    title="FNSE",
    description="API for fractal neural swarm simulations",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "fnse"}


@app.post("/epochs", response_model=CreateEpochResponse, status_code=201)
async def create_epoch(request: CreateEpochRequest) -> CreateEpochResponse:
    roles = []
    if request.agent_roles:
        for role_str in request.agent_roles:
            try:
                roles.append(AgentRole(role_str))
            except ValueError:
                raise HTTPException(400, f"Invalid agent role: {role_str}")

    config = SwarmConfig(
        epoch_id=f"epoch_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        num_agents=request.num_agents,
        agent_roles=roles
        or [
            AgentRole.EXPLORER,
            AgentRole.OPTIMIZER,
            AgentRole.CRITIC,
            AgentRole.SYNTHESIZER,
            AgentRole.COORDINATOR,
        ],
        max_ticks=request.max_ticks,
        global_objective=request.global_objective,
        loss_function=request.loss_function,
        convergence_threshold=request.convergence_threshold,
        checkpoint_interval=request.checkpoint_interval,
    )

    swarm = swarm_manager.create_swarm(config)

    safeguard = SafeguardSystem(
        epoch_id=config.epoch_id, checkpoint_interval=request.checkpoint_interval
    )
    _epoch_safeguards[config.epoch_id] = safeguard

    if request.seed_entities:
        for entity in request.seed_entities:
            graph_rag_manager.get_or_create(config.epoch_id).add_node(
                node_type=entity.get("type", "entity"),
                label=entity.get("label", ""),
                properties=entity.get("properties", {}),
            )

    if request.model:
        swarm._model = request.model
    if request.model_temperature is not None:
        swarm._temperature = request.model_temperature

    logger.info(f"Created epoch: {config.epoch_id} with {len(swarm.agents)} agents")

    return CreateEpochResponse(
        epoch_id=config.epoch_id, status="initialized", config=config.model_dump()
    )


@app.get("/epochs/{epoch_id}", response_model=EpochStatusResponse)
async def get_epoch_status(epoch_id: str) -> EpochStatusResponse:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    safeguard = _epoch_safeguards.get(epoch_id)
    safeguard_status = safeguard.get_status() if safeguard else {}

    agent_states = {}
    for agent_id, agent_node in swarm.agents.items():
        state = agent_node.state
        agent_states[agent_id] = {
            "role": state.role.value,
            "status": state.status.value,
            "tick_count": state.tick_count,
            "success_count": state.success_count,
            "failure_count": state.failure_count,
            "total_tokens_used": state.total_tokens_used,
            "divergence_score": state.divergence_score,
            "current_objective": state.current_objective,
            "active_skill_id": state.active_skill_id,
        }

    return EpochStatusResponse(
        epoch_id=epoch_id,
        running=swarm._running,
        tick_number=swarm.tick_number,
        global_loss=swarm.global_loss_history[-1] if swarm.global_loss_history else 1.0,
        convergence_rate=swarm._compute_convergence_rate(),
        converged=(
            swarm.global_loss_history[-1] < swarm.config.convergence_threshold
            if swarm.global_loss_history
            else False
        ),
        agent_states=agent_states,
        circuit_breakers=safeguard_status.get("circuit_breakers", {}),
        alerts_summary=safeguard_status.get("alerts", {}),
        checkpoints=safeguard_status.get("checkpoints", {}).get("total", 0),
        uptime_seconds=(datetime.now(UTC) - swarm._start_time).total_seconds(),
    )


@app.post("/epochs/{epoch_id}/tick", response_model=TickResponse)
async def step_epoch(epoch_id: str) -> TickResponse:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    if swarm._running:
        raise HTTPException(
            409, "Epoch is already running. Stop it first or wait for completion."
        )

    safeguard = _epoch_safeguards.get(epoch_id)

    tick_packet = swarm.execute_tick()

    alerts = []
    if safeguard:
        alerts = safeguard.on_tick_end(tick_packet)

    alert_dicts = [
        {
            "alert_id": a.alert_id,
            "severity": a.severity.value,
            "source": a.source,
            "message": a.message,
            "details": a.details,
            "timestamp": a.timestamp.isoformat(),
        }
        for a in alerts
    ]

    return TickResponse(
        epoch_id=epoch_id,
        tick_number=swarm.tick_number,
        global_loss=swarm.global_loss_history[-1] if swarm.global_loss_history else 1.0,
        convergence_rate=swarm._compute_convergence_rate(),
        agent_count=len(swarm.agents),
        alerts=alert_dicts,
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.post("/epochs/{epoch_id}/run")
async def run_epoch(epoch_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    if swarm._running:
        raise HTTPException(409, "Epoch is already running")

    if epoch_id in _running_epochs:
        raise HTTPException(409, "Epoch already has a running background task")

    async def run_simulation():
        safeguard = _epoch_safeguards.get(epoch_id)
        try:
            while swarm.tick_number < swarm.config.max_ticks and swarm._running:
                tick_packet = swarm.execute_tick()

                if safeguard:
                    alerts = safeguard.on_tick_end(tick_packet)
                    if any(a.severity == AlertSeverity.EMERGENCY for a in alerts):
                        logger.warning(f"Emergency stop triggered for epoch {epoch_id}")
                        break

                if swarm.global_loss_history and swarm.global_loss_history[-1] < swarm.config.convergence_threshold:
                    logger.info(
                        f"Epoch {epoch_id} converged at tick {swarm.tick_number}"
                    )
                    break

                await asyncio.sleep(0.01)

            swarm._running = False
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            logger.error(f"Error in simulation {epoch_id}: {e}")
            swarm._running = False
        finally:
            _running_epochs.pop(epoch_id, None)

    task = asyncio.create_task(run_simulation())
    _running_epochs[epoch_id] = task

    return {"status": "started", "epoch_id": epoch_id}


@app.post("/epochs/{epoch_id}/stop")
async def stop_epoch(epoch_id: str) -> dict[str, str]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    if epoch_id in _running_epochs:
        _running_epochs[epoch_id].cancel()
        try:
            await _running_epochs[epoch_id]
        except asyncio.CancelledError:
            pass
        _running_epochs.pop(epoch_id, None)

    swarm._running = False

    return {"status": "stopped", "epoch_id": epoch_id}


@app.get("/epochs/{epoch_id}/result", response_model=EpochResultResponse)
async def get_epoch_result(epoch_id: str) -> EpochResultResponse:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    result = swarm.get_epoch_result()

    safeguard = _epoch_safeguards.get(epoch_id)
    safeguard_status = safeguard.get_status() if safeguard else {}

    return EpochResultResponse(
        epoch_id=result.epoch_id,
        converged=result.converged,
        final_global_loss=result.final_global_loss,
        total_ticks=result.total_ticks,
        loss_trajectory=result.loss_trajectory,
        total_duration_seconds=result.total_duration_seconds,
        total_tokens=result.total_tokens,
        top_performers=result.top_performers,
        skills_compiled=len(result.skills_compiled),
        circuit_breaks=safeguard_status.get("circuit_break_count", 0),
        rollbacks_performed=safeguard_status.get("rollback_count", 0),
    )


@app.delete("/epochs/{epoch_id}")
async def delete_epoch(epoch_id: str) -> dict[str, str]:
    if epoch_id in _running_epochs:
        _running_epochs[epoch_id].cancel()
        try:
            await _running_epochs[epoch_id]
        except asyncio.CancelledError:
            pass
        _running_epochs.pop(epoch_id, None)

    if epoch_id in _epoch_safeguards:
        _epoch_safeguards[epoch_id].emergency_stop()
        _epoch_safeguards.pop(epoch_id, None)

    success = swarm_manager.remove_swarm(epoch_id)
    if not success:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    graph_rag_manager.remove(epoch_id)
    skill_registry.remove_compiler(epoch_id)

    return {"status": "deleted", "epoch_id": epoch_id}


@app.get("/epochs")
async def list_epochs() -> list[str]:
    return swarm_manager.list_swarms()


@app.post("/epochs/{epoch_id}/skills")
async def compile_skill(epoch_id: str, request: SkillCompileRequest) -> dict[str, Any]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    compiler = skill_registry.get_compiler(epoch_id)

    result = compiler.compile_skill(
        name=request.name,
        description=request.description,
        source_code=request.source_code,
        author_agent_id=request.author_agent_id,
        compilation_epoch=epoch_id,
        compilation_tick=swarm.tick_number,
        test_cases=request.test_cases,
    )

    if not result.success:
        raise HTTPException(400, result.error)

    return {"skill_id": result.skill_id, "test_results": result.test_results}


@app.get("/epochs/{epoch_id}/skills")
async def list_skills(epoch_id: str) -> list[dict[str, Any]]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    compiler = skill_registry.get_compiler(epoch_id)
    skills = compiler.list_skills()

    return [
        {
            "skill_id": s.skill_id,
            "name": s.name,
            "description": s.description,
            "version": s.version,
            "author_agent_id": s.author_agent_id,
            "invocation_count": s.invocation_count,
            "success_rate": s.success_rate,
            "created_at": s.created_at.isoformat(),
        }
        for s in skills
    ]


@app.get("/epochs/{epoch_id}/skills/{skill_id}")
async def get_skill(epoch_id: str, skill_id: str) -> dict[str, Any]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    compiler = skill_registry.get_compiler(epoch_id)
    manifest = compiler.get_manifest(skill_id)

    if not manifest:
        raise HTTPException(404, f"Skill {skill_id} not found")

    return manifest.model_dump()


@app.post("/epochs/{epoch_id}/graph/seed")
async def seed_graph(epoch_id: str, request: GraphSeedRequest) -> dict[str, Any]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    graph = graph_rag_manager.get_or_create(epoch_id)

    for entity in request.entities:
        graph.add_node(
            node_type=entity.get("type", "entity"),
            label=entity.get("label", ""),
            properties=entity.get("properties", {}),
        )

    return {"status": "seeded", "nodes_added": len(request.entities)}


@app.post("/epochs/{epoch_id}/graph/query")
async def query_graph(epoch_id: str, request: GraphQueryRequest) -> dict[str, Any]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    graph = graph_rag_manager.get_or_create(epoch_id)

    if request.query_vector:
        from engine.state import CompressedVector

        vector = CompressedVector(
            dimensions=len(request.query_vector), values=request.query_vector
        )
        results = graph.semantic_search(
            vector, top_k=request.top_k, node_types=request.node_types
        )
        return {
            "results": [
                {"node": node.model_dump(), "score": score} for node, score in results
            ]
        }

    if request.center_node:
        result = graph.extract_subgraph(request.center_node, radius=request.radius)
        return {
            "nodes": [n.model_dump() for n in result.nodes],
            "edges": [e.model_dump() for e in result.edges],
        }

    return {"stats": graph.get_stats()}


@app.get("/epochs/{epoch_id}/graph/stats")
async def get_graph_stats(epoch_id: str) -> dict[str, Any]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    graph = graph_rag_manager.get_or_create(epoch_id)
    return graph.get_stats()


@app.get("/epochs/{epoch_id}/safeguards/status")
async def get_safeguard_status(epoch_id: str) -> dict[str, Any]:
    safeguard = _epoch_safeguards.get(epoch_id)
    if not safeguard:
        raise HTTPException(404, f"Safeguard system not found for epoch {epoch_id}")

    return safeguard.get_status()


@app.get("/epochs/{epoch_id}/alerts")
async def get_alerts(
    epoch_id: str, severity: AlertSeverity | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    safeguard = _epoch_safeguards.get(epoch_id)
    if not safeguard:
        raise HTTPException(404, f"Safeguard system not found for epoch {epoch_id}")

    alerts = safeguard._alerts
    if severity:
        alerts = [a for a in alerts if a.severity == severity]

    alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)[:limit]

    return [
        {
            "alert_id": a.alert_id,
            "timestamp": a.timestamp.isoformat(),
            "severity": a.severity.value,
            "source": a.source,
            "message": a.message,
            "details": a.details,
            "acknowledged": a.acknowledged,
            "resolved": a.resolved,
        }
        for a in alerts
    ]


@app.post("/epochs/{epoch_id}/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(epoch_id: str, alert_id: str) -> dict[str, str]:
    safeguard = _epoch_safeguards.get(epoch_id)
    if not safeguard:
        raise HTTPException(404, f"Safeguard system not found for epoch {epoch_id}")

    for alert in safeguard._alerts:
        if alert.alert_id == alert_id:
            alert.acknowledged = True
            return {"status": "acknowledged", "alert_id": alert_id}

    raise HTTPException(404, f"Alert {alert_id} not found")


@app.get("/epochs/{epoch_id}/agents")
async def list_agents(epoch_id: str) -> list[dict[str, Any]]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    return [
        {
            "agent_id": agent_id,
            "role": agent_node.role.value,
            "status": agent_node.state.status.value,
            "tick_count": agent_node.state.tick_count,
            "success_count": agent_node.state.success_count,
            "failure_count": agent_node.state.failure_count,
            "total_tokens_used": agent_node.state.total_tokens_used,
            "divergence_score": agent_node.state.divergence_score,
            "current_objective": agent_node.state.current_objective,
            "active_skill_id": agent_node.state.active_skill_id,
        }
        for agent_id, agent_node in swarm.agents.items()
    ]


@app.get("/epochs/{epoch_id}/agents/{agent_id}")
async def get_agent(epoch_id: str, agent_id: str) -> dict[str, Any]:
    swarm = swarm_manager.get_swarm(epoch_id)
    if not swarm:
        raise HTTPException(404, f"Epoch {epoch_id} not found")

    if agent_id not in swarm.agents:
        raise HTTPException(404, f"Agent {agent_id} not found")

    agent_node = swarm.agents[agent_id]
    state = agent_node.state

    return {
        "agent_id": agent_id,
        "role": agent_node.role.value,
        "status": state.status.value,
        "working_memory": (
            state.working_memory.model_dump() if state.working_memory else None
        ),
        "long_term_memory_ref": state.long_term_memory_ref,
        "knowledge_graph_ref": state.knowledge_graph_ref,
        "current_objective": state.current_objective,
        "active_skill_id": state.active_skill_id,
        "skill_stack": state.skill_stack,
        "tick_count": state.tick_count,
        "success_count": state.success_count,
        "failure_count": state.failure_count,
        "total_tokens_used": state.total_tokens_used,
        "avg_latency_ms": state.avg_latency_ms,
        "divergence_score": state.divergence_score,
        "last_checkpoint_tick": state.last_checkpoint_tick,
        "tags": state.tags,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=True,
    )


def run_api():
    """Entry point for fnse-api command."""
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=True,
    )
