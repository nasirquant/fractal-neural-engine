# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-08-27

### Fixed
- **Python 3.10 Compatibility**: Replaced `datetime.UTC` with `datetime.timezone.utc` across `engine/macro_swarm.py`, `engine/safeguards.py`, and `main.py` to restore runtime compatibility on Python 3.10 environments.

### Quality & CI
- Validated code style and type safety against Ruff (`target-version = "py310"`), Black, and MyPy.

## [1.0.0] - 2025-08-25

### Added
- **MacroSwarm** - Hierarchical multi-agent orchestration with 5 agent roles (Explorer, Optimizer, Critic, Synthesizer, Coordinator)
- **GraphRAG** - Graph-based retrieval-augmented generation with NetworkX-backed knowledge graphs and vector similarity search
- **SkillCompiler** - Recursive self-improving skill system with sandboxed dynamic code generation and test-driven compilation
- **SafeguardSystem** - Enterprise-grade safety with circuit breakers, auto-rollback, divergence monitoring, and audit logging
- **FastAPI REST API** - Async epoch management, real-time tick streaming, WebSocket support, OpenAPI docs
- **Observability** - Prometheus metrics, Grafana dashboards, structured JSON logging, distributed tracing
- **Docker Support** - docker-compose.yml for production deployment with Redis, Grafana, Prometheus
- **CLI Interface** - `run_simulation.py` with configurable agents, ticks, roles, and objectives

### Security
- Sandboxed skill execution with restricted imports
- Keyword blocking for dangerous operations
- Resource limits (CPU, memory, recursion depth)
- Circuit breakers for anomalous behavior detection

### Infrastructure
- GitHub Actions CI/CD pipeline
- PyPI package publishing workflow
- Automated release creation on tag push

## [Unreleased]

### Planned for v1.1.0
- Python SDK improvements
- Unit test coverage >80%
- Comprehensive documentation (MkDocs)
- Agent marketplace / plugin system

### Planned for v1.2.0
- Distributed swarm (multi-node support)
- Kubernetes Helm chart
- Web UI dashboard

### Planned for v2.0.0
- LLM-agnostic skill compilation (local models)
- Advanced plugin architecture
- Multi-modal agent support