# 🧠 Fractal Neural Simulation Engine (FNSE)

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.2-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/license-AGPL--3.0-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/fastapi-0.109+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/redis-7.0+-red.svg" alt="Redis">
  <img src="https://img.shields.io/badge/docker-ready-blue.svg" alt="Docker">
  <img src="https://img.shields.io/pypi/v/fnse.svg" alt="PyPI">
  <img src="https://github.com/nasirquant/fractal-neural-engine/workflows/CI/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/security-SECURITY.md-brightgreen.svg" alt="Security Policy">
</p>

<p align="center">
  <strong>A production-grade, self-evolving multi-agent simulation framework with recursive skill compilation, graph-based memory (GraphRAG), and enterprise safeguards.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/fnse/"><strong>PyPI Package</strong></a> •
  <a href="https://github.com/nasirquant/fractal-neural-engine/releases"><strong>Releases</strong></a> •
  <a href="#-quick-start"><strong>Quick Start</strong></a> •
  <a href="https://github.com/nasirquant/fractal-neural-engine#readme"><strong>Documentation</strong></a> •
  <a href="SECURITY.md"><strong>Security Policy</strong></a>
</p>

<p align="center">
  <img src="assets/demo.gif" alt="FNSE Simulation Demo" width="100%" />
  <br/>
  <sub><em>10-Agent Swarm executing GraphRAG traversal, loss convergence, and automated safeguard rollbacks.</em></sub>
</p>

---

## 🏗️ Architecture Overview

```mermaid
```

---

## ✨ Feature Matrix: The 5 Engine Pillars

| Pillar | Description | Key Capabilities |
|--------|-------------|------------------|
| 🏭 **MacroSwarm** | Hierarchical multi-agent orchestration | Role-based agents (Explorer, Optimizer, Critic, Synthesizer, Coordinator), dynamic scaling, tick-based execution, cross-agent consensus |
| 🧠 **GraphRAG** | Graph-based retrieval-augmented generation | Vector similarity search, knowledge graph traversal, entity linking, episodic memory, semantic clustering |
| ⚙️ **SkillCompiler** | Recursive self-improving skill system | Dynamic code generation, sandboxed execution, test-driven compilation, skill versioning, dependency tracking |
| 🛡️ **SafeguardSystem** | Enterprise-grade safety & observability | Circuit breakers, automatic rollbacks, divergence detection, alert management, checkpoint recovery |
| 🌐 **REST API** | Production-ready FastAPI interface | Async epoch management, tick polling, OpenAPI docs, health checks |

---

## 🚀 Quick Start

### Option 1: PyPI Package (Recommended)
```bash
# Install from PyPI (when published)
pip install fnse

# Run simulation
fnse --agents 10 --ticks 100
```

### Option 2: Docker Compose (Production)
```bash
# 1. Clone the repository
git clone https://github.com/nasirquant/fractal-neural-engine.git
cd fractal-neural-engine

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (at minimum OPENAI_API_KEY)

# 3. Start all services
docker compose up -d

# 4. Verify deployment
curl http://localhost:8000/health

# 5. Access API docs
open http://localhost:8000/docs
```

**Services started:**
- **API Server**: http://localhost:8000 (FastAPI + Swagger UI)
- **Redis**: localhost:6379 (State persistence)
- **Worker**: Background simulation processing
- **Grafana**: http://localhost:3000 (admin/admin) - Optional monitoring
- **Prometheus**: http://localhost:9090 - Optional metrics

### Option 3: Python CLI (Development & Testing)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (optional for basic testing)
cp .env.example .env

# 3. Run a quick simulation
python run_simulation.py --agents 5 --ticks 10 --quiet

# 4. Run with custom roles and output
python run_simulation.py \
  --agents 10 \
  --ticks 50 \
  --roles explorer optimizer critic synthesizer coordinator \
  --output results.json

# 5. Full help
python run_simulation.py --help
```

### Option 4: Direct Python API
```python
import asyncio
from run_simulation import run_async_simulation

# Run simulation programmatically
result = await run_async_simulation(
    num_agents=10,
    max_ticks=100,
    global_objective="minimize_loss",
    loss_function="mse",
    convergence_threshold=0.01,
    agent_roles=["explorer", "optimizer", "critic", "synthesizer", "coordinator"],
    verbose=True,
    output_file="simulation_results.json"
)

print(f"Converged: {result['converged']}")
print(f"Final Loss: {result['final_global_loss']}")
```

---

## 📡 REST API Reference

### Base URL
```
http://localhost:8000
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/epochs` | Create new simulation epoch |
| `GET` | `/epochs/{epoch_id}` | Get epoch status |
| `POST` | `/epochs/{epoch_id}/start` | Start simulation |
| `POST` | `/epochs/{epoch_id}/tick` | Execute single tick |
| `POST` | `/epochs/{epoch_id}/stop` | Stop simulation |
| `GET` | `/epochs/{epoch_id}/result` | Get final results |
| `DELETE` | `/epochs/{epoch_id}` | Cleanup epoch |

### GraphRAG Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/graph/query` | Query knowledge graph |
| `POST` | `/graph/seed` | Seed graph with entities |
| `GET` | `/graph/stats` | Get graph statistics |

### Skill Compiler Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/skills/compile` | Compile new skill |
| `GET` | `/skills` | List compiled skills |
| `GET` | `/skills/{skill_id}` | Get skill details |

### Safeguard Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/epochs/{epoch_id}/alerts` | List safety alerts |
| `POST` | `/epochs/{epoch_id}/alerts/{alert_id}/acknowledge` | Acknowledge alert |

### Example: Create & Run Epoch

```bash
# Create epoch
curl -X POST http://localhost:8000/epochs \
  -H "Content-Type: application/json" \
  -d '{
    "num_agents": 10,
    "max_ticks": 100,
    "global_objective": "minimize_loss",
    "loss_function": "mse",
    "convergence_threshold": 0.01
  }'

# Start simulation
curl -X POST http://localhost:8000/epochs/{epoch_id}/start

# Monitor progress (poll)
curl http://localhost:8000/epochs/{epoch_id}

# Get final results
curl http://localhost:8000/epochs/{epoch_id}/result
```
---

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build production image
docker build -t fnse:latest .

# Run with Docker Compose (includes Redis, monitoring)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale workers
docker compose up -d --scale worker=4

# View logs
docker compose logs -f api
```

### Docker Compose Override for Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  api:
    environment:
      - LOG_LEVEL=WARNING
      - API_WORKERS=4
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G

  worker:
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '8'
          memory: 8G

  redis:
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          memory: 1G
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fnse-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fnse-api
  template:
    metadata:
      labels:
        app: fnse-api
    spec:
      containers:
      - name: api
        image: fnse:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: fnse-secrets
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
```


---
apiVersion: v1
kind: Service
metadata:
  name: fnse-api
spec:
  selector:
    app: fnse-api
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

---

## 🔧 Configuration
### Agent Roles

| Role | Purpose | Best For |
|------|---------|----------|
| `explorer` | Discovery & hypothesis generation | Novel problem spaces, research |
| `optimizer` | Parameter tuning & refinement | Known problems, performance tuning |
| `critic` | Validation & error detection | Quality assurance, verification |
| `synthesizer` | Knowledge integration | Cross-domain insights, unification |
| `coordinator` | Task delegation & orchestration | Complex multi-step workflows |

---

## 🏢 Enterprise Use Cases

### 1. **Automated Research & Discovery**
- Deploy explorer/critic swarms for literature review
- Synthesizer agents compile cross-domain insights
- GraphRAG maintains persistent knowledge base

### 2. **Hyperparameter Optimization**
- Optimizer agents search configuration spaces
- Critic agents validate model performance
- SkillCompiler learns optimization strategies

### 3. **Code Generation & Refactoring**
- Explorer agents propose architectural changes
- Critic agents run security/static analysis
- Synthesizer produces final implementation

### 4. **Scientific Simulation**
- Multi-agent parameter sweeps
- Automatic checkpoint/resume
- Divergence detection for numerical stability

### 5. **Decision Support Systems**
- Coordinator orchestrates analysis pipeline
- GraphRAG retrieves relevant precedents
- SafeguardSystem ensures compliance bounds

---

## 📊 Monitoring & Observability

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Redis health
docker exec fnse-redis redis-cli ping

# Full system check
curl http://localhost:8000/health/detailed
```

### Metrics (Prometheus)
```promql
# Simulation throughput
rate(fnse_ticks_total[5m])

# Convergence rate
fnse_convergence_rate

# Agent divergence
fnse_agent_divergence_score

# Circuit breaker status
fnse_circuit_breaker_state
```

### Grafana Dashboards
Pre-built dashboards in `grafana/dashboards/`:
- **FNSE Overview**: Cluster health, active epochs, throughput
- **Agent Performance**: Per-agent metrics, token usage, divergence
- **Safety Monitor**: Alerts, circuit breaks, rollbacks
- **GraphRAG Analytics**: Query latency, cache hit rate, graph growth

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run integration tests
pytest tests/integration/ -v

# Run with coverage
pytest --cov=engine --cov=config tests/

# Load testing
locust -f tests/load_test.py --host=http://localhost:8000
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install dev dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run linters
ruff check .
mypy engine/ config.py
black --check .
```

### Running Tests
```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest --cov=engine --cov=config tests/ --cov-fail-under=50
```

---

## 📦 Releases & Packages

### PyPI Package
The `fnse` package is published to PyPI:
- **Package**: [`fnse`](https://pypi.org/project/fnse/)
- **Install**: `pip install fnse`
- **CLI**: `fnse --help` or `fnse-api` for the FastAPI server

### GitHub Releases
- **Releases**: [GitHub Releases](https://github.com/nasirquant/fractal-neural-engine/releases)
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md) (if exists) or release notes
- **Versioning**: [Semantic Versioning](https://semver.org/)

### Docker Images
```bash
# Build locally
docker build -t fnse:latest .

# Or use pre-built (when available)
docker pull ghcr.io/nasirquant/fractal-neural-engine:latest
```

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

### What this means:
- ✅ **Commercial use** - You may use this software commercially
- ✅ **Modification** - You may modify the source code
- ✅ **Distribution** - You may distribute copies
- ✅ **Patent use** - Patent grants included
- ✅ **Private use** - You may use privately

### Requirements:
- 📋 **License notice** - Include license in distributions
- 📋 **State changes** - Document modifications
- 📋 **Disclose source** - **Network use triggers source disclosure** (key AGPL provision)
- 📋 **Same license** - Derivatives must use AGPL-3.0

### For Enterprise:
If you need a commercial license with different terms (e.g., no source disclosure for SaaS), contact: **contact@fnse.dev**

---

## 🙏 Acknowledgments

- **LiteLLM** - Unified LLM interface
- **FastAPI** - Modern web framework
- **Redis** - High-performance caching
- **NetworkX** - Graph algorithms
- **Pydantic** - Data validation

---

## 📞 Support

- 📖 **Documentation**: https://github.com/nasirquant/fractal-neural-engine#readme
- 🐛 **Issues**: https://github.com/nasirquant/fractal-neural-engine/issues
- 💬 **Discussions**: https://github.com/nasirquant/fractal-neural-engine/discussions
- 📧 **Contact**: contact@fnse.dev

---

<p align="center">
  <strong>Built with ❤️ for the future of autonomous AI systems</strong>
</p>

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_MODEL` | `gpt-4o-mini` | Default LLM model |
| `MODEL_PROVIDER` | `openai` | LLM provider |
| `OPENAI_API_KEY` | - | **Required** OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `MAX_AGENTS` | `100` | Max agents per epoch |
| `MAX_TICKS_PER_EPOCH` | `1000` | Max simulation ticks |
| `GLOBAL_LOSS_THRESHOLD` | `0.01` | Convergence threshold |
| `CHECKPOINT_INTERVAL` | `10` | Checkpoint frequency |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_FORMAT` | `json` | Log format |

See [.env.example](.env.example) for complete list."# Trigger CI"  
