https://github.com/user-attachments/assets/a288265d-03ac-4d7d-b803-b74066267f26



# Project Sid: Many-agent simulations toward AI civilization

Reimplementation of the **PIANO (Parallel Information Aggregation via Neural Orchestration)** architecture from the paper "Project Sid: Many-agent simulations toward AI civilization" by Altera.AL.

## Abstract

AI agents have been evaluated in isolation or within small groups, where interactions remain limited in scope and complexity. Large-scale simulations involving many autonomous agents—reflecting the full spectrum of civilizational processes—have yet to be explored. Here, we demonstrate how 10 – 1000+ AI agents behave and progress within agent societies. We first introduce the PIANO (Parallel Information Aggregation via Neural Orchestration) architecture, which enables agents to interact with humans and other agents in real-time while maintaining coherence across multiple output streams. We then evaluate agent performance in large-scale simulations using civilizational benchmarks inspired by human history. These simulations, set within a Minecraft environment, reveal that agents are capable of meaningful progress—autonomously developing specialized roles, adhering to and changing collective rules, and engaging in cultural and religious transmission. These preliminary results show that agents can achieve significant milestones towards AI civilizations, opening new avenues for large-scale societal simulations, agentic organizational intelligence, and integrating AI into human civilizations.

<img src="./visual_abstract.png" width="1200">

## Project Status

**Phase 0-2 + E2E Complete** — 2,079 tests passing, ruff lint clean. MC server E2E verified (3 bots, 5 ticks).

| Phase | Status | Tests | Key Deliverables |
|-------|--------|-------|------------------|
| Phase 0: MVP | Done | 407 | SAS (Redis), Scheduler (3-tier), CC (template compression), WM/STM, LLM (OpenAI+Mock), ZMQ Bridge, 7 Skills, Action Awareness, Config, Docker, CI |
| Phase 1: Foundation | Done | 1,165 | Goal Generation, Planning, Talking, Self-Reflection, Social Awareness, Big Five Personality, Social Graph, Emotion Tracking, LTM (Qdrant), Memory Consolidation, Model Tiering, LLM Gateway, NN Action Awareness, Checkpoint, Orchestrator (10 agents), Social/Advanced Skills |
| Phase 2: Scaling | Done | 1,979 | Worker Pool, Supervisor, Sharding, Resource Limiter, Prompt Cache, Structured Logging, Prometheus Metrics, Tracing, Collective Intelligence, Influencer Analysis, Governance/Meme/Role Eval, Distributed Checkpoint, K8s Manifests, Network Policies, TLS (Redis/ZMQ/Qdrant), CLI Launcher, Fault Injection Framework, E2E Test Infrastructure, Grafana Alerts |
| E2E Simulation | Done | 2,079 | BridgePerception, ChatBroadcaster, BridgeManager, HealthMonitor, ActionMapper, Multi-bot Launcher (TS), Docker Compose sim, MC server verified |
| Phase 3: Civilization | Pending | - | Specialization, Collective Rules, Cultural Memes, Religious Propagation |

See [docs/implementation/roadmap.md](docs/implementation/roadmap.md) for the full 4-phase plan.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for Redis, PostgreSQL, Minecraft server)

### Setup

```bash
# Clone and install
git clone <repo-url>
cd project-sid
uv sync --dev

# Run tests
uv run pytest tests/

# Lint
uv run ruff check src/ tests/
```

### Run Simulation

```bash
# Single agent with mock LLM (no external dependencies)
uv run piano --agents 1 --ticks 10 --mock-llm --no-bridge

# Multi-agent with real LLM (requires API key)
uv run piano --agents 5 --ticks 100

# Also works via python -m
uv run python -m piano --mock-llm --ticks 10
```

### E2E Simulation (with Minecraft)

```bash
# 1. Start MC server + Redis
docker compose -f docker/docker-compose.sim.yml up -d minecraft redis

# 2. Wait for MC server (~2 min), then start bridge
cd bridge && npm install && npx tsc
NUM_BOTS=3 MC_HOST=localhost node dist/launcher.js

# 3. Run PIANO agents (in another terminal)
uv run piano --agents 3 --ticks 50 --mock-llm
```

See [docs/e2e-setup.md](docs/e2e-setup.md) for detailed setup instructions.

### Docker Services

```bash
# Start Redis + PostgreSQL + Minecraft server
docker compose -f docker/docker-compose.yml up -d

# Start Phase 2 stack (Redis + Qdrant + Prometheus + Grafana)
docker compose -f docker/docker-compose.phase2.yml up -d

# Start E2E simulation stack (MC + Redis + Bridge + Agent)
docker compose -f docker/docker-compose.sim.yml up -d
```

## Architecture

```
┌──────────────────── PIANO Agent ────────────────────┐
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Goal Gen │  │ Planning │  │ Talking  │  SLOW     │
│  │  (LLM)   │  │  (LLM)   │  │  (LLM)   │  Tier    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │                 │
│  ┌────▼──────────────▼──────────────▼────┐           │
│  │     Shared Agent State (Redis)        │           │
│  └────┬──────────────┬──────────────┬────┘           │
│       │              │              │                 │
│  ┌────▼─────┐  ┌─────▼────┐  ┌─────▼────┐           │
│  │ Action   │  │ Social   │  │ Memory   │  FAST/MID │
│  │ Awareness│  │ Awareness│  │  Module  │  Tier     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │                 │
│  ┌────▼──────────────▼──────────────▼────┐           │
│  │  Cognitive Controller (CC)            │           │
│  │  GWT: Compress → Decide → Broadcast   │           │
│  └───────────────────────────────────────┘           │
│                                                      │
└──────────────┬───────────────────────────────────────┘
               │ ZMQ Bridge
    ┌──────────▼──────────┐
    │  Minecraft Server   │
    │  (Paper 1.20.4)     │
    │  via Mineflayer     │
    └─────────────────────┘
```

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12+, TypeScript (bridge) |
| Package Manager | uv |
| Shared State | Redis 7+ |
| LLM | OpenAI SDK |
| Bridge | ZMQ (pyzmq / zeromq.js) |
| Vector DB | Qdrant |
| MC Server | Paper 1.20.4 (Docker) / Pufferfish + Velocity (production) |
| Testing | pytest + pytest-asyncio |
| TLS/Crypto | cryptography, PyNaCl (CurveZMQ) |
| Lint | ruff |
| CI | GitHub Actions |

## Documentation

Detailed technical documentation is available in [docs/implementation/](docs/implementation/):

- [00-overview.md](docs/implementation/00-overview.md) - Project overview and document index
- [01-system-architecture.md](docs/implementation/01-system-architecture.md) - PIANO system architecture
- [02-llm-integration.md](docs/implementation/02-llm-integration.md) - LLM integration strategy
- [03-cognitive-controller.md](docs/implementation/03-cognitive-controller.md) - Cognitive Controller design
- [04-memory-system.md](docs/implementation/04-memory-system.md) - Memory system (WM/STM/LTM)
- [05-minecraft-platform.md](docs/implementation/05-minecraft-platform.md) - Minecraft platform
- [06-social-cognition.md](docs/implementation/06-social-cognition.md) - Social cognition module
- [07-goal-planning.md](docs/implementation/07-goal-planning.md) - Goal and planning system
- [08-infrastructure.md](docs/implementation/08-infrastructure.md) - Infrastructure and scaling
- [09-evaluation.md](docs/implementation/09-evaluation.md) - Evaluation and benchmarks
- [10-devops.md](docs/implementation/10-devops.md) - DevOps workflow
- [e2e-simulation.md](docs/implementation/e2e-simulation.md) - E2E simulation architecture
- [roadmap.md](docs/implementation/roadmap.md) - Implementation roadmap

Setup guides:
- [docs/e2e-setup.md](docs/e2e-setup.md) - E2E simulation setup guide

## Paper

- arXiv: [arXiv:2411.00114](https://arxiv.org/abs/2411.00114)
- PDF: [2024-10-31.pdf](2024-10-31.pdf) (in this repository)

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{altera2024sid,
  title   = {Project Sid: Many-agent simulations toward AI civilization},
  author  = {Altera.AL},
  year    = {2024},
  journal = {arXiv preprint arXiv:2411.00114}
}
```
