https://github.com/user-attachments/assets/a288265d-03ac-4d7d-b803-b74066267f26



# Project Sid: Many-agent simulations toward AI civilization

Reimplementation of the **PIANO (Parallel Information Aggregation via Neural Orchestration)** architecture from the paper "Project Sid: Many-agent simulations toward AI civilization" by Altera.AL.

## Abstract

AI agents have been evaluated in isolation or within small groups, where interactions remain limited in scope and complexity. Large-scale simulations involving many autonomous agents—reflecting the full spectrum of civilizational processes—have yet to be explored. Here, we demonstrate how 10 – 1000+ AI agents behave and progress within agent societies. We first introduce the PIANO (Parallel Information Aggregation via Neural Orchestration) architecture, which enables agents to interact with humans and other agents in real-time while maintaining coherence across multiple output streams. We then evaluate agent performance in large-scale simulations using civilizational benchmarks inspired by human history. These simulations, set within a Minecraft environment, reveal that agents are capable of meaningful progress—autonomously developing specialized roles, adhering to and changing collective rules, and engaging in cultural and religious transmission. These preliminary results show that agents can achieve significant milestones towards AI civilizations, opening new avenues for large-scale societal simulations, agentic organizational intelligence, and integrating AI into human civilizations.

<img src="./visual_abstract.png" width="1200">

## Project Status

**Phase 0 MVP: Complete** - Core PIANO architecture implemented and tested (323 tests passing).

| Module | Status | Description |
|--------|--------|-------------|
| Shared Agent State (SAS) | Done | Redis-backed shared state with capacity limits |
| Module Scheduler | Done | Tick-based parallel execution with 3-tier scheduling |
| Cognitive Controller (CC) | Done | Template compression + LLM decision + broadcast |
| Memory (WM + STM) | Done | Working memory (cap 10) + Short-term memory (cap 100) |
| LLM Abstraction | Done | LiteLLM provider + Mock + Response cache |
| ZMQ Bridge | Done | Python-TypeScript IPC (REQ-REP + PUB-SUB) |
| Skills | Done | Registry + 7 basic Minecraft skills + Executor |
| Action Awareness | Done | Rule-based expectation-outcome comparison |
| Config | Done | pydantic-settings with env var support |
| Docker Compose | Done | Redis + PostgreSQL + Pufferfish MC server |
| CI/CD | Done | GitHub Actions (Python 3.12/3.13, pytest + ruff + mypy) |

See [docs/implementation/roadmap.md](docs/implementation/roadmap.md) for the full 4-phase plan (Phase 0-3).

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

### Docker Services

```bash
# Start Redis + PostgreSQL + Minecraft server
docker compose -f docker/docker-compose.yml up -d
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
    │  (Pufferfish)       │
    │  via Mineflayer     │
    └─────────────────────┘
```

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12+, TypeScript (bridge) |
| Package Manager | uv |
| Shared State | Redis 7+ |
| LLM | LiteLLM (multi-provider) |
| Bridge | ZMQ (pyzmq / zeromq.js) |
| Vector DB | Qdrant (Phase 1) |
| MC Server | Pufferfish + Velocity |
| Testing | pytest + pytest-asyncio |
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
- [roadmap.md](docs/implementation/roadmap.md) - Implementation roadmap

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
