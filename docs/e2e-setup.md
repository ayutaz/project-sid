# E2E Simulation Setup Guide

> Status: **VERIFIED** — Tested with 3 bots x 5 ticks on Paper 1.20.4

## Prerequisites

- Docker and Docker Compose
- Python 3.12+ with uv
- Node.js 20+ (for bridge)

## Quick Start (Hybrid: Docker MC + Local Bridge/Agent)

This is the tested and recommended approach.

### 1. Start MC Server + Redis

```bash
docker compose -f docker/docker-compose.sim.yml up -d minecraft redis
```

Wait ~2 minutes for MC server startup:

```bash
docker compose -f docker/docker-compose.sim.yml logs -f minecraft
# Wait for: [Server] Done! For help, type "help"
```

### 2. Build and Start Bridge

```bash
cd bridge
npm install
npx tsc
NUM_BOTS=3 MC_HOST=localhost node dist/launcher.js
```

You should see:

```
[launcher] Starting 3 bots...
[bridge] Bot PIANOBot_0 spawned at Vec3 { x: ..., y: ..., z: ... }
[bridge] Bot PIANOBot_1 spawned at Vec3 { x: ..., y: ..., z: ... }
[bridge] Bot PIANOBot_2 spawned at Vec3 { x: ..., y: ..., z: ... }
```

### 3. Run PIANO Agent

In a separate terminal:

```bash
uv run piano --agents 3 --ticks 5 --mock-llm
```

You should see:

```
[piano] Starting 3 agents (ticks=5, mock_llm=True)
[piano] Agent agent-001 connected to bridge
[piano] Agent agent-002 connected to bridge
[piano] Agent agent-003 connected to bridge
[piano] All agents completed 5 ticks
```

### 4. Stop Everything

```bash
# Stop bridge (Ctrl+C in bridge terminal)
# Stop Docker services
docker compose -f docker/docker-compose.sim.yml down
```

## Full Docker Setup

To run everything in Docker (not yet fully tested):

```bash
NUM_BOTS=3 MAX_TICKS=50 docker compose -f docker/docker-compose.sim.yml up
```

This starts:
- **Minecraft Server** (Paper 1.20.4, flat world, port 25565)
- **Redis** (port 6379)
- **Bridge** (multi-bot Mineflayer, ports 5555+)
- **Agent** (PIANO simulation)

## Architecture

```
┌─────────────┐    ZMQ REQ/REP    ┌──────────────┐    MC Protocol    ┌─────────────┐
│ PIANO Agent │ ◄──────────────► │   Bridge     │ ◄──────────────► │  Minecraft  │
│  (Python)   │    ZMQ PUB/SUB   │  (Node.js)   │                  │   Server    │
└──────┬──────┘                  └──────────────┘                  └─────────────┘
       │
       │ Redis
       ▼
┌─────────────┐
│    Redis    │
│    (SAS)    │
└─────────────┘
```

### Port Allocation

For N bots, ports are allocated as:
- Bot 0: CMD=5555, EVT=5556
- Bot 1: CMD=5557, EVT=5558
- Bot N: CMD=5555+N*2, EVT=5556+N*2

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_BOTS` | 3 | Number of bots to spawn |
| `MAX_TICKS` | 100 | Maximum simulation ticks |
| `MC_HOST` | localhost | Minecraft server host |
| `MC_PORT` | 25565 | Minecraft server port |
| `MC_VERSION` | 1.20.4 | Minecraft version |
| `PIANO_REDIS__HOST` | localhost | Redis host |
| `PIANO_BRIDGE__HOST` | localhost | Bridge host |
| `PIANO_BRIDGE__BASE_COMMAND_PORT` | 5555 | Base ZMQ command port |
| `PIANO_BRIDGE__BASE_EVENT_PORT` | 5556 | Base ZMQ event port |

## CLI Flags

| Flag | Description |
|------|-------------|
| `--agents N` | Number of agents (default: 1) |
| `--ticks N` | Max ticks before exit (default: unlimited) |
| `--mock-llm` | Use MockLLMProvider instead of real LLM |
| `--no-bridge` | Skip bridge connection entirely |
| `--config PATH` | Path to .env config file |
| `--log-level LEVEL` | Log level: DEBUG, INFO, WARNING, ERROR |

## Troubleshooting

### Windows: ZMQ "Proactor event loop does not implement add_reader"

Install tornado for async event loop support:

```bash
uv add tornado
```

### Bots not connecting to MC server

- Ensure MC server is fully started (check logs for "Done!")
- Verify `online-mode=false` in server.properties
- Check bridge logs for connection errors

### PUFFERFISH type fails in Docker

The `itzg/minecraft-server` image has an unbound variable issue with PUFFERFISH type. Use `TYPE=PAPER` instead (configured in docker-compose.sim.yml).

### server.properties "Read-only file system" error

Do NOT mount server.properties as read-only (`:ro`). The MC server writes to it on startup. The docker-compose.sim.yml is already configured correctly.

### ZMQ timeout errors

- Increase `PIANO_BRIDGE__CONNECT_TIMEOUT_S`
- Verify port mapping in docker-compose
- Check for port conflicts: `netstat -an | findstr 5555`

### Port conflicts (EADDRINUSE)

If bridge ports are still in use after a crash:

```bash
# Windows
taskkill //F //IM node.exe

# Linux/Mac
pkill -f "node dist/launcher.js"
```

Wait a few seconds for ports to be released before restarting.

### High memory usage

- Reduce `VIEW_DISTANCE` and `SIMULATION_DISTANCE` in server.properties
- Lower `NUM_BOTS`
- Allocate more memory to Docker

## Related Documents

- [docs/implementation/e2e-simulation.md](implementation/e2e-simulation.md) — E2E architecture details
- [docs/implementation/roadmap.md](implementation/roadmap.md) — Implementation roadmap
