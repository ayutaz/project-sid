# E2E Simulation Setup Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.12+ with uv
- Node.js 20+ (for bridge development only)

## Quick Start

### 1. Start Infrastructure

```bash
docker compose -f docker/docker-compose.sim.yml up -d
```

This starts:
- **Minecraft Server** (Paper 1.20.4, flat world, port 25565)
- **Redis** (port 6379)
- **Bridge** (multi-bot Mineflayer, ports 5555+)
- **Agent** (PIANO simulation)

### 2. Wait for MC Server

The Minecraft server takes ~2 minutes to start. Check status:

```bash
docker compose -f docker/docker-compose.sim.yml logs -f minecraft
```

Wait for: `[Server] Done! For help, type "help"`

### 3. Run Simulation

With Docker (recommended):
```bash
NUM_BOTS=5 MAX_TICKS=100 docker compose -f docker/docker-compose.sim.yml up agent
```

Or locally (requires bridge running):
```bash
uv run piano --agents 5 --ticks 100 --mock-llm
```

### 4. Monitor

Watch agent logs:
```bash
docker compose -f docker/docker-compose.sim.yml logs -f agent
```

Watch bridge logs:
```bash
docker compose -f docker/docker-compose.sim.yml logs -f bridge
```

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
| `PIANO_REDIS__HOST` | localhost | Redis host |
| `PIANO_BRIDGE__HOST` | localhost | Bridge host |

## Troubleshooting

### Bots not connecting
- Ensure MC server is fully started (check logs for "Done!")
- Verify `online-mode=false` in server.properties
- Check bridge logs for connection errors

### ZMQ timeout errors
- Increase `PIANO_BRIDGE__CONNECT_TIMEOUT_S`
- Verify port mapping in docker-compose

### High memory usage
- Reduce `VIEW_DISTANCE` and `SIMULATION_DISTANCE`
- Lower `NUM_BOTS`
