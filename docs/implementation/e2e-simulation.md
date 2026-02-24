# E2E Simulation Architecture

> PIANO Agent-to-Minecraft E2E simulation connection architecture.
> Status: **VERIFIED** — 3 bots x 5 ticks on Paper 1.20.4, MockLLM
> Last updated: 2026-02-24

## Overview

The PIANO simulation connects AI agents to a live Minecraft server through a multi-layer architecture:
Perception -> Cognition -> Action loop running in real-time.

## Architecture Diagram

```
+--------------------------------------------------+
|                  PIANO Agent                      |
|                                                   |
|  +--------------+  +------------------------+    |
|  |   Scheduler   |  |  Cognitive Controller  |    |
|  |  (FAST/MID/  |  |  (LLM Decision Making) |    |
|  |   SLOW tiers)|  +------------+-----------+    |
|  +------+-------+              | broadcast        |
|         | tick                  v                  |
|  +------+------+  +------------------------+     |
|  |  Perception  |  |    Output Modules      |     |
|  |   Module     |  | +------------------+   |     |
|  | (Bridge->SAS)|  | | SkillExecutor    |   |     |
|  +--------------+  | | ChatBroadcaster  |   |     |
|                    | +------------------+   |     |
|                    +------------+-----------+     |
|                                 |                  |
|  +------------------------------+--------------+  |
|  |              Bridge Manager                  |  |
|  |  (Multi-bot ZMQ connection pool)            |  |
|  +----------------+----------------------------+  |
|                   | ZMQ REQ/REP + PUB/SUB         |
+-------------------+-------------------------------+
                    |
+-------------------+-------------------------------+
|   Bridge          |         (TypeScript/Node.js)   |
|  +----------------+--------------------+           |
|  |  Launcher (Multi-bot)               |           |
|  |  Bot 0: CMD=5555 EVT=5556          |           |
|  |  Bot 1: CMD=5557 EVT=5558          |           |
|  |  Bot N: CMD=5555+N*2               |           |
|  +-----------+-------------------------+           |
|  Handlers: basic, social, combat, advanced         |
|  Perception: enhanced block/entity/inventory scan  |
+-------------------+-------------------------------+
                    | MC Protocol
+-------------------+-------------------------------+
|  Minecraft Server (Paper 1.20.4, flat world)      |
+---------------------------------------------------+
```

## Perception Loop

1. Bridge publishes perception events every 1s via ZMQ PUB
2. `BridgePerceptionModule` receives via SUB callback
3. Events buffered in deque (max 100)
4. Each FAST tick: drain buffer -> update SAS `PerceptData`

### Event Types

| Event | Description | Update Target |
|-------|-------------|---------------|
| `position` | Bot position (x, y, z, yaw, pitch) | `PerceptData.position` |
| `health` | Health and food levels | `PerceptData.health` |
| `nearby_entities` | Entities within 16 blocks | `PerceptData.nearby_entities` |
| `nearby_blocks` | Block scan results | `PerceptData.nearby_blocks` |
| `inventory` | Full inventory snapshot | `PerceptData.inventory` |
| `chat` | Chat messages from other players | `PerceptData.chat_messages` |
| `action_complete` | Skill execution result | `PerceptData.action_results` |

## Cognition Loop

Standard PIANO CC cycle:

1. Modules tick and write to SAS
2. CC reads module results, applies information bottleneck
3. CC produces `CCDecision` (action, action_params, speaking)
4. Decision broadcast to output modules

### Tick Tiers

| Tier | Interval | Modules |
|------|----------|---------|
| FAST | Every tick (~1s) | Perception, Working Memory, Action Awareness |
| MID | Every 3 ticks | STM consolidation, Social Awareness, Skill Executor |
| SLOW | Every 10 ticks | Goal Generation, Planning, Reflection, LTM consolidation |

## Action Loop

1. `SkillExecutor.on_broadcast()`: maps CC action -> skill -> `BridgeCommand` -> ZMQ REQ
2. `ChatBroadcaster.on_broadcast()`: if speaking, reads utterance -> `bridge.chat()`
3. Bridge executes command via Mineflayer bot
4. Bridge publishes `action_complete` event

### Command Flow

```
CCDecision(action="mine", params={"block": "oak_log"})
  -> SkillExecutor maps to MineSkill
  -> MineSkill.execute() creates BridgeCommand(type="dig", target="oak_log")
  -> BridgeManager.send(cmd) via ZMQ REQ to bridge
  -> Bridge bot executes bot.dig(block)
  -> Bridge publishes {type: "action_complete", success: true} via ZMQ PUB
```

## Port Allocation

| Bot Index | Command Port | Event Port | Bot Username |
|-----------|-------------|------------|--------------|
| 0 | 5555 | 5556 | PIANOBot_0 |
| 1 | 5557 | 5558 | PIANOBot_1 |
| 2 | 5559 | 5560 | PIANOBot_2 |
| N | 5555+N*2 | 5556+N*2 | PIANOBot_N |

For 10 agents, ports 5555-5574 are used (10 command + 10 event ports).

## Configuration

All settings via environment variables with `PIANO_` prefix:

| Setting | Env Variable | Default |
|---------|-------------|---------|
| Bridge host | `PIANO_BRIDGE__HOST` | `localhost` |
| Base command port | `PIANO_BRIDGE__BASE_COMMAND_PORT` | `5555` |
| Base event port | `PIANO_BRIDGE__BASE_EVENT_PORT` | `5556` |
| Connect timeout | `PIANO_BRIDGE__CONNECT_TIMEOUT_S` | `30.0` |
| Bot name prefix | `PIANO_BRIDGE__BOT_NAME_PREFIX` | `PIANOBot` |
| Perception interval | `PIANO_BRIDGE__PERCEPTION_INTERVAL_MS` | `1000` |

### Example: 5-agent simulation with mock LLM

```bash
uv run piano --agents 5 --ticks 100 --mock-llm
```

### Example: 10-agent simulation with real LLM

```bash
export PIANO_LLM__API_KEY="sk-..."
export PIANO_LLM__MODEL="gpt-4o-mini"
uv run piano --agents 10 --ticks 500
```

## Docker Services

See `docker/docker-compose.sim.yml` for full service definitions.

### Service Overview

| Service | Image | Purpose |
|---------|-------|---------|
| `minecraft` | `itzg/minecraft-server` (Paper 1.20.4) | Minecraft server (flat world) |
| `redis` | `redis:7-alpine` | Shared Agent State (SAS) |
| `bridge-N` | `piano-bridge` | TypeScript Mineflayer bot per agent |
| `agent` | `piano-agent` | PIANO agent orchestrator |
| `prometheus` | `prom/prometheus` | Metrics collection |
| `grafana` | `grafana/grafana` | Dashboard and alerting |

### Network Topology

```
agent <--ZMQ--> bridge-0 <--MC Protocol--> minecraft
agent <--ZMQ--> bridge-1 <--MC Protocol--> minecraft
agent <--ZMQ--> bridge-N <--MC Protocol--> minecraft
agent <--Redis--> redis
prometheus --scrape--> agent
grafana --query--> prometheus
```

## Startup Sequence

1. Start Minecraft server, wait for world generation
2. Start Redis
3. Start bridge instances (one per bot), wait for MC connection
4. Start PIANO agent orchestrator, which:
   a. Creates Agent instances with bridge connections
   b. Initializes SAS sections for each agent
   c. Begins scheduler tick loop

## Health Checks

| Component | Check | Interval | Timeout |
|-----------|-------|----------|---------|
| Minecraft | TCP port 25565 | 10s | 5s |
| Redis | `PING` command | 5s | 3s |
| Bridge | ZMQ `ping` REQ/REP | 5s | 3s |
| Agent | Internal heartbeat | 10s | 5s |

## Failure Modes and Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Bridge disconnect | ZMQ timeout | Auto-reconnect with exponential backoff |
| MC server crash | Bridge connection lost | Restart MC + re-join bots |
| Redis unavailable | SAS operation timeout | Retry with backoff, checkpoint on recovery |
| LLM API error | Gateway circuit breaker | Fallback to cached/previous decision |
| Agent crash | Supervisor heartbeat | Restart from last checkpoint |

## Observability

### Key Metrics

- `piano_tick_duration_seconds`: Per-tick processing time
- `piano_bridge_rtt_seconds`: ZMQ round-trip time
- `piano_llm_request_duration_seconds`: LLM API latency
- `piano_sas_operation_duration_seconds`: Redis operation latency
- `piano_agent_action_total`: Action execution count by type

### Grafana Dashboards

- Agent Overview: tick rate, action distribution, health
- Bridge Status: connection state, RTT, message throughput
- LLM Gateway: request rate, latency, error rate, cost
- Infrastructure: Redis memory, CPU, network

## Platform Notes

### Windows

- ZMQ on Windows requires `tornado>=6.1` for Proactor event loop `add_reader` support
- Install via: `uv add tornado`

### Docker

- MC server uses `itzg/minecraft-server` with TYPE=PAPER (not PUFFERFISH — itzg image has unbound variable issue)
- `server.properties` must NOT be mounted read-only (MC server writes to it on startup)
- Bridge Dockerfile uses `npm ci` + `npx tsc` for reproducible builds
- Both agent and bridge Dockerfiles use non-root users for security

## Verified Configuration

The following configuration has been tested and confirmed working:

| Component | Version/Config |
|-----------|---------------|
| MC Server | Paper 1.20.4 via itzg/minecraft-server |
| World Type | flat, seed=12345 |
| Bridge | Node.js 20, TypeScript compiled to dist/ |
| Agent | Python 3.12, uv, MockLLMProvider |
| Bots | 3 bots, ports 5555-5560 |
| Ticks | 5 ticks, full Perception→CC→Action cycle |
| Platform | Windows 11 (local bridge) + Docker (MC+Redis) |

## Related Documents

- [00-overview.md](./00-overview.md) -- Architecture overview
- [05-minecraft-platform.md](./05-minecraft-platform.md) -- Minecraft platform details
- [08-infrastructure.md](./08-infrastructure.md) -- Infrastructure and scaling
- [10-devops.md](./10-devops.md) -- DevOps and operations
- [roadmap.md](./roadmap.md) -- Implementation roadmap
- [../e2e-setup.md](../e2e-setup.md) -- E2E setup guide
