/**
 * Mineflayer-ZMQ Bridge
 *
 * Runs a Mineflayer bot that exposes a ZMQ REP socket for commands
 * and a ZMQ PUB socket for event streaming.
 */

import mineflayer, { Bot } from "mineflayer";
import { Vec3 } from "vec3";
import { pathfinder, Movements, goals } from "mineflayer-pathfinder";
import * as zmq from "zeromq";
import { v4 as uuidv4 } from "uuid";

import { getBasicHandlers } from "./handlers/basic";
import { getSocialHandlers } from "./handlers/social";
import { getCombatHandlers } from "./handlers/combat";
import { getAdvancedHandlers } from "./handlers/advanced";
import { collectPerception } from "./handlers/perception";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BridgeCommand {
  id: string;
  action: string;
  params: Record<string, any>;
  timeout_ms?: number;
}

interface BridgeResponse {
  id: string;
  success: boolean;
  data: Record<string, any>;
  error: string | null;
}

interface BridgeEvent {
  event_type: string;
  data: Record<string, any>;
  timestamp: string;
}

type CommandHandler = (
  bot: Bot,
  params: Record<string, any>
) => Promise<Record<string, any>>;

export interface BotBridgeConfig {
  mcHost: string;
  mcPort: number;
  mcVersion: string;
  username: string;
  zmqCmdPort: number;
  zmqEvtPort: number;
  perceptionIntervalMs: number;
  zmqTlsEnabled?: boolean;
  zmqCurveServerPublicKey?: string;
  zmqCurveServerSecretKey?: string;
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

function buildHandlers(): Record<string, CommandHandler> {
  return {
    // Built-in handlers
    ping: async () => ({ pong: true }),

    move: async (bot, params) => {
      const { x, y, z } = params as { x: number; y: number; z: number };
      const movements = new Movements(bot);
      (bot as any).pathfinder.setMovements(movements);
      (bot as any).pathfinder.setGoal(
        new goals.GoalBlock(Math.floor(x), Math.floor(y), Math.floor(z))
      );
      // Wait for goal_reached or timeout
      await new Promise<void>((resolve, reject) => {
        const onGoalReached = () => { clearTimeout(timeout); resolve(); };
        const timeout = setTimeout(() => {
          bot.removeListener("goal_reached" as any, onGoalReached);
          (bot as any).pathfinder.stop();
          reject(new Error("Movement timed out"));
        }, (params.timeout_ms as number) ?? 30000);
        bot.once("goal_reached" as any, onGoalReached);
      });
      const pos = bot.entity.position;
      return { x: pos.x, y: pos.y, z: pos.z };
    },

    mine: async (bot, params) => {
      const { x, y, z } = params as { x: number; y: number; z: number };
      const block = bot.blockAt(
        new Vec3(Math.floor(x), Math.floor(y), Math.floor(z))
      );
      if (!block) {
        throw new Error(`No block at ${x},${y},${z}`);
      }
      await bot.dig(block);
      return { mined: block.name, position: { x, y, z } };
    },

    craft: async (bot, params) => {
      const { item, count } = params as { item: string; count: number };
      const mcData = require("minecraft-data")(bot.version);
      const itemDef = mcData.itemsByName[item];
      if (!itemDef) {
        throw new Error(`Unknown item: ${item}`);
      }
      const recipes = bot.recipesFor(itemDef.id, null, 1, null);
      if (recipes.length === 0) {
        throw new Error(`No recipe for ${item}`);
      }
      await bot.craft(recipes[0], count, undefined as any);
      return { crafted: item, count };
    },

    chat: async (bot, params) => {
      const { message } = params as { message: string };
      bot.chat(message);
      return { sent: message };
    },

    look: async (bot, params) => {
      const { yaw, pitch } = params as { yaw: number; pitch: number };
      await bot.look(yaw, pitch);
      return { yaw, pitch };
    },

    get_position: async (bot) => {
      const pos = bot.entity.position;
      return { x: pos.x, y: pos.y, z: pos.z };
    },

    get_inventory: async (bot) => {
      const items = bot.inventory.items().map((i) => ({
        name: i.name,
        count: i.count,
        slot: i.slot,
      }));
      return { items };
    },

    // Imported handler modules
    ...getBasicHandlers(),
    ...getSocialHandlers(),
    ...getCombatHandlers(),
    ...getAdvancedHandlers(),
  };
}

// ---------------------------------------------------------------------------
// Bot-bridge creation (reusable)
// ---------------------------------------------------------------------------

export async function createBotBridge(
  config: BotBridgeConfig
): Promise<{ bot: Bot; repSocket: zmq.Reply; pubSocket: zmq.Publisher }> {
  const zmqCmdAddr = `tcp://0.0.0.0:${config.zmqCmdPort}`;
  const zmqEvtAddr = `tcp://0.0.0.0:${config.zmqEvtPort}`;

  console.log(
    `[bridge] Creating bot ${config.username} -> ${config.mcHost}:${config.mcPort}`
  );

  const bot = mineflayer.createBot({
    host: config.mcHost,
    port: config.mcPort,
    username: config.username,
    version: config.mcVersion,
  });

  bot.loadPlugin(pathfinder);

  // ZMQ sockets
  const repSocket = new zmq.Reply();
  const pubSocket = new zmq.Publisher();

  // Configure CurveZMQ if TLS is enabled
  if (config.zmqTlsEnabled) {
    if (!config.zmqCurveServerPublicKey || !config.zmqCurveServerSecretKey) {
      throw new Error("CurveZMQ TLS enabled but keys are not provided");
    }
    console.log(`[bridge] CurveZMQ TLS enabled for ${config.username}`);

    repSocket.curveServer = true;
    repSocket.curveSecretKey = config.zmqCurveServerSecretKey ?? "";
    repSocket.curvePublicKey = config.zmqCurveServerPublicKey ?? "";

    pubSocket.curveServer = true;
    pubSocket.curveSecretKey = config.zmqCurveServerSecretKey ?? "";
    pubSocket.curvePublicKey = config.zmqCurveServerPublicKey ?? "";
  }

  await repSocket.bind(zmqCmdAddr);
  await pubSocket.bind(zmqEvtAddr);
  console.log(`[bridge] ZMQ REP bound on ${zmqCmdAddr}`);
  console.log(`[bridge] ZMQ PUB bound on ${zmqEvtAddr}`);

  const handlers = buildHandlers();

  // -- Command loop --------------------------------------------------------
  (async () => {
    for await (const [msg] of repSocket) {
      let cmd: BridgeCommand;
      try {
        cmd = JSON.parse(msg.toString()) as BridgeCommand;
      } catch {
        await repSocket.send(
          JSON.stringify({
            id: "",
            success: false,
            data: {},
            error: "Invalid JSON",
          } satisfies BridgeResponse)
        );
        continue;
      }

      const handler = handlers[cmd.action];
      if (!handler) {
        await repSocket.send(
          JSON.stringify({
            id: cmd.id,
            success: false,
            data: {},
            error: `Unknown action: ${cmd.action}`,
          } satisfies BridgeResponse)
        );
        continue;
      }

      try {
        const timeoutMs = cmd.timeout_ms ?? 30000;
        const data = await Promise.race([
          handler(bot, cmd.params),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error(`Command '${cmd.action}' timed out after ${timeoutMs}ms`)), timeoutMs)
          ),
        ]);
        await repSocket.send(
          JSON.stringify({
            id: cmd.id,
            success: true,
            data,
            error: null,
          } satisfies BridgeResponse)
        );
      } catch (err: any) {
        await repSocket.send(
          JSON.stringify({
            id: cmd.id,
            success: false,
            data: {},
            error: err.message ?? String(err),
          } satisfies BridgeResponse)
        );
      }
    }
  })();

  // -- Perception event publisher ------------------------------------------
  let perceptionInterval: ReturnType<typeof setInterval> | null = null;
  bot.once("spawn", () => {
    console.log(`[bridge] Bot ${config.username} spawned at ${bot.entity.position}`);

    perceptionInterval = setInterval(async () => {
      const perception = collectPerception(bot);

      const event: BridgeEvent = {
        event_type: "perception",
        data: perception,
        timestamp: new Date().toISOString(),
      };

      try {
        await pubSocket.send(JSON.stringify(event));
      } catch {
        // PUB is non-blocking; ignore send errors
      }
    }, config.perceptionIntervalMs);
  });

  // -- Game events -> PUB --------------------------------------------------
  bot.on("chat", async (username, message) => {
    if (username === bot.username) return;
    const event: BridgeEvent = {
      event_type: "chat",
      data: { username, message },
      timestamp: new Date().toISOString(),
    };
    try {
      await pubSocket.send(JSON.stringify(event));
    } catch {
      // ignore
    }
  });

  bot.on("death", async () => {
    const event: BridgeEvent = {
      event_type: "death",
      data: {},
      timestamp: new Date().toISOString(),
    };
    try {
      await pubSocket.send(JSON.stringify(event));
    } catch {
      // ignore
    }
  });

  bot.on("error", (err) => {
    console.error(`[bridge] Bot ${config.username} error: ${err.message}`);
  });

  bot.on("end", (reason) => {
    if (perceptionInterval) clearInterval(perceptionInterval);
    console.log(`[bridge] Bot ${config.username} disconnected: ${reason}`);
  });

  return { bot, repSocket, pubSocket };
}

// ---------------------------------------------------------------------------
// Single-bot entry point (backward compatible)
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const config: BotBridgeConfig = {
    mcHost: process.env.MC_HOST ?? "localhost",
    mcPort: parseInt(process.env.MC_PORT ?? "25565", 10),
    mcVersion: process.env.MC_VERSION ?? "1.20.4",
    username: process.env.MC_USERNAME ?? "PIANOBot",
    zmqCmdPort: parseInt(process.env.ZMQ_CMD_PORT ?? "5555", 10),
    zmqEvtPort: parseInt(process.env.ZMQ_EVT_PORT ?? "5556", 10),
    perceptionIntervalMs: parseInt(
      process.env.PERCEPTION_INTERVAL_MS ?? "1000",
      10
    ),
    zmqTlsEnabled: process.env.ZMQ_TLS_ENABLED === "true",
    zmqCurveServerPublicKey: process.env.ZMQ_CURVE_SERVER_PUBLIC_KEY ?? "",
    zmqCurveServerSecretKey: process.env.ZMQ_CURVE_SERVER_SECRET_KEY ?? "",
  };

  const { bot } = await createBotBridge(config);

  // In single-bot mode, exit on disconnect (backward compatible behavior)
  bot.on("end", () => {
    process.exit(1);
  });
}

if (require.main === module) {
  main().catch((err) => {
    console.error("[bridge] Fatal error:", err);
    process.exit(1);
  });
}
