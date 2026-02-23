/**
 * Mineflayer-ZMQ Bridge
 *
 * Runs a Mineflayer bot that exposes a ZMQ REP socket for commands
 * and a ZMQ PUB socket for event streaming.
 */

import mineflayer, { Bot } from "mineflayer";
import { pathfinder, Movements, goals } from "mineflayer-pathfinder";
import * as zmq from "zeromq";
import { v4 as uuidv4 } from "uuid";

// ---------------------------------------------------------------------------
// Configuration from environment
// ---------------------------------------------------------------------------

const MC_HOST = process.env.MC_HOST ?? "localhost";
const MC_PORT = parseInt(process.env.MC_PORT ?? "25565", 10);
const MC_USERNAME = process.env.MC_USERNAME ?? "PIANOBot";
const MC_VERSION = process.env.MC_VERSION ?? "1.20.4";

const ZMQ_CMD_PORT = parseInt(process.env.ZMQ_CMD_PORT ?? "5555", 10);
const ZMQ_EVT_PORT = parseInt(process.env.ZMQ_EVT_PORT ?? "5556", 10);
const ZMQ_CMD_ADDR = `tcp://0.0.0.0:${ZMQ_CMD_PORT}`;
const ZMQ_EVT_ADDR = `tcp://0.0.0.0:${ZMQ_EVT_PORT}`;

const PERCEPTION_INTERVAL_MS = parseInt(
  process.env.PERCEPTION_INTERVAL_MS ?? "1000",
  10
);

const ZMQ_TLS_ENABLED = process.env.ZMQ_TLS_ENABLED === "true";
const ZMQ_CURVE_SERVER_PUBLIC_KEY =
  process.env.ZMQ_CURVE_SERVER_PUBLIC_KEY ?? "";
const ZMQ_CURVE_SERVER_SECRET_KEY =
  process.env.ZMQ_CURVE_SERVER_SECRET_KEY ?? "";

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

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

type CommandHandler = (
  bot: Bot,
  params: Record<string, any>
) => Promise<Record<string, any>>;

function buildHandlers(): Record<string, CommandHandler> {
  return {
    ping: async () => ({ pong: true }),

    move: async (bot, params) => {
      const { x, y, z } = params as { x: number; y: number; z: number };
      const mcData = require("minecraft-data")(bot.version);
      const movements = new Movements(bot as any, mcData);
      (bot as any).pathfinder.setMovements(movements);
      (bot as any).pathfinder.setGoal(
        new goals.GoalBlock(Math.floor(x), Math.floor(y), Math.floor(z))
      );
      // Wait for goal_reached or timeout
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          (bot as any).pathfinder.stop();
          reject(new Error("Movement timed out"));
        }, (params.timeout_ms as number) ?? 30000);

        bot.once("goal_reached" as any, () => {
          clearTimeout(timeout);
          resolve();
        });
      });
      const pos = bot.entity.position;
      return { x: pos.x, y: pos.y, z: pos.z };
    },

    mine: async (bot, params) => {
      const { x, y, z } = params as { x: number; y: number; z: number };
      const block = bot.blockAt(
        (bot as any).vec3(Math.floor(x), Math.floor(y), Math.floor(z))
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
  };
}

// ---------------------------------------------------------------------------
// Bridge process
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log(`[bridge] Creating bot ${MC_USERNAME} -> ${MC_HOST}:${MC_PORT}`);

  const bot = mineflayer.createBot({
    host: MC_HOST,
    port: MC_PORT,
    username: MC_USERNAME,
    version: MC_VERSION,
  });

  bot.loadPlugin(pathfinder);

  // ZMQ sockets
  const repSocket = new zmq.Reply();
  const pubSocket = new zmq.Publisher();

  // Configure CurveZMQ if TLS is enabled
  if (ZMQ_TLS_ENABLED) {
    console.log("[bridge] CurveZMQ TLS enabled");

    // Server-side: the bridge acts as server, clients connect to it
    repSocket.curveServer = true;
    repSocket.curveSecretKey = ZMQ_CURVE_SERVER_SECRET_KEY;
    repSocket.curvePublicKey = ZMQ_CURVE_SERVER_PUBLIC_KEY;

    pubSocket.curveServer = true;
    pubSocket.curveSecretKey = ZMQ_CURVE_SERVER_SECRET_KEY;
    pubSocket.curvePublicKey = ZMQ_CURVE_SERVER_PUBLIC_KEY;
  }

  await repSocket.bind(ZMQ_CMD_ADDR);
  await pubSocket.bind(ZMQ_EVT_ADDR);
  console.log(`[bridge] ZMQ REP bound on ${ZMQ_CMD_ADDR}`);
  console.log(`[bridge] ZMQ PUB bound on ${ZMQ_EVT_ADDR}`);

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
        const data = await handler(bot, cmd.params);
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
  bot.once("spawn", () => {
    console.log(`[bridge] Bot spawned at ${bot.entity.position}`);

    setInterval(async () => {
      const nearbyPlayers = Object.values(bot.entities)
        .filter(
          (e) =>
            e.type === "player" &&
            e.username !== bot.username &&
            e.position.distanceTo(bot.entity.position) < 32
        )
        .map((e) => ({
          name: e.username,
          distance: Math.round(e.position.distanceTo(bot.entity.position)),
          position: { x: e.position.x, y: e.position.y, z: e.position.z },
        }));

      const pos = bot.entity.position;
      const event: BridgeEvent = {
        event_type: "perception",
        data: {
          position: { x: pos.x, y: pos.y, z: pos.z },
          health: bot.health,
          food: bot.food,
          nearby_players: nearbyPlayers,
          time_of_day: bot.time.timeOfDay,
          is_raining: bot.isRaining,
        },
        timestamp: new Date().toISOString(),
      };

      try {
        await pubSocket.send(JSON.stringify(event));
      } catch {
        // PUB is non-blocking; ignore send errors
      }
    }, PERCEPTION_INTERVAL_MS);
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
    console.error(`[bridge] Bot error: ${err.message}`);
  });

  bot.on("end", (reason) => {
    console.log(`[bridge] Bot disconnected: ${reason}`);
    process.exit(1);
  });
}

main().catch((err) => {
  console.error("[bridge] Fatal error:", err);
  process.exit(1);
});
