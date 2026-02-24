/**
 * Multi-bot launcher for PIANO simulation.
 * Spawns NUM_BOTS bots, each with unique ZMQ port pairs.
 */

import { createBotBridge, BotBridgeConfig } from "./index";

const NUM_BOTS = parseInt(process.env.NUM_BOTS ?? "1", 10);
const MC_HOST = process.env.MC_HOST ?? "localhost";
const MC_PORT = parseInt(process.env.MC_PORT ?? "25565", 10);
const MC_VERSION = process.env.MC_VERSION ?? "1.20.4";
const BOT_PREFIX = process.env.BOT_PREFIX ?? "PIANOBot";
const BASE_CMD_PORT = parseInt(process.env.ZMQ_CMD_PORT ?? "5555", 10);
const BASE_EVT_PORT = parseInt(process.env.ZMQ_EVT_PORT ?? "5556", 10);
const PERCEPTION_INTERVAL_MS = parseInt(process.env.PERCEPTION_INTERVAL_MS ?? "1000", 10);

async function launchAll(): Promise<void> {
  console.log(`[launcher] Starting ${NUM_BOTS} bot(s)`);

  const launches: Promise<void>[] = [];

  for (let i = 0; i < NUM_BOTS; i++) {
    const config: BotBridgeConfig = {
      mcHost: MC_HOST,
      mcPort: MC_PORT,
      mcVersion: MC_VERSION,
      username: `${BOT_PREFIX}_${i}`,
      zmqCmdPort: BASE_CMD_PORT + i * 2,
      zmqEvtPort: BASE_EVT_PORT + i * 2,
      perceptionIntervalMs: PERCEPTION_INTERVAL_MS,
    };

    launches.push(
      createBotBridge(config).then(() => {
        console.log(`[launcher] Bot ${i} (${config.username}) initialized`);
      }).catch((err) => {
        console.error(`[launcher] Bot ${i} failed:`, err);
      })
    );

    // Stagger bot launches by 2s to avoid MC server overload
    if (i < NUM_BOTS - 1) {
      await new Promise<void>((resolve) => setTimeout(resolve, 2000));
    }
  }

  await Promise.allSettled(launches);
  console.log(`[launcher] All ${NUM_BOTS} bot(s) launched`);
}

process.on("SIGINT", () => {
  console.log("[launcher] Shutting down...");
  process.exit(0);
});
process.on("SIGTERM", () => {
  console.log("[launcher] Shutting down...");
  process.exit(0);
});

launchAll().catch((err) => {
  console.error("[launcher] Fatal error:", err);
  process.exit(1);
});
