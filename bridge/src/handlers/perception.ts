/**
 * Enhanced perception collection for Mineflayer bots.
 * Extends the basic perception with nearby blocks, entities, and full inventory.
 */

import { Bot } from "mineflayer";
import * as zmq from "zeromq";
import { BridgeEvent } from "../types";

// Block types we care about for perception
const IMPORTANT_BLOCKS = new Set([
  "diamond_ore", "iron_ore", "gold_ore", "coal_ore", "redstone_ore",
  "lapis_ore", "emerald_ore", "copper_ore",
  "deepslate_diamond_ore", "deepslate_iron_ore", "deepslate_gold_ore",
  "deepslate_coal_ore", "deepslate_redstone_ore", "deepslate_lapis_ore",
  "deepslate_emerald_ore", "deepslate_copper_ore",
  "chest", "crafting_table", "furnace", "lit_furnace",
  "anvil", "enchanting_table", "brewing_stand",
  "wheat", "carrots", "potatoes", "beetroots",
  "water", "lava",
  // Basic resource blocks
  "oak_log", "birch_log", "spruce_log", "jungle_log", "acacia_log", "dark_oak_log",
  "stone", "cobblestone",
  "dirt", "grass_block",
  "sand", "gravel",
]);

// Configurable scan radius via env var (default 4 for performance)
const PERCEPTION_RADIUS = parseInt(process.env.PERCEPTION_RADIUS ?? "4", 10);

/**
 * Collect enhanced perception data from the bot.
 */
export function collectPerception(bot: Bot): Record<string, any> {
  const pos = bot.entity.position;

  // Basic perception
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

  // Nearby blocks (configurable radius, default 4)
  const nearbyBlocks: Array<{ name: string; position: { x: number; y: number; z: number } }> = [];
  const radius = PERCEPTION_RADIUS;
  for (let dx = -radius; dx <= radius; dx++) {
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dz = -radius; dz <= radius; dz++) {
        const blockPos = pos.offset(dx, dy, dz);
        const block = bot.blockAt(blockPos);
        if (block && IMPORTANT_BLOCKS.has(block.name)) {
          nearbyBlocks.push({
            name: block.name,
            position: {
              x: Math.floor(blockPos.x),
              y: Math.floor(blockPos.y),
              z: Math.floor(blockPos.z),
            },
          });
        }
      }
    }
  }

  // Nearby entities (16 block range, sorted by distance)
  const nearbyEntities = Object.values(bot.entities)
    .filter(
      (e) =>
        e !== bot.entity &&
        e.position.distanceTo(bot.entity.position) < 16
    )
    .sort((a, b) => {
      const distA = a.position.distanceTo(bot.entity.position);
      const distB = b.position.distanceTo(bot.entity.position);
      return distA - distB;
    })
    .slice(0, 20) // Cap at 20 closest entities
    .map((e) => ({
      type: e.type,
      name: e.name ?? e.username ?? "unknown",
      distance: Math.round(e.position.distanceTo(bot.entity.position)),
      position: { x: e.position.x, y: e.position.y, z: e.position.z },
    }));

  // Full inventory
  const inventory = bot.inventory.items().map((i) => ({
    name: i.name,
    count: i.count,
    slot: i.slot,
  }));

  return {
    position: { x: pos.x, y: pos.y, z: pos.z },
    health: bot.health,
    food: bot.food,
    nearby_players: nearbyPlayers,
    nearby_blocks: nearbyBlocks.slice(0, 50), // Cap at 50
    nearby_entities: nearbyEntities,
    inventory,
    time_of_day: bot.time.timeOfDay,
    is_raining: bot.isRaining,
  };
}

/**
 * Publish an action_complete event.
 * Wired into the command loop in index.ts to notify subscribers
 * when a command finishes execution.
 */
export async function publishActionComplete(
  pub: zmq.Publisher,
  commandId: string,
  success: boolean,
  data: Record<string, any> = {}
): Promise<void> {
  const event: BridgeEvent = {
    event_type: "action_complete",
    data: { command_id: commandId, success, ...data },
    timestamp: new Date().toISOString(),
  };
  try {
    await pub.send(JSON.stringify(event));
  } catch {
    // PUB is non-blocking; ignore errors
  }
}
