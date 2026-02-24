/**
 * Basic command handlers for Mineflayer bot.
 * These extend the built-in handlers in index.ts with additional actions.
 */

import { Bot } from "mineflayer";
import { Vec3 } from "vec3";

type CommandHandler = (
  bot: Bot,
  params: Record<string, any>
) => Promise<Record<string, any>>;

export const place: CommandHandler = async (bot, params) => {
  const { x, y, z, block_type } = params;
  const referenceBlock = bot.blockAt(
    new Vec3(Math.floor(x), Math.floor(y) - 1, Math.floor(z))
  );
  if (!referenceBlock) {
    throw new Error(`No reference block at ${x},${y - 1},${z}`);
  }
  // Find the block item in inventory
  const mcData = require("minecraft-data")(bot.version);
  const blockItem = mcData.itemsByName[block_type];
  if (blockItem) {
    await bot.equip(blockItem.id, "hand");
  }
  await (bot as any).placeBlock(referenceBlock, new Vec3(0, 1, 0));
  return { placed: block_type, position: { x, y, z } };
};

export const eat: CommandHandler = async (bot, _params) => {
  // Find food in inventory
  const foodItems = bot.inventory.items().filter((item) => {
    const mcData = require("minecraft-data")(bot.version);
    const food = mcData.foodsByName?.[item.name];
    return food !== undefined;
  });
  if (foodItems.length === 0) {
    throw new Error("No food items in inventory");
  }
  await bot.equip(foodItems[0].type, "hand");
  await bot.consume();
  return { ate: foodItems[0].name };
};

export const equip: CommandHandler = async (bot, params) => {
  const { item, destination } = params;
  const mcData = require("minecraft-data")(bot.version);
  const itemDef = mcData.itemsByName[item];
  if (!itemDef) {
    throw new Error(`Unknown item: ${item}`);
  }
  const dest = destination ?? "hand";
  await bot.equip(itemDef.id, dest as any);
  return { equipped: item, destination: dest };
};

export const use: CommandHandler = async (bot, _params) => {
  await bot.activateItem();
  return { used: true };
};

export const drop: CommandHandler = async (bot, params) => {
  const { item, count } = params;
  const mcData = require("minecraft-data")(bot.version);
  const itemDef = mcData.itemsByName[item];
  if (!itemDef) {
    throw new Error(`Unknown item: ${item}`);
  }
  const invItem = bot.inventory.items().find((i) => i.type === itemDef.id);
  if (!invItem) {
    throw new Error(`Item ${item} not in inventory`);
  }
  await bot.toss(invItem.type, null, count);
  return { dropped: item, count: count ?? invItem.count };
};

export function getBasicHandlers(): Record<string, CommandHandler> {
  return { place, eat, equip, use, drop };
}
