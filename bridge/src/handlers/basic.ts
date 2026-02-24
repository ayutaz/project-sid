/**
 * Basic command handlers for Mineflayer bot.
 * These extend the built-in handlers in index.ts with additional actions.
 */

import { Bot } from "mineflayer";
import { Vec3 } from "vec3";
import { CommandHandler, McData } from "../types";

export function getBasicHandlers(mcData: McData): Record<string, CommandHandler> {
  const place: CommandHandler = async (bot, params) => {
    const { x, y, z, block_type, face } = params;
    const referenceBlock = bot.blockAt(
      new Vec3(Math.floor(x), Math.floor(y) - 1, Math.floor(z))
    );
    if (!referenceBlock) {
      throw new Error(`No reference block at ${x},${y - 1},${z}`);
    }
    // Find the block item in inventory
    const blockItem = mcData.itemsByName[block_type];
    if (blockItem) {
      await bot.equip(blockItem.id, "hand");
    }
    // Use provided face vector or default to (0, 1, 0) = top face
    const faceVec = face
      ? new Vec3(face.x ?? 0, face.y ?? 1, face.z ?? 0)
      : new Vec3(0, 1, 0);
    await (bot as any).placeBlock(referenceBlock, faceVec);
    return { placed: block_type, position: { x, y, z } };
  };

  const eat: CommandHandler = async (bot, _params) => {
    // Find food in inventory
    const foodItems = bot.inventory.items().filter((item) => {
      const food = (mcData as any).foodsByName?.[item.name];
      return food !== undefined;
    });
    if (foodItems.length === 0) {
      throw new Error("No food items in inventory");
    }
    await bot.equip(foodItems[0].type, "hand");
    await bot.consume();
    return { ate: foodItems[0].name };
  };

  const equip: CommandHandler = async (bot, params) => {
    const { item, destination } = params;
    const itemDef = mcData.itemsByName[item];
    if (!itemDef) {
      throw new Error(`Unknown item: ${item}`);
    }
    const dest = destination ?? "hand";
    await bot.equip(itemDef.id, dest as any);
    return { equipped: item, destination: dest };
  };

  const use: CommandHandler = async (bot, _params) => {
    await bot.activateItem();
    return { used: true };
  };

  const drop: CommandHandler = async (bot, params) => {
    const { item, count } = params;
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

  return { place, eat, equip, use, drop };
}
