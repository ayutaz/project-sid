import { Bot } from "mineflayer";
import { Vec3 } from "vec3";
import { goals } from "mineflayer-pathfinder";

type CommandHandler = (bot: Bot, params: Record<string, any>) => Promise<Record<string, any>>;

export const smelt: CommandHandler = async (bot, params) => {
  const { input_item, fuel, count } = params;
  // Find nearby furnace
  const furnaceBlock = bot.findBlock({
    matching: (block: any) => block.name === "furnace" || block.name === "lit_furnace",
    maxDistance: 6,
  });
  if (!furnaceBlock) {
    throw new Error("No furnace found nearby");
  }
  const furnace = await (bot as any).openFurnace(furnaceBlock);
  try {
    const mcData = require("minecraft-data")(bot.version);
    const inputDef = mcData.itemsByName[input_item];
    const fuelDef = mcData.itemsByName[fuel || "coal"];
    if (inputDef) await furnace.putInput(inputDef.id, null, count || 1);
    if (fuelDef) await furnace.putFuel(fuelDef.id, null, count || 1);
    // Wait for smelting
    await new Promise<void>((resolve) => setTimeout(resolve, 10000 * (count || 1)));
    return { smelted: input_item, count: count || 1 };
  } finally {
    furnace.close();
  }
};

export const farm: CommandHandler = async (bot, params) => {
  const { action, crop, x, y, z } = params;
  const pos = new Vec3(Math.floor(x), Math.floor(y), Math.floor(z));

  if (action === "harvest") {
    const block = bot.blockAt(pos);
    if (block) {
      await bot.dig(block);
    }
    return { harvested: true, position: { x, y, z } };
  } else {
    // plant
    const block = bot.blockAt(pos);
    if (block) {
      const mcData = require("minecraft-data")(bot.version);
      const seedItem = mcData.itemsByName[crop || "wheat_seeds"];
      if (seedItem) {
        await bot.equip(seedItem.id, "hand");
        await bot.activateBlock(block);
      }
    }
    return { planted: crop, position: { x, y, z } };
  }
};

export const explore: CommandHandler = async (bot, params) => {
  const { direction, distance } = params;
  const dist = distance || 50;
  const pos = bot.entity.position;
  let target: { x: number; y: number; z: number };

  switch (direction) {
    case "north": target = { x: pos.x, y: pos.y, z: pos.z - dist }; break;
    case "south": target = { x: pos.x, y: pos.y, z: pos.z + dist }; break;
    case "east":  target = { x: pos.x + dist, y: pos.y, z: pos.z }; break;
    case "west":  target = { x: pos.x - dist, y: pos.y, z: pos.z }; break;
    default:      target = { x: pos.x + dist, y: pos.y, z: pos.z }; break;
  }

  (bot as any).pathfinder.setGoal(
    new goals.GoalBlock(Math.floor(target.x), Math.floor(target.y), Math.floor(target.z))
  );

  // Wait for goal or timeout
  await new Promise<void>((resolve) => {
    const onGoalReached = () => { clearTimeout(timeout); resolve(); };
    const timeout = setTimeout(() => {
      bot.removeListener("goal_reached" as any, onGoalReached);
      (bot as any).pathfinder.stop();
      resolve(); // Don't error on explore timeout
    }, 30000);
    bot.once("goal_reached" as any, onGoalReached);
  });

  const finalPos = bot.entity.position;
  return {
    explored: direction,
    distance: dist,
    final_position: { x: finalPos.x, y: finalPos.y, z: finalPos.z },
  };
};

export function getAdvancedHandlers(): Record<string, CommandHandler> {
  return { smelt, farm, explore };
}
