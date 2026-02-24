import { Bot } from "mineflayer";
import { Vec3 } from "vec3";
import { goals } from "mineflayer-pathfinder";
import { CommandHandler, McData } from "../types";

export function getAdvancedHandlers(mcData: McData): Record<string, CommandHandler> {
  const smelt: CommandHandler = async (bot, params) => {
    const { input_item, fuel, count } = params;
    const smeltCount = count || 1;
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
      const inputDef = mcData.itemsByName[input_item];
      const fuelDef = mcData.itemsByName[fuel || "coal"];
      if (inputDef) await furnace.putInput(inputDef.id, null, smeltCount);
      if (fuelDef) await furnace.putFuel(fuelDef.id, null, smeltCount);

      // Wait for smelting with event-based check, capped at 25 seconds
      const maxWait = Math.min(10000 * smeltCount, 25000);
      await new Promise<void>((resolve) => {
        let itemsCollected = 0;
        const checkInterval = setInterval(() => {
          const outputSlot = furnace.outputItem();
          if (outputSlot && outputSlot.count >= smeltCount) {
            clearInterval(checkInterval);
            clearTimeout(timeout);
            resolve();
          }
        }, 1000);
        const timeout = setTimeout(() => {
          clearInterval(checkInterval);
          resolve(); // Don't error on smelt timeout, return what we have
        }, maxWait);
      });

      return { smelted: input_item, count: smeltCount };
    } finally {
      furnace.close();
    }
  };

  const farm: CommandHandler = async (bot, params) => {
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
        const seedItem = mcData.itemsByName[crop || "wheat_seeds"];
        if (seedItem) {
          await bot.equip(seedItem.id, "hand");
          await bot.activateBlock(block);
        }
      }
      return { planted: crop, position: { x, y, z } };
    }
  };

  const explore: CommandHandler = async (bot, params) => {
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

    // Use GoalNear instead of GoalBlock to handle terrain height variation
    (bot as any).pathfinder.setGoal(
      new goals.GoalNear(Math.floor(target.x), Math.floor(target.y), Math.floor(target.z), 3)
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

  return { smelt, farm, explore };
}
