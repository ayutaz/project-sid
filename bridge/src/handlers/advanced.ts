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
      // Coal smelts 8 items; compute fuel needed
      const fuelNeeded = Math.ceil(smeltCount / 8);
      if (fuelDef) await furnace.putFuel(fuelDef.id, null, fuelNeeded);

      // Wait for smelting with event-based check, capped at 25 seconds
      const maxWait = Math.min(10000 * smeltCount, 25000);
      await new Promise<void>((resolve) => {
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

      // Take smelted output
      const outputItem = furnace.outputItem();
      if (outputItem) {
        await furnace.takeOutput();
      }

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

  const flee: CommandHandler = async (bot, params) => {
    const { from_x, from_y, from_z, distance } = params;
    const fleeDistance = distance || 20;
    // Move away from the specified position (or stay if no direction)
    const dx = bot.entity.position.x - (from_x ?? bot.entity.position.x);
    const dz = bot.entity.position.z - (from_z ?? bot.entity.position.z);
    const len = Math.sqrt(dx * dx + dz * dz) || 1;
    const targetX = bot.entity.position.x + (dx / len) * fleeDistance;
    const targetZ = bot.entity.position.z + (dz / len) * fleeDistance;

    (bot as any).pathfinder.setGoal(
      new goals.GoalXZ(Math.floor(targetX), Math.floor(targetZ))
    );

    await new Promise<void>((resolve) => {
      const onGoalReached = () => { clearTimeout(timeout); resolve(); };
      const timeout = setTimeout(() => {
        bot.removeListener("goal_reached" as any, onGoalReached);
        (bot as any).pathfinder.setGoal(null);
        resolve();
      }, 10000);
      bot.once("goal_reached" as any, onGoalReached);
    });

    const finalPos = bot.entity.position;
    return {
      success: true,
      final_position: { x: finalPos.x, y: finalPos.y, z: finalPos.z },
    };
  };

  const deposit: CommandHandler = async (bot, params) => {
    const { x, y, z, items } = params;
    const chestBlock = bot.blockAt(new Vec3(x, y, z));
    if (!chestBlock) {
      throw new Error("No block at position");
    }
    const chest = await (bot as any).openContainer(chestBlock);
    try {
      if (items && Array.isArray(items)) {
        for (const item of items) {
          const invItem = bot.inventory.items().find((i: any) => i.name === item.name);
          if (invItem) {
            await chest.deposit(invItem.type, null, item.count || invItem.count);
          }
        }
      }
      return { success: true };
    } finally {
      chest.close();
    }
  };

  const withdraw: CommandHandler = async (bot, params) => {
    const { x, y, z, items } = params;
    const chestBlock = bot.blockAt(new Vec3(x, y, z));
    if (!chestBlock) {
      throw new Error("No block at position");
    }
    const chest = await (bot as any).openContainer(chestBlock);
    try {
      if (items && Array.isArray(items)) {
        for (const item of items) {
          const chestItem = chest.items().find((i: any) => i.name === item.name);
          if (chestItem) {
            await chest.withdraw(chestItem.type, null, item.count || chestItem.count);
          }
        }
      }
      return { success: true };
    } finally {
      chest.close();
    }
  };

  return {
    smelt,
    farm,
    plant: farm,       // Python "plant" action -> farmHandler
    harvest: farm,     // Python "harvest" action -> farmHandler
    explore,
    flee,
    deposit,
    withdraw,
  };
}
