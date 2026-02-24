import { Bot } from "mineflayer";
import { goals } from "mineflayer-pathfinder";
import { CommandHandler, McData } from "../types";

export function getSocialHandlers(mcData: McData): Record<string, CommandHandler> {
  const trade: CommandHandler = async (bot, params) => {
    const { target_agent, offer_items } = params;
    // In MC, trading via item drop at player location
    const target = Object.values(bot.entities).find(
      (e) => e.type === "player" && e.username === target_agent
    );
    if (!target) {
      throw new Error(`Player ${target_agent} not found nearby`);
    }
    // Navigate to target before dropping items
    if (target.position) {
      (bot as any).pathfinder.setGoal(
        new goals.GoalNear(target.position.x, target.position.y, target.position.z, 2)
      );
      await new Promise<void>((resolve) => {
        const onGoalReached = () => { clearTimeout(timeout); resolve(); };
        const timeout = setTimeout(() => {
          bot.removeListener("goal_reached" as any, onGoalReached);
          (bot as any).pathfinder.stop();
          resolve();
        }, 15000);
        bot.once("goal_reached" as any, onGoalReached);
      });
    }
    // Drop offered items near target
    for (const [itemName, count] of Object.entries(offer_items || {})) {
      const itemDef = mcData.itemsByName[itemName];
      if (itemDef) {
        const invItem = bot.inventory.items().find((i) => i.type === itemDef.id);
        if (invItem) {
          await bot.toss(invItem.type, null, count as number);
        }
      }
    }
    return { traded_with: target_agent, success: true };
  };

  const gift: CommandHandler = async (bot, params) => {
    const { target_agent, item, count } = params;
    const target = Object.values(bot.entities).find(
      (e) => e.type === "player" && e.username === target_agent
    );
    if (!target) {
      throw new Error(`Player ${target_agent} not found nearby`);
    }
    // Navigate to target before tossing
    if (target.position) {
      (bot as any).pathfinder.setGoal(
        new goals.GoalNear(target.position.x, target.position.y, target.position.z, 2)
      );
      await new Promise<void>((resolve) => {
        const onGoalReached = () => { clearTimeout(timeout); resolve(); };
        const timeout = setTimeout(() => {
          bot.removeListener("goal_reached" as any, onGoalReached);
          (bot as any).pathfinder.stop();
          resolve();
        }, 15000);
        bot.once("goal_reached" as any, onGoalReached);
      });
    }
    const itemDef = mcData.itemsByName[item];
    if (!itemDef) throw new Error(`Unknown item: ${item}`);
    const invItem = bot.inventory.items().find((i) => i.type === itemDef.id);
    if (!invItem) throw new Error(`Item ${item} not in inventory`);
    await bot.toss(invItem.type, null, count);
    return { gifted: item, count: count ?? 1, to: target_agent };
  };

  const follow: CommandHandler = async (bot, params) => {
    const { target_agent, timeout_ms } = params;
    const target = Object.values(bot.entities).find(
      (e) => e.type === "player" && e.username === target_agent
    );
    if (!target) {
      throw new Error(`Player ${target_agent} not found`);
    }
    (bot as any).pathfinder.setGoal(new goals.GoalFollow(target, 2), true);

    // Follow with a timeout; cancel the persistent goal when it expires
    const followTimeout = timeout_ms ?? 30000;
    await new Promise<void>((resolve) => {
      setTimeout(() => {
        (bot as any).pathfinder.stop();
        resolve();
      }, followTimeout);
    });
    return { following: target_agent, duration_ms: followTimeout };
  };

  const vote: CommandHandler = async (bot, params) => {
    const { proposal_id, choice } = params;
    bot.chat(`[VOTE] ${proposal_id}: ${choice}`);
    return { voted: true, proposal_id, choice };
  };

  const unfollow: CommandHandler = async (bot, params) => {
    (bot as any).pathfinder.setGoal(null);
    return { success: true };
  };

  const send_message: CommandHandler = async (bot, params) => {
    const { target, message } = params;
    bot.chat(`/msg ${target} ${message}`);
    return { success: true };
  };

  const request_help: CommandHandler = async (bot, params) => {
    const { message, target } = params;
    if (target) {
      bot.chat(`/msg ${target} [HELP] ${message || "I need help!"}`);
    } else {
      bot.chat(`[HELP] ${message || "I need help!"}`);
    }
    return { success: true };
  };

  const form_group: CommandHandler = async (bot, params) => {
    const { group_name, members } = params;
    bot.chat(`[GROUP] Forming group "${group_name || "unnamed"}" with: ${(members || []).join(", ")}`);
    return { success: true };
  };

  const leave_group: CommandHandler = async (bot, params) => {
    const { group_name } = params;
    bot.chat(`[GROUP] Leaving group "${group_name || "unnamed"}"`);
    return { success: true };
  };

  return {
    trade, gift, follow, vote,
    unfollow, send_message, request_help, form_group, leave_group,
  };
}
