import { Bot } from "mineflayer";
import { goals } from "mineflayer-pathfinder";

type CommandHandler = (bot: Bot, params: Record<string, any>) => Promise<Record<string, any>>;

export const trade: CommandHandler = async (bot, params) => {
  const { target_agent, offer_items } = params;
  // In MC, trading via item drop at player location
  const target = Object.values(bot.entities).find(
    (e) => e.type === "player" && e.username === target_agent
  );
  if (!target) {
    throw new Error(`Player ${target_agent} not found nearby`);
  }
  // Drop offered items near target
  for (const [itemName, count] of Object.entries(offer_items || {})) {
    const mcData = require("minecraft-data")(bot.version);
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

export const gift: CommandHandler = async (bot, params) => {
  const { target_agent, item, count } = params;
  const target = Object.values(bot.entities).find(
    (e) => e.type === "player" && e.username === target_agent
  );
  if (!target) {
    throw new Error(`Player ${target_agent} not found nearby`);
  }
  const mcData = require("minecraft-data")(bot.version);
  const itemDef = mcData.itemsByName[item];
  if (!itemDef) throw new Error(`Unknown item: ${item}`);
  const invItem = bot.inventory.items().find((i) => i.type === itemDef.id);
  if (!invItem) throw new Error(`Item ${item} not in inventory`);
  await bot.toss(invItem.type, null, count);
  return { gifted: item, count: count ?? 1, to: target_agent };
};

export const follow: CommandHandler = async (bot, params) => {
  const { target_agent } = params;
  const target = Object.values(bot.entities).find(
    (e) => e.type === "player" && e.username === target_agent
  );
  if (!target) {
    throw new Error(`Player ${target_agent} not found`);
  }
  (bot as any).pathfinder.setGoal(new goals.GoalFollow(target, 2), true);
  return { following: target_agent };
};

export const vote: CommandHandler = async (bot, params) => {
  const { proposal_id, choice } = params;
  bot.chat(`[VOTE] ${proposal_id}: ${choice}`);
  return { voted: true, proposal_id, choice };
};

export function getSocialHandlers(): Record<string, CommandHandler> {
  return { trade, gift, follow, vote };
}
