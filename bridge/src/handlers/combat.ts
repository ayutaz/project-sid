import { Bot } from "mineflayer";
import { CommandHandler, McData } from "../types";

export function getCombatHandlers(mcData: McData): Record<string, CommandHandler> {
  const attack: CommandHandler = async (bot, params) => {
    const { target } = params;
    const entity = Object.values(bot.entities).find(
      (e) => e.name === target || e.username === target || String(e.id) === String(target)
    );
    if (!entity) {
      throw new Error(`Entity ${target} not found`);
    }
    await bot.attack(entity);
    return { attacked: target, success: true };
  };

  const defend: CommandHandler = async (bot, params) => {
    const { shield } = params;
    if (shield !== false) {
      // Try to equip shield in off-hand
      const shieldItem = mcData.itemsByName["shield"];
      if (shieldItem) {
        const invShield = bot.inventory.items().find((i) => i.type === shieldItem.id);
        if (invShield) {
          await bot.equip(invShield.type, "off-hand" as any);
        }
      }
      await bot.activateItem(true);
    }
    return { defending: true, shield: shield !== false };
  };

  return { attack, defend };
}
