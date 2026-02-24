/**
 * Shared types for the Mineflayer-ZMQ bridge.
 */

import { Bot } from "mineflayer";

/**
 * Handler function for a single bridge command.
 */
export type CommandHandler = (
  bot: Bot,
  params: Record<string, any>
) => Promise<Record<string, any>>;

/**
 * An event published over the ZMQ PUB socket.
 */
export interface BridgeEvent {
  event_type: string;
  data: Record<string, any>;
  timestamp: string;
}

/**
 * Cached minecraft-data instance for a given bot version.
 * Created once per bot and passed to handler factories.
 */
export type McData = ReturnType<typeof import("minecraft-data")>;
