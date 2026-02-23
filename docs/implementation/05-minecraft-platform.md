# 5. Minecraft基盤技術調査

> 概要: Mineflayer、マルチボット接続、スキル実行、環境インターフェースの技術分析と推奨アーキテクチャ
> 対応論文セクション: 2.1 (Environmental Interaction), Appendix A (Technical Implementation)
> 最終更新: 2026-02-23

---

## 目次

1. [Mineflayerの技術分析](#1-mineflayerの技術分析)
2. [マルチボット接続](#2-マルチボット接続)
3. [スキル実行モジュール](#3-スキル実行モジュール)
4. [環境インターフェース](#4-環境インターフェース)
5. [既存プロジェクトとの比較](#5-既存プロジェクトとの比較)
6. [推奨アーキテクチャ](#6-推奨アーキテクチャ)

---

## 1. Mineflayerの技術分析

### 1.1 概要

[Mineflayer](https://github.com/PrismarineJS/mineflayer)は、PrismarineJSプロジェクトが開発するMinecraft bot向け高レベルJavaScript APIライブラリである。Project Sidの再現実装において、エージェントとMinecraft環境間のインターフェース層として最も有力な選択肢となる。

| 項目 | 詳細 |
|---|---|
| 言語 | JavaScript (Node.js 18+) |
| 対応バージョン | Minecraft 1.8 - 1.21.x |
| プロトコル | Minecraft Protocol (minecraft-protocol パッケージ) |
| プログラミングモデル | イベント駆動 (EventEmitter) |
| ライセンス | MIT |
| パッケージ管理 | npm (`npm install mineflayer`) |

### 1.2 コア機能

#### ボット制御

```javascript
const mineflayer = require('mineflayer');

const bot = mineflayer.createBot({
  host: 'localhost',
  port: 25565,
  username: 'Agent_001',
  version: '1.20.4'
});
```

**移動制御**:
- `bot.entity.position` — 現在位置の取得 (Vec3)
- `bot.setControlState(control, state)` — 低レベル移動 (forward, back, left, right, jump, sprint, sneak)
- `bot.lookAt(position)` — 視線方向の制御
- `bot.navigate` — pathfinderプラグインによる自律ナビゲーション

**採掘 (Digging)**:
- `bot.dig(block, forceLook?)` — 指定ブロックの採掘 (Promise)
- `bot.canDigBlock(block)` — 採掘可能性の判定
- `bot.digTime(block)` — 採掘所要時間の推定

**建築 (Building)**:
- `bot.placeBlock(referenceBlock, faceVector)` — ブロック設置
- `bot.creative.flyTo(position)` — クリエイティブモードでの移動

**インベントリ管理**:
- `bot.inventory` — インベントリウィンドウへのアクセス
- `bot.inventory.items()` — 所持アイテム一覧
- `bot.equip(item, destination)` — アイテムの装備
- `bot.toss(itemType, metadata, count)` — アイテムの投棄
- `bot.transfer(options)` — コンテナ間のアイテム移動

**クラフト**:
- `bot.recipesFor(itemType, metadata, minResultCount, craftingTable)` — 利用可能レシピの取得
- `bot.craft(recipe, count, craftingTable)` — アイテムのクラフト (Promise)
- `bot.recipesAll(itemType, metadata, craftingTable)` — 全レシピの列挙

**エンティティ操作**:
- `bot.entities` — 周辺エンティティの取得
- `bot.nearestEntity(match?)` — 最近接エンティティの検索
- `bot.attack(entity)` — エンティティへの攻撃
- `bot.useOn(entity)` — エンティティへのアイテム使用

#### イベントシステム

Mineflayerのイベント駆動モデルはProject Sidの**行動認識モジュール**のフィードバックループに直接活用できる。

```javascript
// 主要イベント例
bot.on('chat', (username, message) => { /* チャット受信 */ });
bot.on('health', () => { /* 体力変化 */ });
bot.on('death', () => { /* 死亡 */ });
bot.on('spawn', () => { /* スポーン */ });
bot.on('playerJoined', (player) => { /* プレイヤー参加 */ });
bot.on('playerLeft', (player) => { /* プレイヤー退出 */ });
bot.on('entitySpawn', (entity) => { /* エンティティ出現 */ });
bot.on('blockUpdate', (oldBlock, newBlock) => { /* ブロック変更 */ });
bot.on('diggingCompleted', (block) => { /* 採掘完了 */ });
bot.on('goal_reached', () => { /* 目標到達(pathfinder) */ });
```

### 1.3 プラグインエコシステム

Mineflayerはプラグイン形式の拡張をサポートし、PrismarineJSコミュニティが多数のプラグインを提供している。

| プラグイン | 機能 | Project Sidでの用途 |
|---|---|---|
| **mineflayer-pathfinder** | A*ベースの自律ナビゲーション、静的/動的/複合ゴール設定 | エージェントの移動制御、探索 |
| **mineflayer-collectblock** | ブロック採集の高レベルAPI (パスファインディング+ツール選択+採掘+回収を自動化) | リソース収集タスク |
| **mineflayer-pvp** | 戦闘AI (近接+遠距離) | 警備員ロール、防衛行動 |
| **mineflayer-tool** | 最適ツールの自動選択 | 効率的な採掘 |
| **mineflayer-auto-eat** | 空腹時の自動食事 | 生存維持 |
| **mineflayer-armor-manager** | 防具の自動装備 | 防御力の最適化 |
| **mineflayer-hawkeye** | 弓の自動照準 | 遠距離戦闘 |
| **mineflayer-viewer** | ブラウザベースの3Dビュー | デバッグ・可視化 |

#### pathfinderの詳細

```javascript
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');

bot.loadPlugin(pathfinder);

const mcData = require('minecraft-data')(bot.version);
const movements = new Movements(bot, mcData);
movements.allowSprinting = true;
movements.canDig = true;

bot.pathfinder.setMovements(movements);
bot.pathfinder.setGoal(new goals.GoalBlock(x, y, z));
```

**GoalTypes**:
- `GoalBlock(x, y, z)` — 特定座標への到達
- `GoalNear(x, y, z, range)` — 範囲内への接近
- `GoalXZ(x, z)` — XZ平面での到達
- `GoalFollow(entity, range)` — エンティティの追跡
- `GoalGetToBlock(x, y, z)` — ブロック隣接位置への到達
- `GoalCompositeAny(goals)` — 複数ゴールのOR結合
- `GoalCompositeAll(goals)` — 複数ゴールのAND結合

### 1.4 制約と課題

| 制約 | 詳細 | 影響度 |
|---|---|---|
| **メモリ消費** | 各ボットがワールド状態を個別に保持するため、大量接続時にRAMが急増 | 高 |
| **シングルスレッド** | Node.jsのシングルスレッドモデルにより、1プロセスでの大量ボット制御に限界 | 高 |
| **視覚情報なし** | テキストベースの知覚のみ、画像/映像は取得不可 | 中 |
| **空間推論の限界** | ブロック座標ベースの情報のみ、3D空間認識は限定的 | 中 |
| **サーバー依存** | サーバー側の性能がボトルネックになりうる | 高 |

---

## 2. マルチボット接続

### 2.1 500-1000ボットの同時接続設計

Project Sidの500-1000体規模のシミュレーションを実現するには、単純な1プロセス方式では不可能であり、分散アーキテクチャが必須となる。

#### アーキテクチャ戦略

```
                    [プロキシ層 (Velocity)]
                    /       |        \
            [Paper #1] [Paper #2] [Paper #3]  ... [Paper #N]
            (Town A)   (Town B)   (Town C)       (Rural)
               |          |          |              |
         [Bot Pool]  [Bot Pool] [Bot Pool]    [Bot Pool]
         (33 bots)   (33 bots)  (33 bots)    (~100 bots)
```

#### プロセス分離モデル

大量ボットの安定稼働には、以下の階層的プロセス分離が必要:

**Level 1: Worker Threads (1プロセス内)**
```javascript
const { Worker } = require('worker_threads');

// 各Worker Threadが5-10ボットを管理
for (let i = 0; i < botCount; i += BOTS_PER_WORKER) {
  const worker = new Worker('./bot-worker.js', {
    workerData: {
      startIndex: i,
      count: Math.min(BOTS_PER_WORKER, botCount - i),
      serverHost: 'localhost',
      serverPort: 25565
    }
  });
}
```

**Level 2: マルチプロセス (1マシン内)**
- Node.jsの`cluster`モジュールまたは個別プロセスとして起動
- 各プロセスが10-20ボットを管理
- IPC (Inter-Process Communication) でコーディネーション

**Level 3: マルチマシン (分散)**
- 複数のマシンにBotプロセスを分散
- WebSocket/gRPCでオーケストレータと通信
- 各マシンが50-100ボットを担当

#### メモリ見積もり

| 構成 | ボット/プロセス | メモリ/ボット | 合計(500ボット) |
|---|---|---|---|
| 素朴な実装 | 500 | ~100-200MB | 50-100GB |
| Worker Thread分離 | 10 | ~50-80MB | 25-40GB |
| ワールド状態共有 | 10 | ~20-40MB | 10-20GB |
| 最適化済み | 10 | ~15-30MB | 7.5-15GB |

**メモリ最適化手法**:
- チャンク読み込み範囲の最小化 (`viewDistance: 2-4`)
- 不要なエンティティトラッキングの無効化
- ワールド状態の共有メモリ化 (SharedArrayBuffer)
- 非アクティブボットの一時休止

### 2.2 サーバーソフトウェアの選定

#### 比較表

| サーバー | ベース | 特徴 | 500+ボット適性 |
|---|---|---|---|
| **Paper** | Spigot fork | 最も広く使われる高性能サーバー、豊富なプラグイン | 良好 |
| **Pufferfish** | Paper fork | 性能特化、エンティティ最適化、非同期パスファインディング | 最適 |
| **Purpur** | Pufferfish fork | 追加設定オプション、実験的機能 | 良好 |
| **Fabric** | 独立実装 | MOD対応、軽量 | 限定的 |
| **Vanilla** | 公式 | 基準実装 | 不適 |

**推奨: Pufferfish (または Pufferfish+)**
- エンティティトラッキングの最適化
- 完全非同期パスファインディング
- 大規模サーバー向けに設計
- Paper互換プラグインが利用可能

**Pufferfish選定に伴うリスク**:

| リスク | 詳細 | 緩和策 |
|---|---|---|
| **Pufferfish+（有料版）依存** | 非同期エンティティトラッキング等の高度な最適化はPufferfish+でのみ利用可能 | Phase 2で無料版の性能を評価し、有料版が必要かを判断。必要なら予算に組み込む |
| **コミュニティ規模の小ささ** | Paperと比較してユーザー数・プラグイン検証報告が少なく、プラグイン互換性問題時のサポートが限定的 | Paper互換のプラグインのみを使用し、Pufferfish固有のAPIには依存しない設計にする |
| **Paper本体の最適化進展** | Paper自体の性能最適化が進んでおり、Pufferfishの性能優位性が縮小傾向にある | Phase 2-3でPaper vs Pufferfishのベンチマーク比較を実施し、優位性が確認できない場合はPaperにフォールバック |
| **プロジェクト継続性** | Paper本体のフォークであるため、Paper側の大幅な変更に追従が遅れる可能性 | Paperへのフォールバックパスを常に維持。Pufferfish固有の設定は最小限にとどめる |

#### サーバーチューニング

**server.properties**:
```properties
view-distance=4          # デフォルト10 -> 4に削減 (ボットに視覚は不要)
simulation-distance=4    # チャンクシミュレーション範囲の縮小
max-players=600          # 最大接続数
network-compression-threshold=256
entity-broadcast-range-percentage=50
```

**paper-world-defaults.yml (Paper/Pufferfish)**:
```yaml
chunks:
  max-auto-save-chunks-per-tick: 6   # 自動保存の制限
  delay-chunk-unloads-by: 5s
entities:
  armor-stands:
    tick: false                       # 不要なエンティティティックの無効化
  marker:
    tick: false
collisions:
  max-entity-collisions: 2           # 衝突判定の簡略化
spawn-limits:
  monsters: 20                       # Mob生成の大幅制限
  animals: 5
  water-animals: 2
  ambient: 1
tick-rates:
  mob-spawner: 4                     # スポナーティックの削減
```

**JVM最適化フラグ (Aikar's Flags)**:
```bash
java -Xms8G -Xmx8G \
  -XX:+UseG1GC \
  -XX:+ParallelRefProcEnabled \
  -XX:MaxGCPauseMillis=200 \
  -XX:+UnlockExperimentalVMOptions \
  -XX:+DisableExplicitGC \
  -XX:+AlwaysPreTouch \
  -XX:G1NewSizePercent=30 \
  -XX:G1MaxNewSizePercent=40 \
  -XX:G1HeapRegionSize=8M \
  -XX:G1ReservePercent=20 \
  -XX:G1HeapWastePercent=5 \
  -XX:G1MixedGCCountTarget=4 \
  -XX:InitiatingHeapOccupancyPercent=15 \
  -XX:G1MixedGCLiveThresholdPercent=90 \
  -XX:G1RSetUpdatingPauseTimePercent=5 \
  -XX:SurvivorRatio=32 \
  -XX:+PerfDisableSharedMem \
  -XX:MaxTenuringThreshold=1 \
  -jar pufferfish.jar
```

### 2.3 マルチサーバー分散構成

500-1000体規模では、単一サーバーでは処理限界に達するため、プロキシベースのマルチサーバー構成が必要となる。

#### Velocity プロキシ

[Velocity](https://github.com/PaperMC/Velocity) は PaperMCチームが開発する次世代プロキシサーバーである。

| 特徴 | 詳細 |
|---|---|
| パフォーマンス | 低レイテンシ、高スループット |
| セキュリティ | Modern player information forwarding |
| API | 拡張性の高いプラグインAPI |
| 互換性 | Paper/Pufferfish/Purpur と完全互換 |

#### 分散構成の設計

```
[500 Mineflayer Bots]
        |
  [Velocity Proxy]  ← ルーティング・ロードバランシング
   /    |    |    \
[Sv1] [Sv2] [Sv3] [Sv4]
Town   Town  Town  Rural
A,B    C,D   E,F   Areas

各サーバー: Pufferfish, 80-150ボット
```

**Velocity構成 (velocity.toml)**:
```toml
[servers]
town-ab = "server1:25566"
town-cd = "server2:25567"
town-ef = "server3:25568"
rural   = "server4:25569"

try = ["town-ab"]

[forced-hosts]
"lobby.example.com" = ["town-ab"]
```

#### サーバー間通信

ボットが町間を移動する場合、Velocityがサーバー間転送を処理する。ただし、Project Sidの文脈では以下の追加考慮が必要:

- **エージェント状態の永続化**: サーバー移動時にエージェントの記憶・目標・社会関係データを維持
- **クロスサーバーチャット**: 異なるサーバー上のエージェント間コミュニケーション
- **グローバルイベント**: 投票・税制などシミュレーション全体に関わるイベントの同期

**推奨通信レイヤー**:
```
[Agent Orchestra (中央制御)]
    |--- Redis/NATS (メッセージング)
    |--- PostgreSQL (永続状態)
    |
[Server 1] <-> [Server 2] <-> [Server 3] <-> [Server 4]
```

### 2.4 ネットワーク帯域とレイテンシ

| 項目 | 値 (推定) |
|---|---|
| ボット1体あたり帯域 | 5-20 KB/s (view-distance=4) |
| 500ボット総帯域 | 2.5-10 MB/s |
| アクション遅延許容 | 50-200ms (ゲーム内1-4ティック) |
| LLM呼び出し遅延 | 500-5000ms (GPT-4o) |

LLM呼び出し遅延がMinecraftのティックレート(50ms)よりはるかに大きいため、LLM応答待ちの間にゲーム内アクションを非同期で処理する設計が必須。

---

## 3. スキル実行モジュール

### 3.1 論文におけるスキル実行の位置づけ

PIANOアーキテクチャにおいて、スキル実行モジュールは「高レベル意図を環境操作に変換する」コンポーネントであり、ベースラインアーキテクチャにも含まれる基本モジュールである。

```
[認知コントローラ (CC)]
    |
    | 高レベル意図: "鉄インゴットを作りたい"
    v
[スキル実行モジュール]
    |
    | 分解:
    |   1. 鉄鉱石を見つける
    |   2. 鉄鉱石を採掘する
    |   3. かまどを見つける/クラフトする
    |   4. 石炭を取得する
    |   5. かまどで精錬する
    v
[Mineflayer API呼び出し]
    |
    v
[Minecraft環境]
```

### 3.2 高レベル意図から低レベルアクションへの変換設計

#### 変換パイプライン

```
意図 (Intent)
  |
  v
目標分解 (Goal Decomposition)
  |  LLMによるサブゴール分解
  v
アクションプラン (Action Plan)
  |  順序付きアクションリスト
  v
プリミティブアクション (Primitive Actions)
  |  Mineflayer API呼び出し
  v
実行 & フィードバック (Execute & Feedback)
  |  結果の検証、行動認識モジュールへの通知
  v
完了 or リトライ
```

#### プリミティブアクション一覧

| カテゴリ | アクション | Mineflayer API |
|---|---|---|
| **移動** | 指定座標への移動 | `pathfinder.setGoal()` |
| | エンティティへの追従 | `GoalFollow(entity, range)` |
| | 方向転換・視線制御 | `bot.lookAt()` |
| **採掘** | ブロックの採掘 | `bot.dig(block)` |
| | ブロックの探索 | `bot.findBlocks()` |
| | アイテムの回収 | `collectblock.collect()` |
| **建築** | ブロックの設置 | `bot.placeBlock()` |
| | 構造物の建築 | カスタム関数 (座標配列+placeBlock) |
| **クラフト** | レシピの検索 | `bot.recipesFor()` |
| | アイテムのクラフト | `bot.craft()` |
| | 精錬 | かまど操作 (openFurnace) |
| **インベントリ** | アイテムの装備 | `bot.equip()` |
| | アイテムの投棄 | `bot.toss()` |
| | チェストへの格納 | `bot.openContainer()` + transfer |
| **戦闘** | 近接攻撃 | `bot.attack()` |
| | 遠距離攻撃 | `hawkeye.autoAttack()` |
| | 逃走 | `pathfinder.setGoal(awayFrom)` |
| **社会** | チャット送信 | `bot.chat()` |
| | ウィスパー送信 | `bot.whisper()` |
| | 近接プレイヤーの検出 | `bot.nearestEntity()` |

### 3.3 Voyager風スキルライブラリ設計

VoyagerのスキルライブラリアプローチをProject Sidに適応させることで、エージェントが経験的にスキルを蓄積・再利用できるシステムを構築できる。

#### スキルの定義

```typescript
interface Skill {
  name: string;                    // "mine_iron_ore"
  description: string;            // "Find and mine iron ore blocks"
  code: string;                   // 実行可能なJavaScriptコード
  embedding: number[];            // description の埋め込みベクトル
  dependencies: string[];         // 前提スキル ["find_block", "equip_pickaxe"]
  successRate: number;            // 成功率の統計
  avgExecutionTime: number;       // 平均実行時間
  requiredItems: string[];        // 必要アイテム ["stone_pickaxe"]
  producedItems: string[];        // 生成アイテム ["raw_iron"]
}
```

#### スキルライブラリの構成

```
[Skill Library]
  |
  |-- Base Skills (事前定義)
  |     |-- move_to(x, y, z)
  |     |-- mine_block(blockType)
  |     |-- place_block(blockType, position)
  |     |-- craft_item(itemName)
  |     |-- attack_entity(entityType)
  |     |-- chat(message)
  |     |-- open_container(containerType)
  |
  |-- Composite Skills (合成スキル)
  |     |-- gather_wood()          = find_tree + mine_block("log")
  |     |-- make_planks()          = gather_wood + craft_item("planks")
  |     |-- make_crafting_table()  = make_planks + craft_item("crafting_table")
  |     |-- build_shelter()        = gather_wood + make_planks + place_block(...)
  |
  |-- Learned Skills (LLM生成)
        |-- 実行時にLLMが生成し、成功したら保存
        |-- 類似タスクでベクトル検索により再利用
```

#### スキル検索と実行フロー

```
1. 高レベル目標を受信: "ダイヤモンドのツルハシを作る"
2. 目標のembeddingを計算
3. ベクトルDBで類似スキルをtop-5検索
4. 該当スキルがあれば実行を試行
5. なければLLMにスキルコード生成を依頼
6. 環境フィードバックを収集しながら実行
7. 成功 → スキルライブラリに保存
8. 失敗 → エラー情報をLLMにフィードバックして改善
```

### 3.4 アクション実行のフィードバックループ

論文が強調する**行動認識モジュール (Action Awareness)** との統合が、スキル実行の信頼性に決定的に重要である。

```
[スキル実行]
    |
    | アクション実行
    v
[Minecraft環境]
    |
    | 結果イベント
    v
[行動認識モジュール]
    |
    | 期待結果 vs 実際結果の比較
    |
    |--- 一致 → 次のアクションへ
    |
    |--- 不一致 (ハルシネーション検出)
           |
           | エラー種別の分類:
           |   - アイテム未取得 (誤認)
           |   - 位置ずれ
           |   - ツール不足
           |   - 環境変化
           |
           v
         [修正アクション]
           |
           |--- リトライ (同一アクション)
           |--- 代替手段 (別のアプローチ)
           |--- 目標変更 (前提条件の充足から)
           |--- エスカレーション (CCへ報告)
```

#### 実装例: 行動認識によるフィードバック

```javascript
class ActionAwarenessChecker {
  async verifyAction(action, expectedResult) {
    const actualInventory = bot.inventory.items();
    const actualPosition = bot.entity.position;
    const actualHealth = bot.health;

    const discrepancies = [];

    // インベントリの期待値チェック
    if (expectedResult.newItems) {
      for (const item of expectedResult.newItems) {
        const found = actualInventory.find(i => i.name === item.name);
        if (!found || found.count < item.count) {
          discrepancies.push({
            type: 'ITEM_NOT_ACQUIRED',
            expected: item,
            actual: found || null
          });
        }
      }
    }

    // 位置の期待値チェック
    if (expectedResult.position) {
      const dist = actualPosition.distanceTo(expectedResult.position);
      if (dist > expectedResult.positionTolerance || 5) {
        discrepancies.push({
          type: 'POSITION_MISMATCH',
          expected: expectedResult.position,
          actual: actualPosition,
          distance: dist
        });
      }
    }

    return {
      success: discrepancies.length === 0,
      discrepancies,
      timestamp: Date.now()
    };
  }
}
```

### 3.5 エラー検出と修正戦略

| エラー種別 | 検出方法 | 修正戦略 |
|---|---|---|
| **ハルシネーション (誤認)** | インベントリ/位置の期待値との乖離 | アクションのリトライ、状態のリセット |
| **無限ループ** | 同一アクションの反復検出 (N回以上) | ループブレーク、代替目標の生成 |
| **ツール不足** | クラフトレシピのチェック | 前提ツールの作成から再開 |
| **到達不能** | pathfinderのタイムアウト/失敗 | 代替経路の探索、目標位置の変更 |
| **リソース枯渇** | 近辺の対象ブロック数チェック | 探索範囲の拡大、場所の移動 |
| **敵対エンティティ** | 近接敵Mobの検出 | 戦闘または逃走 |
| **体力低下** | `bot.health` の監視 | 食事、安全地帯への退避 |

---

## 4. 環境インターフェース

### 4.1 エージェント → 環境 (アクション実行)

エージェントが環境に対して実行できるアクションの全カテゴリ:

```
[PIANOエージェント]
    |
    v
[スキル実行モジュール]
    |
    |--- 移動アクション: pathfinder API
    |--- 採掘アクション: dig API
    |--- 建築アクション: placeBlock API
    |--- クラフトアクション: craft API
    |--- 戦闘アクション: attack/pvp API
    |--- 社会アクション: chat/whisper API
    |--- インベントリアクション: equip/toss/transfer API
    |--- 環境操作: useBlock (ドア、レバー、ボタン等)
    |
    v
[Mineflayer → Minecraft Protocol → サーバー]
```

#### アクション実行の非同期処理

LLM呼び出し(500-5000ms)とMinecraftのゲームティック(50ms)の時間スケール差を吸収するため、アクション実行は完全に非同期で設計する必要がある。

```javascript
class ActionExecutor {
  constructor(bot) {
    this.bot = bot;
    this.actionQueue = [];       // 待機中のアクション
    this.currentAction = null;   // 実行中のアクション
    this.isExecuting = false;
  }

  async enqueue(action) {
    this.actionQueue.push(action);
    if (!this.isExecuting) {
      await this.processQueue();
    }
  }

  async processQueue() {
    this.isExecuting = true;
    while (this.actionQueue.length > 0) {
      this.currentAction = this.actionQueue.shift();
      try {
        const result = await this.executeAction(this.currentAction);
        this.currentAction.resolve(result);
      } catch (error) {
        this.currentAction.reject(error);
      }
    }
    this.isExecuting = false;
  }

  // CCからの割り込み (高優先度アクション)
  async interrupt(urgentAction) {
    // 現在のアクションをキャンセルし、緊急アクションを即時実行
    if (this.currentAction) {
      this.bot.pathfinder.stop();
      this.actionQueue.unshift(this.currentAction); // 現在のを先頭に戻す
    }
    return await this.executeAction(urgentAction);
  }
}
```

### 4.2 環境 → エージェント (知覚情報取得)

エージェントが環境から取得できる情報の一覧:

| カテゴリ | 情報 | 取得方法 | 更新頻度 |
|---|---|---|---|
| **自己状態** | 位置、体力、空腹度、経験値 | `bot.entity`, `bot.health`, `bot.food` | 毎ティック |
| **インベントリ** | 所持アイテム、装備状態 | `bot.inventory.items()` | イベント駆動 |
| **周辺ブロック** | ブロック種別、座標 | `bot.blockAt(pos)`, `bot.findBlocks()` | オンデマンド |
| **エンティティ** | 近接プレイヤー/Mob/アイテム | `bot.entities`, `bot.nearestEntity()` | 毎ティック |
| **天候/時刻** | 雨/雷/晴れ、ゲーム内時刻 | `bot.time`, `bot.isRaining` | イベント駆動 |
| **チャット** | 他プレイヤーからのメッセージ | `bot.on('chat')` | イベント駆動 |
| **バイオーム** | 現在地のバイオーム | `bot.world.getBiome()` | オンデマンド |

#### 知覚情報のテキスト変換

LLMベースのPIANOアーキテクチャに送信するため、環境情報をテキスト形式に変換する必要がある。

```javascript
class PerceptionFormatter {
  formatState(bot) {
    const nearbyPlayers = Object.values(bot.entities)
      .filter(e => e.type === 'player' && e.username !== bot.username)
      .filter(e => e.position.distanceTo(bot.entity.position) < 32)
      .map(e => ({
        name: e.username,
        distance: Math.round(e.position.distanceTo(bot.entity.position)),
        direction: this.getDirection(bot.entity.position, e.position)
      }));

    const nearbyBlocks = this.scanNearbyBlocks(bot, 8);
    const inventory = bot.inventory.items().map(i => `${i.name} x${i.count}`);

    return `
[自己状態]
位置: (${Math.round(bot.entity.position.x)}, ${Math.round(bot.entity.position.y)}, ${Math.round(bot.entity.position.z)})
体力: ${bot.health}/20
空腹度: ${bot.food}/20
時刻: ${bot.time.timeOfDay > 13000 ? '夜' : '昼'}

[インベントリ]
${inventory.join(', ') || '空'}

[近くのプレイヤー]
${nearbyPlayers.map(p => `${p.name} (${p.distance}m ${p.direction})`).join('\n') || 'なし'}

[近くの注目ブロック]
${nearbyBlocks.join('\n') || '特になし'}
`.trim();
  }
}
```

### 4.3 チャット/会話の処理

Project Sidにおいて、チャットは社会的相互作用の主要チャネルである。

#### チャット処理フロー

```
[他エージェントのチャット]
    |
    v
[bot.on('chat') イベント]
    |
    v
[メッセージフィルタリング]
    |  - 自分宛か？
    |  - 近接範囲内か？
    |  - システムメッセージか？
    v
[社会認識モジュール]
    |  - 感情分析
    |  - 意図推定
    |  - 関係性の更新
    v
[記憶モジュール (STM)]
    |  - 会話履歴の保存
    v
[認知コントローラ (CC)]
    |  - 応答の必要性判断
    |  - 行動との整合性チェック
    v
[発話モジュール]
    |  - 応答テキスト生成
    v
[bot.chat() / bot.whisper()]
```

#### チャット範囲の実装

Project Sidでは地理的近接性に基づくチャット範囲の制限が重要:

```javascript
const CHAT_RANGE = 32;  // ブロック単位
const WHISPER_RANGE = 8;

bot.on('chat', (username, message) => {
  if (username === bot.username) return; // 自分のメッセージは無視

  const sender = bot.players[username];
  if (!sender || !sender.entity) return;

  const distance = bot.entity.position.distanceTo(sender.entity.position);

  if (distance <= CHAT_RANGE) {
    // 聞こえる範囲 → 社会認識モジュールに渡す
    socialAwareness.processMessage(username, message, distance);
  }
});
```

### 4.4 テクノロジー依存ツリーの実装

論文では、Minecraftの約1000種類のアイテムによる「テクノロジー依存ツリー (Technology Dependency Tree)」を累積的進歩の定量指標として使用している。

#### 依存ツリーのデータ構造

```javascript
// minecraft-data パッケージからレシピ情報を取得
const mcData = require('minecraft-data')('1.20.4');

class TechTree {
  constructor() {
    this.recipes = mcData.recipes;
    this.items = mcData.items;
    this.blocks = mcData.blocks;
    this.dependencyGraph = this.buildDependencyGraph();
  }

  buildDependencyGraph() {
    const graph = new Map();

    for (const [resultId, recipeList] of Object.entries(this.recipes)) {
      const resultItem = this.items[resultId] || this.blocks[resultId];
      if (!resultItem) continue;

      const dependencies = new Set();
      for (const recipe of recipeList) {
        if (recipe.ingredients) {
          recipe.ingredients.forEach(ing => {
            if (ing) dependencies.add(ing.id || ing);
          });
        }
        if (recipe.inShape) {
          recipe.inShape.flat().forEach(ing => {
            if (ing) dependencies.add(ing.id || ing);
          });
        }
      }

      graph.set(resultId, {
        name: resultItem.name,
        dependencies: [...dependencies],
        tier: -1  // 後で計算
      });
    }

    this.computeTiers(graph);
    return graph;
  }

  computeTiers(graph) {
    // トポロジカルソートによるティア計算
    const visited = new Set();
    const computeTier = (id) => {
      if (visited.has(id)) return graph.get(id)?.tier || 0;
      visited.add(id);

      const node = graph.get(id);
      if (!node || node.dependencies.length === 0) {
        if (node) node.tier = 0;
        return 0;
      }

      const maxDepTier = Math.max(
        ...node.dependencies.map(dep => computeTier(dep))
      );
      node.tier = maxDepTier + 1;
      return node.tier;
    };

    for (const id of graph.keys()) {
      computeTier(id);
    }
  }

  // エージェントの進歩度測定
  measureProgress(inventory) {
    const uniqueItems = new Set(inventory.map(i => i.name));
    const maxTier = Math.max(
      ...Array.from(uniqueItems)
        .map(name => this.getItemTier(name))
        .filter(t => t >= 0)
    );

    return {
      uniqueItemCount: uniqueItems.size,
      maxTier,
      items: [...uniqueItems]
    };
  }
}
```

#### ツリーの視覚例

```
Tier 0 (原始的): 丸太、土、石、砂 ...
    |
Tier 1 (加工): 板材、棒、丸石 ...
    |
Tier 2 (道具): 木のツルハシ、作業台、かまど ...
    |
Tier 3 (中級): 石のツルハシ、石炭 ...
    |
Tier 4 (上級): 鉄インゴット、鉄のツルハシ ...
    |
Tier 5 (高級): ダイヤモンド、ダイヤのツルハシ、エンチャントテーブル ...
    |
Tier 6+ (最上級): ネザーアイテム、エンドアイテム ...
```

---

## 5. 既存プロジェクトとの比較

### 5.1 Voyager

[Voyager](https://github.com/MineDojo/Voyager) (2023, NVIDIA/MineDojo) は、LLMを活用した初のオープンエンド型Minecraftエージェントである。

| 項目 | Voyager | Project Sid |
|---|---|---|
| **LLM** | GPT-4 | GPT-4o |
| **エージェント数** | 1体 | 25-1000体 |
| **アクション空間** | Mineflayer JavaScript コード生成 | スキル実行モジュール |
| **スキル管理** | ベクトルDB + コード保存 | (類似アプローチ推定) |
| **目標生成** | 自動カリキュラム | 目標生成モジュール (グラフ推論) |
| **社会性** | なし (単一エージェント) | 社会認識モジュール |
| **行動検証** | Self-verification (LLM) | 行動認識モジュール (非LLM NN) |
| **学習方式** | In-context learning | マルチモジュール並行処理 |

**再現実装への示唆**:
- Voyagerのスキルライブラリ (コード生成+ベクトル検索) アプローチは直接参考にできる
- ただしVoyagerは単一エージェント向けであり、マルチエージェント環境でのスキル共有機構が追加で必要
- Voyagerのiterative prompting (環境フィードバック+エラー情報+Self-verification) はProject Sidの行動認識モジュールの設計に応用可能

### 5.2 DEPS (Describe, Explain, Plan and Select)

[DEPS](https://arxiv.org/abs/2302.01560) (2023) は、LLMベースのインタラクティブプランニングアプローチである。

| 項目 | DEPS | Project Sid |
|---|---|---|
| **アプローチ** | 記述→説明→計画→選択の4段階 | PIANOの並行モジュール |
| **エラー処理** | 実行過程の記述+自己説明によるフィードバック | 行動認識モジュール |
| **目標選択** | 訓練可能なGoal Selector | 目標生成モジュール |
| **タスク達成** | 70+タスクのゼロショット達成 | 320種のアイテム取得 |
| **マルチエージェント** | なし | 500-1000体 |

**再現実装への示唆**:
- DEPSの4段階プロセスは、認知コントローラ(CC)の意思決定フローに統合可能
- Goal Selectorの訓練可能なランキング機構は、目標生成モジュールの効率化に参考になる
- エラー発生時の自己説明メカニズムは、ハルシネーション検出の改善に有用

### 5.3 MineDojo

[MineDojo](https://github.com/MineDojo/MineDojo) (2022, NVIDIA) は、Minecraft AI研究向けの包括的フレームワークである。

| 項目 | MineDojo | Project Sid |
|---|---|---|
| **目的** | 汎用AIベンチマーク | 多エージェント社会シミュレーション |
| **API** | OpenAI Gym互換 | Mineflayer |
| **言語** | Python | JavaScript (推定) |
| **観測空間** | 画像 (ピクセル) + テキスト | テキスト (構造化知覚) |
| **行動空間** | キーボード/マウス (低レベル) | 高レベルスキル |
| **データセット** | 75万YouTube動画 + Wiki + Reddit | なし (オンライン実行) |
| **タスク数** | 3142 | オープンエンド |
| **エージェント数** | 1体 | 25-1000体 |

**再現実装への示唆**:
- MineDojo/MineRLは低レベル行動空間(キーボード/マウス)のため、Project Sidの高レベルスキル実行とは異なるアプローチ
- ただしMineDojo付属のMineCLIP (動画とテキストの対応学習) は、将来的なマルチモーダル統合に有用
- ベンチマークタスク設計の参考としては有用

### 5.4 MineRL

[MineRL](https://github.com/minerllabs/minerl) (CMU) は、強化学習ベースのMinecraft AI研究プラットフォームである。

| 項目 | MineRL | Project Sid |
|---|---|---|
| **アプローチ** | 強化学習 + 模倣学習 | LLMベースの認知アーキテクチャ |
| **データ** | 6000万状態-行動ペア | なし |
| **観測** | ピクセル画像 + インベントリ | テキスト (構造化知覚) |
| **行動空間** | キーボード/マウス | 高レベルスキル |
| **学習** | オフライン事前学習 + オンラインRL | In-context learning |

**再現実装への示唆**:
- MineRLのデータセットは行動パターンの分析に有用だが、直接のアーキテクチャ参考にはならない
- VPT (Video PreTraining) モデルとの統合は将来的な視覚能力付与に有用
- MineStudio (2024) が最新の統合フレームワークとして注目

### 5.5 STEVE-1

[STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1) (2023, NeurIPS) は、テキスト・画像指示に基づくMinecraftエージェントである。

| 項目 | STEVE-1 | Project Sid |
|---|---|---|
| **アプローチ** | 生成モデル (text-to-behavior) | LLMベースの認知アーキテクチャ |
| **入力** | テキスト/画像指示 + ピクセル | テキスト (構造化知覚) |
| **出力** | キーボード/マウス操作 | 高レベルスキル |
| **基盤モデル** | VPT + MineCLIP | GPT-4o |
| **学習方式** | 自己教師あり行動クローニング | In-context learning |
| **計算コスト** | ~$60 | 高額 (GPT-4o API) |

**再現実装への示唆**:
- STEVE-1の低コスト学習は魅力的だが、言語的推論・社会性の点ではLLMベースのアプローチが優位
- 将来的にVisual-Language Modelと統合する際、STEVE-1/VPTの知見が参考になる
- マルチモーダル能力の付与 (論文の「今後の課題」) に直結

### 5.6 比較まとめ

| フレームワーク | スキル表現 | マルチエージェント | 社会性 | 再現実装との関連度 |
|---|---|---|---|---|
| **Voyager** | JSコード生成 | 不可 | なし | **最高** (スキルライブラリ) |
| **DEPS** | プラン+サブゴール | 不可 | なし | **高** (プランニング) |
| **MineDojo** | 低レベル行動 | 不可 | なし | 中 (ベンチマーク) |
| **MineRL** | 低レベル行動 | 不可 | なし | 低 (データセット) |
| **STEVE-1** | 低レベル行動 | 不可 | なし | 低 (マルチモーダル) |

---

## 6. 推奨アーキテクチャ

### 6.1 全体構成図

```
                          [中央オーケストレータ]
                          /        |         \
                   [Redis/NATS]  [PostgreSQL]  [Vector DB]
                   (メッセージ)   (永続状態)    (スキル検索)
                         |
                   [Velocity Proxy]
                  /    |    |    \
          [Pufferfish] [Pufferfish] [Pufferfish] [Pufferfish]
          Server 1     Server 2     Server 3     Server 4
             |            |            |            |
        [Bot Manager] [Bot Manager] [Bot Manager] [Bot Manager]
        (~125 bots)   (~125 bots)   (~125 bots)   (~125 bots)
             |            |            |            |
        [Worker Pool] [Worker Pool] [Worker Pool] [Worker Pool]
        (N workers)   (N workers)   (N workers)   (N workers)
```

### 6.2 ボット1体の内部構成

```
[Mineflayer Bot Instance]
    |
    |--- [Perception Layer]       ← 環境情報のテキスト変換
    |       |--- SelfState
    |       |--- NearbyEntities
    |       |--- NearbyBlocks
    |       |--- ChatMessages
    |
    |--- [Action Executor]        ← アクションキュー + 非同期実行
    |       |--- ActionQueue
    |       |--- InterruptHandler
    |       |--- FeedbackCollector
    |
    |--- [Skill Library Client]   ← ベクトルDB検索 + スキル実行
    |       |--- SkillSearch
    |       |--- SkillExecutor
    |       |--- SkillSaver
    |
    |--- [Action Awareness]       ← 期待値 vs 実際値の比較
    |       |--- ExpectationChecker
    |       |--- HallucinationDetector
    |       |--- LoopBreaker
    |
    |--- [PIANO Interface]        ← 上位モジュールとの通信
            |--- SharedAgentState
            |--- CCBroadcastReceiver
            |--- EventEmitter → 上位モジュール
```

### 6.3 技術スタック推奨

| レイヤー | 推奨技術 | 理由 |
|---|---|---|
| **Minecraftサーバー** | Pufferfish 1.20.x | 大規模ボット向け最適化 |
| **プロキシ** | Velocity | PaperMC公式、高性能 |
| **Botフレームワーク** | Mineflayer | 唯一の実用的選択肢、Voyager実績 |
| **Bot実行環境** | Node.js 20 LTS + Worker Threads | 安定性、パフォーマンス |
| **プロセスオーケストレーション** | PM2 / Kubernetes | 500+プロセスの管理 |
| **メッセージング** | Redis Pub/Sub or NATS | 低レイテンシ、軽量 |
| **永続化** | PostgreSQL | 信頼性、JSON対応 |
| **スキルベクトルDB** | Qdrant or ChromaDB | 軽量、高速 |
| **モニタリング** | Grafana + Prometheus | リアルタイム監視 |

### 6.4 スケーリングロードマップ

| フェーズ | ボット数 | サーバー構成 | 目的 |
|---|---|---|---|
| Phase 1 | 5-10 | 単一Paper | スキル実行モジュールの開発・テスト |
| Phase 2 | 25 | 単一Pufferfish | 単一エージェント実験の再現 (30分) |
| Phase 3 | 49 | 単一Pufferfish (チューニング済み) | 長期実験の再現 (4時間) |
| Phase 4 | 100-200 | Velocity + 2-3 Pufferfish | マルチサーバー構成の検証 |
| Phase 5 | 500 | Velocity + 4 Pufferfish | 文明実験の再現 |
| Phase 6 | 1000+ | Velocity + 8+ Pufferfish (分散) | フルスケール実験 |

### 6.5 リスクと緩和策

| リスク | 影響度 | 緩和策 |
|---|---|---|
| Mineflayerのメモリ消費がスケールしない | 高 | Worker Thread分離、ワールド状態共有、view-distance最小化 |
| サーバーが500+ボットに耐えられない | 高 | Velocity分散、Pufferfish最適化、Mob/エンティティ削減 |
| サーバー間転送時のエージェント状態消失 | 中 | Redis/PostgreSQLによる状態永続化、転送前後のチェックポイント |
| LLM API遅延によるゲーム内タイムアウト | 中 | アクション非同期キュー、LLM呼び出しの並列化 |
| Mineflayer APIの変更・非互換 | 低 | バージョン固定、抽象化レイヤーの導入 |
| スキルコード生成の安全性 | 中 | サンドボックス実行、API ホワイトリスト |

---

## 参考資料

- [PrismarineJS/mineflayer GitHub](https://github.com/PrismarineJS/mineflayer)
- [Mineflayer API Documentation](https://github.com/PrismarineJS/mineflayer/blob/master/docs/api.md)
- [mineflayer-pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder)
- [mineflayer-collectblock](https://github.com/PrismarineJS/mineflayer-collectblock)
- [Voyager: An Open-Ended Embodied Agent with LLMs](https://github.com/MineDojo/Voyager)
- [DEPS: Describe, Explain, Plan and Select](https://arxiv.org/abs/2302.01560)
- [MineDojo Framework](https://github.com/MineDojo/MineDojo)
- [MineRL](https://github.com/minerllabs/minerl)
- [STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1)
- [PaperMC / Velocity](https://github.com/PaperMC/Velocity)
- [Paper Server Optimization Guide](https://paper-chan.moe/paper-optimization/)
- [Minecraft Server Optimization Guide](https://github.com/YouHaveTrouble/minecraft-optimization)
- [Pufferfish Server](https://docs.pufferfish.host/)

---
## 関連ドキュメント
- [06-social-cognition.md](./06-social-cognition.md) — 社会認知モジュール
- [07-goal-planning.md](./07-goal-planning.md) — 目標・計画システム
- [08-infrastructure.md](./08-infrastructure.md) — インフラ設計
