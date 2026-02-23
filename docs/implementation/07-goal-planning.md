# 7. 目標・計画システム実装設計

> 概要: 目標生成、計画、行動認識、自己省察の4モジュールによる認知ループの実装設計
> 対応論文セクション: 3.1 (Cognitive Controller), Appendix B (Planning & Reflection)
> 最終更新: 2026-02-23

---

## 1. 目標生成モジュール（Goal Generation）

### 1.1 設計概要

目標生成モジュールは、エージェントの経験と環境相互作用に基づいて新しい目的を再帰的に創出する**低速モジュール**である。論文では「グラフ構造上での推論による目的創出」と記述されている。

### 1.2 目標の階層構造

```
[長期目標（Long-term Goal）]
  ├── "村を発展させ、コミュニティを守る"
  │
  ├── [中期目標（Mid-term Goal）]
  │     ├── "防衛設備を整える"
  │     └── "食料供給を安定させる"
  │
  └── [短期目標（Short-term Goal / Social Goal）]
        ├── "フェンスを作成する"   → 5-10秒ごとに生成
        ├── "小麦の種を集める"
        └── [即時行動（Immediate Action）]
              ├── "craft fence 4"
              └── "collect wheat_seeds"
```

| レベル | 生成間隔 | 持続時間 | 生成方法 |
|---|---|---|---|
| 長期目標 | エージェント初期化時 / 自己省察時 | シミュレーション全体 | プロンプト設定 + LLM推論 |
| 中期目標 | 数分ごと / イベント駆動 | 数分〜数十分 | LLMによるグラフ推論 |
| 社会的目標 | **5-10秒ごと** | 5-10秒（ローリングウィンドウ） | LLM + 環境コンテキスト |
| 即時行動 | 社会的目標に応じてリアルタイム | 即時 | 計画モジュール連携 |

### 1.3 グラフ構造推論の設計

目標生成には2種類のグラフ構造を活用する。

#### 1.3.1 知識グラフ（Knowledge Graph）

エージェントの環境認識と関係性を構造化する。

```typescript
interface KnowledgeNode {
  id: string;
  type: "agent" | "item" | "location" | "structure" | "concept";
  name: string;
  properties: Record<string, unknown>;
  lastUpdated: number;
}

interface KnowledgeEdge {
  source: string;  // node id
  target: string;  // node id
  relation: string;
  // 例: "owns", "needs", "located_at", "knows", "likes", "near"
  weight: number;
  timestamp: number;
}

interface KnowledgeGraph {
  nodes: Map<string, KnowledgeNode>;
  edges: KnowledgeEdge[];

  // クエリメソッド
  getNeighbors(nodeId: string, relationType?: string): KnowledgeNode[];
  getSubgraph(centerNodeId: string, depth: number): KnowledgeGraph;
  shortestPath(from: string, to: string): KnowledgeNode[];
}
```

#### 1.3.2 因果グラフ（Causal / Dependency Graph）

Minecraftの技術依存ツリーを反映し、目標の前提条件と因果関係を表現する。

```typescript
interface GoalNode {
  id: string;
  description: string;
  status: "pending" | "active" | "completed" | "failed" | "blocked";
  priority: number;
  prerequisites: string[];  // 前提となるGoalNode.id
  estimatedDifficulty: number; // 0.0 - 1.0
  category: "survival" | "social" | "exploration" | "crafting" | "building";
}

interface DependencyGraph {
  goals: Map<string, GoalNode>;

  // 実行可能な目標（前提条件がすべてcompletedのもの）
  getExecutableGoals(): GoalNode[];

  // 目標の優先度を再計算
  recalculatePriorities(agentState: AgentState): void;

  // 新しい目標を追加
  addGoal(goal: GoalNode, dependencies: string[]): void;
}
```

### 1.4 社会的目標の生成メカニズム

5-10秒ごとに生成される社会的目標は、エージェントの**即時行動を駆動する最も動的な目標層**である。論文では、これらの目標がローリングウィンドウ（5目標セット）で管理され、役割の専門化を駆動することが示されている。

> **責務分担**: 社会的目標の生成ロジックは本モジュール（目標生成モジュール）が一元的に担当する。社会認識モジュール（06-social-cognition.md）は社会的シグナル（好感度変化、社会的機会、コミュニティニーズ等）の検出・評価を担当し、本モジュールへの入力として社会的コンテキストを提供する。

```typescript
interface SocialGoalGenerator {
  // 5目標のローリングウィンドウ
  currentGoals: SocialGoal[];
  maxGoals: number; // = 5

  // 5-10秒間隔で呼び出される
  generateGoal(context: GoalGenerationContext): Promise<SocialGoal>;

  // 古い目標を淘汰し新しい目標を追加
  updateRollingWindow(newGoal: SocialGoal): void;
}

interface SocialGoal {
  id: string;
  description: string;
  generatedAt: number;
  expiresAt: number;
  category: GoalCategory;
  socialContext: {
    nearbyAgents: string[];
    communityGoal: string;
    recentInteractions: string[];
  };
}

type GoalCategory =
  | "farming"      // 農民
  | "mining"       // 鉱夫
  | "engineering"  // 技術者
  | "guarding"     // 警備員
  | "exploring"    // 探検家
  | "crafting"     // 鍛冶屋
  | "social"       // 社会的交流
  | "other";
```

### 1.5 LLMプロンプト設計

#### 社会的目標生成プロンプト

```
You are {agent_name}, a member of {community_name}.

## Your Current State
- Location: {current_location}
- Inventory: {inventory_summary}
- Health/Hunger: {status}
- Current Role Tendency: {inferred_role}

## Community Context
- Community Goal: {community_goal}
- Nearby Agents: {nearby_agents_and_activities}
- Recent Events: {recent_events_summary}

## Your Recent Goals (Rolling Window)
{recent_goals_list}

## Your Knowledge
- Known Resources: {known_resources}
- Known Needs: {community_needs}

## Task
Generate your next immediate social goal (to be pursued for the next 5-10 seconds).
Your goal should:
1. Be consistent with your emerging role and recent behavior patterns
2. Consider what nearby agents are doing (avoid redundancy, seek complementarity)
3. Contribute to the community's overall objective
4. Be achievable within 5-10 seconds of game time

Respond in JSON format:
{
  "goal": "<concise goal description>",
  "category": "<farming|mining|engineering|guarding|exploring|crafting|social>",
  "reasoning": "<brief explanation of why this goal>",
  "expected_actions": ["<action1>", "<action2>"]
}
```

#### 中期目標生成プロンプト

```
You are {agent_name}. Reflect on your recent experiences and generate
a medium-term goal (next 5-10 minutes).

## Your Experience Summary
- Completed Goals: {completed_goals}
- Failed Goals: {failed_goals}
- Skills Acquired: {skills}
- Social Relationships: {relationship_summary}

## Environment State
- Available Resources: {resources}
- Community Status: {community_status}
- Threats/Opportunities: {threats_and_opportunities}

## Dependency Graph (relevant subgraph)
{dependency_subgraph_description}

Generate a medium-term goal that:
1. Builds on your strengths and recent successes
2. Addresses gaps or opportunities in the community
3. Has clear sub-steps that can be decomposed into social goals

Respond in JSON:
{
  "goal": "<goal description>",
  "priority": <1-10>,
  "prerequisites": ["<prerequisite1>", ...],
  "sub_goals": ["<sub_goal1>", "<sub_goal2>", ...],
  "estimated_duration_seconds": <number>
}
```

### 1.6 実装アーキテクチャ

```typescript
class GoalGenerationModule {
  private knowledgeGraph: KnowledgeGraph;
  private dependencyGraph: DependencyGraph;
  private socialGoalGenerator: SocialGoalGenerator;
  private llmClient: LLMClient;

  // メインループ: 5-10秒間隔で実行
  async runSocialGoalLoop(agentState: AgentState): Promise<void> {
    const interval = 5000 + Math.random() * 5000; // 5-10秒

    const context = this.buildGoalContext(agentState);
    const newGoal = await this.socialGoalGenerator.generateGoal(context);

    // ローリングウィンドウを更新
    this.socialGoalGenerator.updateRollingWindow(newGoal);

    // 共有エージェント状態に書き込み
    agentState.currentGoals = this.socialGoalGenerator.currentGoals;

    // 知識グラフを更新
    this.knowledgeGraph.updateFromGoal(newGoal);

    // 次回の実行をスケジュール
    setTimeout(() => this.runSocialGoalLoop(agentState), interval);
  }

  // 中期目標の生成: イベント駆動 + 定期実行
  async generateMidTermGoal(agentState: AgentState): Promise<GoalNode> {
    const subgraph = this.knowledgeGraph.getSubgraph(
      agentState.agentId, 3
    );
    const executableGoals = this.dependencyGraph.getExecutableGoals();

    const prompt = this.buildMidTermPrompt(agentState, subgraph, executableGoals);
    const response = await this.llmClient.generate(prompt);
    const goalData = JSON.parse(response);

    const goalNode: GoalNode = {
      id: generateId(),
      description: goalData.goal,
      status: "pending",
      priority: goalData.priority,
      prerequisites: goalData.prerequisites,
      estimatedDifficulty: this.estimateDifficulty(goalData),
      category: this.inferCategory(goalData),
    };

    this.dependencyGraph.addGoal(goalNode, goalData.prerequisites);
    return goalNode;
  }

  private buildGoalContext(agentState: AgentState): GoalGenerationContext {
    return {
      agentName: agentState.name,
      location: agentState.currentLocation,
      inventory: agentState.inventory,
      nearbyAgents: agentState.nearbyAgents,
      communityGoal: agentState.communityGoal,
      recentGoals: this.socialGoalGenerator.currentGoals,
      recentEvents: agentState.recentEvents,
      inferredRole: agentState.inferredRole,
      knowledgeSubgraph: this.knowledgeGraph.getSubgraph(
        agentState.agentId, 2
      ),
    };
  }
}
```

---

## 2. 計画モジュール（Planning）

### 2.1 設計概要

計画モジュールは目標を実行可能なアクションシーケンスに分解する**低速モジュール**である。論文では具体的なアルゴリズムの詳細は限定的だが、「ゆっくり考える」プロセスとして高速モジュールの動作をブロックしない並行実行が強調されている。

### 2.2 計画立案アプローチ

先行研究（Voyager, DEPS, Generative Agents）の知見を統合し、PIANOに適した計画方式を設計する。

#### 2.2.1 HTN（Hierarchical Task Network）ベースの分解

Minecraftの技術依存ツリーは本質的に階層的であり、HTNベースの計画が自然に適合する。

```typescript
interface TaskNode {
  id: string;
  name: string;
  type: "compound" | "primitive";
  // compound: さらに分解可能な抽象タスク
  // primitive: 直接実行可能なMinecraftアクション
}

interface Method {
  // compoundタスクの分解規則
  taskId: string;
  preconditions: Condition[];
  subtasks: TaskNode[];  // 順序付き
}

interface PlanStep {
  action: string;          // Minecraftアクション (例: "mine_block", "craft_item")
  parameters: Record<string, unknown>;
  expectedOutcome: OutcomeExpectation;
  estimatedDuration: number;
}
```

#### 2.2.2 LLMベースの動的計画

静的なHTN分解では対応できない動的な状況に対して、LLMによる柔軟な計画修正を行う。DEPSアプローチを参考に、**Describe-Explain-Plan-Select**パターンを適用する。

```typescript
interface PlanningPipeline {
  // 1. Describe: 現在の状況を記述
  describeCurrentState(agentState: AgentState): string;

  // 2. Explain: 失敗の原因を分析（再計画時）
  explainFailure(failedPlan: Plan, error: ActionError): string;

  // 3. Plan: 目標に対する計画を生成
  generatePlan(goal: GoalNode, context: string): Promise<Plan>;

  // 4. Select: 複数の候補計画から最適なものを選択
  selectBestPlan(candidates: Plan[], agentState: AgentState): Plan;
}
```

### 2.3 計画の動的修正

PIANOの並行実行特性により、計画は実行中に動的に修正される必要がある。

```typescript
interface PlanExecutionState {
  originalPlan: Plan;
  currentStepIndex: number;
  completedSteps: PlanStep[];
  failedSteps: PlanStep[];
  modifications: PlanModification[];
}

interface PlanModification {
  reason: "action_failure" | "environment_change" | "new_goal" | "social_event";
  timestamp: number;
  originalStep: PlanStep;
  replacement: PlanStep[] | null;  // null = ステップ削除
}

class PlanningModule {
  private llmClient: LLMClient;
  private htnDecomposer: HTNDecomposer;
  private activePlans: Map<string, PlanExecutionState>;

  // 目標から計画を生成
  async createPlan(goal: GoalNode, agentState: AgentState): Promise<Plan> {
    // まずHTN分解を試行
    const htnPlan = this.htnDecomposer.decompose(goal, agentState);

    if (htnPlan && htnPlan.isComplete()) {
      return htnPlan;
    }

    // HTN分解が不完全な場合、LLMで補完
    const prompt = this.buildPlanningPrompt(goal, agentState, htnPlan);
    const response = await this.llmClient.generate(prompt);
    return this.parsePlan(response);
  }

  // 行動認識からのフィードバックによる再計画
  async replan(
    planState: PlanExecutionState,
    feedback: ActionAwarenessFeedback,
    agentState: AgentState
  ): Promise<Plan> {
    const failureDescription = this.describeFailure(feedback);
    const explanation = await this.explainFailure(planState, failureDescription);

    // 部分的再計画: 失敗したステップ以降のみ再生成
    const remainingGoal = this.extractRemainingGoal(planState);
    const newPlan = await this.createPlan(remainingGoal, agentState);

    // 完了済みステップを保持しつつ新計画を統合
    return this.mergePlans(planState.completedSteps, newPlan);
  }

  // 計画ステップの次のアクションを取得（認知コントローラから呼ばれる）
  getNextAction(planId: string): PlanStep | null {
    const state = this.activePlans.get(planId);
    if (!state) return null;

    const nextIndex = state.currentStepIndex;
    if (nextIndex >= state.originalPlan.steps.length) return null;

    return state.originalPlan.steps[nextIndex];
  }
}
```

### 2.4 計画プロンプト設計

```
You are {agent_name}. Create a step-by-step plan to achieve the following goal.

## Goal
{goal_description}

## Current State
- Location: {location}
- Inventory: {inventory}
- Nearby: {nearby_entities}
- Available Tools: {tools}

## Known Recipes/Dependencies
{relevant_crafting_recipes}

## Previous Plan Failures (if any)
{failure_history}

Create a plan with concrete, executable steps.
Each step should be a single Minecraft action.

Respond in JSON:
{
  "plan_id": "<unique_id>",
  "steps": [
    {
      "action": "<minecraft_action>",
      "parameters": { ... },
      "expected_outcome": "<what should happen>",
      "fallback": "<what to do if this fails>"
    }
  ],
  "estimated_total_duration": <seconds>,
  "risk_factors": ["<potential issues>"]
}
```

### 2.5 部分計画と再計画戦略

| 再計画トリガー | 対応戦略 | 応答速度 |
|---|---|---|
| アクション失敗（行動認識が検出） | 失敗ステップ以降のみ再計画 | 低速（LLM） |
| 環境変化（新しい脅威など） | 計画全体を再評価 | 低速（LLM） |
| 新しい社会的目標の発生 | 目標優先度を再計算、必要なら計画切替 | 中速 |
| リソース不足の発見 | 前提条件タスクを挿入 | 低速（LLM） |

---

## 3. 行動認識モジュール（Action Awareness）

### 3.1 設計概要

行動認識は、PIANOアーキテクチャにおいて**最も重要なモジュール**として特定されている。エージェントの期待されるアクション結果と実際の結果をリアルタイムで比較し、LLMのハルシネーションに起因するエラーを検出・修正する**高速モジュール**である。

論文の重要な知見:
- このモジュールの除去でアイテム取得性能が**大幅に低下**
- 「少量のハルシネーションが連鎖的なLLM呼び出しの下流の行動を汚染しうる」
- **非LLMニューラルネットワーク**で実装され、LLMの推論に依存しない

### 3.2 ハルシネーションの問題パターン

```
エージェントの典型的なハルシネーション連鎖:

1. LLMが「鉄インゴットを3つ持っている」と誤認（実際は0）
2. 「鉄インゴットの採集」目標を達成済みと判断 → 目標をスキップ
3. 「鉄のツルハシを作る」計画に進む
4. クラフト失敗 → LLMが別の原因を推測（さらなるハルシネーション）
5. 意味のないアクションのループに陥る

行動認識による解決:
1. 「鉄インゴットを3つ持っている」→ 実際のインベントリを確認 → 不一致検出
2. 修正アクション「鉄インゴットの採集を再開」を生成
3. 正しい状態に基づいて下流の計画が継続
```

### 3.3 期待 vs 実際の比較ロジック

```typescript
interface ActionOutcome {
  actionId: string;
  action: string;
  parameters: Record<string, unknown>;
  timestamp: number;
}

interface ExpectedOutcome extends ActionOutcome {
  expectedInventoryDelta: InventoryDelta;
  expectedPositionDelta: PositionDelta;
  expectedWorldChange: WorldChange;
  confidence: number;  // LLM/計画モジュールの確信度
}

interface ActualOutcome extends ActionOutcome {
  actualInventoryDelta: InventoryDelta;
  actualPosition: Position;
  actualWorldState: WorldState;
  success: boolean;
  errorMessage?: string;
}

interface DiscrepancyReport {
  actionId: string;
  discrepancyType: DiscrepancyType;
  severity: "low" | "medium" | "high" | "critical";
  expected: unknown;
  actual: unknown;
  suggestedCorrection: CorrectionAction;
}

type DiscrepancyType =
  | "inventory_mismatch"     // インベントリの不一致
  | "position_mismatch"      // 位置の不一致
  | "action_no_effect"       // アクションが効果なし
  | "unexpected_failure"     // 予期しない失敗
  | "phantom_success"        // 成功したと誤認（最も危険）
  | "repeated_action_loop";  // 同じアクションの繰り返し
```

### 3.4 非LLMニューラルネットワークの設計

論文が「小規模な非LLMニューラルネットワーク」と記述していることから、軽量で高速推論可能なアーキテクチャを設計する。

#### 3.4.1 モデルアーキテクチャ

```
入力層:
  ├── 期待されるアクション結果のエンコーディング (d=64)
  ├── 実際のアクション結果のエンコーディング   (d=64)
  ├── 直近N回のアクション履歴のエンコーディング (d=128)
  └── エージェント状態コンテキスト              (d=32)
      合計: d=288

隠れ層:
  ├── Dense(288 → 128, ReLU)
  ├── Dropout(0.2)
  ├── Dense(128 → 64, ReLU)
  └── Dropout(0.2)

出力層:
  ├── 不一致検出ヘッド: Dense(64 → 1, Sigmoid)    -- 不一致あり/なしの二値分類
  ├── 不一致分類ヘッド: Dense(64 → 6, Softmax)    -- DiscrepancyType分類
  ├── 重大度ヘッド: Dense(64 → 4, Softmax)         -- severity分類
  └── 修正アクション推薦ヘッド: Dense(64 → K, Softmax) -- 修正アクション候補
```

**推論速度の目標**: < 10ms（CPU推論）、全体の行動認識ループで < 100ms

#### 3.4.2 エンコーディング設計

```typescript
interface ActionEncoder {
  // アクション種別の埋め込み
  actionEmbedding: EmbeddingLayer;  // vocab_size=50, dim=16

  // 数値特徴量の正規化
  inventoryEncoder(inventory: Inventory): Float32Array;  // dim=32
  positionEncoder(position: Position): Float32Array;      // dim=8
  contextEncoder(context: AgentContext): Float32Array;     // dim=8

  // 期待/実際のペアをエンコード
  encodePair(
    expected: ExpectedOutcome,
    actual: ActualOutcome
  ): Float32Array;  // dim=128
}
```

#### 3.4.3 ループ検出サブモジュール

連続的な同一アクションの繰り返し（ハルシネーションループ）を検出する専用コンポーネント。

```typescript
class ActionLoopDetector {
  private actionHistory: ActionRecord[];
  private windowSize: number = 20;  // 直近20アクション
  private loopThreshold: number = 3; // 3回以上の繰り返しで検出

  detectLoop(): LoopDetectionResult {
    const recent = this.actionHistory.slice(-this.windowSize);

    // n-gram分析による繰り返しパターン検出
    for (let n = 1; n <= 5; n++) {
      const ngrams = this.extractNgrams(recent, n);
      const maxFreq = Math.max(...Object.values(ngrams));
      if (maxFreq >= this.loopThreshold) {
        return {
          detected: true,
          patternLength: n,
          repetitions: maxFreq,
          pattern: this.getMostFrequentPattern(ngrams),
        };
      }
    }

    return { detected: false };
  }
}
```

### 3.5 Phase 0ブートストラップ戦略

行動認識NNの学習にはエージェントの行動ログが必要だが、Phase 0ではエージェントシステム自体がまだ稼働していないため、学習データが存在しない。この「鶏と卵」問題に対し、段階的なブートストラップ戦略を採用する。

#### Phase 0: ルールベース行動認識（NNなし）

Phase 0では、NNモデルの代わりにルールベースの行動認識で代替する。Minecraftサーバーから取得できる確定的な情報を直接比較することで、ハルシネーション検出の基本機能を実現する。

```typescript
class RuleBasedActionAwareness {
  /**
   * Phase 0用のルールベース行動認識。
   * NNモデルの学習データが蓄積されるまでの代替実装。
   */

  evaluate(
    expected: ExpectedOutcome,
    actual: ActualOutcome,
    agentState: AgentState
  ): ActionAwarenessFeedback {
    const discrepancies: DiscrepancyReport[] = [];

    // ルール1: インベントリの直接比較
    // サーバーAPIから取得した実際のインベントリとLLMの認識を比較
    if (!this.inventoryMatches(expected.expectedInventoryDelta, actual.actualInventoryDelta)) {
      discrepancies.push({
        discrepancyType: "inventory_mismatch",
        severity: "high",
        expected: expected.expectedInventoryDelta,
        actual: actual.actualInventoryDelta,
      });
    }

    // ルール2: アクション結果の成否判定
    // mine_block, place_block, craft_item 等の結果をサーバーAPIで直接確認
    if (expected.confidence > 0.5 && !actual.success) {
      discrepancies.push({
        discrepancyType: "unexpected_failure",
        severity: "medium",
        expected: "success",
        actual: actual.errorMessage,
      });
    }

    // ルール3: 位置の不一致検出
    if (this.positionDivergence(expected.expectedPositionDelta, actual.actualPosition) > 5.0) {
      discrepancies.push({
        discrepancyType: "position_mismatch",
        severity: "low",
        expected: expected.expectedPositionDelta,
        actual: actual.actualPosition,
      });
    }

    // ルール4: ループ検出（既存のActionLoopDetectorを再利用）
    // n-gramベースのパターンマッチングはNNに依存しない

    return discrepancies.length > 0
      ? { status: "discrepancy_detected", reports: discrepancies }
      : { status: "ok" };
  }

  private inventoryMatches(
    expectedDelta: InventoryDelta,
    actualDelta: InventoryDelta
  ): boolean {
    // アイテムID・数量の完全一致を確認
    for (const [item, expectedQty] of Object.entries(expectedDelta)) {
      const actualQty = actualDelta[item] ?? 0;
      if (expectedQty !== actualQty) return false;
    }
    return true;
  }

  private positionDivergence(
    expectedDelta: PositionDelta,
    actualPos: Position
  ): number {
    // ユークリッド距離で位置の乖離を計算
    return Math.sqrt(
      (expectedDelta.dx - actualPos.x) ** 2 +
      (expectedDelta.dy - actualPos.y) ** 2 +
      (expectedDelta.dz - actualPos.z) ** 2
    );
  }
}
```

**ルールベース vs NN の比較**:

| 観点 | ルールベース（Phase 0） | NN（Phase 1以降） |
|---|---|---|
| ハルシネーション検出 | インベントリ・位置の直接比較のみ | パターン認識による予測的検出 |
| phantom_success検出 | 限定的（サーバーAPIで確認可能な範囲） | 文脈を考慮した高精度検出 |
| 推論速度 | < 1ms | < 10ms |
| 対応可能な不一致タイプ | 4種類（上記ルール） | 6種類全て |
| 学習データの必要性 | なし | 数千サンプル以上 |

#### Phase 0 → Phase 1 移行計画

1. **Phase 0稼働中**: ルールベース行動認識で運用しつつ、全アクションの期待/実際ペアをログに記録
2. **データ蓄積**: 正常実行ログ（自動ラベリング）+ エラーログ（サーバーAPI検出）を収集
3. **合成データ生成**: 収集した正常ログのインベントリ・位置を意図的に改ざんし、不一致サンプルを生成
4. **NNモデル学習**: 蓄積データ（目標: 5,000サンプル以上）でNNモデルを初期訓練
5. **段階的切替**: ルールベースとNNを並行実行し、NNの精度が閾値（F1 > 0.85）を超えたらNNに移行

### 3.6 学習データの収集方法

行動認識ニューラルネットワークの学習には、エージェントの行動ログからの教師データが必要となる。

#### 3.6.1 データ収集パイプライン

```
[エージェント実行] ──→ [アクションログ収集]
                             ↓
                    [期待/実際ペアの記録]
                             ↓
                    [人手/LLMによるラベリング]
                             ↓
                    [学習データセット構築]
                             ↓
                    [モデル訓練・検証]
```

#### 3.6.2 データ収集戦略

| データソース | ラベリング方法 | 期待されるデータ量 |
|---|---|---|
| 正常実行ログ | 自動ラベリング（結果一致 = 正常） | 大量（90%以上） |
| エラー発生ログ | MinecraftサーバーAPIのエラー検出 | 中程度 |
| ハルシネーション事例 | LLMによる事後分析 + 人手確認 | 少量だが重要 |
| 合成データ | インベントリ/位置を意図的に改ざん | 補足用 |

#### 3.6.3 継続的学習

```typescript
interface OnlineLearningConfig {
  // 実行中に発見された不一致からの継続学習
  collectFromRuntime: boolean;
  batchSize: number;          // 64
  updateInterval: number;      // 1000アクションごと
  learningRate: number;        // 1e-4
  maxBufferSize: number;       // 10000サンプル

  // 検証用ホールドアウト
  validationSplit: number;     // 0.1
}
```

### 3.7 自己修正アクションの生成

不一致が検出された場合の修正パターン。

```typescript
interface CorrectionAction {
  type: CorrectionType;
  description: string;
  actions: PlanStep[];  // 実行すべき修正アクション
}

type CorrectionType =
  | "retry"              // 同じアクションを再試行
  | "prerequisite_fix"   // 前提条件を満たすアクションを挿入
  | "goal_reset"         // 目標を最初から再開
  | "plan_invalidate"    // 計画全体を無効化して再計画を要求
  | "state_sync";        // エージェント状態を実際の環境と同期

// 修正ロジック
class CorrectionGenerator {
  generateCorrection(
    discrepancy: DiscrepancyReport,
    agentState: AgentState
  ): CorrectionAction {
    switch (discrepancy.discrepancyType) {
      case "phantom_success":
        // 最も危険: エージェントが成功を誤認
        // → 目標をリセットし、状態を実環境と同期
        return {
          type: "state_sync",
          description: `Agent believed action succeeded but it did not. ` +
            `Syncing state with actual environment.`,
          actions: [
            { action: "sync_inventory", parameters: {} },
            { action: "reassess_goals", parameters: {} },
          ],
        };

      case "repeated_action_loop":
        // ループ検出 → 計画を無効化して全く異なるアプローチを要求
        return {
          type: "plan_invalidate",
          description: `Action loop detected. Invalidating current plan.`,
          actions: [
            { action: "abandon_current_plan", parameters: {} },
            { action: "request_replan", parameters: { avoidPrevious: true } },
          ],
        };

      case "inventory_mismatch":
        // インベントリ不一致 → 状態同期後に再試行
        return {
          type: "state_sync",
          description: `Inventory mismatch detected. Syncing and retrying.`,
          actions: [
            { action: "sync_inventory", parameters: {} },
          ],
        };

      case "action_no_effect":
        // アクションが効果なし → 前提条件を確認
        return {
          type: "prerequisite_fix",
          description: `Action had no effect. Checking prerequisites.`,
          actions: [
            { action: "check_prerequisites", parameters: {
              targetAction: discrepancy.actionId
            }},
          ],
        };

      default:
        return {
          type: "retry",
          description: `Retrying action after discrepancy.`,
          actions: [],
        };
    }
  }
}
```

### 3.8 全体フロー

```typescript
class ActionAwarenessModule {
  private model: ActionAwarenessModel;  // 非LLMニューラルネット
  private loopDetector: ActionLoopDetector;
  private correctionGenerator: CorrectionGenerator;

  // 高速実行ループ: 各アクション完了後に呼ばれる
  async evaluate(
    expected: ExpectedOutcome,
    actual: ActualOutcome,
    agentState: AgentState
  ): ActionAwarenessFeedback {
    // 1. ニューラルネットによる不一致検出（< 10ms）
    const encoding = this.encode(expected, actual, agentState);
    const prediction = this.model.predict(encoding);

    // 2. ループ検出（< 1ms）
    this.loopDetector.record(actual);
    const loopResult = this.loopDetector.detectLoop();

    // 3. 不一致が検出された場合
    if (prediction.hasDiscrepancy || loopResult.detected) {
      const discrepancy: DiscrepancyReport = {
        actionId: actual.actionId,
        discrepancyType: loopResult.detected
          ? "repeated_action_loop"
          : prediction.discrepancyType,
        severity: prediction.severity,
        expected: expected,
        actual: actual,
        suggestedCorrection: this.correctionGenerator.generateCorrection(
          { discrepancyType: prediction.discrepancyType } as DiscrepancyReport,
          agentState
        ),
      };

      // 4. 共有エージェント状態に不一致レポートを書き込み
      agentState.actionAwarenessReports.push(discrepancy);

      return {
        status: "discrepancy_detected",
        report: discrepancy,
        shouldReplan: discrepancy.severity === "high" ||
                      discrepancy.severity === "critical",
      };
    }

    return { status: "ok" };
  }
}
```

---

## 4. 自己省察モジュール（Self-Reflection）

### 4.1 設計概要

自己省察モジュールは、エージェントが自身の行動と結果を振り返り、知識と戦略を更新する**低速モジュール**である。Reflexionフレームワークの知見を取り入れつつ、PIANOの並行実行モデルに適合させる。

### 4.2 振り返りのタイミングとトリガー

| トリガー種別 | 条件 | 優先度 |
|---|---|---|
| **定期実行** | 一定時間経過（例: 2-5分ごと） | 通常 |
| **目標完了** | 中期目標が完了または失敗した時 | 高 |
| **重大イベント** | 死亡、大量アイテムロス、新バイオーム発見 | 最高 |
| **社会的イベント** | 重要な対話、協力/対立の発生 | 高 |
| **行動認識アラート** | 重大な不一致やループが検出された時 | 高 |
| **アイドル時** | 実行すべきアクションがない時 | 低 |

```typescript
interface ReflectionTrigger {
  type: "periodic" | "goal_completion" | "critical_event"
       | "social_event" | "action_awareness_alert" | "idle";
  priority: number;
  context: Record<string, unknown>;
}

class ReflectionScheduler {
  private triggerQueue: PriorityQueue<ReflectionTrigger>;
  private lastReflectionTime: number;
  private minInterval: number = 120_000; // 最低2分間隔

  shouldReflect(): boolean {
    const now = Date.now();
    if (now - this.lastReflectionTime < this.minInterval) {
      return false; // クールダウン中
    }
    return !this.triggerQueue.isEmpty();
  }
}
```

### 4.3 振り返りプロセス

Reflexionアーキテクチャを参考にした3段階プロセス。

```
[行動履歴] ──→ [要約（Summarize）] ──→ [評価（Evaluate）] ──→ [教訓抽出（Extract Lessons）]
                                                                        ↓
                                                              [記憶への統合]
                                                              [目標の修正]
                                                              [戦略の更新]
```

### 4.4 LLMプロンプト設計

#### 振り返りプロンプト

```
You are {agent_name}. Reflect on your recent actions and experiences.

## Recent Action History
{action_history_summary}

## Goals Attempted
- Completed: {completed_goals}
- Failed: {failed_goals}
- In Progress: {active_goals}

## Action Awareness Reports
{discrepancy_reports_summary}

## Social Interactions
{interaction_summary}

## Current Role
{current_inferred_role}

Reflect on the following:
1. What went well? What actions were effective?
2. What went wrong? Why did certain goals fail?
3. What patterns do you notice in your behavior?
4. What should you do differently next time?
5. Are there any new insights about the environment or other agents?

Respond in JSON:
{
  "summary": "<brief summary of the reflection period>",
  "successes": ["<what worked>"],
  "failures": ["<what did not work>"],
  "lessons": [
    {
      "insight": "<key lesson learned>",
      "category": "strategy|social|environment|skill",
      "actionable": "<specific behavioral change>"
    }
  ],
  "goal_adjustments": [
    {
      "current_goal": "<goal to modify>",
      "adjustment": "keep|modify|abandon",
      "reason": "<why>",
      "new_goal": "<if modify, the new version>"
    }
  ],
  "role_assessment": {
    "current_role": "<inferred role>",
    "satisfaction": <1-10>,
    "should_change": <boolean>,
    "reason": "<why keep or change>"
  }
}
```

### 4.5 振り返り結果の記憶への統合

```typescript
interface ReflectionResult {
  summary: string;
  successes: string[];
  failures: string[];
  lessons: Lesson[];
  goalAdjustments: GoalAdjustment[];
  roleAssessment: RoleAssessment;
  timestamp: number;
}

class SelfReflectionModule {
  private llmClient: LLMClient;
  private scheduler: ReflectionScheduler;

  async reflect(agentState: AgentState): Promise<ReflectionResult> {
    const trigger = this.scheduler.getNextTrigger();
    if (!trigger) return null;

    // 振り返り対象期間の情報を収集
    const context = this.gatherReflectionContext(agentState, trigger);
    const prompt = this.buildReflectionPrompt(context);

    const response = await this.llmClient.generate(prompt);
    const result = JSON.parse(response) as ReflectionResult;

    // 記憶システムへの統合
    await this.integrateIntoMemory(result, agentState);

    // 目標の修正
    await this.applyGoalAdjustments(result.goalAdjustments, agentState);

    return result;
  }

  private async integrateIntoMemory(
    result: ReflectionResult,
    agentState: AgentState
  ): Promise<void> {
    // 教訓を長期記憶に保存
    for (const lesson of result.lessons) {
      await agentState.memory.storeLongTerm({
        type: "reflection_lesson",
        content: lesson.insight,
        category: lesson.category,
        actionable: lesson.actionable,
        timestamp: result.timestamp,
        importance: this.assessImportance(lesson),
      });
    }

    // 振り返り要約を短期記憶に保存
    await agentState.memory.storeShortTerm({
      type: "reflection_summary",
      content: result.summary,
      timestamp: result.timestamp,
    });
  }

  private async applyGoalAdjustments(
    adjustments: GoalAdjustment[],
    agentState: AgentState
  ): Promise<void> {
    for (const adj of adjustments) {
      switch (adj.adjustment) {
        case "abandon":
          agentState.goalModule.abandonGoal(adj.currentGoal);
          break;
        case "modify":
          agentState.goalModule.modifyGoal(adj.currentGoal, adj.newGoal);
          break;
        case "keep":
          // 変更なし
          break;
      }
    }
  }
}
```

---

## 5. 専門化を駆動する目標生成

### 5.1 文脈依存的な社会的目標

専門化は、社会的目標の生成パターンが時間とともに特定のカテゴリに収束することで自然に創発する。論文では以下のメカニズムが示されている。

```
初期状態: 全エージェントが同一のミッション・性格
                ↓
[社会的目標の生成] × 5-10秒間隔
                ↓
[他のエージェントの行動を観察]（社会認識モジュール）
                ↓
[自分の強み/成功パターンの認識]（自己省察モジュール）
                ↓
[特定カテゴリの目標が増加] ← 正のフィードバックループ
                ↓
[役割の安定化] -- 農民、鉱夫、技術者、警備員、探検家、鍛冶屋
```

### 5.2 役割持続性のメカニズム

```typescript
interface RolePersistenceTracker {
  // 直近N個の社会的目標のカテゴリ分布
  goalCategoryHistory: GoalCategory[];
  historyWindowSize: number;  // = 30 (5目標 x 6回分)

  // 現在の推定役割
  inferredRole: GoalCategory | null;

  // 役割の安定度（0.0-1.0）
  roleStability: number;

  // カテゴリ分布を計算
  getCategoryDistribution(): Map<GoalCategory, number>;

  // 支配的カテゴリが閾値以上なら役割として確定
  inferRole(threshold: number): GoalCategory | null;
}
```

**役割持続性の数理モデル**:

```
P(category_t = c | history) ∝ α * freq(c, recent_goals) + β * success_rate(c) + γ * community_need(c)
```

- `α`: 慣性項（最近の目標カテゴリの頻度）-- 役割の持続を促進
- `β`: 成功報酬（そのカテゴリの目標の成功率）-- 得意分野への収束
- `γ`: 社会的圧力（コミュニティがそのカテゴリを必要としている度合い）-- 社会認識モジュールから

### 5.3 村落構成による分化パターン

論文では3種類の村落構成で異なる専門化パターンが観察されている。

| 村落構成 | 初期コミュニティ目標 | 創発する専門役割 |
|---|---|---|
| **通常（Normal）** | 効率的な村を作り、コミュニティを守る | 農民、鉱夫、技術者、警備員、探検家、鍛冶屋 |
| **武闘（Martial）** | 防衛に重点を置いた軍事的社会 | 偵察兵（Scout）、戦略家（Strategist）、警備員 |
| **芸術（Artistic）** | 芸術と文化を重視する社会 | 学芸員（Curator）、収集家（Collector）、装飾家 |

```typescript
interface CommunityConfig {
  name: string;
  type: "normal" | "martial" | "artistic";
  communityGoal: string;

  // 初期コミュニティ目標に応じた目標生成バイアス
  goalBias: Map<GoalCategory, number>;
}

// コミュニティ設定例
const normalCommunity: CommunityConfig = {
  name: "Village Alpha",
  type: "normal",
  communityGoal: "Build an efficient village and protect the community",
  goalBias: new Map([
    ["farming", 1.0], ["mining", 1.0], ["engineering", 1.0],
    ["guarding", 1.0], ["exploring", 1.0], ["crafting", 1.0],
  ]),  // 均等 → 相互作用から自然に分化
};

const martialCommunity: CommunityConfig = {
  name: "Fortress Beta",
  type: "martial",
  communityGoal: "Establish a fortified military outpost and defend against all threats",
  goalBias: new Map([
    ["farming", 0.5], ["mining", 0.8], ["engineering", 1.2],
    ["guarding", 2.0], ["exploring", 1.5], ["crafting", 1.2],
  ]),  // 防衛・偵察にバイアス → 軍事的役割が分化
};

const artisticCommunity: CommunityConfig = {
  name: "Haven Gamma",
  type: "artistic",
  communityGoal: "Create a beautiful settlement celebrating art and culture",
  goalBias: new Map([
    ["farming", 0.8], ["mining", 0.5], ["engineering", 1.0],
    ["guarding", 0.3], ["exploring", 1.2], ["crafting", 1.5],
  ]),  // 建築・工芸にバイアス → 芸術的役割が分化
};
```

### 5.4 専門化の評価

論文に基づき、専門化の質を以下の指標で評価する。

```typescript
interface SpecializationMetrics {
  // 役割エントロピー: 高いほど多様な役割が存在
  roleEntropy: number;

  // 役割持続時間: 個々のエージェントの役割が安定している度合い
  rolePersistenceDuration: Map<string, number>;

  // 行動-役割相関: 推定された役割と実際のアクションの一致度
  behaviorRoleCorrelation: number;

  // 役割カバレッジ: コミュニティ内で必要な役割がどれだけ充足されているか
  roleCoverage: number;
}
```

---

## 6. モジュール間の統合

### 6.1 データフロー

```
[目標生成] ──goals──→ [計画] ──plan──→ [スキル実行] ──action──→ [環境]
                                                                    ↓
                                                              [行動認識]
                                                                    ↓
                                            ┌── discrepancy ──→ [計画（再計画）]
                                            └── feedback ──→ [自己省察]
                                                                    ↓
                                                              [目標修正]
                                                              [記憶更新]
```

### 6.2 共有エージェント状態への読み書き

各モジュールはステートレスに動作し、共有エージェント状態を介して間接的に通信する。

```typescript
interface SharedAgentState {
  // 目標関連
  longTermGoals: GoalNode[];
  midTermGoals: GoalNode[];
  currentSocialGoals: SocialGoal[];     // 5目標ローリングウィンドウ
  inferredRole: GoalCategory | null;

  // 計画関連
  activePlan: Plan | null;
  planExecutionState: PlanExecutionState | null;

  // 行動認識関連
  actionAwarenessReports: DiscrepancyReport[];
  recentActionHistory: ActionRecord[];

  // 自己省察関連
  lastReflection: ReflectionResult | null;
  lessons: Lesson[];

  // 環境状態
  inventory: Inventory;
  currentPosition: Position;
  nearbyEntities: Entity[];
  communityGoal: string;
}
```

### 6.3 並行実行とタイミング

```
時間 →
                     0s    5s   10s   15s   20s   ...  120s  125s
行動認識（高速）     ████  ████  ████  ████  ████       ████  ████
社会的目標（5-10s)   █──────█──────█──────█──────         █──────█
計画（低速）         ██████████                           ██████
自己省察（低速）                                          ████████████
認知コントローラ     ██  ██  ██  ██  ██  ██  ██     ██  ██  ██  ██

█ = 実行中、─ = 待機中
```

**重要な制約**: 低速モジュール（目標生成、計画、自己省察）が実行中でも、高速モジュール（行動認識）は中断されず継続実行する。これにより「ゆっくり考え、素早く行動する」を実現する。

---

## 7. 実装上の考慮事項

### 7.1 技術選定

| コンポーネント | 推奨技術 | 理由 |
|---|---|---|
| 目標生成/計画/自己省察 | GPT-4o API / Claude API | LLMベースの推論が必要 |
| 行動認識ニューラルネット | ONNX Runtime / TensorFlow.js | 軽量・高速推論 |
| 知識グラフ | Neo4j / 自前グラフ構造 | 関係性の効率的クエリ |
| 依存グラフ | 自前DAG実装 | Minecraft固有の依存関係 |
| 並行実行 | Node.js Worker Threads / async | 非ブロッキング並行処理 |

### 7.2 LLMコスト最適化

| 最適化手法 | 適用対象 | 期待される削減率 |
|---|---|---|
| プロンプトキャッシュ | 類似コンテキストでの目標生成 | 30-50% |
| バッチ処理 | 複数エージェントの自己省察 | 20-40% |
| 軽量モデルへのフォールバック | 低優先度の目標生成 | 50-70% |
| 行動認識の非LLM化 | 全アクション検証 | 95%以上（LLM不使用） |

### 7.3 先行研究との比較

| 機能 | Project Sid (PIANO) | Generative Agents | Voyager | DEPS |
|---|---|---|---|---|
| 目標生成 | グラフ推論 + 社会的目標 | 記憶からの計画 | 自動カリキュラム | N/A |
| 計画 | HTN + LLM動的修正 | 日次/時間ベース計画 | タスク分解 | Describe-Explain-Plan-Select |
| 行動検証 | **非LLMニューラルネット** | なし | 自己検証コード | 失敗フィードバック |
| 自己省察 | 低速定期振り返り | 定期的反省 | なし | 失敗時の説明 |
| 並行実行 | **マルチスレッド並行** | 逐次実行 | 逐次実行 | 逐次実行 |
| 専門化 | **自発的役割分化** | 固定的性格 | N/A | N/A |

### 7.4 実装優先度

| 優先度 | モジュール/機能 | 理由 |
|---|---|---|
| **P0（必須）** | 行動認識モジュール | 論文で最重要モジュールと特定。除去で性能大幅低下 |
| **P0（必須）** | 社会的目標生成（5-10秒ループ） | エージェントの行動駆動の中核 |
| **P1（重要）** | 計画モジュール（基本版） | 目標を実行可能なアクションに分解 |
| **P1（重要）** | 行動認識の学習パイプライン | モデルの精度向上に不可欠 |
| **P2（推奨）** | 自己省察モジュール | 長期的な行動改善 |
| **P2（推奨）** | 知識グラフ / 依存グラフ | 目標生成の質向上 |
| **P3（将来）** | HTN分解の完全実装 | LLMコスト削減、計画の信頼性向上 |
| **P3（将来）** | 継続的オンライン学習 | 行動認識モデルの適応的改善 |

---

## 8. 参考文献・先行研究

- **Project Sid**: Altera.AL (2024). "Project Sid: Many-agent simulations toward AI civilization." [arXiv:2411.00114](https://arxiv.org/abs/2411.00114)
- **Reflexion**: Shinn et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- **Generative Agents**: Park et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." [ACM UIST 2023](https://dl.acm.org/doi/10.1145/3586183.3606763)
- **Voyager**: Wang et al. (2023). "Voyager: An Open-Ended Embodied Agent with Large Language Models." [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
- **DEPS**: Wang et al. (2023). "Describe, Explain, Plan and Select: Interactive Planning with LLMs." [arXiv:2302.01560](https://arxiv.org/abs/2302.01560)
- **ChatHTN**: LLMとHTN計画の統合フレームワーク. [arXiv:2505.11814](https://arxiv.org/abs/2505.11814)
- **VillagerAgent**: グラフベースのマルチエージェントフレームワーク for Minecraft. [ACL 2024](https://aclanthology.org/2024.findings-acl.964/)
- **Graphiti**: リアルタイム知識グラフ構築フレームワーク. [GitHub](https://github.com/getzep/graphiti)

---
## 関連ドキュメント
- [05-minecraft-platform.md](./05-minecraft-platform.md) — Minecraft基盤技術
- [06-social-cognition.md](./06-social-cognition.md) — 社会認知モジュール
- [08-infrastructure.md](./08-infrastructure.md) — インフラ設計
