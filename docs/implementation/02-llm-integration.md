# 02. LLM統合戦略

> 概要: PIANOの並列モジュールに対応するLLM APIアブストラクション層、モジュール別プロンプト設計、モデル選択戦略、コスト最適化の実装方針
> 対応論文セクション: Section 2 (PIANOアーキテクチャ)、Section 4 (モジュール詳細)
> 最終更新: 2026-02-23

---

## 目次

1. [LLM APIアブストラクション層](#1-llm-apiアブストラクション層)
2. [モジュール別プロンプト設計](#2-モジュール別プロンプト設計)
3. [モデル選択戦略](#3-モデル選択戦略)
4. [コスト最適化](#4-コスト最適化)
5. [非LLMニューラルネットワーク](#5-非llmニューラルネットワーク)

---

## 1. LLM APIアブストラクション層

### 1.1 設計目標

PIANOアーキテクチャは約10個のモジュールが並列に動作し、各モジュールがLLMまたは非LLMニューラルネットを使用する。LLM呼び出しを統一的に管理するアブストラクション層が必要となる。

**要件**:

- 複数プロバイダ（OpenAI, Anthropic, Google, ローカルモデル）の統一インターフェース
- モジュールごとのモデルルーティング
- フォールバック・リトライの自動処理
- レート制限・並行制御の管理
- コスト追跡・監視

### 1.2 統一インターフェース設計

> **言語方針**: 00-overview.mdでPython 3.12+が主要言語と決定されているため、LLM APIアブストラクション層を含むバックエンド全体はPythonで実装する。TypeScriptはMineflayerとの連携（05-minecraft-platform.md参照）に限定して使用し、Python-TypeScript間はZMQ/gRPC/RESTによるブリッジで接続する（ブリッジ方式はPhase 0で決定）。

```python
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Literal


class PIANOModule(str, Enum):
    """PIANOモジュール種別"""
    COGNITIVE_CONTROLLER = "cognitive_controller"
    GOAL_GENERATION = "goal_generation"
    PLANNING = "planning"
    SOCIAL_AWARENESS = "social_awareness"
    ACTION_AWARENESS = "action_awareness"
    TALKING = "talking"
    SELF_REFLECTION = "self_reflection"
    MEMORY = "memory"
    SKILL_EXECUTION = "skill_execution"


@dataclass
class RequestMetadata:
    module: PIANOModule
    agent_id: str
    priority: Literal["high", "normal", "low"] = "normal"


@dataclass
class CompletionRequest:
    messages: list[dict]
    model: str | None = None        # 省略時はルーターが決定
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: Literal["text", "json"] = "text"
    metadata: RequestMetadata | None = None


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0


@dataclass
class CompletionResponse:
    content: str
    usage: TokenUsage
    model: str
    latency_ms: float
    cost: float  # USD


class LLMProvider:
    """LLM呼び出しの統一インターフェース（抽象基底クラス）"""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise NotImplementedError

    async def stream_complete(self, request: CompletionRequest) -> AsyncIterator[str]:
        raise NotImplementedError

    def get_model_info(self) -> dict:
        raise NotImplementedError

    def estimate_cost(self, request: CompletionRequest) -> float:
        raise NotImplementedError
```

### 1.3 プロバイダアダプタ

各LLMプロバイダに対してアダプタを実装する。

| プロバイダ | アダプタ | 用途 |
|---|---|---|
| **OpenAI** | `OpenAIAdapter` | GPT-4o, GPT-4o-mini。論文のベースモデル |
| **Anthropic** | `AnthropicAdapter` | Claude Sonnet/Haiku。代替モデル |
| **Google** | `GoogleAdapter` | Gemini Flash/Pro。低コスト高速処理 |
| **Ollama** | `OllamaAdapter` | Llama, Gemma等ローカルモデル。開発・テスト用 |
| **vLLM** | `VLLMAdapter` | 本番ローカル推論。高スループット |

**推奨ライブラリ**: プロバイダ抽象化には以下を検討。

| ライブラリ | 特徴 | 推奨度 |
|---|---|---|
| **LiteLLM** | 100+モデルの統一API、フォールバック内蔵 | 高（初期開発） |
| **OpenRouter** | 623+モデル、単一API | 中（プロトタイプ） |
| **自前実装** | 完全な制御、PIANO固有の最適化 | 高（本番） |

初期開発ではLiteLLMを使用し、本番移行時にPIANO固有の要件（モジュール別ルーティング、エージェント単位のコスト追跡等）に合わせたカスタム実装へ移行するハイブリッドアプローチを推奨する。

### 1.4 モデルルーター

PIANOの各モジュールは異なる速度・品質要件を持つため、タスク種類に応じてモデルを自動選択するルーターを実装する。

```python
@dataclass
class RoutingDecision:
    primary_model: str
    fallback_models: list[str]
    rationale: str


@dataclass
class RoutingRule:
    tier: Literal["high", "medium", "low", "none"]
    max_latency_ms: int
    models: list[str]


class ModelRouter:
    """PIANOモジュール別にモデルを自動選択するルーター"""

    async def route(self, request: CompletionRequest) -> RoutingDecision:
        raise NotImplementedError


# ルーティングルール（設定ファイルで管理）
ROUTING_CONFIG: dict[PIANOModule, RoutingRule] = {
    PIANOModule.COGNITIVE_CONTROLLER: RoutingRule(
        tier="high",         # 高品質モデル必須
        max_latency_ms=5000,
        models=["gpt-4o", "claude-sonnet-4-5"],
    ),
    PIANOModule.GOAL_GENERATION: RoutingRule(
        tier="high",
        max_latency_ms=10000,  # 低速モジュール、遅延許容
        models=["gpt-4o", "claude-sonnet-4-5"],
    ),
    PIANOModule.PLANNING: RoutingRule(
        tier="high",
        max_latency_ms=10000,
        models=["gpt-4o", "claude-sonnet-4-5"],
    ),
    PIANOModule.SOCIAL_AWARENESS: RoutingRule(
        tier="medium",
        max_latency_ms=5000,
        models=["gpt-4o-mini", "claude-haiku-4-5", "gemini-2.0-flash"],
    ),
    PIANOModule.TALKING: RoutingRule(
        tier="medium",
        max_latency_ms=2000,   # 高速応答が必要
        models=["gpt-4o-mini", "claude-haiku-4-5"],
    ),
    PIANOModule.SELF_REFLECTION: RoutingRule(
        tier="medium",
        max_latency_ms=10000,
        models=["gpt-4o-mini", "claude-haiku-4-5"],
    ),
    PIANOModule.MEMORY: RoutingRule(
        tier="low",          # 検索・要約は軽量モデルで十分
        max_latency_ms=3000,
        models=["gpt-4o-mini", "gemini-2.0-flash-lite"],
    ),
    PIANOModule.ACTION_AWARENESS: RoutingRule(
        tier="none",         # 非LLM（専用NN）
        max_latency_ms=5,
        models=[],
    ),
    PIANOModule.SKILL_EXECUTION: RoutingRule(
        tier="none",         # 非LLM（ルールベース + 軽量NN）
        max_latency_ms=10,
        models=[],
    ),
}
```

### 1.5 フォールバック・リトライ戦略

マルチエージェント環境ではAPI障害の影響が増幅されるため、堅牢なフォールバック設計が不可欠。

```
リクエスト → プライマリモデル
              ↓ (失敗/タイムアウト)
            フォールバック1（同プロバイダ別モデル）
              ↓ (失敗)
            フォールバック2（別プロバイダ）
              ↓ (失敗)
            フォールバック3（ローカルモデル）
              ↓ (失敗)
            デグレード応答（キャッシュ or デフォルト）
```

| 戦略 | 説明 | 適用場面 |
|---|---|---|
| **指数バックオフ** | 初回1s、2s、4s...で再試行（最大3回） | レート制限（429） |
| **同プロバイダフォールバック** | GPT-4o → GPT-4o-mini | 品質低下許容時 |
| **クロスプロバイダフォールバック** | OpenAI → Anthropic → Google | プロバイダ障害 |
| **ローカルフォールバック** | API全滅 → Ollama/vLLM | 完全API障害 |
| **デグレード応答** | 直近のキャッシュ結果を返す | 全モデル不通時 |

**並行制御**: 500-1000エージェント x 複数モジュールからの同時リクエストに対し、プロバイダごとのレート制限に従うセマフォ/トークンバケットを実装する。

#### レート制限問題と対策

500体規模ではCC（10秒間隔）だけで1分あたり3,000リクエスト、全モジュール合計で約2.5M-5M TPMが必要と推定される。一方、GPT-4o単一アカウントのレート制限はTier 5でも800K TPM程度であり、単一プロバイダでは処理しきれない。

**具体的な対策**:

| 戦略 | 説明 | 効果 |
|---|---|---|
| **マルチプロバイダ分散** | Tier 1をOpenAI/Anthropic/Googleに分散。例: CC→GPT-4o、目標生成→Claude Sonnet 4.5、計画→Gemini 2.0 Pro | TPM上限を3倍に拡大 |
| **Tier別プロバイダ分離** | Tier 1はAPI（GPT-4o等）、Tier 2-3はローカルモデル（vLLM）で処理 | API側のTPM消費をTier 1のみに限定 |
| **マルチアカウント** | 同一プロバイダで複数Organization/Projectを使用し、レート制限枠を拡大 | 単一プロバイダのTPMをN倍に |
| **ローカルモデル併用** | 100体以上ではTier 2-3をvLLM+Llama 3.3でローカル処理 | APIレート制限から完全に独立 |
| **階層的処理** | 軽量分類器で変化なし/軽微を判定し、LLM呼び出し自体を削減 | リクエスト数を30-50%削減 |

**規模別のレート制限対応方針**:

- **10-50体**: 単一アカウント（GPT-4o Tier 3-4: 450K-800K TPM）で対応可能
- **100-500体**: マルチプロバイダ分散 + Tier 2-3ローカル化が必須
- **500-1000体**: ローカルモデル中心 + CCのみマルチプロバイダAPI

```python
@dataclass
class RateLimiterConfig:
    provider: str
    max_requests_per_minute: int   # RPM
    max_tokens_per_minute: int     # TPM

class RateLimiter:
    """プロバイダごとのレート制限管理（トークンバケット方式）"""

    async def acquire(self) -> None:
        """枠が空くまで待機"""
        ...

    def release(self) -> None:
        ...
```

---

## 2. モジュール別プロンプト設計

### 2.1 プロンプトテンプレート管理方式

プロンプトはコードと分離して管理し、バージョニング・A/Bテストを容易にする。

```
prompts/
  ├── cognitive_controller/
  │   ├── main.prompt.yaml
  │   └── broadcast.prompt.yaml
  ├── goal_generation/
  │   ├── generate.prompt.yaml
  │   └── evaluate.prompt.yaml
  ├── planning/
  │   └── plan.prompt.yaml
  ├── social_awareness/
  │   ├── analyze.prompt.yaml
  │   └── sentiment.prompt.yaml
  ├── talking/
  │   ├── generate_speech.prompt.yaml
  │   └── interpret.prompt.yaml
  ├── self_reflection/
  │   └── reflect.prompt.yaml
  └── memory/
      ├── summarize.prompt.yaml
      └── retrieve.prompt.yaml
```

**プロンプト言語方針**:

| 用途 | 言語 | 理由 |
|---|---|---|
| **内部推論（CC、目標生成、計画、自己省察）** | **英語** | LLMの推論精度は英語で最適化されている。GPT-4o、Claude等の主要モデルは英語プロンプトで最高性能を発揮する |
| **エージェント発話** | **任意言語** | 発話モジュールの出力はエージェントの「言語」設定に従う。日本語・英語等、シミュレーションの設定に応じて変更可能 |
| **システム指示（行動原則等）** | **英語** | 安定した指示遵守のため |
| **入力データ（記憶、環境情報）** | **英語** | データの一貫性のため |

> **注**: 以降のプロンプトテンプレート例は説明の便宜上日本語で記述しているが、実装時は上記方針に従い英語化する。

**テンプレート形式**:

```yaml
# prompts/cognitive_controller/main.prompt.yaml
name: cognitive_controller_main
version: "1.0.0"
model_tier: high
description: "認知コントローラの統合意思決定プロンプト"

system: |
  あなたは{{agent_name}}のCognitive Controller（認知コントローラ）です。
  あなたの役割は、複数のモジュールからの情報を統合し、
  一貫した高レベルの意思決定を行うことです。

  ## あなたの性格特性
  {{personality_traits}}

  ## 行動原則
  - 言葉と行動の一貫性を維持すること
  - 現在の目標と社会的文脈を考慮すること
  - 自分の行動が他者に与える影響を考慮すること

user: |
  ## 現在の状態
  - 位置: {{position}}
  - 体力: {{health}} / 満腹度: {{food}}
  - インベントリ: {{inventory_summary}}

  ## モジュールからの入力
  ### 目標生成モジュール
  {{goal_output}}

  ### 社会認識モジュール
  {{social_output}}

  ### 行動認識モジュール
  {{action_output}}

  ### 記憶（作業記憶）
  {{working_memory}}

  ## 判断を求めるタスク
  上記の情報を統合し、次の行動方針を決定してください。
  応答はJSON形式で返してください。

response_format: json
json_schema:
  type: object
  properties:
    decision:
      type: string
      description: "高レベルの意思決定（1-2文）"
    action:
      type: string
      enum: [explore, gather, craft, build, talk, trade, rest, help, flee]
    target:
      type: string
      description: "行動の対象"
    speech_intent:
      type: string
      description: "発話がある場合のその意図"
    priority:
      type: string
      enum: [urgent, normal, low]
```

### 2.2 認知コントローラ（CC）用プロンプト

CCはPIANOの中核であり、情報ボトルネック + ブロードキャストの二重機能を持つ。

**設計原則**:

1. **情報ボトルネック**: 全モジュールの出力を受け取り、エージェント状態から得られる情報を統合・圧縮する
2. **ブロードキャスト**: CCの意思決定を全出力モジュール（発話、スキル実行等）に一斉配信する
3. **一貫性保証**: 発話と行動が矛盾しないよう、言語的コミットメントを強く条件付ける

**入力構造**: 各モジュールの出力を構造化JSONとして受け取る。トークン数の制御のため、各モジュール出力を要約形式（100-200トークン以内）に圧縮して渡す。

**出力構造**: ブロードキャスト用に全出力モジュールが参照可能な標準形式で決定を返す。

**Few-shot例**（プロンプトに含める入出力例）:

```yaml
# prompts/cognitive_controller/few_shot_examples.yaml
examples:
  - name: "cooperation_scenario"
    input: |
      ## Current State
      - Position: (120, 64, -45), Health: 18/20, Food: 15/20
      - Inventory: oak_log x12, cobblestone x32, iron_pickaxe x1

      ## Module Inputs
      ### Goal Generation: Build a shared storage for the village (priority: 0.8)
      ### Social Awareness: Agent_Bob is nearby, likeability=7, offering to help build
      ### Action Awareness: Last action (mine_block) succeeded. Confidence: 0.95
      ### Working Memory: Promised Agent_Bob to cooperate on village storage yesterday
    output: |
      {
        "decision": "Accept Bob's help and begin building the shared storage together, fulfilling yesterday's commitment",
        "action": "build",
        "target": "shared_storage_with_Agent_Bob",
        "speech_intent": "Accept Bob's offer and coordinate building tasks",
        "priority": "normal"
      }

  - name: "threat_scenario"
    input: |
      ## Current State
      - Position: (80, 63, -30), Health: 6/20, Food: 3/20
      - Inventory: wooden_sword x1, bread x2

      ## Module Inputs
      ### Goal Generation: Explore the eastern cave (priority: 0.6)
      ### Social Awareness: No agents nearby
      ### Action Awareness: Last action (attack zombie) partially succeeded but took 8 damage. Confidence: 0.7
      ### Working Memory: Health is critically low after zombie encounter
    output: |
      {
        "decision": "Abandon exploration goal and prioritize survival by eating and retreating to safe area",
        "action": "rest",
        "target": "eat_bread_and_retreat_to_village",
        "speech_intent": null,
        "priority": "urgent"
      }
```

> **注**: Few-shot例はシステムプロンプトの末尾に配置し、プロンプトキャッシングの恩恵を受けられるようにする。2-3例を含め、典型的なパターン（協力、脅威対応、目標切替）をカバーする。

### 2.3 目標生成・計画用プロンプト

論文によると、目標生成はグラフ構造上での推論による目的創出を行う低速モジュールである。

**目標生成プロンプト設計**:

```yaml
# prompts/goal_generation/generate.prompt.yaml
name: goal_generation
version: "1.0.0"
model_tier: high

system: |
  あなたは{{agent_name}}の目標生成モジュールです。
  エージェントの現在の状況、社会的文脈、過去の経験に基づいて、
  次に追求すべき目標を生成してください。

  ## エージェントの性格
  {{personality_traits}}

  ## コミュニティ目標
  {{community_goals}}

user: |
  ## 現在の目標（5目標ローリングセット）
  {{current_goals}}

  ## 最近の経験
  {{recent_experiences}}

  ## 社会的状況
  - 近くにいるエージェント: {{nearby_agents}}
  - 最近の会話: {{recent_conversations_summary}}
  - 他者からの評価: {{social_perception}}

  ## 環境状況
  - 利用可能な資源: {{available_resources}}
  - 達成済みの進捗: {{achievements}}

  新しい目標を提案してください。既存の目標を修正・置換しても構いません。

response_format: json
json_schema:
  type: object
  properties:
    goals:
      type: array
      maxItems: 5
      items:
        type: object
        properties:
          id: { type: string }
          description: { type: string }
          type: { type: string, enum: [survival, social, exploration, craft, community] }
          priority: { type: number, minimum: 0, maximum: 1 }
          estimated_steps: { type: number }
    reasoning: { type: string }
```

**計画プロンプト**: 目標が決定された後、具体的なステップに分解する。Minecraftの技術依存ツリー（木のツルハシ → 石のツルハシ → ...）を考慮した計画生成が必要。

### 2.4 社会認識用プロンプト

社会認識モジュールは選択的起動モジュールであり、他のエージェントとの対話時にのみ実行される。アブレーション実験で、このモジュールの除去により社会的進化が完全に停止することが示されている。

**設計ポイント**:

- 他者の好感度（likeability）を0-10の離散スケールで推定
- 他者の意図・感情状態の推測
- 社会的文脈に基づく適応行動の提案

```yaml
# prompts/social_awareness/analyze.prompt.yaml
name: social_awareness_analyze
version: "1.0.0"
model_tier: medium

system: |
  あなたは{{agent_name}}の社会認識モジュールです。
  他のエージェントとの対話から社会的手がかりを解釈し、
  関係性と感情状態を分析してください。

user: |
  ## 対話相手: {{target_agent_name}}

  ## 過去の対話履歴
  {{interaction_history}}

  ## 対話相手の最近の行動
  {{target_recent_actions}}

  ## 対話相手への現在の好感度
  {{current_likeability_score}} / 10

  上記の情報に基づき、社会的分析を行ってください。

response_format: json
json_schema:
  type: object
  properties:
    likeability_update:
      type: number
      minimum: 0
      maximum: 10
    perceived_intent:
      type: string
    emotional_state:
      type: string
      enum: [friendly, neutral, hostile, anxious, excited, sad]
    recommended_approach:
      type: string
      enum: [cooperate, cautious, avoid, help, trade, socialize]
    reasoning:
      type: string
```

### 2.5 発話生成用プロンプト

発話モジュールはCCの決定に条件付けされて動作し、言行一致を保証する。

**設計ポイント**:

- CCからのブロードキャスト（decision, speech_intent）を必ず入力に含める
- エージェントの性格特性を反映した口調
- 文脈に適した長さ（短い挨拶〜詳細な説明）

**ストリーミング活用方針**:

発話モジュールはストリーミング応答を活用して体感レイテンシを改善する。

| モジュール | ストリーミング | 理由 |
|---|---|---|
| **発話生成** | **使用する** | エージェントの発話をトークン単位で逐次表示することで、チャット画面やMinecraft内の吹き出しで「考えながら話している」自然な印象を実現。TTFT（Time To First Token）の短縮が体感速度に直結する |
| CC | 使用しない | JSON構造化出力のため、完全なレスポンスが必要 |
| 目標生成・計画 | 使用しない | JSON出力 + 低速モジュールのため不要 |
| 社会認識 | 使用しない | JSON出力のため不要 |
| 記憶要約 | 使用しない | 内部処理のため不要 |

```yaml
# prompts/talking/generate_speech.prompt.yaml
name: talking_generate
version: "1.0.0"
model_tier: medium

system: |
  あなたは{{agent_name}}の発話生成モジュールです。
  認知コントローラの決定に基づき、適切な発話を生成してください。
  発話はエージェントの性格と一致し、行動と矛盾してはなりません。

  ## 性格特性
  {{personality_traits}}

  ## コミュニケーションスタイル
  {{communication_style}}

user: |
  ## 認知コントローラの決定
  {{cc_broadcast}}

  ## 会話の文脈
  - 相手: {{conversation_partner}}
  - 直前のやり取り: {{recent_dialogue}}

  ## 現在の行動
  {{current_action}}

  自然で性格に合った発話を生成してください。
  行動と矛盾する発言は絶対に避けてください。

response_format: json
json_schema:
  type: object
  properties:
    speech:
      type: string
      description: "生成された発話テキスト"
    tone:
      type: string
      enum: [friendly, formal, casual, urgent, playful, serious]
```

---

## 3. モデル選択戦略

### 3.1 PIANOモジュールとモデルティアのマッピング

論文ではGPT-4oが基盤モデルとして使用されているが、再現実装では**コスト最適化のためモジュールごとに異なるモデルを割り当てる**。

| モジュール | 速度要件 | 推論品質要件 | 推奨ティア | LLM使用 |
|---|---|---|---|---|
| 認知コントローラ（CC） | 中速 | 最高 | Tier 1（高性能） | Yes |
| 目標生成 | 低速 | 高 | Tier 1 | Yes |
| 計画 | 低速 | 高 | Tier 1 | Yes |
| 社会認識 | 選択的 | 中〜高 | Tier 2（中性能） | Yes |
| 発話 | 高速 | 中 | Tier 2 | Yes |
| 自己省察 | 低速 | 中 | Tier 2 | Yes |
| 記憶（要約・検索） | 可変 | 中 | Tier 2-3 | Yes |
| 行動認識 | 高速 | - | 非LLM | No |
| スキル実行 | 高速 | - | 非LLM | No |

### 3.2 具体的モデル選択肢（2026年2月時点）

#### Tier 1: 高性能モデル（CC、目標生成、計画）

| モデル | 入力/MTok | 出力/MTok | 特徴 | 推奨度 |
|---|---|---|---|---|
| **GPT-4o** | $2.50 | $10.00 | 論文のベースモデル。再現性最高 | 最高 |
| **Claude Sonnet 4.5** | $3.00 | $15.00 | 高品質推論。構造化出力に強い | 高 |
| **Gemini 2.0 Pro** | $1.25 | $5.00 | コスト効率が高い | 中 |
| **Llama 3.3 70B (vLLM)** | 自前GPU | 自前GPU | ローカル実行。大規模時に有利 | 中（大規模時） |

#### Tier 2: 中性能モデル（社会認識、発話、自己省察）

| モデル | 入力/MTok | 出力/MTok | 特徴 | 推奨度 |
|---|---|---|---|---|
| **GPT-4o-mini** | $0.15 | $0.60 | 最もコスト効率が高い | 最高 |
| **Claude Haiku 4.5** | $1.00 | $5.00 | 高速で高品質 | 中 |
| **Gemini 2.0 Flash** | $0.10 | $0.40 | 超低コスト | 高 |
| **Gemma 3 9B (Ollama)** | 自前GPU | 自前GPU | ローカル高速推論 | 中 |

#### Tier 3: 軽量モデル（記憶要約、簡易分類）

| モデル | 入力/MTok | 出力/MTok | 特徴 | 推奨度 |
|---|---|---|---|---|
| **Gemini 2.0 Flash Lite** | $0.08 | $0.30 | 最安値クラス | 高 |
| **GPT-4o-mini** | $0.15 | $0.60 | 安定性重視 | 高 |
| **Llama 3.2 8B (Ollama)** | 自前GPU | 自前GPU | ローカル超軽量 | 中 |

### 3.3 ローカルモデル vs APIモデルのトレードオフ

| 観点 | APIモデル | ローカルモデル |
|---|---|---|
| **初期コスト** | なし | GPU購入/レンタル（A100: $2,000-3,000/月） |
| **従量課金** | トークン単位で課金 | 電力・メンテのみ |
| **損益分岐点** | ~50エージェント以下で有利 | ~100エージェント以上で有利 |
| **レイテンシ** | ネットワーク依存（50-500ms） | ローカル（10-100ms） |
| **スループット** | プロバイダのレート制限に依存 | GPU数に比例してスケール |
| **モデル品質** | 最先端モデルが利用可能 | OSSモデルのみ（やや劣る） |
| **可用性** | プロバイダ障害リスク | ハードウェア障害のみ |
| **プライバシー** | データがプロバイダに送信される | データがローカルに留まる |

**推奨アプローチ**: ハイブリッド構成

- 10-50エージェント規模: API中心（GPT-4o + GPT-4o-mini）
- 100-500エージェント規模: Tier 1のみAPI、Tier 2-3はローカル（vLLM + Llama）
- 500-1000エージェント規模: 可能な限りローカル、CCのみAPI

### 3.4 非LLMモジュールの設計指針

行動認識（Action Awareness）とスキル実行（Skill Execution）は非LLMニューラルネットワークで実装する。詳細は[セクション5](#5-非llmニューラルネットワーク)を参照。

---

## 4. コスト最適化

### 4.1 コスト構造の分析

PIANOアーキテクチャにおけるLLM呼び出しのコスト構造:

```
総コスト = エージェント数 × LLMモジュール数 × 呼び出し頻度 × 呼び出し単価
```

**モジュール別の呼び出し頻度推定**（論文の実験パラメータから推定）:

| モジュール | 呼び出し間隔 | 1分あたり呼び出し数 | 平均入力トークン | 平均出力トークン |
|---|---|---|---|---|
| 認知コントローラ | 10秒 | 6回 | 1,500 | 300 |
| 目標生成 | 60秒 | 1回 | 2,000 | 500 |
| 計画 | 60秒 | 1回 | 1,500 | 400 |
| 社会認識 | 5-10秒（対話時のみ） | 3回（平均） | 1,000 | 200 |
| 発話 | 5秒（対話時のみ） | 4回（平均） | 800 | 150 |
| 自己省察 | 120秒 | 0.5回 | 1,200 | 300 |
| 記憶（要約） | 60秒 | 1回 | 1,000 | 200 |

### 4.2 規模別コスト見積もり

#### シナリオ条件
- 実行時間: 1時間
- 全モジュールがAPI呼び出し（最適化前のベースライン）
- 全モジュールでGPT-4o使用（$2.50/MTok入力、$10.00/MTok出力）

#### ベースライン試算（最適化前、全モジュールGPT-4o）

| 規模 | エージェント数 | 1時間あたりLLM呼び出し数 | 推定入力トークン | 推定出力トークン | 推定コスト/時間 |
|---|---|---|---|---|---|
| 小規模 | 10体 | ~9,900 | ~15.8M | ~3.5M | **$74.5** |
| 中規模 | 50体 | ~49,500 | ~79.2M | ~17.3M | **$371** |
| 大規模 | 500体 | ~495,000 | ~792M | ~173M | **$3,710** |
| 超大規模 | 1,000体 | ~990,000 | ~1,584M | ~346M | **$7,420** |

> 注: 論文の500体実験（2.5時間）はこのベースラインで約$9,275となり、研究予算としては妥当だが、再現研究や反復実験には高コスト。

#### 最適化後試算（モデルミックス + キャッシング + バッチ処理）

最適化戦略を適用した場合の推定:

| 規模 | 最適化前 | 最適化後 | 削減率 |
|---|---|---|---|
| 10体 | $74.5/h | **$8-15/h** | ~80-90% |
| 50体 | $371/h | **$35-65/h** | ~83-91% |
| 500体 | $3,710/h | **$250-500/h** | ~87-93% |
| 1,000体 | $7,420/h | **$450-900/h** | ~88-94% |

**最適化の内訳**:

| 最適化手法 | 削減率 | 適用対象 |
|---|---|---|
| モデルミックス（Tier分け） | 50-70% | Tier 2/3モジュールを低コストモデルへ |
| プロンプトキャッシング | 50-90% | システムプロンプトの再利用 |
| 階層的処理 | 20-40% | 軽量分類器による不要なLLM呼び出しスキップ |
| 選択的起動 | 30-50% | 社会認識は対話時のみ等 |
| ローカルモデル（大規模時） | 60-80% | Tier 2-3のローカル移行 |

> **注**: 各最適化手法の削減率は独立ではなく相互依存があるため、単純な乗算は適用できない。保守的な見積もりでは全体60-70%削減、楽観的な見積もりでは80-90%削減と想定する。上記の最適化後試算はこの範囲で算出している。

### 4.3 プロンプトキャッシング戦略

プロンプトキャッシングは最も効果的なコスト削減手段の一つ。PIANOでは各モジュールのシステムプロンプト（性格特性、行動原則等）が全呼び出しで共通するため、キャッシュヒット率が極めて高い。

**OpenAI プロンプトキャッシング**:
- 1,024トークン以上のプロンプトで自動適用
- キャッシュヒット時 50-90% 割引（モデルにより異なる）
- 追加設定不要

**Anthropic プロンプトキャッシング**:
- `cache_control` ブレークポイントを明示的に指定
- 5分間キャッシュ: 書き込み1.25倍、読み取り0.1倍（90%割引）
- 1時間キャッシュ: 書き込み2倍、読み取り0.1倍

**PIANOでの適用**:

```
システムプロンプト（~800トークン、キャッシュ対象）
  + 性格特性テンプレート（~200トークン、キャッシュ対象）
  + エージェント固有情報（~100トークン、動的）
  + モジュール入力データ（~500-1500トークン、動的）
```

各エージェントのシステムプロンプト + 性格特性テンプレートは同一エージェントの全呼び出しで共通のため、1,000トークン以上のプレフィックスがキャッシュされ、入力トークンの60-70%が割引対象となる。

### 4.4 セマンティックキャッシュ（Phase 2以降で検討）

> **Phase 0-1では導入しない**: セマンティックキャッシュ（GPTCacheベース）は追加インフラ（ベクトルDB + Redis等）の運用コストがかかり、PIANOのような動的環境ではキャッシュヒット率が低い（閾値0.95では「ほぼ同一のクエリ」のみヒット）。Phase 0-1ではプロンプトキャッシング（4.3）とモデルミックスによるコスト削減に集中し、十分なログデータが蓄積されるPhase 2以降でヒット率の実測値に基づき導入を判断する。

**Phase 2以降の導入時の設計方針**:

同一または類似のクエリに対するLLM応答をキャッシュし、再利用する。

```
クエリ → 埋め込みベクトル生成
  → ベクトルDB検索（類似度閾値: 0.95以上）
    → ヒット: キャッシュ結果を返却（LLM呼び出しスキップ）
    → ミス: LLM呼び出し → 結果をキャッシュに保存
```

**PIANOでの適用場面（ヒット率は実測が必要）**:

| 場面 | キャッシュヒット率（推定） | 理由 |
|---|---|---|
| 記憶要約 | 40-60% | 類似状況の繰り返し |
| 社会認識（初見判断） | 20-30% | エージェント間の関係パターンが類似 |
| 目標生成 | 10-20% | 文脈依存性が高くヒット率は低め |
| 認知コントローラ | 5-10% | 状態の組み合わせが多様 |

**注意点**: セマンティックキャッシュは創造性や多様性を損なうリスクがある。特に社会的行動の多様性が重要なPIANOでは、キャッシュヒット率を意図的に下げる（閾値を高くする）ことも検討すべき。

### 4.5 バッチ処理設計

#### 4.5.1 マイクロバッチ（リアルタイム用）

低速モジュール（目標生成、計画、自己省察）は数秒〜数十秒の遅延が許容されるため、複数エージェントのリクエストを2-5秒バッファリングして一括送信する**マイクロバッチ**方式でスループットを向上させる。これはLLM API側のバッチ処理ではなく、クライアント側でリクエストを蓄積して並行送信する方式である。

```python
@dataclass
class MicroBatchConfig:
    max_batch_size: int = 50          # 最大バッチサイズ
    max_wait_seconds: float = 5.0     # 最大待機時間（秒）
    eligible_modules: list[str] = field(default_factory=lambda: [
        "goal_generation", "planning", "self_reflection", "memory"
    ])

class MicroBatchProcessor:
    """複数エージェントのリクエストをバッファリングして一括送信"""

    async def enqueue(self, request: CompletionRequest) -> asyncio.Future:
        """リクエストをバッファに追加し、結果のFutureを返す"""
        ...

    async def flush(self) -> list[CompletionResponse]:
        """バッファ内のリクエストを並行送信"""
        # asyncio.gatherで並行実行し、スループットを最大化
        ...
```

**マイクロバッチの効果**: API側のバッチ割引は適用されないが、リクエストの並行送信によりスループットが向上し、レート制限の枠を効率的に使用できる。

**マイクロバッチ対象モジュール**:

| モジュール | バッチ対象 | バッファリング時間 | 理由 |
|---|---|---|---|
| 目標生成 | Yes | 2-5秒 | 低速。数秒の遅延許容 |
| 計画 | Yes | 2-5秒 | 低速 |
| 自己省察 | Yes | 2-5秒 | 低速 |
| 記憶要約 | Yes | 2-5秒 | 非リアルタイム |
| 認知コントローラ | **No** | - | リアルタイム性が必要 |
| 発話 | **No** | - | 即時応答が必要 |
| 社会認識 | **No** | - | 対話中のリアルタイム分析 |

#### 4.5.2 Batch API（オフライン分析用）

OpenAI Batch API（50%割引、24時間以内に結果返却）やAnthropic Message Batches API（50%割引）は、リアルタイムシミュレーション中のモジュールには使用できない（24時間の返却時間はリアルタイム要件と矛盾するため）。これらはシミュレーション後のオフライン処理に限定して使用する。

**Batch API適用場面**:
- シミュレーション後のログ分析・要約
- 大量の記憶データの一括再要約・再分類
- 実験結果の評価指標の一括計算
- プロンプト最適化のためのA/Bテストデータ生成

### 4.6 トークン削減テクニック

| テクニック | 削減率 | 説明 |
|---|---|---|
| **構造化JSON出力** | 15-25% | 自由形式テキストよりJSON指定で出力トークン削減 |
| **入力圧縮** | 20-40% | 記憶・履歴の要約化、不要情報の除外 |
| **プロンプト最適化** | 10-20% | 冗長な指示の削除、簡潔な表現 |
| **選択的起動** | 30-50% | 社会認識は対話時のみ、自己省察は一定間隔のみ |
| **差分入力** | 10-30% | 前回からの変更部分のみを入力 |
| **階層的処理** | 20-40% | 簡易判断を低コストモデルで実施し、複雑な場合のみ高性能モデルへエスカレート |

**階層的処理の例（認知コントローラ）**:

```
環境変化検出 → 軽量分類器（GPT-4o-mini）
  → 「変化なし」→ 前回の決定を維持（LLM呼び出しスキップ）
  → 「軽微な変化」→ GPT-4o-miniで判断
  → 「重大な変化」→ GPT-4oで詳細推論
```

### 4.7 コスト監視・アラート

大規模マルチエージェント環境ではコストが急激に増加するリスクがあるため、リアルタイム監視が必須。

```python
@dataclass
class CostAlertConfig:
    hourly_budget: float        # 1時間あたりの予算上限（USD）
    per_agent_budget: float     # エージェントあたりの予算上限
    per_module_budget: float    # モジュールあたりの予算上限

@dataclass
class CostReport:
    total_cost: float
    cost_by_provider: dict[str, float]
    cost_by_module: dict[str, float]
    cost_by_agent: dict[str, float]
    total_input_tokens: int
    total_output_tokens: int
    cached_input_tokens: int

class CostMonitor:
    """リアルタイムコスト追跡・アラート"""

    def track_cost(self, response: CompletionResponse) -> None:
        """LLM応答のコストを記録"""
        ...

    def set_alerts(self, config: CostAlertConfig) -> None:
        """アラート閾値を設定"""
        ...

    def get_report(self) -> CostReport:
        """コストレポートを生成"""
        ...
```

---

## 5. 非LLMニューラルネットワーク

### 5.1 行動認識モジュール（Action Awareness）

論文で「単一エージェントの進歩において最も重要なモジュール」として特定された行動認識モジュールは、非LLMの小規模ニューラルネットワークで実装する。

**機能**: 期待されるアクション結果と実際の結果を比較し、ハルシネーションに起因するエラーを検出・修正する。

**設計**:

```
入力:
  - 実行したアクション（action_type: string）
  - 期待される結果（expected_outcome: struct）
  - 実際の環境状態（actual_state: struct）
  - インベントリ変化（inventory_diff: struct）

出力:
  - 成功/失敗の判定（success: boolean）
  - 不一致の種類（mismatch_type: enum）
  - 修正アクションの提案（correction: string）
  - 信頼度スコア（confidence: float）
```

**入力ベクトル設計（128次元）**:

Minecraftの状態を128次元の固定長ベクトルに変換する。各次元の構成は以下の通り。

| 特徴量グループ | 次元数 | 内容 |
|---|---|---|
| **アクション種別** | 16 | one-hotエンコーディング（mine_block, place_block, craft_item, pickup_item, equip, eat, attack, interact, move_to, use_item, drop, fish, trade, sleep, respawn, other） |
| **位置変化** | 6 | 実行前XYZ座標(3) + 実行後XYZ座標(3)、正規化 |
| **インベントリ差分** | 32 | 主要アイテムカテゴリ（木材系, 石系, 鉄系, 食料系, ツール系, 武器系, 建材系, その他）× 4値（期待増減, 実際増減, 期待保有量, 実際保有量） |
| **対象ブロック/エンティティ** | 16 | ブロックID埋め込み(8) + エンティティタイプ埋め込み(8) |
| **体力・満腹度変化** | 4 | 実行前HP(1) + 実行後HP(1) + 実行前食料(1) + 実行後食料(1) |
| **周辺環境** | 24 | 隣接6方向のブロックタイプ埋め込み(6×4) |
| **時間特徴** | 4 | ゲーム内時刻(1) + アクション経過時間(1) + 昼夜フラグ(1) + 天候(1) |
| **アクション履歴** | 16 | 直前4アクションのタイプ埋め込み(4×4) |
| **期待結果フラグ** | 10 | 成功期待(1) + アイテム取得期待(1) + ブロック変化期待(1) + HP変化期待(1) + 位置変化期待(1) + 予備(5) |
| **合計** | **128** | |

> **注**: ブロックID・エンティティタイプ・アクション履歴の埋め込みは学習時に獲得する低次元表現。初期実装ではハッシュエンコーディングで固定長ベクトルに変換する。

**ニューラルネットワークアーキテクチャ**:

```
入力層（状態ベクトル: 128次元）
  → 全結合層（256ユニット, ReLU）
  → 全結合層（128ユニット, ReLU）
  → ドロップアウト（0.2）
  → 出力層（分類ヘッド + 回帰ヘッド）

パラメータ数: ~100K（非常に軽量）
推論時間: <1ms
```

**代替アプローチ**: ルールベースシステム

行動認識はアクションと結果の対応関係が比較的決定的であるため、ルールベースのシステムでも実装可能。初期実装ではルールベースで開始し、エッジケースの蓄積に応じてNNに移行することを推奨する。

```python
def check_action_result(action: Action, expected: State, actual: State) -> ActionResult:
    """ルールベース行動認識の例"""
    # アイテム取得アクションの検証
    if action.type == "mine_block":
        expected_item = action.target_block.drop_item
        has_item = expected_item in actual.inventory
        if not has_item and expected_item in expected.inventory:
            return ActionResult(
                success=False,
                mismatch_type="hallucinated_item_acquisition",
                correction=f"Retry mining {action.target_block.name}",
                confidence=0.95,
            )
    # ... その他のルール
```

### 5.2 学習データの収集・生成方法

NN版行動認識モジュールの学習データは、シミュレーション実行のログから収集する。

**データ収集パイプライン**:

```
Minecraftシミュレーション実行
  → アクション・状態ペアのログ記録
  → 成功/失敗のラベル付け（ルールベースで自動、エッジケースは手動）
  → 学習データセットの構築
```

**データ量の目安**:

| ステージ | データ量 | 期待精度 |
|---|---|---|
| 初期 | 10,000サンプル | ~85% |
| 中期 | 100,000サンプル | ~92% |
| 成熟期 | 1,000,000サンプル | ~97% |

**データ生成方法**:

1. **シミュレーションログ**: 通常のエージェント実行から自動収集
2. **意図的失敗生成**: ランダムにアクション結果を改竄し、「失敗」ケースを増やす
3. **合成データ**: Minecraftのアイテム・ブロックのメタデータから状態遷移を合成

### 5.3 スキル実行モジュール

スキル実行は高レベル意図を環境操作に変換する。主にMineflayerのAPIを介して実装され、LLMは使用しない。

**構成**:

```
CCからの高レベル指示（例: "木を伐採"）
  → スキルライブラリからの検索（コード検索、非NN）
  → Mineflayerアクションシーケンスへの変換
  → 実行 + 行動認識フィードバック
```

スキルライブラリの設計については、Minecraft基盤技術調査ドキュメント（05-minecraft-foundation.md）を参照。

### 5.4 推論速度の要件と最適化

| モジュール | 推論時間要件 | 実行頻度 | 最適化手法 |
|---|---|---|---|
| 行動認識 | <5ms | 毎ティック（50ms） | ONNX Runtime、量子化（INT8） |
| スキル実行 | <10ms | アクション実行時 | ルックアップテーブル、事前計算 |

**最適化手法**:

- **ONNX Runtime**: PyTorch/TensorFlowモデルをONNX形式に変換し、CPU推論を高速化
- **INT8量子化**: モデルサイズを1/4に削減、推論速度を2-4倍に向上
- **バッチ推論**: 複数エージェントの行動認識をバッチ処理（GPUがある場合）

---

## 付録: 技術選定サマリー

### 推奨技術スタック

| コンポーネント | 推奨技術 | 代替案 |
|---|---|---|
| LLM API抽象化 | LiteLLM（初期）→ カスタム（本番） | OpenRouter, LangChain |
| プライマリLLM | GPT-4o | Claude Sonnet 4.5 |
| 高速LLM | GPT-4o-mini | Gemini 2.0 Flash |
| ローカルLLM | vLLM + Llama 3.3 | Ollama（開発時） |
| セマンティックキャッシュ | GPTCache + Redis（Phase 2以降） | カスタム実装 |
| プロンプト管理 | YAMLテンプレート | Jinja2, Handlebars |
| コスト監視 | カスタムダッシュボード | LangSmith, Helicone |
| 行動認識NN | PyTorch → ONNX Runtime | TensorFlow Lite |
| モデルルーティング | カスタムルーター | RouteLLM |

### リスクと緩和策

| リスク | 影響度 | 緩和策 |
|---|---|---|
| API価格変動 | 中 | マルチプロバイダ対応、ローカルモデルへの切替 |
| モデル廃止 | 中 | アブストラクション層による交換容易性 |
| レート制限 | 高 | 複数アカウント、ローカルフォールバック |
| 品質低下（低コストモデル使用時） | 中 | A/Bテスト、品質監視、段階的移行 |
| コスト超過 | 高 | リアルタイム監視、自動停止機構 |

---
## 関連ドキュメント
- [01-system-architecture.md](./01-system-architecture.md) — PIANO全体設計
- [03-cognitive-controller.md](./03-cognitive-controller.md) — 認知コントローラ
- [04-memory-system.md](./04-memory-system.md) — 記憶システム
