# 03. 認知コントローラ（Cognitive Controller）実装方針

[トップ](../index.md) | [PIANOアーキテクチャ分析](../02-piano-architecture.md)

---

## 目次

1. [Global Workspace Theoryの実装](#1-global-workspace-theoryの実装)
2. [情報ボトルネック設計](#2-情報ボトルネック設計)
3. [ブロードキャスト機構](#3-ブロードキャスト機構)
4. [実装設計](#4-実装設計)
5. [既存認知アーキテクチャとの比較](#5-既存認知アーキテクチャとの比較)
6. [推奨実装ロードマップ](#6-推奨実装ロードマップ)

---

## 1. Global Workspace Theoryの実装

### 1.1 理論的背景

**Global Workspace Theory（GWT）** は、Bernard Baarsが1988年に提唱した意識の理論である。脳内の多数の専門モジュール（視覚、聴覚、運動制御など）が並列に動作し、そのうち「意識的」な情報のみが**グローバルワークスペース**を通じて全モジュールに一斉配信（broadcast）されるという構造を持つ。

PIANOアーキテクチャの認知コントローラ（CC）は、まさにこのGWTの計算論的実装に対応する：

| GWT概念 | PIANO対応 |
|---|---|
| 専門モジュール群 | Memory, Goal Generation, Social Awareness等の並列モジュール |
| グローバルワークスペース | 認知コントローラ（CC） |
| 意識へのアクセス競争 | 情報ボトルネック（情報選択・圧縮） |
| 意識的ブロードキャスト | CCの意思決定を全出力モジュールに一斉配信 |
| 無意識的処理 | 各モジュールの独立した並列処理 |

### 1.2 計算モデルとしてのGWT

GWTの計算モデルは以下の3フェーズから構成される：

```
フェーズ1: 競争（Competition）
  複数のモジュールが情報をワークスペースに投入 → 重要度に基づく競争

フェーズ2: 点火（Ignition）
  勝者の情報がワークスペースを「占有」 → 閾値を超えた情報が選択

フェーズ3: ブロードキャスト（Broadcast）
  選択された情報が全モジュールに一斉配信 → 行動の一貫性を確保
```

### 1.3 ソフトウェアアーキテクチャへの変換方針

GWTをソフトウェアシステムに変換する際の設計原則：

**原則1: Observer/Listenerパターンの採用**
LIDAフレームワーク（後述）では、`BroadcastListener`インターフェースを実装したモジュールのみがブロードキャストを受信する。PIANOでも同様に、出力モジュールがCC決定を受信するためのリスナー機構を導入する。

**原則2: 非同期メッセージキューによるモジュール間通信**
各モジュールはステートレスに動作し、共有エージェント状態（Shared Agent State）を介して間接的に通信する。CCへの情報投入もキューを通じて非同期に行う。

**原則3: 情報の明示的な優先度付け**
ワークスペースへのアクセス競争を、LLMプロンプト内での情報の順序付けと要約で実現する。

### 1.4 LIDAフレームワークからの知見

LIDA（Learning Intelligent Decision Agent）は、Stan Franklinらがメンフィス大学で開発したGWTの最も代表的なオープンソース実装（Java）である。

**LIDAの認知サイクル（約300ms/サイクルに対応）**：

```
知覚 → 知覚連想記憶 → ワークスペース → 注意コードレット →
グローバルワークスペース → ブロードキャスト → 手続き記憶 →
行動選択 → 運動記憶 → 行動実行
```

**PIANOへの適用可能な知見：**

| LIDAの設計 | PIANOへの適用 |
|---|---|
| 注意コードレット（Attention Codelets） | 情報ボトルネックでの重要度評価関数 |
| ワークスペースの連合（Coalitions） | 関連情報のグループ化・統合 |
| `BroadcastListener`インターフェース | 出力モジュールのCC購読機構 |
| 認知サイクルの時間管理 | CCの「中速」実行タイミング制御 |
| 構造構築コードレット | 入力情報のパターン認識・構造化 |

---

## 2. 情報ボトルネック設計

### 2.1 設計目標

情報ボトルネックの目的は、全モジュールからの情報を統合・圧縮し、CCが「意思決定に必要な情報」のみに基づいて判断することである。これにより：

- LLMのコンテキストウィンドウを効率的に使用
- 無関係な情報によるノイズを排除
- 設計者が情報の流れを明示的に制御可能

### 2.2 入力情報の分類と統合

CCが受け取る情報源と、その統合方法：

```
┌─────────────────────────────────────────────────┐
│              情報ボトルネック                      │
│                                                   │
│  入力                          出力               │
│  ┌──────────────┐                                 │
│  │ 知覚情報      │──┐                             │
│  │ (環境状態)    │  │                             │
│  ├──────────────┤  │    ┌──────────────┐         │
│  │ 記憶情報      │──┤    │ 圧縮された   │         │
│  │ (WM/STM/LTM) │  ├──→│ 状況モデル   │──→ CC   │
│  ├──────────────┤  │    │ (プロンプト) │         │
│  │ 社会情報      │──┤    └──────────────┘         │
│  │ (他者の状態)  │  │                             │
│  ├──────────────┤  │                             │
│  │ 目標情報      │──┤                             │
│  │ (現在の計画)  │  │                             │
│  ├──────────────┤  │                             │
│  │ 行動結果      │──┘                             │
│  │ (AA出力)     │                                 │
│  └──────────────┘                                 │
└─────────────────────────────────────────────────┘
```

### 2.3 情報選択アルゴリズム — GWT点火プロセスの実装

GWTの3フェーズ（競争→点火→ブロードキャスト）のうち、**競争フェーズ**と**点火フェーズ**を情報選択アルゴリズムとして実装する。従来の静的フィルタリング（全情報をプロンプトに注入）ではなく、GWTの「意識へのアクセス競争」に対応する動的メカニズムを導入する。

#### 2.3.1 競争フェーズ（Competition）

各モジュールからの情報アイテムに対して、3軸で「活性化スコア（activation score）」を計算する。これはLIDAの「注意コードレット」に相当する：

| 評価軸 | 説明 | 重み（推奨初期値） |
|---|---|---|
| **緊急性（Urgency）** | 時間的制約のある情報を優先。敵の攻撃、会話応答要求など | 0.4 |
| **関連性（Relevance）** | 現在の目標・状況に関連する情報を優先 | 0.35 |
| **新規性（Novelty）** | 既知でない新しい情報を優先。繰り返し情報を抑制 | 0.25 |

```python
def compute_activation(info_item: InfoItem, current_goals: list[Goal],
                       recent_broadcasts: list[Broadcast]) -> float:
    """情報アイテムの活性化スコアを計算する（GWT競争フェーズ）"""
    urgency = evaluate_urgency(info_item)         # 0.0-1.0
    relevance = evaluate_relevance(info_item, current_goals)  # 0.0-1.0
    novelty = evaluate_novelty(info_item, recent_broadcasts)  # 0.0-1.0

    return (WEIGHT_URGENCY * urgency +
            WEIGHT_RELEVANCE * relevance +
            WEIGHT_NOVELTY * novelty)
```

#### 2.3.2 点火フェーズ（Ignition）

GWTの点火プロセスでは、**閾値を超えた情報のみ**がグローバルワークスペースを「占有」する。これは、全情報をCCに流し込む静的アプローチとは本質的に異なる：

- **点火閾値（ignition threshold）**: 活性化スコアがこの閾値を超えた情報のみがCCのワークスペースに進入する
- **ワークスペース容量制約**: 同時に占有できる情報スロット数に上限を設ける（LLMコンテキストの有効活用）
- **動的閾値調整**: 状況に応じて閾値を動的に変動させる

```python
class IgnitionGate:
    """GWT点火ゲート — 閾値を超えた情報のみがワークスペースに進入する"""

    # 点火閾値の初期値
    BASE_THRESHOLD = 0.3

    # ワークスペースの最大スロット数（同時に保持できる情報グループ数）
    MAX_WORKSPACE_SLOTS = 7  # ミラーの法則: 7±2

    # 閾値の動的調整範囲
    MIN_THRESHOLD = 0.15  # 情報不足時に下がる
    MAX_THRESHOLD = 0.6   # 情報過多時に上がる

    def __init__(self):
        self._threshold = self.BASE_THRESHOLD
        self._previous_occupancy: float = 0.0  # 前回の占有率

    def apply_ignition(
        self,
        candidates: list[tuple[InfoItem, float]],  # (情報, 活性化スコア)
    ) -> list[InfoItem]:
        """
        点火プロセス: 閾値を超えた情報を選択し、
        ワークスペース容量内に収める。

        GWTの「勝者が意識を占有する」メカニズムに対応。
        """
        # Step 1: 閾値による足切り（点火判定）
        ignited = [
            (item, score) for item, score in candidates
            if score >= self._threshold
        ]

        # Step 2: 活性化スコアの降順でソート（競争の勝者決定）
        ignited.sort(key=lambda x: x[1], reverse=True)

        # Step 3: ワークスペース容量でクリップ
        selected = [item for item, _ in ignited[:self.MAX_WORKSPACE_SLOTS]]

        # Step 4: 次回サイクルに向けた閾値の動的調整
        self._adapt_threshold(len(candidates), len(ignited))

        return selected

    def _adapt_threshold(self, total_candidates: int, ignited_count: int):
        """
        閾値を動的に調整する。

        - 点火された情報が多すぎる → 閾値を上げる（競争を激化）
        - 点火された情報が少なすぎる → 閾値を下げる（情報不足を防止）

        これはGWTにおける「注意の調節」に対応する。
        神経科学では、覚醒度やタスク負荷に応じて
        意識へのアクセス閾値が変動することが知られている。
        """
        if total_candidates == 0:
            return

        occupancy = ignited_count / self.MAX_WORKSPACE_SLOTS

        if occupancy > 1.0:
            # 情報過多: 閾値を上げて競争を激化
            self._threshold = min(
                self.MAX_THRESHOLD,
                self._threshold + 0.05
            )
        elif occupancy < 0.3:
            # 情報不足: 閾値を下げて情報を通す
            self._threshold = max(
                self.MIN_THRESHOLD,
                self._threshold - 0.03
            )
        # else: 適正範囲なら閾値を維持

        self._previous_occupancy = occupancy
```

#### 2.3.3 GWT点火プロセスの全体フロー

```
各モジュール出力
     │
     ▼
┌─────────────────────────────┐
│ 競争フェーズ（Competition）   │
│ 各情報アイテムに活性化スコア   │
│ を計算（緊急性/関連性/新規性） │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 点火フェーズ（Ignition）      │
│ 1. 閾値による足切り           │
│ 2. スコア降順で勝者決定       │
│ 3. ワークスペース容量でクリップ │
│ 4. 次回の閾値を動的調整       │
└─────────────┬───────────────┘
              │ 選択された情報のみ
              ▼
┌─────────────────────────────┐
│ CCプロンプト構築              │
│ 点火された情報のみで          │
│ BottleneckInputを構成        │
└─────────────────────────────┘
```

**GWTとの対応関係:**

| GWT概念 | 本設計での実装 | 理論的根拠 |
|---|---|---|
| 注意コードレット | `compute_activation`の3軸評価 | LIDAの注意コードレットがワークスペースへの投入を制御 |
| 活性化の競争 | 活性化スコアの降順ソート | 複数モジュールの出力が限られた「意識の帯域幅」を巡って競争 |
| 点火閾値 | `IgnitionGate.BASE_THRESHOLD` | 閾値を超えた情報のみが「意識的」になる（Dehaene et al., 2003の全か無か仮説） |
| ワークスペース容量 | `MAX_WORKSPACE_SLOTS = 7` | Millerの法則（7±2）に対応する情報チャンク制限 |
| 閾値の動的変動 | `_adapt_threshold` | 覚醒度・タスク負荷に応じた意識閾値の変動（Baars, 2005） |

### 2.4 圧縮アルゴリズム

情報圧縮は以下の階層で実施する：

**レベル1: フィルタリング（情報の取捨選択）**
- 優先度スコアが閾値未満の情報を除外
- 重複する情報を統合（最新のみ保持）

**レベル2: 要約（LLMによる圧縮）**
- 各カテゴリの情報を短い要約に圧縮
- 「何が・いつ・なぜ重要か」の3要素に集約

**レベル3: テンプレート化（構造化プロンプト生成）**
- 圧縮された情報を定型フォーマットに配置
- CCのLLMプロンプトとして最適化

### 2.5 圧縮率と情報損失のトレードオフ

| 圧縮レベル | トークン数目安 | 情報保持率 | 用途 |
|---|---|---|---|
| 低圧縮 | 2000-3000 | 90%+ | 低負荷時、重要な意思決定時 |
| 中圧縮（推奨） | 800-1500 | 70-80% | 通常運用 |
| 高圧縮 | 300-600 | 50-60% | 高負荷時、多数エージェント同時実行時 |

圧縮レベルは動的に調整可能とし、システム負荷やエージェント数に応じて自動選択する。

### 2.6 情報提示フォーマット（CCへの入力プロンプト）

```
## 現在の状況サマリ
- 場所: [場所名] / 周囲: [環境概要]
- 体力: [HP/MaxHP] / インベントリ: [主要アイテム要約]

## 緊急事項
- [緊急度の高い情報があればここに記載]

## 現在の目標
- 主目標: [目標名] / 進捗: [進捗概要]
- 副目標: [副目標リスト]

## 社会的状況
- 近くのエージェント: [名前リスト]
- 最近の会話: [要約]
- 未解決の社会的コミットメント: [約束した内容]

## 最近の行動結果
- [直近の行動とその結果]
- [期待と実際の乖離があれば記載]

## 記憶からの関連情報
- [現在の状況に関連する過去の経験]
```

---

## 3. ブロードキャスト機構

### 3.1 ブロードキャスト設計の概要

CCの意思決定結果を全出力モジュールに一斉配信する機構。PIANOの一貫性（Coherence）を実現する中核要素。

```
           ┌──────────────────┐
           │  認知コントローラ  │
           │   (CC決定出力)    │
           └────────┬─────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Talking  │ │ Skill   │ │ その他   │
   │ (発話)   │ │ Exec    │ │ 出力    │
   │          │ │ (実行)  │ │ モジュール│
   └─────────┘ └─────────┘ └─────────┘
   条件付け強  条件付け中   条件付け弱
```

### 3.2 ブロードキャストの内容構造

CCのブロードキャスト出力は以下の構造を持つ：

```python
@dataclass
class CognitiveControllerOutput:
    """CCのブロードキャスト出力"""

    # 高レベル意思決定
    decision: str              # 「何をすべきか」の要約
    reasoning: str             # 意思決定の理由

    # 発話への条件付け（強い制約）
    speech_directive: SpeechDirective  # 発話内容・トーン・意図
    speech_constraints: list[str]     # 発話で守るべき制約

    # 行動への条件付け（中程度の制約）
    action_directive: ActionDirective  # 推奨アクション
    action_priority: float            # 行動の優先度 0.0-1.0

    # コンテキスト情報（弱い制約）
    context_summary: str       # 状況要約（全モジュール共有）
    active_commitments: list[str]  # 社会的コミットメント

    # メタデータ
    timestamp: float
    cycle_id: int
    confidence: float          # CC自身の確信度 0.0-1.0
```

### 3.3 発話モジュールへの条件付け強度調整

論文で強調される「言行一致」は、発話モジュールへの条件付けを他モジュールより強くすることで実現する。

**条件付け強度の階層：**

| 出力モジュール | 条件付け強度 | 理由 |
|---|---|---|
| Talking（発話） | **強** (0.9) | 言語的コミットメントが行動の一貫性の基盤 |
| Skill Execution（スキル実行） | **中** (0.6) | 行動は発話と整合する必要があるが、環境制約もある |
| その他出力モジュール | **弱** (0.3) | CCの方向性を参考にしつつ独自判断も許容 |

**実装方法：**

発話モジュールのプロンプトにCCの決定を**必須コンテキスト**として注入し、スキル実行モジュールには**推奨コンテキスト**として注入する。

```python
def build_talking_prompt(cc_output: CognitiveControllerOutput,
                         conversation_context: str) -> str:
    """発話モジュール用プロンプトを構築（CCの条件付け強）"""
    return f"""
あなたは以下の意思決定に**必ず従って**発話してください。

## CC意思決定（必須遵守）
決定: {cc_output.decision}
理由: {cc_output.reasoning}
発話指示: {cc_output.speech_directive}
制約: {cc_output.speech_constraints}

## 守るべきコミットメント
{cc_output.active_commitments}

## 会話コンテキスト
{conversation_context}

上記を踏まえ、適切な発話を生成してください。
"""

def build_skill_exec_prompt(cc_output: CognitiveControllerOutput,
                            environment_state: str) -> str:
    """スキル実行モジュール用プロンプトを構築（CCの条件付け中）"""
    return f"""
あなたは以下の意思決定を**参考にして**行動を選択してください。
環境の制約がある場合はそちらを優先してよいですが、
可能な限りCCの方針に沿った行動を取ってください。

## CC意思決定（参考）
決定: {cc_output.decision}
推奨行動: {cc_output.action_directive}
優先度: {cc_output.action_priority}

## 環境状態
{environment_state}

上記を踏まえ、適切な行動を選択してください。
"""
```

### 3.4 ブロードキャストの非同期処理

ブロードキャストは非同期で配信し、出力モジュールは最新のCC決定をキャッシュとして保持する：

```python
class BroadcastManager:
    """ブロードキャスト管理"""

    def __init__(self):
        self._listeners: dict[str, BroadcastListener] = {}
        self._latest_broadcast: CognitiveControllerOutput | None = None
        self._broadcast_queue: asyncio.Queue = asyncio.Queue()

    def register_listener(self, module_id: str, listener: BroadcastListener):
        """出力モジュールをリスナーとして登録"""
        self._listeners[module_id] = listener

    async def broadcast(self, cc_output: CognitiveControllerOutput):
        """全リスナーに非同期でブロードキャスト"""
        self._latest_broadcast = cc_output
        tasks = [
            listener.on_broadcast(cc_output)
            for listener in self._listeners.values()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_latest(self) -> CognitiveControllerOutput | None:
        """最新のブロードキャストを取得（キャッシュ）"""
        return self._latest_broadcast
```

**非同期処理の利点：**
- 高速モジュール（Talking, Skill Exec）はCCの完了を待たずに直前のブロードキャストを参照可能
- CCの処理中でもエージェントは環境に反応し続ける
- モジュール間のデッドロックを防止

---

## 4. 実装設計

### 4.1 CCの入出力インターフェース

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

class CognitiveController(ABC):
    """認知コントローラの抽象基底クラス"""

    @abstractmethod
    async def process(self, bottleneck_input: BottleneckInput) -> CognitiveControllerOutput:
        """
        情報ボトルネックを通過した入力を処理し、
        ブロードキャスト用の出力を生成する。
        """
        ...

    @abstractmethod
    def get_execution_interval(self) -> float:
        """CCの実行間隔（秒）を返す。「中速」に対応"""
        ...


@dataclass
class BottleneckInput:
    """情報ボトルネックを通過したCCへの入力"""

    # 環境情報（圧縮済み）
    environment_summary: str

    # 現在の目標と計画（圧縮済み）
    current_goals: list[GoalSummary]
    active_plan: str | None

    # 社会的状況（圧縮済み）
    social_context: SocialContext

    # 行動認識の結果
    action_awareness: ActionAwarenessResult | None

    # 記憶からの関連情報
    relevant_memories: list[MemoryItem]

    # 前回のCC出力（自己参照）
    previous_output: CognitiveControllerOutput | None

    # 緊急イベント
    urgent_events: list[UrgentEvent] = field(default_factory=list)

    # メタデータ
    timestamp: float = 0.0
    agent_id: str = ""


@dataclass
class GoalSummary:
    goal_text: str
    priority: float
    progress: float  # 0.0-1.0
    source: str       # "self", "social", "system"


@dataclass
class SocialContext:
    nearby_agents: list[str]
    active_conversations: list[ConversationSummary]
    commitments: list[str]      # 他者への約束
    social_goals: list[str]     # 社会的目標


@dataclass
class ConversationSummary:
    partner: str
    topic: str
    last_exchange: str
    requires_response: bool


@dataclass
class UrgentEvent:
    description: str
    urgency_level: float  # 0.0-1.0
    source: str
```

### 4.2 CC実装クラス

```python
class LLMCognitiveController(CognitiveController):
    """LLMベースの認知コントローラ実装"""

    def __init__(self, llm_client, agent_profile: AgentProfile,
                 execution_interval: float = 5.0):
        self._llm = llm_client
        self._profile = agent_profile
        self._interval = execution_interval
        self._cycle_count = 0
        self._history: list[CognitiveControllerOutput] = []

    async def process(self, bottleneck_input: BottleneckInput) -> CognitiveControllerOutput:
        """CCの中核処理: 情報統合 → 意思決定 → ブロードキャスト出力生成"""
        self._cycle_count += 1

        # 1. プロンプト構築
        prompt = self._build_prompt(bottleneck_input)

        # 2. LLM呼び出し
        response = await self._llm.generate(
            system_prompt=self._build_system_prompt(),
            user_prompt=prompt,
            response_format=CognitiveControllerOutput,  # 構造化出力
            temperature=0.7,
        )

        # 3. 出力の検証と補正（詳細は4.2.1節を参照）
        output = self._validate_and_correct(response, bottleneck_input)

        # 4. 履歴に追加（傾向検出付き、詳細は4.2.2節を参照）
        self._history_manager.record(output)

        return output

    def get_execution_interval(self) -> float:
        return self._interval

    def _build_system_prompt(self) -> str:
        return f"""あなたは「{self._profile.name}」というエージェントの認知コントローラです。

あなたの役割:
1. 提示された状況情報を統合的に理解する
2. エージェントが次に何をすべきか高レベルの意思決定を行う
3. 特に「言ったことと行動」の一貫性を保つ
4. 社会的コミットメント（約束）を守る方向に導く

出力形式:
- decision: 次にすべきことの簡潔な要約
- reasoning: なぜその決定をしたかの理由
- speech_directive: 発話すべき内容とトーン
- speech_constraints: 発話で守るべき制約
- action_directive: 推奨する行動
- action_priority: 行動の優先度(0.0-1.0)
- context_summary: 全モジュールに共有する状況要約
- active_commitments: 現在有効な社会的コミットメント
- confidence: この決定への確信度(0.0-1.0)

エージェントのプロフィール:
名前: {self._profile.name}
性格: {self._profile.personality}
"""

    def _build_prompt(self, inp: BottleneckInput) -> str:
        sections = []

        # 緊急事項（最優先）
        if inp.urgent_events:
            urgent_text = "\n".join(
                f"- [{e.urgency_level:.1f}] {e.description}"
                for e in sorted(inp.urgent_events,
                               key=lambda x: x.urgency_level, reverse=True)
            )
            sections.append(f"## 緊急事項\n{urgent_text}")

        # 環境情報
        sections.append(f"## 現在の環境\n{inp.environment_summary}")

        # 目標と計画
        goals_text = "\n".join(
            f"- [{g.priority:.1f}] {g.goal_text} (進捗: {g.progress:.0%})"
            for g in sorted(inp.current_goals,
                           key=lambda x: x.priority, reverse=True)
        )
        sections.append(f"## 現在の目標\n{goals_text}")
        if inp.active_plan:
            sections.append(f"## アクティブな計画\n{inp.active_plan}")

        # 社会的状況
        social = inp.social_context
        if social.active_conversations:
            conv_text = "\n".join(
                f"- {c.partner}: {c.topic} (応答必要: {c.requires_response})"
                for c in social.active_conversations
            )
            sections.append(f"## 会話状況\n{conv_text}")
        if social.commitments:
            sections.append(
                f"## 社会的コミットメント\n" +
                "\n".join(f"- {c}" for c in social.commitments)
            )

        # 行動認識
        if inp.action_awareness:
            sections.append(
                f"## 行動認識結果\n{inp.action_awareness.summary}"
            )

        # 関連記憶
        if inp.relevant_memories:
            mem_text = "\n".join(
                f"- {m.content}" for m in inp.relevant_memories[:5]
            )
            sections.append(f"## 関連する記憶\n{mem_text}")

        # 前回の決定（自己参照）
        if inp.previous_output:
            sections.append(
                f"## 前回の決定\n"
                f"決定: {inp.previous_output.decision}\n"
                f"確信度: {inp.previous_output.confidence:.1f}"
            )

        return "\n\n".join(sections)
```

### 4.2.1 出力検証・補正ロジック（`_validate_and_correct`）

CCのLLM出力は不正なJSON、欠損フィールド、意味的に矛盾する指示を含む可能性がある。特に`speech_constraints`と`action_directive`の矛盾検出は一貫性保証の核心部分であるため、3段階の検証と補正を行う：

```python
class OutputValidator:
    """CC出力の検証と補正"""

    # --- 意味的矛盾ルール ---
    # (action, speech_intent) の禁止組み合わせ
    CONTRADICTION_RULES: list[tuple[str, str, str]] = [
        # (action_pattern, speech_intent_pattern, 矛盾の説明)
        ("flee|retreat|escape", "invite_to_cooperate|offer_help",
         "逃走行動と協力の申し出は矛盾"),
        ("attack|fight|combat", "express_friendship|compliment",
         "攻撃行動と友好的発話は矛盾"),
        ("ignore|leave|abandon", "promise_to_help|commit_to_task",
         "離脱行動と支援の約束は矛盾"),
        ("hide|avoid", "seek_attention|initiate_conversation",
         "隠れる行動と注意を引く発話は矛盾"),
    ]

    def validate_and_correct(
        self,
        response: CognitiveControllerOutput,
        bottleneck_input: BottleneckInput,
        previous_output: CognitiveControllerOutput | None,
    ) -> CognitiveControllerOutput:
        """
        3段階の検証と補正を行う:
        1. 構造検証（JSON Schema）
        2. 意味的整合性チェック
        3. フォールバック補正
        """
        # Stage 1: 構造検証
        structural_issues = self._validate_structure(response)
        if structural_issues:
            response = self._fix_structural_issues(response, structural_issues)

        # Stage 2: 意味的整合性チェック
        contradictions = self._detect_contradictions(response)
        if contradictions:
            response = self._resolve_contradictions(
                response, contradictions, bottleneck_input
            )

        # Stage 3: コミットメント整合性チェック
        response = self._enforce_commitments(
            response, bottleneck_input.social_context
        )

        return response

    def _validate_structure(
        self, output: CognitiveControllerOutput
    ) -> list[str]:
        """
        Stage 1: 構造検証

        - 必須フィールドの存在確認
        - 値の範囲チェック（confidence: 0.0-1.0, action_priority: 0.0-1.0）
        - 文字列フィールドの空チェック
        """
        issues = []

        # 必須フィールドの空チェック
        if not output.decision or not output.decision.strip():
            issues.append("decision_empty")
        if not output.reasoning or not output.reasoning.strip():
            issues.append("reasoning_empty")

        # 数値範囲の検証
        if not (0.0 <= output.confidence <= 1.0):
            issues.append("confidence_out_of_range")
        if not (0.0 <= output.action_priority <= 1.0):
            issues.append("action_priority_out_of_range")

        return issues

    def _fix_structural_issues(
        self,
        output: CognitiveControllerOutput,
        issues: list[str],
    ) -> CognitiveControllerOutput:
        """構造的な問題をデフォルト値で補正"""
        if "decision_empty" in issues:
            output.decision = "continue_current_activity"
        if "reasoning_empty" in issues:
            output.reasoning = "LLM output was incomplete; defaulting."
        if "confidence_out_of_range" in issues:
            output.confidence = max(0.0, min(1.0, output.confidence))
        if "action_priority_out_of_range" in issues:
            output.action_priority = max(0.0, min(1.0, output.action_priority))
        return output

    def _detect_contradictions(
        self, output: CognitiveControllerOutput
    ) -> list[str]:
        """
        Stage 2: 意味的整合性チェック

        行動指示と発話意図の矛盾を検出する。
        例: action=flee + speech_intent=invite_to_cooperate は矛盾。
        """
        contradictions = []
        action_text = str(output.action_directive).lower()
        speech_text = str(output.speech_directive).lower()

        for action_pattern, speech_pattern, description in self.CONTRADICTION_RULES:
            action_match = re.search(action_pattern, action_text)
            speech_match = re.search(speech_pattern, speech_text)
            if action_match and speech_match:
                contradictions.append(description)

        return contradictions

    def _resolve_contradictions(
        self,
        output: CognitiveControllerOutput,
        contradictions: list[str],
        bottleneck_input: BottleneckInput,
    ) -> CognitiveControllerOutput:
        """
        矛盾の解決: 緊急性が高い方を優先する。

        - 緊急イベントがある場合 → action_directiveを優先、speech_directiveを調整
        - 会話が進行中の場合 → speech_directiveを優先、action_directiveを調整
        - それ以外 → action_directiveを優先（行動の安全性を重視）
        """
        has_urgent = len(bottleneck_input.urgent_events) > 0
        has_conversation = any(
            c.requires_response
            for c in bottleneck_input.social_context.active_conversations
        )

        if has_urgent or not has_conversation:
            # 行動を優先: 発話を行動と整合させる
            output.speech_constraints.append(
                f"行動との矛盾を検出（{contradictions[0]}）: "
                f"発話を行動に合わせて調整すること"
            )
        else:
            # 発話を優先: 行動を発話と整合させる
            output.action_priority *= 0.5  # 行動の優先度を下げる

        output.confidence *= 0.7  # 矛盾検出時は確信度を下げる
        return output

    def _enforce_commitments(
        self,
        output: CognitiveControllerOutput,
        social_context: SocialContext,
    ) -> CognitiveControllerOutput:
        """
        Stage 3: 社会的コミットメントとの整合性確認

        未解決の約束がある場合、active_commitmentsに反映されていることを確認する。
        """
        if social_context.commitments:
            existing = set(output.active_commitments)
            for commitment in social_context.commitments:
                if commitment not in existing:
                    output.active_commitments.append(commitment)
        return output
```

### 4.2.2 CC決定履歴の管理と傾向検出

直近10件のリスト保持だけでは、CC決定の長期的な傾向（同じ行動の繰り返し、目標の停滞等）を検出できない。以下のメカニズムで履歴を管理し、パターン検出を行う：

```python
@dataclass
class DecisionTrend:
    """検出された決定傾向"""
    pattern_type: str          # "repetition" | "goal_stagnation" | "oscillation"
    description: str
    severity: float            # 0.0-1.0
    suggested_intervention: str

class CCHistoryManager:
    """
    CC決定履歴の管理と傾向検出。

    短期履歴（直近10件）: 即座の文脈参照用
    集約統計（長期）: 傾向検出用（全決定のカウントを保持）
    """

    SHORT_HISTORY_SIZE = 10

    def __init__(self):
        self._short_history: list[CognitiveControllerOutput] = []
        self._action_counts: dict[str, int] = {}      # action -> 出現回数
        self._decision_counts: dict[str, int] = {}     # decision -> 出現回数
        self._total_decisions: int = 0
        self._last_goal_change_cycle: int = 0
        self._current_primary_goal: str = ""

    def record(self, output: CognitiveControllerOutput):
        """決定を記録し、集約統計を更新"""
        self._short_history.append(output)
        if len(self._short_history) > self.SHORT_HISTORY_SIZE:
            self._short_history.pop(0)

        action_key = str(output.action_directive).lower().strip()
        self._action_counts[action_key] = (
            self._action_counts.get(action_key, 0) + 1
        )
        decision_key = output.decision.lower().strip()
        self._decision_counts[decision_key] = (
            self._decision_counts.get(decision_key, 0) + 1
        )
        self._total_decisions += 1

    def detect_trends(self) -> list[DecisionTrend]:
        """
        現在の履歴から異常な傾向を検出する。
        検出結果はCCプロンプトに注入し、LLMに自己修正を促す。
        """
        trends = []

        # 1. 行動の繰り返し検出
        repetition = self._detect_repetition()
        if repetition:
            trends.append(repetition)

        # 2. 目標停滞の検出
        stagnation = self._detect_goal_stagnation()
        if stagnation:
            trends.append(stagnation)

        # 3. 行動の振動検出（A→B→A→B...）
        oscillation = self._detect_oscillation()
        if oscillation:
            trends.append(oscillation)

        return trends

    def _detect_repetition(self) -> DecisionTrend | None:
        """直近の短期履歴で同じ行動が連続しすぎていないかチェック"""
        if len(self._short_history) < 4:
            return None

        recent_actions = [
            str(o.action_directive).lower().strip()
            for o in self._short_history[-5:]
        ]

        # 直近5件中4件以上が同じ行動なら異常
        from collections import Counter
        counter = Counter(recent_actions)
        most_common, count = counter.most_common(1)[0]
        if count >= 4:
            return DecisionTrend(
                pattern_type="repetition",
                description=f"直近5回中{count}回が同じ行動: '{most_common}'",
                severity=count / 5.0,
                suggested_intervention=(
                    "同じ行動を繰り返しています。"
                    "状況が変化していないか再評価し、"
                    "別のアプローチを検討してください。"
                ),
            )
        return None

    def _detect_goal_stagnation(self) -> DecisionTrend | None:
        """
        目標が長期間変化していない場合を検出。
        CCサイクル20回以上同じ主目標が続いている場合に警告。
        """
        if len(self._short_history) < 2:
            return None

        current_goal = self._short_history[-1].decision
        cycles_since_change = (
            self._total_decisions - self._last_goal_change_cycle
        )

        if current_goal != self._current_primary_goal:
            self._current_primary_goal = current_goal
            self._last_goal_change_cycle = self._total_decisions
            return None

        if cycles_since_change >= 20:
            return DecisionTrend(
                pattern_type="goal_stagnation",
                description=(
                    f"目標が{cycles_since_change}サイクル間変化なし: "
                    f"'{self._current_primary_goal}'"
                ),
                severity=min(cycles_since_change / 40.0, 1.0),
                suggested_intervention=(
                    "長期間同じ目標に取り組んでいます。"
                    "進捗がない場合は、サブゴールへの分解や"
                    "別の目標への切り替えを検討してください。"
                ),
            )
        return None

    def _detect_oscillation(self) -> DecisionTrend | None:
        """行動がA→B→A→B...と振動していないかチェック"""
        if len(self._short_history) < 6:
            return None

        recent = [
            str(o.action_directive).lower().strip()
            for o in self._short_history[-6:]
        ]

        # A-B-A-B パターンの検出
        if (recent[0] == recent[2] == recent[4] and
            recent[1] == recent[3] == recent[5] and
            recent[0] != recent[1]):
            return DecisionTrend(
                pattern_type="oscillation",
                description=(
                    f"行動が振動: '{recent[0]}' ↔ '{recent[1]}'"
                ),
                severity=0.7,
                suggested_intervention=(
                    "2つの行動の間で迷っています。"
                    "どちらかに決定し、一貫して実行してください。"
                ),
            )
        return None

    @property
    def recent_history(self) -> list[CognitiveControllerOutput]:
        return list(self._short_history)
```

**傾向検出結果のCCプロンプトへの注入:**

検出された傾向は、`_build_prompt`で「自己認識」セクションとしてCCプロンプトに追加する。これにより、LLMが自己修正的な意思決定を行えるようになる：

```python
# _build_prompt内での傾向注入（追加セクション）
trends = self._history_manager.detect_trends()
if trends:
    trend_text = "\n".join(
        f"- [{t.severity:.1f}] {t.description}\n  → {t.suggested_intervention}"
        for t in trends
    )
    sections.append(f"## 自己認識（行動傾向の検出）\n{trend_text}")
```

### 4.3 実行頻度・タイミングの決定ロジック

CCは「中速」で実行される。具体的な実行タイミングは以下のロジックで決定する：

```python
class CCScheduler:
    """CC実行スケジューラ"""

    # 基本間隔（秒）
    BASE_INTERVAL = 5.0

    # 最小・最大間隔
    MIN_INTERVAL = 1.0   # 緊急時の最速
    MAX_INTERVAL = 15.0  # 安静時の最遅

    # コスト制約: 1分あたりの最大CC実行回数
    # LLM呼び出しコストの急増を防止する。
    # MIN_INTERVAL=1秒だけでは保護が不十分（LLMレイテンシ1-5秒を考慮すると、
    # 完了直後に次が起動する連鎖が発生する）。
    # 最大12回/分 = 平均5秒間隔を下回らないことを保証。
    MAX_EXECUTIONS_PER_MINUTE = 12

    def __init__(self):
        self._execution_timestamps: list[float] = []

    def compute_next_interval(self,
                              urgent_events: list[UrgentEvent],
                              active_conversations: list[ConversationSummary],
                              action_awareness_alert: bool,
                              system_load: SystemLoad) -> float:
        """次のCC実行までの間隔を動的に計算"""
        interval = self.BASE_INTERVAL

        # --- 短縮因子（加算方式に変更: 乗算による過剰短縮を防止） ---
        # 旧方式: interval *= 0.2 * 0.5 * 0.6 → 0.06倍（過剰短縮）
        # 新方式: 各因子を加算して最大短縮率をクリップ
        reduction = 0.0

        # 緊急イベントがあれば短縮
        if urgent_events:
            max_urgency = max(e.urgency_level for e in urgent_events)
            reduction += max_urgency * 0.4  # 最大0.4

        # アクティブな会話があれば短縮
        if any(c.requires_response for c in active_conversations):
            reduction += 0.25  # 固定

        # 行動認識がアラートを出していれば短縮
        if action_awareness_alert:
            reduction += 0.15  # 固定

        # 合計短縮率を最大70%にクリップ（最低でもBASEの30%は残る）
        reduction = min(reduction, 0.7)
        interval *= (1.0 - reduction)

        # --- 延長因子 ---
        # システム負荷に応じて延長
        interval *= (1.0 + system_load.normalized_score * 0.5)

        # --- コスト制約によるレート制限 ---
        interval = self._apply_rate_limit(interval)

        return max(self.MIN_INTERVAL, min(self.MAX_INTERVAL, interval))

    def _apply_rate_limit(self, desired_interval: float) -> float:
        """
        コスト制約: 直近1分間のCC実行回数が上限に達している場合、
        間隔を強制的に延長する。

        これはLLMコストの急激な増大を防止するガードレール。
        例: 緊急イベント+会話+AAアラートが同時に発生しても、
        1分あたりMAX_EXECUTIONS_PER_MINUTE回を超えない。
        """
        now = time.time()
        # 1分以上前のタイムスタンプを除去
        self._execution_timestamps = [
            t for t in self._execution_timestamps
            if now - t < 60.0
        ]

        if len(self._execution_timestamps) >= self.MAX_EXECUTIONS_PER_MINUTE:
            # レート上限到達: 最も古い実行から60秒後まで待機
            oldest = self._execution_timestamps[0]
            forced_wait = (oldest + 60.0) - now
            return max(desired_interval, forced_wait)

        return desired_interval

    def record_execution(self):
        """CC実行完了時にタイムスタンプを記録（process()から呼び出す）"""
        self._execution_timestamps.append(time.time())
```

#### system_loadの定義

`system_load`は複数の計測指標を正規化した複合スコアとして定義する：

```python
@dataclass
class SystemLoad:
    """
    システム負荷の複合指標。
    CCスケジューラが実行間隔を延長すべきかの判断に使用する。
    """
    # LLM APIの応答キュー長（未処理リクエスト数）
    llm_queue_depth: int = 0

    # 同時アクティブなエージェント数
    active_agent_count: int = 0

    # LLM APIの直近レイテンシ（秒、移動平均）
    llm_latency_avg: float = 0.0

    @property
    def normalized_score(self) -> float:
        """
        正規化された負荷スコア（0.0-1.0）。

        計算方法:
        - LLMキュー深度: 0-50の範囲で正規化（50以上は飽和）
        - アクティブエージェント数: 想定最大数で正規化
        - LLMレイテンシ: 正常(1秒)〜高負荷(10秒)で正規化

        各指標を重み付き平均で統合する。
        """
        queue_score = min(self.llm_queue_depth / 50.0, 1.0)
        agent_score = min(self.active_agent_count / 500.0, 1.0)
        latency_score = min(max(self.llm_latency_avg - 1.0, 0.0) / 9.0, 1.0)

        return (
            0.4 * queue_score +    # LLMキューが最も直接的な指標
            0.3 * agent_score +    # エージェント数は全体の負荷
            0.3 * latency_score    # レイテンシは実際の応答性能
        )
```

**設計判断の根拠:**
- **乗算方式から加算方式への変更**: 旧方式では緊急(0.2倍)×会話(0.5倍)×AA(0.6倍)=0.06倍となり、`5.0 * 0.06 = 0.3秒`に短縮されMIN_INTERVALでクリップされるが、CC完了後にすぐ次のCCが起動する連鎖を引き起こす。加算方式では最大70%短縮（`5.0 * 0.3 = 1.5秒`）に抑制される
- **レート制限の追加**: MIN_INTERVALだけでは「LLMレイテンシ＋MIN_INTERVAL」の周期で連続起動が可能であり、コスト制約として不十分。1分あたりの実行回数上限を設けることで、LLMコストの上限を保証する
- **system_loadの複合指標化**: 単一の`float`では何を示すか曖昧だったため、具体的な計測指標（LLMキュー深度、アクティブエージェント数、LLMレイテンシ）を明示し、その正規化方法を定義した

**タイミング設計の理由：**

| 速度カテゴリ | モジュール例 | 間隔 | 理由 |
|---|---|---|---|
| 高速 | Action Awareness, Talking | 0.5-2秒 | 環境への即時反応 |
| **中速** | **CC** | **1-15秒** | **情報統合に十分な時間、かつ応答性を確保** |
| 低速 | Goal Generation, Planning | 15-60秒 | 深い推論に時間が必要 |

### 4.4 状態遷移図

CCの内部状態遷移：

```
                    ┌─────────────┐
                    │   IDLE      │
                    │ (待機中)    │
                    └──────┬──────┘
                           │ スケジュール到来
                           ▼
                    ┌─────────────┐
                    │  COLLECTING  │
            ┌──────│ (情報収集中) │
            │      └──────┬──────┘
            │             │ 情報収集完了
   タイムアウト            ▼
            │      ┌─────────────┐
            │      │ COMPRESSING  │
            │      │ (情報圧縮中) │
            │      └──────┬──────┘
            │             │ 圧縮完了
            │             ▼
            │      ┌─────────────┐
            ├──────│  DECIDING    │
            │      │ (意思決定中) │──────────┐
            │      └──────┬──────┘          │
            │             │ LLM応答受信     │ LLMエラー
            │             ▼                 │
            │      ┌─────────────┐          │
            │      │BROADCASTING │          │
            │      │(ブロードキャスト│         │
            │      │       中)   │          │
            │      └──────┬──────┘          │
            │             │ 配信完了         │
            │             ▼                 ▼
            │      ┌─────────────┐  ┌──────────┐
            └─────→│  IDLE       │  │ FALLBACK │
                   │ (待機中)    │  │(前回決定  │
                   └─────────────┘  │  再利用)  │
                                    └──────────┘
```

**フォールバック戦略：**
LLM呼び出しが失敗・タイムアウトした場合、前回のCC出力を再利用する。これにより、一時的な障害でもエージェントの一貫性が維持される。

### 4.5 エージェント状態との統合

```python
class SharedAgentState:
    """共有エージェント状態（全モジュールからアクセス）"""

    # CC関連の状態
    latest_cc_output: CognitiveControllerOutput | None = None
    cc_cycle_count: int = 0
    cc_last_execution: float = 0.0

    # 環境状態
    environment: EnvironmentState
    inventory: Inventory
    health: HealthStatus
    position: Position

    # 記憶状態
    working_memory: WorkingMemory
    short_term_memory: ShortTermMemory
    long_term_memory: LongTermMemory

    # 社会状態
    social_state: SocialState

    # 目標・計画状態
    goal_stack: GoalStack
    active_plan: Plan | None

    # 行動認識状態
    action_awareness_state: ActionAwarenessState

    def read_for_cc(self) -> BottleneckInput:
        """CCが必要とする情報を読み取り、BottleneckInputに変換"""
        return BottleneckInput(
            environment_summary=self._compress_environment(),
            current_goals=self._extract_goal_summaries(),
            active_plan=str(self.active_plan) if self.active_plan else None,
            social_context=self._extract_social_context(),
            action_awareness=self.action_awareness_state.latest_result,
            relevant_memories=self._query_relevant_memories(),
            previous_output=self.latest_cc_output,
            urgent_events=self._collect_urgent_events(),
            timestamp=time.time(),
            agent_id=self.agent_id,
        )
```

---

## 5. 既存認知アーキテクチャとの比較

### 5.1 比較表

| 特性 | PIANO CC | LIDA | ACT-R | Soar | CLARION |
|---|---|---|---|---|---|
| **理論的基盤** | GWT | GWT | ACT理論 | 統一認知理論 | 二重プロセス理論 |
| **意識モデル** | ボトルネック+ブロードキャスト | 意識的ブロードキャスト | 宣言的焦点 | ワーキングメモリ | 明示的/暗黙的処理 |
| **記憶構造** | WM/STM/LTM | 知覚連想/エピソード/手続き | 宣言的/手続き | 意味/エピソード/手続き | 宣言的/手続き/メタ |
| **実行モデル** | 非同期並列 | 認知サイクル(逐次) | 産出規則(逐次) | 産出規則(逐次) | 逐次+並列混合 |
| **学習** | LLM依存(in-context) | 知覚/エピソード/手続き学習 | チャンク形成/強化学習 | チャンキング/強化学習 | 連想学習+規則学習 |
| **並行性** | 明示的マルチスレッド | 単一認知サイクル | 単一サイクル | 単一サイクル | 制限付き並列 |
| **スケーラビリティ** | 1000+エージェント | 単一エージェント | 単一エージェント | 単一エージェント | 単一エージェント |
| **LLM統合** | 中核的に使用 | なし(従来型) | 最近の研究で統合 | 最近の研究で統合 | なし |
| **実装言語** | Python(想定) | Java | Lisp/Python | Java/C++ | Java |
| **オープンソース** | 計画中 | あり | あり | あり | あり |

### 5.2 各アーキテクチャからの再利用可能な設計パターン

**LIDAからの借用:**

| パターン | LIDA実装 | PIANO適用 |
|---|---|---|
| BroadcastListenerインターフェース | Observerパターンによるモジュール通信 | 出力モジュールのCC購読機構 |
| 注意コードレット | 情報の重要度評価 | 情報ボトルネックの優先度計算 |
| 認知サイクル時間管理 | 固定周期のサイクル実行 | CCの動的実行間隔スケジューラ |
| 連合（Coalition）形成 | 関連情報のグループ化 | 入力情報の構造化・カテゴリ化 |

**ACT-Rからの借用:**

| パターン | ACT-R実装 | PIANO適用 |
|---|---|---|
| 活性化拡散 | 宣言的記憶の関連アイテム検索 | 記憶モジュールからの関連情報検索 |
| ユーティリティ学習 | 産出規則の有用性評価 | CC決定の過去の成功率によるフィードバック |
| 目標バッファ | 現在の目標を保持・管理 | 目標スタックの管理 |

**Soarからの借用:**

| パターン | Soar実装 | PIANO適用 |
|---|---|---|
| インパス検出 | 問題解決の行き詰まり検出 | 行動認識モジュールによるハルシネーション検出 |
| サブゴール生成 | 行き詰まり時の自動サブゴール化 | 計画モジュールでの目標分解 |
| エピソード記憶 | 過去の経験の文脈的検索 | STM/LTMからの類似状況検索 |

**CLARIONからの借用:**

| パターン | CLARION実装 | PIANO適用 |
|---|---|---|
| 二重プロセス | 明示的(規則)+暗黙的(NN)処理 | 高速(非LLM)+低速(LLM)モジュール |
| メタ認知サブシステム | 自己の認知プロセスの監視 | 自己省察モジュールとの連携 |
| 動機サブシステム | 内発的動機の生成 | 将来的な内発的動機付け機能への拡張ポイント |

---

## 6. 推奨実装ロードマップ

### Phase 1: 最小構成CC（MVP）

**目標:** ベースライン構成（CC + Memory + Skill Execution）の動作確認

```
実装項目:
├── CognitiveController基底クラス
├── LLMCognitiveController（基本実装）
├── BottleneckInput / CognitiveControllerOutput データクラス
├── 固定間隔スケジューラ（5秒固定）
├── 単純なブロードキャスト（同期配信）
└── 基本的な情報圧縮（テンプレート方式のみ）
```

### Phase 2: 情報ボトルネックの高度化

**目標:** 情報選択・圧縮の品質向上

```
実装項目:
├── 3軸評価（緊急性/関連性/新規性）による情報選択
├── LLMベースの情報要約・圧縮
├── 動的圧縮レベル調整
└── 情報提示フォーマットの最適化
```

### Phase 3: ブロードキャスト機構の完成

**目標:** 全出力モジュールへの非同期配信と条件付け

```
実装項目:
├── BroadcastManagerの非同期実装
├── BroadcastListenerインターフェース
├── 発話モジュールへの強条件付け
├── スキル実行モジュールへの中条件付け
├── 動的実行間隔スケジューラ
└── フォールバック戦略
```

### Phase 4: 高度化と最適化

**目標:** パフォーマンス最適化とスケーラビリティ確保

```
実装項目:
├── CC決定のキャッシュと再利用
├── 複数エージェント同時実行時の負荷分散
├── CC決定品質のメトリクス収集
├── プロンプトの自動最適化
└── 確信度に基づく適応的スケジューリング
```

---

## 付録: 主要参考文献・リソース

### 認知アーキテクチャ

- **LIDA Framework**: [CCRG - University of Memphis](https://ccrg.cs.memphis.edu/framework.html)
- **ACT-R**: [An Analysis and Comparison of ACT-R and Soar](https://arxiv.org/abs/2201.09305)
- **Soar**: [Introduction to the Soar Cognitive Architecture](https://arxiv.org/pdf/2205.03854)
- **CLARION**: [Comparing four cognitive architectures](https://roboticsbiz.com/comparing-four-cognitive-architectures-soar-act-r-clarion-and-dual/)

### Global Workspace Theory

- **GWT概要**: [Wikipedia - Global Workspace Theory](https://en.wikipedia.org/wiki/Global_workspace_theory)
- **LIDA Tutorial**: [The LIDA Tutorial (PDF)](https://ccrg.cs.memphis.edu/assets/framework/The-LIDA-Tutorial.pdf)
- **GWT数理モデル**: [Mathematical and Computational Approaches](https://www.cmup.pt/sites/default/files/2025-09/Global%20Workspace%20Theory%20GWT_V8_EN_0.pdf)

### LLMエージェントの認知設計パターン

- **Cognitive Design Patterns for LLM Agents**: [Applying Cognitive Design Patterns to General LLM Agents](https://arxiv.org/html/2505.07087v2)
- **Cognitive Workspace**: [Active Memory Management for LLMs](https://arxiv.org/abs/2508.13171)
- **GWTマーカー評価**: [Evaluating Global Workspace Markers in Contemporary LLM Systems](https://www.preprints.org/manuscript/202601.1683)

### マルチエージェント設計パターン

- **Event-Driven Multi-Agent Systems**: [Four Design Patterns](https://www.confluent.io/blog/event-driven-multi-agent-systems/)
- **AI Agent Orchestration Patterns**: [Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- **Google Cloud Agentic AI Patterns**: [Architecture Center](https://docs.google.com/architecture/choose-design-pattern-agentic-ai-system)
