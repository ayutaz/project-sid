# 6. 社会的認知モジュール実装設計

[トップ](../index.md) | [PIANOアーキテクチャ](../02-piano-architecture.md) | [マルチエージェント](../04-multi-agent.md) | [文明的進化](../05-civilization.md)

---

## 概要

社会認識モジュール（Social Awareness Module / SAS: Social Awareness System）は、PIANOアーキテクチャにおいて**文明創発に不可欠**なモジュールである。除去すると他者の意図推測能力を完全喪失し、専門化が失敗し、社会的進化が停止する。本ドキュメントでは、論文の知見に基づく再現実装の設計方針を定義する。

> **用語注記**: 論文原文では "Social Awareness System (SAS)" という用語が使用されている。本ドキュメントおよび実装では、PIANOアーキテクチャ内の1モジュールとしての位置づけを明確にするため「Social Awareness Module」の表記を併用する。両者は同一の概念を指す。

### 論文における主要な定量結果

| 指標 | 値 |
|---|---|
| 社会的認知精度（50体） | r = 0.807 |
| 観察者閾値11での精度 | r = 0.907 |
| モジュール除去時の精度 | r = 0.617 |
| 回帰勾配（モジュールあり） | 0.373 |
| 回帰勾配（モジュールなし） | 0.161 |
| 好感度スケール | 0-10（離散） |
| 社会目標生成頻度 | 5-10秒ごと |

---

## 1. 社会認識モジュール設計

### 1.1 入出力インターフェース

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ActivationTrigger(Enum):
    """社会認識モジュールの起動トリガー種別"""
    CONVERSATION_START = "conversation_start"
    CONVERSATION_ONGOING = "conversation_ongoing"
    NEARBY_AGENT_DETECTED = "nearby_agent_detected"
    SOCIAL_GOAL_CYCLE = "social_goal_cycle"


@dataclass
class SocialAwarenessInput:
    """社会認識モジュールへの入力"""
    agent_id: str
    # 現在のコンテキスト
    current_conversation: Optional[str]          # 進行中の会話テキスト
    nearby_agent_ids: list[str]                   # 近傍エージェントID
    # 共有状態からの読み取り
    agent_personality: "PersonalityProfile"        # 自身の性格特性
    memory_context: str                            # 記憶モジュールからの関連記憶
    cognitive_controller_broadcast: str            # CCからのブロードキャスト
    # 既存の社会情報
    social_graph: "SocialGraph"                    # 現在の社会グラフ
    activation_trigger: ActivationTrigger          # 起動トリガー


@dataclass
class SocialAwarenessOutput:
    """社会認識モジュールの出力 -> 共有エージェント状態に書き込み"""
    # 好感度更新
    sentiment_updates: dict[str, float]            # {target_agent_id: new_score}
    # 他者の意図推測
    inferred_intents: dict[str, str]               # {agent_id: inferred_intent}
    # 社会的シグナル（目標生成モジュールへの入力）
    # 注記: 社会的目標の生成自体は目標生成モジュール（07）が担当。
    # 本モジュールはシグナル（誰が助けを必要としているか、社会的機会、
    # 未充足のコミュニティニーズ等）の検出・提供に専念する。
    social_signals: list[str]                      # 検出された社会的シグナル
    # 対話戦略
    interaction_strategy: str                      # 推奨される対話方針
    # 社会グラフ更新
    graph_updates: list["GraphEdgeUpdate"]          # グラフエッジ更新リスト
```

### 1.2 選択的起動条件

論文では社会認識モジュールは**選択的起動（selective activation）**型であり、対話時のみ実行される。これにより計算リソースを効率化する。

```python
class SocialAwarenessActivationPolicy:
    """選択的起動ポリシー"""

    # 起動条件の閾値
    PROXIMITY_THRESHOLD: float = 16.0    # Minecraft内16ブロック以内
    CONVERSATION_ACTIVE: bool = True      # 会話中は常に起動
    SOCIAL_GOAL_INTERVAL: float = 7.5     # 5-10秒の中央値

    def should_activate(
        self,
        agent_state: "SharedAgentState",
        elapsed_since_last: float,
    ) -> tuple[bool, ActivationTrigger | None]:
        """
        起動すべきか判定する。

        Returns:
            (should_activate, trigger_type)
        """
        # 条件1: 会話中は常に起動
        if agent_state.is_in_conversation:
            return True, ActivationTrigger.CONVERSATION_ONGOING

        # 条件2: 会話開始イベント
        if agent_state.has_new_conversation_request:
            return True, ActivationTrigger.CONVERSATION_START

        # 条件3: 近傍にエージェントがいる場合
        nearby = agent_state.get_nearby_agents(self.PROXIMITY_THRESHOLD)
        if nearby:
            return True, ActivationTrigger.NEARBY_AGENT_DETECTED

        # 条件4: 社会目標サイクル（5-10秒周期）
        if elapsed_since_last >= self.SOCIAL_GOAL_INTERVAL:
            return True, ActivationTrigger.SOCIAL_GOAL_CYCLE

        return False, None
```

### 1.3 LLMプロンプト設計

社会認識モジュールのコア処理は、LLMを用いて社会的手がかりを解釈する。論文では、好感度スコアは「エージェントが他者について生成したサマリーに対するLLM呼び出し」で評価される。

```python
SOCIAL_AWARENESS_SYSTEM_PROMPT = """
You are the social awareness module of an AI agent named {agent_name}.
Your role is to interpret social cues from interactions and update
your understanding of relationships with other agents.

Your personality traits:
{personality_description}

Current social context:
- Location: {location}
- Nearby agents: {nearby_agents}
- Recent interactions summary: {interaction_summary}
"""

SENTIMENT_EVALUATION_PROMPT = """
Based on the following interaction history and social context,
evaluate how {agent_name} feels about {target_name}.

Interaction history:
{interaction_history}

{agent_name}'s personality: {personality}
{agent_name}'s current summary of {target_name}: {agent_summary_of_target}

Previous sentiment score: {previous_score}/10

Rate the sentiment on a scale of 0-10:
- 0: Complete hatred/animosity
- 1-2: Strong dislike
- 3-4: Mild dislike / distrust
- 5: Neutral
- 6-7: Mild liking / friendliness
- 8-9: Strong liking / close friendship
- 10: Deep love / complete trust

Provide:
1. Updated sentiment score (0-10, integer)
2. Reasoning for the score change (1-2 sentences)
3. Inferred intent of {target_name} (1 sentence)

Respond in JSON format:
{{
  "sentiment_score": <int>,
  "reasoning": "<string>",
  "inferred_intent": "<string>"
}}
"""

SOCIAL_CONTEXT_SUMMARY_PROMPT = """
Based on your social awareness of the community, summarize the current
social context to inform goal generation.

Your relationships:
{relationship_summaries}

Community context:
{community_context}

Your current goals:
{current_goals}

Summarize the social signals relevant to goal generation:
1. Who needs help or attention?
2. What social opportunities exist?
3. What community needs are unmet?

Respond in JSON format:
{{
  "social_signals": ["<signal1>", "<signal2>", ...],
  "priority_interaction": "<agent_name or null>",
  "interaction_strategy": "<brief strategy>"
}}
"""

# 注記: 社会的目標の生成ロジック自体は目標生成モジュール（07-goal-planning.md）に
# 一元化されている。本モジュールは社会的コンテキスト（好感度、関係性、社会的シグナル）
# の検出・評価・提供に専念し、目標生成モジュールへの入力を構築する。
```

### 1.4 実行フローとリソース効率

```
社会認識モジュール実行フロー:

1. 起動判定 (< 1ms, CPU)
   |
   v
2. コンテキスト収集 (< 5ms, Memory読み取り)
   |-- 共有状態から会話履歴を取得
   |-- 社会グラフから既存関係を取得
   |-- 記憶モジュールから関連記憶を取得
   |
   v
3. LLM推論 (~500-2000ms, API call)
   |-- 好感度評価プロンプト
   |-- 社会的目標生成プロンプト
   |-- ※バッチ化可能: 複数ターゲットを1回のLLM呼び出しで処理
   |
   v
4. 結果の書き込み (< 5ms, 共有状態更新)
   |-- 好感度スコア更新
   |-- 社会グラフエッジ更新
   |-- 社会的目標を共有状態に設定
```

**リソース最適化戦略**:

| 戦略 | 説明 | 効果 |
|---|---|---|
| 選択的起動 | 対話時・近傍エージェント検出時のみ | LLM呼び出し70-80%削減 |
| バッチ評価 | 複数ターゲットを1プロンプトに統合 | API呼び出し回数削減 |
| スコアキャッシュ | 直近の評価結果を一定時間保持 | 冗長な再評価を回避 |
| 差分更新 | 新規対話がある関係のみ更新 | 不要な計算を回避 |

---

## 2. 感情追跡システム

### 2.1 好感度スコアの更新アルゴリズム

論文では0-10の離散整数スケールが使用される。0がhate、10がloveに対応する。

```python
from dataclasses import dataclass
import time
import math


@dataclass
class SentimentRecord:
    """好感度記録"""
    score: int                  # 0-10の離散スコア
    timestamp: float            # 更新時刻
    reasoning: str              # LLMによる根拠
    interaction_count: int      # 対話回数累計
    confidence: float           # 評価の確信度 (0.0-1.0)


class SentimentTracker:
    """
    エージェント間の好感度を追跡するシステム。

    論文の知見:
    - 0-10の離散スケール（0=hate, 10=love）
    - 非対称的な関係（片想い）が自然発生
    - 社会認識モジュールが有効な場合のみ正確に追跡
    """

    DEFAULT_SCORE: int = 5         # 初期値: 中立
    TIME_DECAY_RATE: float = 0.01  # 時間減衰率 (中立方向への回帰) [本実装独自の拡張]
    MIN_CHANGE_THRESHOLD: int = 1  # 最小変更幅

    def __init__(self):
        # {(source_agent, target_agent): SentimentRecord}
        self._sentiments: dict[tuple[str, str], SentimentRecord] = {}

    def get_score(self, source: str, target: str) -> int:
        """現在の好感度スコアを取得"""
        key = (source, target)
        if key not in self._sentiments:
            return self.DEFAULT_SCORE
        record = self._sentiments[key]
        return self._apply_time_decay(record)

    def update_score(
        self,
        source: str,
        target: str,
        llm_evaluated_score: int,
        reasoning: str,
        confidence: float = 0.8,
    ) -> SentimentRecord:
        """
        LLM評価に基づきスコアを更新する。

        急激な変動を抑制するスムージングを適用:
        - 高確信度の評価はより大きな影響
        - 対話回数が多いほどスコアが安定
        """
        key = (source, target)
        current = self._sentiments.get(key)

        if current is None:
            # 初回評価
            new_record = SentimentRecord(
                score=llm_evaluated_score,
                timestamp=time.time(),
                reasoning=reasoning,
                interaction_count=1,
                confidence=confidence,
            )
        else:
            # スムージング: 既存スコアとLLM評価の加重平均
            stability = min(current.interaction_count / 20.0, 0.7)
            weight_new = confidence * (1.0 - stability)
            weight_old = 1.0 - weight_new

            raw_score = weight_old * current.score + weight_new * llm_evaluated_score
            new_score = max(0, min(10, round(raw_score)))

            new_record = SentimentRecord(
                score=new_score,
                timestamp=time.time(),
                reasoning=reasoning,
                interaction_count=current.interaction_count + 1,
                confidence=confidence,
            )

        self._sentiments[key] = new_record
        return new_record

    def _apply_time_decay(self, record: SentimentRecord) -> int:
        """
        時間減衰を適用: 長時間対話がないと中立(5)に回帰。

        **本実装独自の拡張**: 論文では好感度スコアの時間減衰については
        言及されていない。この機能は、長期間対話がないエージェント間の
        関係が自然に中立に回帰するという仮定に基づく独自設計である。
        TIME_DECAY_RATEは実験的チューニングが必要なパラメータ。

        decay = score + (5 - score) * (1 - e^(-rate * elapsed))
        """
        elapsed = time.time() - record.timestamp
        decay_factor = 1.0 - math.exp(-self.TIME_DECAY_RATE * elapsed)
        decayed = record.score + (self.DEFAULT_SCORE - record.score) * decay_factor
        return max(0, min(10, round(decayed)))

    def get_relationship_asymmetry(self, agent_a: str, agent_b: str) -> int:
        """非対称性を計算: |score(A->B) - score(B->A)|"""
        score_ab = self.get_score(agent_a, agent_b)
        score_ba = self.get_score(agent_b, agent_a)
        return abs(score_ab - score_ba)
```

### 2.2 感情推移のモデリング

> **論文根拠と独自拡張の区分**: 以下のモデルにおいて、α（LLM評価の重み）とβ（慣性の重み）は論文の「LLM呼び出しによる好感度評価」に基づく。γ（性格バイアス）とδ（時間減衰）は**本実装独自の拡張**であり、論文に直接の記述はない。stabilityのパラメータ（20.0, 0.7）も実験的チューニングが必要な値である。

```
感情スコアの時間変動モデル:

  Score(t+1) = clamp(0, 10,
      α * LLM_eval(t)                    # LLM評価からの新規入力
    + β * Score(t)                        # 直前スコアの慣性
    + γ * personality_bias                 # 性格特性バイアス [独自拡張]
    - δ * time_decay(t - t_last)          # 時間減衰（中立回帰）[独自拡張]
  )

  ここで:
    α = confidence * (1 - stability)     # 新規評価の重み
    β = 1 - α                            # 慣性の重み
    γ = 0.1                              # 性格バイアス強度 [独自拡張、要チューニング]
    δ = decay_rate                        # 減衰率 [独自拡張、要チューニング]
    stability = min(interaction_count / 20, 0.7)  # [パラメータは要チューニング]
```

### 2.3 感情から行動への変換ロジック

論文の食料配分実験では、エージェントが「自分を最も高く評価していると感じた相手」に選択的に食料を配分した。

```python
class SentimentToActionConverter:
    """感情スコアに基づく行動選択支援"""

    # 閾値定義
    HOSTILE_THRESHOLD: int = 3      # 敵対的と見なすスコア
    NEUTRAL_LOW: int = 4
    NEUTRAL_HIGH: int = 6
    FRIENDLY_THRESHOLD: int = 7     # 友好的と見なすスコア
    ALLY_THRESHOLD: int = 9         # 強い同盟関係

    def get_interaction_disposition(
        self,
        sentiment_score: int,
    ) -> str:
        """好感度に基づく対話態度を返す"""
        if sentiment_score <= self.HOSTILE_THRESHOLD:
            return "avoidant_or_defensive"
        elif sentiment_score <= self.NEUTRAL_HIGH:
            return "neutral_polite"
        elif sentiment_score <= self.FRIENDLY_THRESHOLD:
            return "friendly_cooperative"
        elif sentiment_score <= self.ALLY_THRESHOLD:
            return "warm_supportive"
        else:
            return "deeply_trusting"

    def rank_resource_recipients(
        self,
        agent_id: str,
        candidate_ids: list[str],
        sentiment_tracker: SentimentTracker,
    ) -> list[tuple[str, int]]:
        """
        資源配分の優先順位を感情スコアで決定。
        食料配分実験の再現: 高評価相手を優先。

        Returns:
            [(agent_id, score)] 降順ソート
        """
        scored = [
            (cid, sentiment_tracker.get_score(agent_id, cid))
            for cid in candidate_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def should_cooperate(
        self,
        my_score_of_them: int,
        their_perceived_score_of_me: int,
    ) -> bool:
        """互恵性に基づく協力判定"""
        avg = (my_score_of_them + their_perceived_score_of_me) / 2
        return avg >= self.NEUTRAL_HIGH
```

---

## 3. 社会グラフ

### 3.1 グラフデータ構造の設計

論文では50体実験で有向性・非対称性・性格依存性を持つ社会グラフが自己組織化された。

```python
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class EdgeAttributes:
    """グラフエッジの属性"""
    sentiment: int = 5              # 好感度 (0-10)
    interaction_count: int = 0      # 対話回数
    trust: float = 0.5              # 信頼度 (0.0-1.0)
    last_interaction: float = 0.0   # 最終対話時刻
    relationship_type: str = "acquaintance"  # 関係種別


@dataclass
class NodeAttributes:
    """グラフノードの属性"""
    agent_id: str
    personality: "PersonalityProfile"
    role: Optional[str] = None             # 創発した役割
    community_id: Optional[str] = None     # 所属コミュニティ
    influence_score: float = 0.0           # 影響力スコア


class SocialGraph:
    """
    有向重み付き社会グラフ。

    論文の知見:
    - 有向グラフ: 関係は非対称的（A->Bの好感度 != B->Aの好感度）
    - 重み付き: 好感度・対話頻度・信頼度がエッジ属性
    - 動的: 対話に基づきリアルタイム更新
    - 性格依存: 内向的/外向的でin-degree接続パターンが異なる
    """

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_agent(self, agent_id: str, personality: "PersonalityProfile"):
        """エージェントノードを追加"""
        attrs = NodeAttributes(agent_id=agent_id, personality=personality)
        self._graph.add_node(agent_id, data=attrs)

    def update_edge(
        self,
        source: str,
        target: str,
        sentiment: int,
        trust_delta: float = 0.0,
    ):
        """
        エッジ（有向）を更新。存在しなければ作成。

        Args:
            source: 評価する側のエージェントID
            target: 評価される側のエージェントID
            sentiment: 新しい好感度スコア (0-10)
            trust_delta: 信頼度の変化量
        """
        now = time.time()

        if self._graph.has_edge(source, target):
            edge = self._graph[source][target]["data"]
            edge.sentiment = sentiment
            edge.interaction_count += 1
            edge.trust = max(0.0, min(1.0, edge.trust + trust_delta))
            edge.last_interaction = now
            # 関係種別の自動更新
            edge.relationship_type = self._classify_relationship(edge)
        else:
            edge = EdgeAttributes(
                sentiment=sentiment,
                interaction_count=1,
                trust=0.5 + trust_delta,
                last_interaction=now,
            )
            edge.relationship_type = self._classify_relationship(edge)
            self._graph.add_edge(source, target, data=edge)

    def _classify_relationship(self, edge: EdgeAttributes) -> str:
        """対話回数と好感度に基づく関係分類"""
        if edge.interaction_count < 3:
            return "acquaintance"
        if edge.sentiment >= 8 and edge.trust >= 0.7:
            return "close_friend"
        if edge.sentiment >= 6:
            return "friend"
        if edge.sentiment <= 3:
            return "rival"
        return "acquaintance"

    # --- クエリメソッド ---

    def get_friends(self, agent_id: str, min_sentiment: int = 7) -> list[str]:
        """好感度が閾値以上のエージェントを返す"""
        friends = []
        for _, target, data in self._graph.out_edges(agent_id, data=True):
            if data["data"].sentiment >= min_sentiment:
                friends.append(target)
        return friends

    def get_in_degree(self, agent_id: str) -> int:
        """入次数: このエージェントに好意を持つエージェント数"""
        return self._graph.in_degree(agent_id)

    def get_asymmetric_pairs(self, threshold: int = 3) -> list[tuple[str, str, int]]:
        """
        非対称的関係（片想い）のペアを検出。
        |score(A->B) - score(B->A)| >= threshold

        Returns:
            [(agent_a, agent_b, asymmetry)]
        """
        pairs = []
        seen = set()
        for u, v, d in self._graph.edges(data=True):
            pair_key = tuple(sorted([u, v]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            score_uv = d["data"].sentiment
            if self._graph.has_edge(v, u):
                score_vu = self._graph[v][u]["data"].sentiment
            else:
                score_vu = 5  # 未評価は中立
            asymmetry = abs(score_uv - score_vu)
            if asymmetry >= threshold:
                pairs.append((u, v, asymmetry))
        return pairs

    def get_influencers(self, top_k: int = 5) -> list[tuple[str, float]]:
        """
        PageRankベースのインフルエンサー検出。
        好感度をエッジ重みとして使用。
        """
        weight_graph = nx.DiGraph()
        for u, v, d in self._graph.edges(data=True):
            weight_graph.add_edge(u, v, weight=d["data"].sentiment / 10.0)

        pagerank = nx.pagerank(weight_graph, weight="weight")
        sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
        return sorted_pr[:top_k]

    def get_community_clusters(self) -> list[set[str]]:
        """
        コミュニティ検出（Louvainアルゴリズム）。
        有向グラフを無向化して適用。
        """
        undirected = self._graph.to_undirected()
        for u, v in undirected.edges():
            # 双方向の好感度の平均をエッジ重みに
            w1 = (
                self._graph[u][v]["data"].sentiment
                if self._graph.has_edge(u, v)
                else 5
            )
            w2 = (
                self._graph[v][u]["data"].sentiment
                if self._graph.has_edge(v, u)
                else 5
            )
            undirected[u][v]["weight"] = (w1 + w2) / 2.0

        communities = nx.community.louvain_communities(
            undirected, weight="weight", seed=42
        )
        return communities

    # --- 可視化 ---

    def export_for_visualization(self) -> dict:
        """
        D3.js / vis.js 等での可視化用にエクスポート。

        Returns:
            {"nodes": [...], "edges": [...]}
        """
        nodes = []
        for nid, ndata in self._graph.nodes(data=True):
            node_info = {
                "id": nid,
                "personality": str(ndata.get("data", {}).personality
                    if hasattr(ndata.get("data", {}), "personality") else "unknown"),
                "in_degree": self._graph.in_degree(nid),
                "out_degree": self._graph.out_degree(nid),
            }
            nodes.append(node_info)

        edges = []
        for u, v, d in self._graph.edges(data=True):
            edge_info = {
                "source": u,
                "target": v,
                "sentiment": d["data"].sentiment,
                "interaction_count": d["data"].interaction_count,
                "trust": d["data"].trust,
                "relationship_type": d["data"].relationship_type,
            }
            edges.append(edge_info)

        return {"nodes": nodes, "edges": edges}
```

### 3.2 グラフの動的更新

```python
class SocialGraphUpdater:
    """対話イベントに基づくグラフ自動更新"""

    # 信頼度変化のマッピング
    TRUST_DELTA_MAP = {
        "cooperative_action": 0.05,     # 協力的行動
        "resource_sharing": 0.08,       # 資源共有
        "conversation": 0.02,           # 通常会話
        "conflict": -0.1,               # 対立
        "betrayal": -0.2,               # 裏切り
    }

    def on_interaction(
        self,
        graph: SocialGraph,
        tracker: SentimentTracker,
        source: str,
        target: str,
        interaction_type: str,
        llm_sentiment: int,
    ):
        """
        対話発生時にグラフと好感度を同時更新。

        このメソッドは社会認識モジュールのLLM推論結果を受けて呼ばれる。
        """
        # 好感度の更新
        tracker.update_score(
            source=source,
            target=target,
            llm_evaluated_score=llm_sentiment,
            reasoning=f"Updated after {interaction_type}",
        )

        # グラフエッジの更新
        trust_delta = self.TRUST_DELTA_MAP.get(interaction_type, 0.01)
        graph.update_edge(
            source=source,
            target=target,
            sentiment=llm_sentiment,
            trust_delta=trust_delta,
        )
```

---

## 4. 性格モデリング

### 4.1 性格特性フレームワーク

論文では各エージェントに「固有の性格特性」が割り当てられ、内向的/外向的エージェントの接続パターンに明確な差異が観察された。Big Fiveモデルをベースに実装する。

```python
from dataclasses import dataclass


@dataclass
class PersonalityProfile:
    """
    Big Five性格特性モデル。
    各特性は0.0-1.0のスケール。

    論文との対応:
    - extraversion（外向性）がネットワーク接続数に直接影響
    - agreeableness（協調性）が協力行動と好感度バイアスに影響
    """
    openness: float = 0.5           # 経験への開放性
    conscientiousness: float = 0.5  # 誠実性
    extraversion: float = 0.5       # 外向性
    agreeableness: float = 0.5      # 協調性
    neuroticism: float = 0.5        # 神経症的傾向

    def to_prompt_description(self) -> str:
        """LLMプロンプトに組み込むための自然言語記述を生成"""
        traits = []

        if self.extraversion > 0.7:
            traits.append("highly outgoing and sociable")
        elif self.extraversion < 0.3:
            traits.append("introverted and reserved")

        if self.agreeableness > 0.7:
            traits.append("very cooperative and trusting")
        elif self.agreeableness < 0.3:
            traits.append("competitive and skeptical")

        if self.openness > 0.7:
            traits.append("curious and creative")
        elif self.openness < 0.3:
            traits.append("practical and conventional")

        if self.conscientiousness > 0.7:
            traits.append("organized and disciplined")
        elif self.conscientiousness < 0.3:
            traits.append("flexible and spontaneous")

        if self.neuroticism > 0.7:
            traits.append("emotionally sensitive")
        elif self.neuroticism < 0.3:
            traits.append("emotionally stable and calm")

        if not traits:
            return "balanced personality with moderate traits"
        return ", ".join(traits)


def generate_random_personality(
    introvert_ratio: float = 0.4,
    rng=None,
) -> PersonalityProfile:
    """
    手続き的な性格生成。

    Args:
        introvert_ratio: 内向的エージェントの比率
        rng: 乱数生成器（再現性のため）
    """
    import random
    r = rng or random

    # 外向性は二峰性分布を模倣（内向/外向の明確な分離）
    is_introvert = r.random() < introvert_ratio
    if is_introvert:
        extraversion = r.uniform(0.1, 0.35)
    else:
        extraversion = r.uniform(0.65, 0.95)

    return PersonalityProfile(
        openness=r.uniform(0.2, 0.9),
        conscientiousness=r.uniform(0.2, 0.9),
        extraversion=extraversion,
        agreeableness=r.uniform(0.2, 0.9),
        neuroticism=r.uniform(0.1, 0.8),
    )
```

### 4.2 性格から行動パターンへのマッピング

```python
@dataclass
class SocialBehaviorParams:
    """性格から導出される社会的行動パラメータ"""
    max_concurrent_conversations: int    # 同時会話数上限
    conversation_initiation_rate: float  # 会話開始頻度（回/分）
    preferred_group_size: int            # 好むグループサイズ
    sentiment_volatility: float          # 感情変動の大きさ
    cooperation_threshold: float         # 協力を開始する好感度閾値
    exploration_tendency: float          # 新規エージェントへの接触傾向


def personality_to_behavior(p: PersonalityProfile) -> SocialBehaviorParams:
    """
    性格特性から行動パラメータを導出。

    論文の知見:
    - 外向的エージェント: 高いin-degree、ハブとして機能
    - 内向的エージェント: 少ないin-degree接続
    """
    return SocialBehaviorParams(
        max_concurrent_conversations=int(1 + p.extraversion * 4),
        conversation_initiation_rate=0.1 + p.extraversion * 0.9,
        preferred_group_size=int(2 + p.extraversion * 6),
        sentiment_volatility=0.3 + p.neuroticism * 0.7,
        cooperation_threshold=int(7 - p.agreeableness * 3),
        exploration_tendency=0.2 + p.openness * 0.6,
    )
```

### 4.3 内向的/外向的エージェントの接続パターン制御

```python
class PersonalityDrivenInteraction:
    """性格に基づく対話相手選択"""

    def select_interaction_target(
        self,
        agent_id: str,
        personality: PersonalityProfile,
        social_graph: SocialGraph,
        available_agents: list[str],
        rng=None,
    ) -> str | None:
        """
        性格に基づいて対話相手を選択。

        外向的: 新規エージェントと積極的に対話（探索）
        内向的: 既知の好感度が高いエージェントを選好（活用）
        """
        import random
        r = rng or random

        if not available_agents:
            return None

        behavior = personality_to_behavior(personality)

        # 探索 vs 活用のバランス
        explore = r.random() < behavior.exploration_tendency

        if explore:
            # 未対話または対話回数が少ないエージェントを優先
            candidates = []
            for aid in available_agents:
                if social_graph._graph.has_edge(agent_id, aid):
                    edge = social_graph._graph[agent_id][aid]["data"]
                    candidates.append((aid, -edge.interaction_count))
                else:
                    candidates.append((aid, 100))  # 未対話は最高優先度

            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0] if candidates else None
        else:
            # 好感度が高いエージェントを優先
            friends = social_graph.get_friends(agent_id, min_sentiment=6)
            available_friends = [f for f in friends if f in available_agents]
            if available_friends:
                return r.choice(available_friends)
            # フレンドがいなければランダム
            return r.choice(available_agents)
```

---

## 5. 集合知メカニズム

### 5.1 複数エージェントの評価集約方法

論文では、観察者閾値を上げる（多くのエージェントが評価に参加する）ほど精度が向上し、r = 0.646（1観察者）からr = 0.907（11観察者）への改善が観測された。

```python
import numpy as np
from dataclasses import dataclass


@dataclass
class WisdomOfCrowdsResult:
    """集合知評価の結果"""
    target_agent: str
    aggregated_score: float          # 集約されたスコア
    individual_scores: list[float]   # 個別スコア
    observer_count: int              # 観察者数
    standard_deviation: float        # 標準偏差
    confidence: float                # 信頼度


class WisdomOfCrowdsAggregator:
    """
    集合知による好感度推定。

    論文の知見:
    - 観察者閾値1: r = 0.646, n = 46
    - 観察者閾値5: r = 0.807
    - 観察者閾値11: r = 0.907, n = 18

    より多くのエージェントが評価に参加するほど精度が向上。
    """

    MIN_OBSERVERS: int = 1
    CONFIDENCE_THRESHOLD: int = 5  # この観察者数で信頼性が十分

    def aggregate_likeability(
        self,
        target_agent: str,
        social_graph: SocialGraph,
        sentiment_tracker: SentimentTracker,
        min_observers: int = 1,
    ) -> WisdomOfCrowdsResult | None:
        """
        複数のエージェントによるターゲットの好感度を集約。

        集約方法: 重み付き平均
        重み = 対話回数（より多く対話したエージェントの評価を重視）
        """
        observers = []
        scores = []
        weights = []

        # ターゲットを評価しているすべてのエージェントを収集
        for source in social_graph._graph.predecessors(target_agent):
            edge = social_graph._graph[source][target_agent]["data"]
            if edge.interaction_count >= min_observers:
                score = sentiment_tracker.get_score(source, target_agent)
                observers.append(source)
                scores.append(float(score))
                weights.append(float(edge.interaction_count))

        if not scores:
            return None

        scores_arr = np.array(scores)
        weights_arr = np.array(weights)
        weights_arr = weights_arr / weights_arr.sum()  # 正規化

        aggregated = float(np.average(scores_arr, weights=weights_arr))
        std_dev = float(np.std(scores_arr))

        # 信頼度: 観察者数と一致度に基づく
        observer_confidence = min(len(observers) / self.CONFIDENCE_THRESHOLD, 1.0)
        agreement_confidence = max(0.0, 1.0 - std_dev / 5.0)
        confidence = 0.6 * observer_confidence + 0.4 * agreement_confidence

        return WisdomOfCrowdsResult(
            target_agent=target_agent,
            aggregated_score=aggregated,
            individual_scores=scores,
            observer_count=len(observers),
            standard_deviation=std_dev,
            confidence=confidence,
        )

    def estimate_true_likeability(
        self,
        target_agent: str,
        social_graph: SocialGraph,
        sentiment_tracker: SentimentTracker,
    ) -> tuple[float, float]:
        """
        真の好感度を推定（集合知の活用）。

        観察者閾値を段階的に上げ、十分なサンプルがある
        最高閾値での集約結果を返す。

        Returns:
            (estimated_score, confidence)
        """
        best_result = None

        for threshold in [1, 3, 5, 7, 11]:
            result = self.aggregate_likeability(
                target_agent, social_graph, sentiment_tracker,
                min_observers=threshold,
            )
            if result and result.observer_count >= threshold:
                best_result = result
            else:
                break  # これ以上閾値を上げてもサンプル不足

        if best_result is None:
            return 5.0, 0.0  # デフォルト

        return best_result.aggregated_score, best_result.confidence
```

### 5.2 集団的認知の数学的モデル

```
集合知精度モデル:

  個々のエージェントの好感度推定:
    estimated_i = true_score + noise_i
    where noise_i ~ N(0, σ²)

  n人のエージェントによる集約推定:
    estimated_agg = (1/n) Σ estimated_i
                  = true_score + (1/n) Σ noise_i

  集約推定の分散:
    Var(estimated_agg) = σ² / n

  したがって:
    - n=1:  Var = σ² (個別推定)
    - n=5:  Var = σ²/5
    - n=11: Var = σ²/11

  相関係数の理論的関係:
    r(n) ≈ 1 / sqrt(1 + σ²/(n * Var(true)))

  論文のデータとの対応:
    r(1) = 0.646 → σ² ≈ 推定可能
    r(5) = 0.807
    r(11) = 0.907
    n→∞: r → 1.0 (理論的上限)

  重要な前提:
    - 各エージェントの評価誤差が独立（diversity条件）
    - 系統的バイアスが小さい（性格による偏りは存在）
```

---

## 6. インフルエンサーと感情伝播

### 6.1 インフルエンサーの影響力モデル

論文の集団ルール実験では、3体のインフルエンサーが25体の構成員の投票行動を形成した。

```python
@dataclass
class InfluencerProfile:
    """インフルエンサーの属性"""
    agent_id: str
    stance: str                      # "pro_tax", "anti_tax", etc.
    influence_radius: float          # 影響範囲
    persuasion_strength: float       # 説得力 (0.0-1.0)
    charisma: float                  # カリスマ性 (0.0-1.0)


class InfluenceModel:
    """
    インフルエンサーの影響力モデル。

    論文の知見:
    - 反税インフルエンサー: 税率 20% -> 9% に低下
    - 賛税インフルエンサー: 税率維持〜引き上げ
    - フィードバック→修正案→投票→行動変化の連鎖
    """

    # 好感度→影響受容度のマッピング
    INFLUENCE_ACCEPTANCE_CURVE = {
        # sentiment_score: acceptance_rate
        0: 0.0, 1: 0.0, 2: 0.05,
        3: 0.1, 4: 0.2, 5: 0.3,
        6: 0.5, 7: 0.65, 8: 0.8,
        9: 0.9, 10: 0.95,
    }

    def calculate_influence_strength(
        self,
        influencer: InfluencerProfile,
        target_agent: str,
        sentiment_tracker: SentimentTracker,
        social_graph: SocialGraph,
    ) -> float:
        """
        インフルエンサーがターゲットに与える影響の強さを計算。

        influence = acceptance_rate * persuasion * charisma * proximity_factor
        """
        # ターゲットがインフルエンサーに持つ好感度
        sentiment = sentiment_tracker.get_score(target_agent, influencer.agent_id)
        acceptance = self.INFLUENCE_ACCEPTANCE_CURVE.get(sentiment, 0.3)

        # 対話回数による親密度補正
        if social_graph._graph.has_edge(target_agent, influencer.agent_id):
            edge = social_graph._graph[target_agent][influencer.agent_id]["data"]
            familiarity = min(edge.interaction_count / 10.0, 1.0)
        else:
            familiarity = 0.1

        influence = (
            acceptance
            * influencer.persuasion_strength
            * influencer.charisma
            * (0.5 + 0.5 * familiarity)
        )
        return min(influence, 1.0)

    def apply_influence_to_opinion(
        self,
        current_opinion: float,
        influencer_stance_value: float,
        influence_strength: float,
    ) -> float:
        """
        意見をインフルエンサーのスタンスに引き寄せる。

        new_opinion = current + strength * (stance - current)
        """
        shift = influence_strength * (influencer_stance_value - current_opinion)
        return current_opinion + shift
```

### 6.2 感情伝播のシミュレーション

```python
class EmotionalContagionSimulator:
    """
    感情伝播シミュレーション。

    論文との対応:
    - 宗教伝播: SIRモデル的な感染拡大ダイナミクス
    - 文化ミーム: 人口密度依存の伝播
    - インフルエンサー効果: 意見の連鎖的伝播
    """

    CONTAGION_RATE: float = 0.1      # 基本伝播率
    RECOVERY_RATE: float = 0.02      # 感情の自然減衰率

    def propagate_sentiment(
        self,
        social_graph: SocialGraph,
        sentiment_tracker: SentimentTracker,
        source_agent: str,
        sentiment_topic: str,
        sentiment_value: float,
        max_hops: int = 3,
    ) -> dict[str, float]:
        """
        感情をグラフ上で伝播させる。

        伝播メカニズム:
        1. sourceがtopic についてsentiment_valueの感情を持つ
        2. 隣接エージェントに好感度に比例した強度で伝播
        3. 各ホップで減衰

        Returns:
            {agent_id: received_sentiment_value}
        """
        affected = {source_agent: sentiment_value}
        current_layer = {source_agent: sentiment_value}

        for hop in range(max_hops):
            next_layer = {}
            decay = (1.0 - self.CONTAGION_RATE) ** (hop + 1)

            for agent, value in current_layer.items():
                for _, neighbor, data in social_graph._graph.out_edges(
                    agent, data=True
                ):
                    if neighbor in affected:
                        continue

                    edge = data["data"]
                    # 好感度が高いほど伝播しやすい
                    transmission = (edge.sentiment / 10.0) * self.CONTAGION_RATE
                    received = value * transmission * decay

                    if abs(received) > 0.1:  # 閾値以上の影響のみ
                        next_layer[neighbor] = received
                        affected[neighbor] = received

            current_layer = next_layer
            if not current_layer:
                break

        return affected
```

### 6.3 投票行動への影響メカニズム

```python
@dataclass
class VoteDecision:
    """投票の意思決定"""
    agent_id: str
    vote: str                    # "approve" / "reject"
    confidence: float            # 確信度
    primary_influence: str       # 主な影響源


class VotingInfluenceModel:
    """
    投票行動への影響モデル。

    論文タイムライン:
    1. 修正前フェーズ (0-600秒): 120秒間隔で5回の課税シーズン、初期法規に従う
    2. 民主的プロセス (300-360秒): フィードバック提出、修正案処理
    3. 投票 (420秒): 構成員が修正案に投票
    4. 修正後フェーズ (600-1200秒): 憲法更新後、5回の追加課税シーズン、行動変化

    注記: 課税シーズンは120秒間隔で実施される（全体1200秒で前半5回・後半5回）。
    29体のエージェント（25構成員 + 3インフルエンサー + 1選挙管理者）で実施。
    """

    def determine_vote(
        self,
        agent_id: str,
        amendment_text: str,
        sentiment_tracker: SentimentTracker,
        social_graph: SocialGraph,
        influencer_stances: dict[str, str],
        agent_personality: PersonalityProfile,
    ) -> VoteDecision:
        """
        投票を決定する。

        要因:
        1. 自身の経験（税負担の感覚）
        2. インフルエンサーからの影響
        3. 周囲のエージェントの意見
        4. 性格特性
        """
        influence_model = InfluenceModel()

        # 各インフルエンサーからの影響を集約
        total_pro = 0.0
        total_anti = 0.0

        for inf_id, stance in influencer_stances.items():
            inf_profile = InfluencerProfile(
                agent_id=inf_id,
                stance=stance,
                influence_radius=32.0,
                persuasion_strength=0.7,
                charisma=0.8,
            )
            strength = influence_model.calculate_influence_strength(
                inf_profile, agent_id, sentiment_tracker, social_graph,
            )
            if stance == "pro_tax":
                total_pro += strength
            else:
                total_anti += strength

        # 性格バイアス: 協調性が高いほど現状維持に傾く
        status_quo_bias = agent_personality.agreeableness * 0.3

        # 意思決定
        pro_score = total_pro + status_quo_bias
        anti_score = total_anti

        if pro_score > anti_score:
            vote = "approve"
            confidence = min(pro_score / (pro_score + anti_score + 0.01), 1.0)
            primary = "pro_influencer"
        else:
            vote = "reject"
            confidence = min(anti_score / (pro_score + anti_score + 0.01), 1.0)
            primary = "anti_influencer"

        return VoteDecision(
            agent_id=agent_id,
            vote=vote,
            confidence=confidence,
            primary_influence=primary,
        )
```

---

## 7. 統合: 社会認識モジュールのメインループ

```python
class SocialAwarenessModule:
    """
    社会認識モジュールの統合実装。

    PIANOアーキテクチャ内での位置:
    - 種別: 選択的起動モジュール
    - 共有状態: 読み書き
    - 起動条件: 対話時、近傍エージェント検出時、社会目標サイクル
    """

    def __init__(
        self,
        llm_client,  # LLM APIクライアント
        sentiment_tracker: SentimentTracker,
        social_graph: SocialGraph,
        graph_updater: SocialGraphUpdater,
        woc_aggregator: WisdomOfCrowdsAggregator,
    ):
        self._llm = llm_client
        self._tracker = sentiment_tracker
        self._graph = social_graph
        self._updater = graph_updater
        self._woc = woc_aggregator
        self._activation_policy = SocialAwarenessActivationPolicy()
        self._last_execution: dict[str, float] = {}

    async def execute(
        self,
        agent_id: str,
        shared_state: "SharedAgentState",
    ) -> SocialAwarenessOutput | None:
        """
        モジュールのメイン実行。

        ステートレス操作: 共有状態を読み取り、処理し、結果を書き戻す。
        """
        # Step 1: 起動判定
        elapsed = time.time() - self._last_execution.get(agent_id, 0)
        should_run, trigger = self._activation_policy.should_activate(
            shared_state, elapsed
        )
        if not should_run:
            return None

        self._last_execution[agent_id] = time.time()

        # Step 2: コンテキスト収集
        personality = shared_state.personality
        conversation = shared_state.current_conversation
        nearby = shared_state.get_nearby_agents(16.0)
        memory_ctx = shared_state.get_relevant_memories("social")
        cc_broadcast = shared_state.cognitive_controller_output

        # Step 3: 対話相手の好感度評価（LLM呼び出し）
        sentiment_updates = {}
        inferred_intents = {}

        interaction_targets = []
        if conversation:
            interaction_targets = [conversation.partner_id]
        interaction_targets.extend(nearby)
        interaction_targets = list(set(interaction_targets))

        for target_id in interaction_targets:
            # 対話履歴の取得
            interaction_history = shared_state.get_interaction_history(
                agent_id, target_id, limit=10
            )
            previous_score = self._tracker.get_score(agent_id, target_id)
            agent_summary = shared_state.get_agent_summary(target_id)

            # LLMによる好感度評価
            prompt = SENTIMENT_EVALUATION_PROMPT.format(
                agent_name=agent_id,
                target_name=target_id,
                interaction_history=interaction_history,
                personality=personality.to_prompt_description(),
                agent_summary_of_target=agent_summary,
                previous_score=previous_score,
            )

            response = await self._llm.generate(
                system=SOCIAL_AWARENESS_SYSTEM_PROMPT.format(
                    agent_name=agent_id,
                    personality_description=personality.to_prompt_description(),
                    location=shared_state.location,
                    nearby_agents=", ".join(nearby),
                    interaction_summary=memory_ctx,
                ),
                user=prompt,
                response_format="json",
            )

            parsed = self._parse_sentiment_response(response)
            if parsed:
                sentiment_updates[target_id] = parsed["sentiment_score"]
                inferred_intents[target_id] = parsed["inferred_intent"]

                # グラフとトラッカーを更新
                self._updater.on_interaction(
                    self._graph, self._tracker,
                    source=agent_id,
                    target=target_id,
                    interaction_type="conversation" if conversation else "observation",
                    llm_sentiment=parsed["sentiment_score"],
                )

        # Step 4: 社会的コンテキストの構築
        # 注記: 社会的目標の生成ロジックは目標生成モジュール（07）に一元化。
        # 本モジュールは社会的シグナルの検出・評価に専念し、
        # 目標生成モジュールへの入力となる社会的コンテキストを提供する。
        relationship_summaries = self._build_relationship_summaries(
            agent_id, interaction_targets
        )
        context_prompt = SOCIAL_CONTEXT_SUMMARY_PROMPT.format(
            relationship_summaries=relationship_summaries,
            community_context=shared_state.community_context,
            current_goals=shared_state.current_goals,
        )

        context_response = await self._llm.generate(
            system=SOCIAL_AWARENESS_SYSTEM_PROMPT.format(
                agent_name=agent_id,
                personality_description=personality.to_prompt_description(),
                location=shared_state.location,
                nearby_agents=", ".join(nearby),
                interaction_summary=memory_ctx,
            ),
            user=context_prompt,
            response_format="json",
        )

        context_parsed = self._parse_context_response(context_response)

        # Step 5: 出力の構築
        return SocialAwarenessOutput(
            sentiment_updates=sentiment_updates,
            inferred_intents=inferred_intents,
            social_signals=context_parsed.get("social_signals", []),
            interaction_strategy=context_parsed.get("interaction_strategy", ""),
            graph_updates=[],  # 既にon_interactionで更新済み
        )

    def _parse_sentiment_response(self, response: str) -> dict | None:
        """LLMレスポンスのJSONをパース"""
        import json
        try:
            data = json.loads(response)
            score = int(data["sentiment_score"])
            score = max(0, min(10, score))
            return {
                "sentiment_score": score,
                "reasoning": data.get("reasoning", ""),
                "inferred_intent": data.get("inferred_intent", ""),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _parse_context_response(self, response: str) -> dict:
        """社会的コンテキストレスポンスのパース"""
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"social_signals": [], "interaction_strategy": "observe"}

    def _build_relationship_summaries(
        self,
        agent_id: str,
        targets: list[str],
    ) -> str:
        """対話相手との関係サマリーを構築"""
        summaries = []
        for target in targets:
            score = self._tracker.get_score(agent_id, target)
            if self._graph._graph.has_edge(agent_id, target):
                edge = self._graph._graph[agent_id][target]["data"]
                summaries.append(
                    f"- {target}: sentiment={score}/10, "
                    f"interactions={edge.interaction_count}, "
                    f"trust={edge.trust:.2f}, "
                    f"relationship={edge.relationship_type}"
                )
            else:
                summaries.append(f"- {target}: sentiment={score}/10 (new contact)")
        return "\n".join(summaries) if summaries else "No active relationships"
```

---

## 8. アブレーション対応と評価指標

### 8.1 モジュール除去時の影響

論文のアブレーション結果を再現するための評価フレームワーク。

```python
@dataclass
class SocialCognitionMetrics:
    """社会的認知の評価指標"""
    pearson_correlation: float      # ピアソン相関係数
    regression_slope: float         # 回帰勾配
    asymmetry_count: int            # 非対称関係の数
    role_entropy: float             # 役割エントロピー（専門化の多様性）
    avg_sentiment_variance: float   # 感情スコアの平均分散

    @staticmethod
    def expected_with_module() -> "SocialCognitionMetrics":
        """論文の社会認識モジュールあり条件の期待値"""
        return SocialCognitionMetrics(
            pearson_correlation=0.807,
            regression_slope=0.373,
            asymmetry_count=20,       # 50体中の推定
            role_entropy=1.8,         # 6役割以上
            avg_sentiment_variance=4.0,
        )

    @staticmethod
    def expected_without_module() -> "SocialCognitionMetrics":
        """論文の社会認識モジュールなし条件の期待値"""
        return SocialCognitionMetrics(
            pearson_correlation=0.617,
            regression_slope=0.161,
            asymmetry_count=5,
            role_entropy=0.5,         # 役割が均質
            avg_sentiment_variance=1.0,
        )
```

---

## 9. 依存関係と技術スタック

### 9.1 Python依存パッケージ

```
# 社会グラフ
networkx>=3.0
python-louvain>=0.16    # コミュニティ検出

# 数値計算
numpy>=1.24

# LLM統合
openai>=1.0             # GPT-4o API (or compatible)

# 可視化（オプション）
matplotlib>=3.7
pyvis>=0.3              # インタラクティブグラフ可視化

# 動的グラフ（オプション）
dynetx>=0.3             # 時系列グラフ
```

### 9.2 他モジュールとの依存関係

```
社会認識モジュール
├── 入力依存
│   ├── 記憶モジュール (Memory) -- 対話履歴・関連記憶の取得
│   ├── 認知コントローラ (CC) -- ブロードキャスト情報の受信
│   └── 環境知覚 -- 近傍エージェント検出
├── 出力先
│   ├── 共有エージェント状態 -- 好感度・社会的シグナルの書き込み
│   ├── 認知コントローラ (CC) -- 社会的コンテキストの提供
│   ├── 目標生成モジュール -- 社会的シグナル・コンテキストの提供（目標生成は07が担当）
│   └── 発話モジュール (Talking) -- 対話戦略の条件付け
└── 内部コンポーネント
    ├── SentimentTracker -- 好感度追跡
    ├── SocialGraph -- 社会グラフ管理
    ├── WisdomOfCrowdsAggregator -- 集合知推定
    └── PersonalityProfile -- 性格モデル
```

---

## 10. 実装優先度

| 優先度 | コンポーネント | 理由 |
|---|---|---|
| P0 | SentimentTracker | 全ての社会的機能の基盤 |
| P0 | SocialGraph (基本) | エージェント関係の記録に必須 |
| P0 | LLM好感度評価プロンプト | コア機能 |
| P1 | 選択的起動ポリシー | リソース効率に重要 |
| P1 | PersonalityProfile | 行動多様性の源泉 |
| P1 | 社会的目標生成 | 5-10秒周期の目標更新 |
| P2 | WisdomOfCrowdsAggregator | 集合知精度の検証に必要 |
| P2 | InfluenceModel | 文明実験の再現に必要 |
| P2 | EmotionalContagionSimulator | 大規模実験の再現に必要 |
| P3 | グラフ可視化 | デバッグ・分析用 |
| P3 | VotingInfluenceModel | 集団ルール実験の再現 |

---

## 参考文献

- Park, J.S., et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." UIST '23.
- Altera.AL (2024). "Project Sid: Many-agent simulations toward AI civilization." arXiv:2411.00114.
- Surowiecki, J. (2004). "The Wisdom of Crowds."
- Baars, B.J. (1988). "A Cognitive Theory of Consciousness" (Global Workspace Theory).
- NetworkX Documentation: https://networkx.org/documentation/stable/
- DyNetx Documentation: https://dynetx.readthedocs.io/

---

[トップ](../index.md) | [PIANOアーキテクチャ](../02-piano-architecture.md) | [マルチエージェント](../04-multi-agent.md)
