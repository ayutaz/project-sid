# 09. 評価・ベンチマークシステム実装ガイド

[トップ](../index.md) | [論文の評価分析](../08-evaluation.md)

---

## 目次

1. [概要](#1-概要)
2. [専門化指標の実装](#2-専門化指標の実装)
3. [統治指標の実装](#3-統治指標の実装)
4. [文化伝播指標の実装](#4-文化伝播指標の実装)
5. [インフラ発展指標の実装](#5-インフラ発展指標の実装)
6. [社会的認知精度の計測](#6-社会的認知精度の計測)
7. [アイテム収集ベンチマーク](#7-アイテム収集ベンチマーク)
8. [データ収集パイプライン](#8-データ収集パイプライン)
9. [可視化・ダッシュボード](#9-可視化ダッシュボード)
10. [LLM依存評価の代替案](#10-llm依存評価の代替案)
11. [評価パイプラインのタイミング方針](#11-評価パイプラインのタイミング方針)
12. [技術スタック推奨](#12-技術スタック推奨)

---

## 1. 概要

Project Sidの文明的ベンチマークは、AIエージェント社会の進歩を4つの文明軸と2つの補助軸で評価する。

**文明的ベンチマーク4軸**（論文 Section 5 で定義）:

| 軸 | 測定対象 | 主要指標 |
|---|---|---|
| **専門化** | 役割分化の創発度 | 役割エントロピー、役割持続時間、行動-役割相関 |
| **制度的統治** | 集団ルールの機能度 | 税遵守率、憲法修正の行動反映度、投票行動 |
| **文化的伝播** | 情報・信念の拡散 | ミーム発生数、地理的伝播範囲、改宗率 |
| **インフラ発展** | 制度の機能的運用度 | 建造物数・種類、共有施設利用率、空間的接続性 |

**補助評価軸**:

| 軸 | 測定対象 | 主要指標 |
|---|---|---|
| **社会的認知** | 他者の感情推測精度 | ピアソン相関、観察者閾値による精度変化 |
| **個体能力** | 単一エージェントの進歩 | ユニークアイテム数、依存ツリー深度、飽和時間 |

### 設計原則

- **創発性重視**: 事前にプログラムされた行動ではなく、自律的に創発する現象を測定
- **多次元評価**: 単一指標ではなく、複数軸を同時に評価
- **スケーラビリティ**: 30体から1000体以上への規模拡張に対応
- **再現性**: 定量的・自動化された測定パイプラインにより再現性を確保

---

## 2. 専門化指標の実装

### 2.1 役割エントロピー（Role Entropy）

Shannonエントロピーを用いて、エージェント集団の役割多様性を定量化する。

#### 数学的定義

```
H(R) = -Σ p(r_i) * log₂(p(r_i))
```

- `R`: 観察された役割の集合
- `p(r_i)`: 役割 `r_i` に分類されたエージェントの比率
- `H(R) = 0`: 全エージェントが同一役割（専門化なし）
- `H(R) = log₂(N)`: 全役割が均等分布（最大多様性）

#### 実装設計

```python
from collections import Counter
import math
from dataclasses import dataclass

@dataclass
class RoleEntropyResult:
    entropy: float
    max_entropy: float
    normalized_entropy: float  # 0-1にスケール
    role_distribution: dict[str, float]
    num_unique_roles: int

class RoleEntropyCalculator:
    """エージェント集団の役割エントロピーを計算する。"""

    def calculate(self, role_assignments: list[str]) -> RoleEntropyResult:
        """
        Args:
            role_assignments: 各エージェントに割り当てられた役割のリスト
                例: ["farmer", "miner", "farmer", "guard", "explorer"]
        """
        total = len(role_assignments)
        counts = Counter(role_assignments)
        probabilities = {role: count / total for role, count in counts.items()}

        entropy = -sum(p * math.log2(p) for p in probabilities.values() if p > 0)
        num_roles = len(counts)
        max_entropy = math.log2(num_roles) if num_roles > 1 else 0
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        return RoleEntropyResult(
            entropy=entropy,
            max_entropy=max_entropy,
            normalized_entropy=normalized,
            role_distribution=probabilities,
            num_unique_roles=num_roles,
        )

    def calculate_over_time(
        self, timestamped_roles: list[tuple[float, list[str]]]
    ) -> list[tuple[float, RoleEntropyResult]]:
        """時系列でのエントロピー変化を計算する。"""
        return [
            (timestamp, self.calculate(roles))
            for timestamp, roles in timestamped_roles
        ]
```

#### 解釈ガイドライン

| 正規化エントロピー | 解釈 |
|---|---|
| 0.0 - 0.2 | 専門化なし（全員が同質的） |
| 0.2 - 0.5 | 部分的専門化（支配的役割あり） |
| 0.5 - 0.8 | 中程度の専門化 |
| 0.8 - 1.0 | 高度な専門化（多様な役割が均等分布） |

**注意**: 論文では社会認識モジュール除去時にエントロピーが低下し、専門化の失敗を示した。高いエントロピーが望ましいかどうかは文脈依存（極端に均等な分布は非現実的な場合もある）。

### 2.2 GPT-4o役割推論パイプライン

エージェントの社会的目標（social goals）からLLMを使って役割を推論するパイプライン。

#### アーキテクチャ

```
エージェントの社会的目標ログ（5目標ローリングセット）
  ↓
前処理・バッチ化
  ↓
LLM推論（GPT-4o / Claude等）
  ↓
役割分類結果
  ↓
集計・エントロピー計算
```

#### 実装設計

```python
from dataclasses import dataclass
from enum import Enum

class AgentRole(Enum):
    FARMER = "farmer"
    MINER = "miner"
    ENGINEER = "engineer"
    GUARD = "guard"
    EXPLORER = "explorer"
    BLACKSMITH = "blacksmith"
    SCOUT = "scout"         # 軍事的社会
    STRATEGIST = "strategist"
    CURATOR = "curator"     # 芸術的社会
    COLLECTOR = "collector"
    OTHER = "other"

@dataclass
class RoleInferenceRequest:
    agent_id: str
    recent_goals: list[str]  # 直近5件の社会的目標
    timestamp: float

@dataclass
class RoleInferenceResult:
    agent_id: str
    inferred_role: AgentRole
    confidence: float  # 0.0 - 1.0
    reasoning: str

ROLE_INFERENCE_PROMPT = """
以下のエージェントの直近5件の社会的目標を分析し、最も適切な役割を推論してください。

社会的目標:
{goals}

使用可能な役割: {available_roles}

出力形式（JSON）:
{{
  "role": "<役割名>",
  "confidence": <0.0-1.0>,
  "reasoning": "<推論の根拠>"
}}
"""

class RoleInferencePipeline:
    """
    LLMを用いた役割推論パイプライン。
    論文では5-10秒ごとに生成される社会的目標の直近5件を使用。
    """

    def __init__(self, llm_client, batch_size: int = 20):
        self.llm_client = llm_client
        self.batch_size = batch_size

    async def infer_roles(
        self, requests: list[RoleInferenceRequest]
    ) -> list[RoleInferenceResult]:
        """バッチ処理で複数エージェントの役割を推論する。"""
        results = []
        for batch in self._chunk(requests, self.batch_size):
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
        return results

    async def _process_batch(
        self, batch: list[RoleInferenceRequest]
    ) -> list[RoleInferenceResult]:
        """1バッチ分のLLM推論を実行する。"""
        tasks = []
        for req in batch:
            prompt = ROLE_INFERENCE_PROMPT.format(
                goals="\n".join(f"- {g}" for g in req.recent_goals),
                available_roles=", ".join(r.value for r in AgentRole),
            )
            tasks.append(self.llm_client.generate(prompt))

        responses = await asyncio.gather(*tasks)
        return [
            self._parse_response(req.agent_id, resp)
            for req, resp in zip(batch, responses)
        ]

    def _chunk(self, items, size):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _parse_response(self, agent_id, response) -> RoleInferenceResult:
        """LLMレスポンスをパースしてRoleInferenceResultに変換する。"""
        # JSON解析とバリデーション
        data = json.loads(response)
        return RoleInferenceResult(
            agent_id=agent_id,
            inferred_role=AgentRole(data["role"]),
            confidence=float(data["confidence"]),
            reasoning=data["reasoning"],
        )
```

#### コスト最適化

| 手法 | 説明 | 削減効果 |
|---|---|---|
| バッチ推論 | 複数エージェントをまとめてAPI呼出 | API呼出回数の削減 |
| キャッシュ | 同一目標セットの結果を再利用 | 重複推論の排除 |
| サンプリング | 全エージェントでなく代表サンプルを分析 | 分析対象の削減 |
| 軽量モデル | 高信頼度のケースには小型モデル使用 | コスト/トークン削減 |
| ローカルモデル | 微調整されたローカルモデルでの代替 | API費用の排除 |

### 2.3 行動-役割相関分析

推論された役割と実際のアクション頻度の相関を定量化する。

#### 実装設計

```python
import numpy as np
from scipy import stats

@dataclass
class ActionRoleCorrelation:
    role: str
    action_type: str
    correlation: float      # ピアソン相関係数
    p_value: float
    sample_size: int

class ActionRoleAnalyzer:
    """
    各役割に期待されるアクションの頻度パターンとの相関を計算する。
    論文では役割内・役割間のアクションカウントを正規化して比較。
    """

    # 各役割の期待されるアクション分布（正規化済み）
    EXPECTED_PATTERNS: dict[str, dict[str, float]] = {
        "farmer": {"collect_seeds": 0.4, "plant": 0.3, "harvest": 0.2, "trade": 0.1},
        "miner": {"mine": 0.5, "smelt": 0.2, "craft_tool": 0.2, "trade": 0.1},
        "guard": {"craft_fence": 0.3, "patrol": 0.3, "craft_weapon": 0.2, "guard": 0.2},
        "explorer": {"move": 0.4, "explore": 0.3, "map": 0.2, "report": 0.1},
    }

    def compute_correlation(
        self,
        agent_actions: dict[str, int],  # {action_type: count}
        inferred_role: str,
    ) -> float:
        """
        エージェントの実際のアクション分布と
        期待されるアクション分布の相関を計算する。
        """
        expected = self.EXPECTED_PATTERNS.get(inferred_role, {})
        if not expected:
            return 0.0

        all_actions = set(agent_actions.keys()) | set(expected.keys())
        total_actions = sum(agent_actions.values())
        if total_actions == 0:
            return 0.0

        observed_vec = []
        expected_vec = []
        for action in sorted(all_actions):
            observed_vec.append(agent_actions.get(action, 0) / total_actions)
            expected_vec.append(expected.get(action, 0.0))

        if len(observed_vec) < 2:
            return 0.0

        corr, p_value = stats.pearsonr(observed_vec, expected_vec)
        return corr
```

### 2.4 役割持続時間の追跡

#### 実装設計

```python
@dataclass
class RolePersistence:
    agent_id: str
    role: str
    start_time: float
    end_time: float | None  # Noneは現在も継続中
    duration: float

class RolePersistenceTracker:
    """エージェントの役割が時間経過とともにどの程度安定するかを追跡する。"""

    def __init__(self):
        self._current_roles: dict[str, tuple[str, float]] = {}  # agent_id -> (role, start)
        self._history: list[RolePersistence] = []

    def update(self, agent_id: str, role: str, timestamp: float):
        """役割の更新をイベントとして記録する。"""
        if agent_id in self._current_roles:
            current_role, start = self._current_roles[agent_id]
            if current_role != role:
                # 役割変更 -> 前の役割期間を記録
                self._history.append(RolePersistence(
                    agent_id=agent_id,
                    role=current_role,
                    start_time=start,
                    end_time=timestamp,
                    duration=timestamp - start,
                ))
                self._current_roles[agent_id] = (role, timestamp)
        else:
            self._current_roles[agent_id] = (role, timestamp)

    def get_average_persistence(self) -> float:
        """全エージェントの平均役割持続時間を返す。"""
        if not self._history:
            return 0.0
        return sum(r.duration for r in self._history) / len(self._history)

    def get_stability_score(self, agent_id: str) -> float:
        """
        エージェントの役割安定性スコア（0-1）を返す。
        長い持続時間 / 合計時間 = 安定性
        """
        agent_records = [r for r in self._history if r.agent_id == agent_id]
        if not agent_records:
            return 1.0  # 変更なし = 最大安定
        total_time = agent_records[-1].end_time - agent_records[0].start_time
        if total_time == 0:
            return 1.0
        max_duration = max(r.duration for r in agent_records)
        return max_duration / total_time
```

---

## 3. 統治指標の実装

### 3.1 税遵守率の計測

#### 実験パラメータ（論文準拠）

| パラメータ | 値 |
|---|---|
| 構成員数 | 25体 |
| 課税シーズン数 | 10回（修正前5回 + 修正後5回） |
| 課税ウィンドウ | 各20秒 |
| 初期税率 | 所持品の20% |

#### 実装設計

```python
from dataclasses import dataclass, field

@dataclass
class TaxEvent:
    agent_id: str
    timestamp: float
    season: int
    required_amount: float   # 納付すべき額
    actual_amount: float     # 実際に納付した額
    compliance: bool         # 遵守したか

@dataclass
class TaxComplianceMetrics:
    overall_rate: float            # 全体遵守率
    pre_amendment_rate: float      # 修正前遵守率
    post_amendment_rate: float     # 修正後遵守率
    rate_by_season: list[float]    # シーズン別遵守率
    amendment_impact: float        # 修正前後の変化量

class TaxComplianceTracker:
    """
    税遵守率をリアルタイムで計測する。
    論文では120秒間隔の課税シーズンで遵守行動を記録。
    """

    def __init__(self, initial_tax_rate: float = 0.20, num_pre_seasons: int = 5):
        self.tax_rate = initial_tax_rate
        self.num_pre_seasons = num_pre_seasons
        self.events: list[TaxEvent] = []

    def record_tax_event(
        self, agent_id: str, timestamp: float, season: int,
        inventory_value: float, paid_amount: float,
    ):
        """課税イベントを記録する。"""
        required = inventory_value * self.tax_rate
        # 遵守判定: 要求額の90%以上を納付した場合に遵守とみなす
        compliance = paid_amount >= required * 0.9
        self.events.append(TaxEvent(
            agent_id=agent_id,
            timestamp=timestamp,
            season=season,
            required_amount=required,
            actual_amount=paid_amount,
            compliance=compliance,
        ))

    def update_tax_rate(self, new_rate: float):
        """憲法修正後の新税率を設定する。"""
        self.tax_rate = new_rate

    def compute_metrics(self) -> TaxComplianceMetrics:
        """遵守率メトリクスを計算する。"""
        if not self.events:
            return TaxComplianceMetrics(0, 0, 0, [], 0)

        pre = [e for e in self.events if e.season <= self.num_pre_seasons]
        post = [e for e in self.events if e.season > self.num_pre_seasons]

        pre_rate = sum(e.compliance for e in pre) / len(pre) if pre else 0
        post_rate = sum(e.compliance for e in post) / len(post) if post else 0

        seasons = sorted(set(e.season for e in self.events))
        rate_by_season = []
        for s in seasons:
            season_events = [e for e in self.events if e.season == s]
            rate_by_season.append(
                sum(e.compliance for e in season_events) / len(season_events)
            )

        return TaxComplianceMetrics(
            overall_rate=sum(e.compliance for e in self.events) / len(self.events),
            pre_amendment_rate=pre_rate,
            post_amendment_rate=post_rate,
            rate_by_season=rate_by_season,
            amendment_impact=post_rate - pre_rate,
        )
```

### 3.2 憲法修正プロセスの自動化

#### プロセスフロー

```
Phase 1: フィードバック収集（300-360秒）
  ├─ 構成員がフィードバックを提出
  └─ インフルエンサーが意見を拡散

Phase 2: 修正案処理（360-420秒）
  ├─ 選挙管理者がフィードバックを集約
  └─ 修正案をフォーマット化

Phase 3: 投票（420秒）
  ├─ 構成員が賛否投票
  └─ 過半数で可決

Phase 4: 憲法更新（420-600秒）
  └─ 新ルールを全エージェントに通知・適用
```

#### 実装設計

```python
from enum import Enum

class VoteChoice(Enum):
    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"

@dataclass
class Amendment:
    id: str
    proposer: str
    description: str
    old_value: str
    new_value: str
    timestamp: float

@dataclass
class VoteRecord:
    voter_id: str
    amendment_id: str
    choice: VoteChoice
    timestamp: float

@dataclass
class AmendmentResult:
    amendment: Amendment
    votes_for: int
    votes_against: int
    votes_abstain: int
    passed: bool
    participation_rate: float

class ConstitutionManager:
    """
    憲法の管理と修正プロセスを制御する。
    論文の民主的プロセスを再現。
    """

    def __init__(self, initial_rules: dict[str, str], voter_ids: list[str]):
        self.rules = dict(initial_rules)
        self.voter_ids = set(voter_ids)
        self.amendments: list[Amendment] = []
        self.vote_records: list[VoteRecord] = []
        self.results: list[AmendmentResult] = []

    def submit_amendment(
        self, proposer: str, rule_key: str, new_value: str, timestamp: float,
    ) -> Amendment:
        """修正案を提出する。"""
        amendment = Amendment(
            id=f"amend_{len(self.amendments) + 1}",
            proposer=proposer,
            description=f"Change {rule_key}",
            old_value=self.rules.get(rule_key, ""),
            new_value=new_value,
            timestamp=timestamp,
        )
        self.amendments.append(amendment)
        return amendment

    def cast_vote(self, voter_id: str, amendment_id: str, choice: VoteChoice, timestamp: float):
        """投票を記録する。"""
        if voter_id not in self.voter_ids:
            raise ValueError(f"Invalid voter: {voter_id}")
        self.vote_records.append(VoteRecord(voter_id, amendment_id, choice, timestamp))

    def tally_votes(self, amendment_id: str) -> AmendmentResult:
        """投票結果を集計し、可決/否決を判定する。"""
        amendment = next(a for a in self.amendments if a.id == amendment_id)
        votes = [v for v in self.vote_records if v.amendment_id == amendment_id]

        votes_for = sum(1 for v in votes if v.choice == VoteChoice.FOR)
        votes_against = sum(1 for v in votes if v.choice == VoteChoice.AGAINST)
        votes_abstain = sum(1 for v in votes if v.choice == VoteChoice.ABSTAIN)
        total_voters = len(self.voter_ids)
        participation = len(votes) / total_voters if total_voters > 0 else 0

        passed = votes_for > votes_against  # 単純過半数

        result = AmendmentResult(
            amendment=amendment,
            votes_for=votes_for,
            votes_against=votes_against,
            votes_abstain=votes_abstain,
            passed=passed,
            participation_rate=participation,
        )
        self.results.append(result)

        if passed:
            # 憲法を更新
            rule_key = amendment.description.replace("Change ", "")
            self.rules[rule_key] = amendment.new_value

        return result
```

### 3.3 インフルエンサー影響度の測定

```python
@dataclass
class InfluencerImpact:
    influencer_id: str
    stance: str  # "pro_tax" or "anti_tax"
    influenced_agents: list[str]
    vote_alignment_rate: float  # 影響を受けたエージェントの投票一致率
    sentiment_shift: float      # 影響前後の感情変化量

class InfluencerAnalyzer:
    """
    インフルエンサーが構成員の投票行動にどの程度影響を与えたかを測定する。
    論文では賛税/反税インフルエンサーの3体を配置。
    """

    def measure_impact(
        self,
        influencer_id: str,
        influencer_stance: str,
        interaction_log: list[dict],  # インフルエンサーと構成員の会話記録
        vote_results: list[VoteRecord],
    ) -> InfluencerImpact:
        """
        インフルエンサーとの会話履歴と投票結果を突合し影響度を算出する。
        """
        # インフルエンサーと会話したエージェントを特定
        interacted = set()
        for entry in interaction_log:
            if entry["speaker"] == influencer_id:
                interacted.add(entry["listener"])
            elif entry["listener"] == influencer_id:
                interacted.add(entry["speaker"])

        # 会話したエージェントの投票傾向を分析
        aligned = 0
        total = 0
        expected_choice = (
            VoteChoice.AGAINST if influencer_stance == "anti_tax" else VoteChoice.FOR
        )
        for vote in vote_results:
            if vote.voter_id in interacted:
                total += 1
                if vote.choice == expected_choice:
                    aligned += 1

        alignment_rate = aligned / total if total > 0 else 0.0

        return InfluencerImpact(
            influencer_id=influencer_id,
            stance=influencer_stance,
            influenced_agents=list(interacted),
            vote_alignment_rate=alignment_rate,
            sentiment_shift=0.0,  # 別途感情分析で算出
        )
```

---

## 4. 文化伝播指標の実装

### 4.1 ミーム検出アルゴリズム

論文では2つの手法を併用している: キーワードベース検出とLLMベース分析。

#### キーワードベース検出

```python
import re
from dataclasses import dataclass, field

@dataclass
class MemeDetection:
    meme_id: str
    keyword: str
    agent_id: str
    timestamp: float
    location: tuple[float, float]  # (x, z) in Minecraft coords
    context: str  # 検出元テキスト

class KeywordMemeDetector:
    """
    キーワードマッチングによるミーム検出。
    論文のキーワード例: "eco", "dance", "meditation", "Pastafarian"
    """

    def __init__(self, meme_keywords: dict[str, list[str]]):
        """
        Args:
            meme_keywords: {meme_id: [keywords]}
                例: {"eco": ["eco", "environment", "green"],
                     "dance": ["dance", "dancing", "choreography"]}
        """
        self.meme_keywords = meme_keywords
        self._patterns = {
            meme_id: re.compile("|".join(re.escape(kw) for kw in keywords), re.IGNORECASE)
            for meme_id, keywords in meme_keywords.items()
        }

    def detect(
        self, agent_id: str, text: str, timestamp: float, location: tuple[float, float],
    ) -> list[MemeDetection]:
        """テキストからミームを検出する。"""
        detections = []
        for meme_id, pattern in self._patterns.items():
            match = pattern.search(text)
            if match:
                detections.append(MemeDetection(
                    meme_id=meme_id,
                    keyword=match.group(),
                    agent_id=agent_id,
                    timestamp=timestamp,
                    location=location,
                    context=text,
                ))
        return detections
```

#### LLMベースミーム分析

```python
MEME_ANALYSIS_PROMPT = """
以下のエージェントの目標履歴を分析し、文化的ミーム（繰り返し出現するテーマ、
共有された行動パターン、流行）を検出してください。

エージェント目標:
{goals}

出力形式（JSON配列）:
[
  {{
    "meme": "<ミーム名>",
    "confidence": <0.0-1.0>,
    "evidence": ["<根拠1>", "<根拠2>"]
  }}
]
"""

class LLMMemeAnalyzer:
    """
    LLMを用いた高次ミーム検出。
    論文では2時間ごとにエージェントの統合目標を要約して分析。
    """

    def __init__(self, llm_client, analysis_interval: float = 7200.0):
        self.llm_client = llm_client
        self.analysis_interval = analysis_interval

    async def analyze_agent_goals(
        self, agent_id: str, goal_history: list[str],
    ) -> list[dict]:
        """エージェントの目標履歴からミームを抽出する。"""
        prompt = MEME_ANALYSIS_PROMPT.format(
            goals="\n".join(f"- {g}" for g in goal_history)
        )
        response = await self.llm_client.generate(prompt)
        return json.loads(response)

    async def analyze_population(
        self, all_goals: dict[str, list[str]],
    ) -> dict[str, int]:
        """全エージェントの目標を分析し、ミームの出現頻度を集計する。"""
        meme_counts: dict[str, int] = {}
        for agent_id, goals in all_goals.items():
            memes = await self.analyze_agent_goals(agent_id, goals)
            for m in memes:
                name = m["meme"]
                meme_counts[name] = meme_counts.get(name, 0) + 1
        return meme_counts
```

### 4.2 地理的伝播範囲の可視化

#### ヒートマップ生成

```python
import numpy as np

@dataclass
class SpatialSpreadMetrics:
    total_area: float         # 伝播した総面積
    density_map: np.ndarray   # 密度マップ（2Dグリッド）
    centroid: tuple[float, float]  # 伝播の重心
    spread_radius: float      # 伝播半径

class GeographicSpreadVisualizer:
    """
    ミーム/宗教の地理的伝播をヒートマップで可視化する。
    論文の1000x1200ブロック環境を対象。
    """

    def __init__(self, world_size: tuple[int, int] = (1000, 1200), grid_resolution: int = 50):
        self.world_size = world_size
        self.grid_resolution = grid_resolution
        self.grid_x = world_size[0] // grid_resolution
        self.grid_z = world_size[1] // grid_resolution

    def compute_density_map(
        self, detections: list[MemeDetection],
    ) -> np.ndarray:
        """検出イベントから密度マップを生成する。"""
        density = np.zeros((self.grid_x, self.grid_z))
        for d in detections:
            gx = min(int(d.location[0] / self.grid_resolution), self.grid_x - 1)
            gz = min(int(d.location[1] / self.grid_resolution), self.grid_z - 1)
            density[gx][gz] += 1
        return density

    def compute_spread_over_time(
        self, detections: list[MemeDetection], time_window: float = 300.0,
    ) -> list[tuple[float, SpatialSpreadMetrics]]:
        """時間窓ごとの伝播範囲を計算する。"""
        if not detections:
            return []

        sorted_detections = sorted(detections, key=lambda d: d.timestamp)
        min_t = sorted_detections[0].timestamp
        max_t = sorted_detections[-1].timestamp

        results = []
        t = min_t
        while t <= max_t:
            window = [d for d in sorted_detections if d.timestamp <= t]
            density = self.compute_density_map(window)
            active_cells = np.count_nonzero(density)
            area = active_cells * (self.grid_resolution ** 2)

            # 重心計算
            if window:
                cx = np.mean([d.location[0] for d in window])
                cz = np.mean([d.location[1] for d in window])
            else:
                cx, cz = 0, 0

            results.append((t, SpatialSpreadMetrics(
                total_area=area,
                density_map=density,
                centroid=(cx, cz),
                spread_radius=np.sqrt(area / np.pi) if area > 0 else 0,
            )))
            t += time_window

        return results

    def render_heatmap(self, density: np.ndarray, title: str = "Meme Spread"):
        """matplotlibでヒートマップを描画する。"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        im = ax.imshow(
            density.T, origin="lower", cmap="YlOrRd",
            extent=[0, self.world_size[0], 0, self.world_size[1]],
            aspect="auto",
        )
        ax.set_xlabel("X (blocks)")
        ax.set_ylabel("Z (blocks)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Detection count")
        return fig
```

### 4.3 宗教伝播追跡

#### SIRモデルとの対応

論文では宗教伝播が疫学モデル（SIR）と構造的に類似していることを指摘。

| SIRモデル | 宗教伝播 | 状態 |
|---|---|---|
| S (Susceptible) | 未接触 | 宗教的キーワードなし |
| I (Infected) | 改宗者 | 宗教的キーワードを使用 |
| R (Recovered) | 脱宗教 | キーワード使用を停止（論文では未観察） |

#### 実装設計

```python
from enum import Enum

class ConversionLevel(Enum):
    NONE = "none"
    INDIRECT = "indirect"  # "Pasta", "Spaghetti"
    DIRECT = "direct"      # "Pastafarian", "Spaghetti Monster"

@dataclass
class ConversionEvent:
    agent_id: str
    level: ConversionLevel
    timestamp: float
    location: tuple[float, float]
    source_agent: str | None  # 誰から影響を受けたか（推定）
    keyword_matched: str

class ReligionSpreadTracker:
    """
    パスタファリアニズムの伝播を追跡する。
    論文の改宗レベル分類を再現。
    """

    DIRECT_KEYWORDS = ["pastafarian", "spaghetti monster", "flying spaghetti"]
    INDIRECT_KEYWORDS = ["pasta", "spaghetti", "noodle", "marinara"]

    def __init__(self, priest_ids: set[str]):
        self.priest_ids = priest_ids
        self.conversions: list[ConversionEvent] = []
        self._agent_status: dict[str, ConversionLevel] = {}

    def check_utterance(
        self, agent_id: str, text: str, timestamp: float, location: tuple[float, float],
    ) -> ConversionEvent | None:
        """エージェントの発話を分析し、改宗イベントを検出する。"""
        if agent_id in self.priest_ids:
            return None  # 司祭は除外

        text_lower = text.lower()

        # 直接改宗の判定
        for kw in self.DIRECT_KEYWORDS:
            if kw in text_lower:
                return self._record_conversion(
                    agent_id, ConversionLevel.DIRECT, timestamp, location, kw,
                )

        # 間接改宗の判定
        for kw in self.INDIRECT_KEYWORDS:
            if kw in text_lower:
                return self._record_conversion(
                    agent_id, ConversionLevel.INDIRECT, timestamp, location, kw,
                )

        return None

    def _record_conversion(
        self, agent_id, level, timestamp, location, keyword,
    ) -> ConversionEvent:
        """改宗イベントを記録する。"""
        # 上位レベルへのアップグレードのみ記録
        current = self._agent_status.get(agent_id, ConversionLevel.NONE)
        if level == ConversionLevel.DIRECT or current == ConversionLevel.NONE:
            self._agent_status[agent_id] = level

        event = ConversionEvent(
            agent_id=agent_id,
            level=level,
            timestamp=timestamp,
            location=location,
            source_agent=None,  # 近接エージェントから推定
            keyword_matched=keyword,
        )
        self.conversions.append(event)
        return event

    def get_conversion_counts(self) -> dict[str, int]:
        """現在の改宗者数を返す。"""
        counts = {"direct": 0, "indirect": 0, "none": 0}
        for level in self._agent_status.values():
            counts[level.value] = counts.get(level.value, 0) + 1
        return counts

    def get_conversion_timeline(
        self, interval: float = 300.0,
    ) -> list[tuple[float, dict[str, int]]]:
        """時系列での改宗者数推移を返す。"""
        if not self.conversions:
            return []

        sorted_events = sorted(self.conversions, key=lambda e: e.timestamp)
        min_t = sorted_events[0].timestamp
        max_t = sorted_events[-1].timestamp

        timeline = []
        t = min_t
        while t <= max_t:
            cumulative = {"direct": 0, "indirect": 0}
            seen_agents: dict[str, ConversionLevel] = {}
            for e in sorted_events:
                if e.timestamp <= t:
                    current = seen_agents.get(e.agent_id, ConversionLevel.NONE)
                    if e.level.value > current.value or current == ConversionLevel.NONE:
                        seen_agents[e.agent_id] = e.level
            for level in seen_agents.values():
                if level != ConversionLevel.NONE:
                    cumulative[level.value] += 1
            timeline.append((t, cumulative))
            t += interval

        return timeline
```

### 4.4 時系列分析

```python
class TimeSeriesAnalyzer:
    """ミームおよび宗教伝播の時系列パターンを分析する。"""

    @staticmethod
    def compute_growth_rate(
        timeline: list[tuple[float, int]],
    ) -> list[tuple[float, float]]:
        """時系列データから成長率を計算する。"""
        rates = []
        for i in range(1, len(timeline)):
            dt = timeline[i][0] - timeline[i - 1][0]
            if dt > 0 and timeline[i - 1][1] > 0:
                rate = (timeline[i][1] - timeline[i - 1][1]) / (timeline[i - 1][1] * dt)
                rates.append((timeline[i][0], rate))
        return rates

    @staticmethod
    def detect_saturation(
        timeline: list[tuple[float, int]], window: int = 5, threshold: float = 0.01,
    ) -> float | None:
        """
        成長が飽和したタイムスタンプを検出する。
        論文では宗教伝播が2時間で飽和しなかったことを報告。
        """
        if len(timeline) < window:
            return None
        for i in range(window, len(timeline)):
            recent = timeline[i - window : i]
            values = [v for _, v in recent]
            if max(values) - min(values) < threshold * max(values):
                return timeline[i][0]
        return None  # 飽和なし

    @staticmethod
    def fit_sir_model(
        timeline: list[tuple[float, int]], population: int,
    ) -> dict:
        """
        SIRモデルをフィッティングし、伝播パラメータを推定する。
        beta (感染率), gamma (回復率) を推定。
        """
        from scipy.optimize import curve_fit

        times = np.array([t for t, _ in timeline])
        infected = np.array([v for _, v in timeline])
        times = times - times[0]  # 正規化

        def sir_infected(t, beta, gamma):
            from scipy.integrate import odeint
            def deriv(y, t, N, beta, gamma):
                S, I, R = y
                dS = -beta * S * I / N
                dI = beta * S * I / N - gamma * I
                dR = gamma * I
                return [dS, dI, dR]
            I0 = infected[0] if infected[0] > 0 else 1
            y0 = [population - I0, I0, 0]
            sol = odeint(deriv, y0, t, args=(population, beta, gamma))
            return sol[:, 1]

        try:
            popt, _ = curve_fit(sir_infected, times, infected, p0=[0.3, 0.1], maxfev=5000)
            return {"beta": popt[0], "gamma": popt[1], "R0": popt[0] / popt[1]}
        except Exception:
            return {"beta": None, "gamma": None, "R0": None}
```

---

## 5. インフラ発展指標の実装

### 5.1 概要

論文の文明的ベンチマーク4軸の1つである「インフラ発展（Infrastructure Development）」は、エージェント社会における制度の機能的運用度を測定する。論文（Section 5）では具体的な定量指標の詳細な定義は示されていないが、文明の物理的・社会的インフラがどの程度発展したかを評価する次元として位置づけられている。

> **論文との対応**: 05-civilization.md L187 の「インフラ発展 -- 制度の機能的運用度」に対応する実装。論文では定量的指標が十分に定義されていないため、以下の指標は人間文明のインフラ発展指標を参考にした**独自拡張**を含む。

### 5.2 測定指標

| 指標 | 測定方法 | 文明的意味 |
|---|---|---|
| **建造物数・種類** | ブロック配置イベントの集計 | 物理的インフラの量と多様性 |
| **共有施設の利用率** | 特定座標範囲でのエージェント滞在・アクション集計 | 制度が実際に機能しているか |
| **空間的接続性** | 建造物間のパス到達性分析 | インフラのネットワーク性 |
| **農地・生産施設の面積** | 農耕系ブロック（耕地、作物）の面積集計 | 経済的インフラの規模 |
| **インフラ投資率** | 建築系アクション / 全アクション の比率 | 社会のインフラ志向度 |

### 5.3 実装設計

```python
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

@dataclass
class Structure:
    structure_id: str
    structure_type: str  # "house", "farm", "road", "bridge", "shared_facility"
    builder_id: str
    location: tuple[float, float, float]  # 中心座標
    block_count: int
    timestamp_start: float
    timestamp_end: float | None  # 建設完了時刻

@dataclass
class InfrastructureMetrics:
    total_structures: int
    structure_types: dict[str, int]       # 種類別の建造物数
    shared_facility_usage_rate: float     # 共有施設利用率 (0-1)
    spatial_connectivity: float           # 空間的接続性スコア (0-1)
    farmland_area: int                    # 農地ブロック数
    infrastructure_investment_rate: float # インフラ投資率
    builders_count: int                   # 建築に携わったエージェント数
    builder_ratio: float                  # 建築者 / 全エージェント

class InfrastructureTracker:
    """
    エージェント社会のインフラ発展を追跡する。
    論文の「制度の機能的運用度」を具体的な指標で計測。

    注意: 論文では定量的指標が詳細に定義されていないため、
    以下の実装は人間文明のインフラ指標を参考にした独自設計を含む。
    """

    # Minecraftにおけるインフラ関連ブロック
    FARM_BLOCKS = {"farmland", "wheat", "carrots", "potatoes", "beetroots", "melon_stem", "pumpkin_stem"}
    ROAD_BLOCKS = {"cobblestone", "stone_bricks", "gravel", "cobblestone_slab"}
    BUILDING_BLOCKS = {"planks", "oak_planks", "spruce_planks", "cobblestone", "stone_bricks", "glass"}

    def __init__(self, total_agents: int):
        self.total_agents = total_agents
        self._structures: list[Structure] = []
        self._block_placements: list[dict] = []  # 全ブロック配置イベント
        self._facility_visits: dict[str, list[str]] = defaultdict(list)  # facility_id -> [agent_ids]
        self._total_actions: int = 0
        self._build_actions: int = 0
        self._builders: set[str] = set()

    def record_block_placement(
        self, agent_id: str, block_type: str,
        location: tuple[float, float, float], timestamp: float,
    ):
        """ブロック配置イベントを記録する。"""
        self._block_placements.append({
            "agent_id": agent_id,
            "block_type": block_type,
            "location": location,
            "timestamp": timestamp,
        })
        self._build_actions += 1
        self._builders.add(agent_id)

    def record_action(self):
        """全アクション数をインクリメントする（インフラ投資率計算用）。"""
        self._total_actions += 1

    def record_facility_visit(self, facility_id: str, agent_id: str):
        """共有施設への訪問を記録する。"""
        self._facility_visits[facility_id].append(agent_id)

    def register_structure(self, structure: Structure):
        """識別された建造物を登録する。"""
        self._structures.append(structure)

    def compute_farmland_area(self) -> int:
        """農地関連ブロックの総数を返す。"""
        return sum(
            1 for bp in self._block_placements
            if bp["block_type"] in self.FARM_BLOCKS
        )

    def compute_shared_facility_usage(self) -> float:
        """
        共有施設の平均利用率を計算する。
        利用率 = 施設を利用したユニークエージェント数 / 全エージェント数
        """
        if not self._facility_visits:
            return 0.0
        usage_rates = []
        for facility_id, visitors in self._facility_visits.items():
            unique_visitors = len(set(visitors))
            usage_rates.append(unique_visitors / self.total_agents)
        return np.mean(usage_rates)

    def compute_spatial_connectivity(self) -> float:
        """
        建造物間の空間的接続性を計算する。
        接続性 = 一定距離内にある建造物ペアの割合。
        """
        if len(self._structures) < 2:
            return 0.0

        connection_distance = 50.0  # ブロック単位の接続判定距離
        connected_pairs = 0
        total_pairs = 0

        for i in range(len(self._structures)):
            for j in range(i + 1, len(self._structures)):
                total_pairs += 1
                loc_a = self._structures[i].location
                loc_b = self._structures[j].location
                dist = np.sqrt(
                    (loc_a[0] - loc_b[0]) ** 2 +
                    (loc_a[2] - loc_b[2]) ** 2  # XZ平面距離
                )
                if dist <= connection_distance:
                    connected_pairs += 1

        return connected_pairs / total_pairs if total_pairs > 0 else 0.0

    def compute_metrics(self) -> InfrastructureMetrics:
        """インフラ発展の総合メトリクスを計算する。"""
        type_counts: dict[str, int] = defaultdict(int)
        for s in self._structures:
            type_counts[s.structure_type] += 1

        return InfrastructureMetrics(
            total_structures=len(self._structures),
            structure_types=dict(type_counts),
            shared_facility_usage_rate=self.compute_shared_facility_usage(),
            spatial_connectivity=self.compute_spatial_connectivity(),
            farmland_area=self.compute_farmland_area(),
            infrastructure_investment_rate=(
                self._build_actions / self._total_actions
                if self._total_actions > 0 else 0.0
            ),
            builders_count=len(self._builders),
            builder_ratio=len(self._builders) / self.total_agents if self.total_agents > 0 else 0.0,
        )
```

### 5.4 解釈ガイドライン

| 指標 | 低い値 | 高い値 |
|---|---|---|
| 建造物数 | 遊牧的・未定着 | 都市化・定住化 |
| 種類の多様性 | 単機能的 | 多機能的社会 |
| 共有施設利用率 | 制度が形骸化 | 制度が機能的に運用 |
| 空間的接続性 | 散在・孤立的 | ネットワーク化・計画的 |
| インフラ投資率 | 個人的活動が主 | 社会的投資が活発 |

### 5.5 制約と今後の拡張

論文ではインフラ発展の定量的指標が詳細に定義されていないため、上記の実装は以下の点で独自拡張を含む:

- **建造物の自動識別**: 隣接ブロックのクラスタリングによる建造物検出は、閾値設定に依存する。Phase 0ではエージェントの明示的な建築目標と対応させて手動ラベリングする方針とする。
- **共有施設の定義**: どのブロック構造が「共有施設」であるかの判定はドメイン知識に依存する。初期実装では、エージェントの目標に「共有」「public」等のキーワードが含まれる建築物を共有施設として扱う。
- **接続性の閾値**: 50ブロックの接続判定距離は調整が必要なパラメータ。

---

## 6. 社会的認知精度の計測

### 6.1 概要

社会的認知精度は、エージェントが他のエージェントの好感度（likeability）をどれだけ正確に推測できるかを測定する指標。論文では50体のエージェントによる4時間超のシミュレーションで評価し、ピアソン相関 r=0.807 を達成した。

| 条件 | 回帰勾配 | ピアソン相関係数 |
|---|---|---|
| 社会認識モジュールあり | 0.373 | **r = 0.807** |
| 社会認識モジュールなし | 0.161 | r = 0.617 |

### 6.2 観察者閾値メカニズム

論文の重要な発見は、多くのエージェントが評価に参加するほど精度が向上する「集合知（Wisdom of Crowds）」効果である。

**「観察者閾値（threshold of observers）」の定義**:

論文における「観察者閾値 N」とは、**各ターゲットエージェントについて、少なくとも N 人の他のエージェントが好感度を評価しているターゲットのみを分析対象に含める**というフィルタリング条件である。これはサンプル品質のフィルタであり、閾値を上げるほど多くのエージェントから評価されたターゲットだけが残るため集合知効果により精度が向上するが、同時にサンプルサイズが減少する。

具体的には:
- 観察者閾値 = 1: 1人以上から評価されたターゲット全てを含む（最大範囲、低精度）
- 観察者閾値 = 5: 5人以上から評価されたターゲットのみ含む（論文のデフォルト）
- 観察者閾値 = 11: 11人以上から評価されたターゲットのみ含む（最小範囲、高精度）

> **注意**: この「観察者閾値」は、エージェントの対話回数の閾値（06-social-cognition.mdの`aggregate_likeability`における`min_observers`パラメータ）とは異なる概念である。前者は**評価分析時のサンプルフィルタリング条件**であり、後者は**社会認識モジュール内部での集約条件**である。

| 観察者閾値 | 相関係数 (r) | サンプルサイズ (n) | 解釈 |
|---|---|---|---|
| 1（最小） | 0.646 | 46 | 少数評価、低精度だが広範囲 |
| 5 | 0.807 | - | 中程度の閾値でバランス |
| 11（最大） | **0.907** | 18 | 高精度だがサンプル減少 |

### 6.3 実装設計

```python
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class SocialCognitionResult:
    pearson_r: float
    p_value: float
    regression_slope: float
    regression_intercept: float
    sample_size: int
    observer_threshold: int

class SocialCognitionEvaluator:
    """
    エージェントの社会的認知精度を評価する。
    各エージェントが他エージェントに対して推測する好感度と、
    実際の好感度スコアの相関を計算する。
    """

    def __init__(self):
        # estimated_scores[a][b] = エージェントaがエージェントbに推測するスコア
        self._estimated_scores: dict[str, dict[str, float]] = {}
        # actual_scores[a][b] = エージェントaのエージェントbに対する実際のスコア
        self._actual_scores: dict[str, dict[str, float]] = {}

    def record_estimated_score(
        self, observer_id: str, target_id: str, score: float,
    ):
        """エージェントが推測した好感度を記録する。"""
        if observer_id not in self._estimated_scores:
            self._estimated_scores[observer_id] = {}
        self._estimated_scores[observer_id][target_id] = score

    def record_actual_score(
        self, agent_id: str, target_id: str, score: float,
    ):
        """実際の好感度スコアを記録する。"""
        if agent_id not in self._actual_scores:
            self._actual_scores[agent_id] = {}
        self._actual_scores[agent_id][target_id] = score

    def compute_accuracy(
        self, observer_threshold: int = 5,
    ) -> SocialCognitionResult:
        """
        社会的認知精度を計算する。

        Args:
            observer_threshold: あるターゲットエージェントを分析対象に含めるために
                必要な、そのターゲットを評価した観察者の最小数。
                論文の「threshold of observers」に対応するサンプルフィルタリング条件。
                閾値を上げるとサンプル数は減少するが、集合知効果により精度が向上する。
                論文のデフォルトは5。
        """
        # ターゲットごとに、複数の観察者の推測値を集約
        target_estimates: dict[str, list[float]] = {}
        target_actuals: dict[str, list[float]] = {}

        for observer_id, estimates in self._estimated_scores.items():
            for target_id, est_score in estimates.items():
                if target_id not in target_estimates:
                    target_estimates[target_id] = []
                target_estimates[target_id].append(est_score)

                # 対応する実際のスコア（ターゲットが観察者をどう思うか）
                actual = self._actual_scores.get(target_id, {}).get(observer_id)
                if actual is not None:
                    if target_id not in target_actuals:
                        target_actuals[target_id] = []
                    target_actuals[target_id].append(actual)

        # 観察者閾値を適用
        estimated_means = []
        actual_means = []
        for target_id in target_estimates:
            if (
                len(target_estimates[target_id]) >= observer_threshold
                and target_id in target_actuals
                and len(target_actuals[target_id]) >= observer_threshold
            ):
                estimated_means.append(np.mean(target_estimates[target_id]))
                actual_means.append(np.mean(target_actuals[target_id]))

        if len(estimated_means) < 2:
            return SocialCognitionResult(
                pearson_r=0.0, p_value=1.0,
                regression_slope=0.0, regression_intercept=0.0,
                sample_size=len(estimated_means),
                observer_threshold=observer_threshold,
            )

        # ピアソン相関の計算
        r, p_value = stats.pearsonr(estimated_means, actual_means)

        # 線形回帰
        slope, intercept, _, _, _ = stats.linregress(estimated_means, actual_means)

        return SocialCognitionResult(
            pearson_r=r,
            p_value=p_value,
            regression_slope=slope,
            regression_intercept=intercept,
            sample_size=len(estimated_means),
            observer_threshold=observer_threshold,
        )

    def compute_threshold_sweep(
        self, min_threshold: int = 1, max_threshold: int = 15,
    ) -> list[SocialCognitionResult]:
        """
        観察者閾値を段階的に変化させ、精度の推移を分析する。
        論文のFigure再現に使用。
        """
        return [
            self.compute_accuracy(observer_threshold=t)
            for t in range(min_threshold, max_threshold + 1)
        ]
```

### 6.4 アブレーション比較

```python
class SocialCognitionAblation:
    """
    社会認識モジュールの有無による認知精度の比較。
    論文ではモジュール除去時にr=0.807->0.617に低下。
    """

    @staticmethod
    def compare_conditions(
        full_piano: SocialCognitionResult,
        ablated: SocialCognitionResult,
    ) -> dict:
        """2条件間の社会的認知精度を比較する。"""
        # Fisher z変換による相関係数の比較検定
        z_full = np.arctanh(full_piano.pearson_r)
        z_ablated = np.arctanh(ablated.pearson_r)

        n1 = full_piano.sample_size
        n2 = ablated.sample_size
        se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        z_diff = (z_full - z_ablated) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_diff)))

        return {
            "full_piano_r": full_piano.pearson_r,
            "ablated_r": ablated.pearson_r,
            "r_difference": full_piano.pearson_r - ablated.pearson_r,
            "z_statistic": z_diff,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
```

---

## 7. アイテム収集ベンチマーク

### 7.1 概要

単一エージェントの基本性能を評価する指標。Minecraftの約1,000種類のアイテムに対して、エージェントがどれだけユニークなアイテムを収集できるかを測定する。

| 実験 | エージェント数 | 時間 | 結果 |
|---|---|---|---|
| 短期 | 25体 | 30分 | 平均約17種のユニークアイテム |
| 長期 | 49体 | 4時間 | 累積約320種で飽和 |

### 7.2 技術依存ツリー（Dependency Tree）

Minecraftのアイテム取得には依存関係が存在する。この依存ツリーの進行度が技術的進歩の定量的指標となる。

```
木のツルハシ → 石のツルハシ → 鉄のツルハシ → ダイヤモンドの採掘
                                    |
                          かまど + 石炭 → 鉄インゴットの精錬
```

### 7.3 実装設計

```python
from dataclasses import dataclass, field
import time as time_module

@dataclass
class ItemCollectionSnapshot:
    agent_id: str
    timestamp: float
    unique_items: set[str]
    total_count: int

@dataclass
class ItemCollectionMetrics:
    unique_item_count: int
    total_items_collected: int
    collection_rate: float  # items per minute
    dependency_depth: int   # 依存ツリーの最大深度
    saturation_time: float | None  # 飽和到達時間

class ItemCollectionBenchmark:
    """
    エージェントのアイテム収集性能を計測する。
    論文では30分で平均17種、4時間で約320種（49体累積）。
    """

    # Minecraftアイテムの依存ツリー（簡略版）
    DEPENDENCY_TREE: dict[str, list[str]] = {
        "wooden_pickaxe": ["planks", "stick"],
        "stone_pickaxe": ["cobblestone", "stick"],
        "iron_pickaxe": ["iron_ingot", "stick"],
        "diamond": ["iron_pickaxe"],
        "iron_ingot": ["raw_iron", "furnace", "coal"],
        "furnace": ["cobblestone"],
        "planks": ["log"],
        "stick": ["planks"],
        "cobblestone": ["wooden_pickaxe"],
        "raw_iron": ["stone_pickaxe"],
        "coal": ["wooden_pickaxe"],
    }

    def __init__(self):
        self._agent_items: dict[str, set[str]] = {}
        self._collection_timeline: list[ItemCollectionSnapshot] = []
        self._start_time: float | None = None

    def record_item(self, agent_id: str, item_name: str, timestamp: float):
        """アイテム取得を記録する。"""
        if self._start_time is None:
            self._start_time = timestamp

        if agent_id not in self._agent_items:
            self._agent_items[agent_id] = set()
        self._agent_items[agent_id].add(item_name)

        self._collection_timeline.append(ItemCollectionSnapshot(
            agent_id=agent_id,
            timestamp=timestamp,
            unique_items=set(self._agent_items[agent_id]),
            total_count=len(self._agent_items[agent_id]),
        ))

    def get_unique_count(self, agent_id: str) -> int:
        """エージェントのユニークアイテム数を返す。"""
        return len(self._agent_items.get(agent_id, set()))

    def get_cumulative_unique(self) -> int:
        """全エージェントの累積ユニークアイテム数を返す。"""
        all_items: set[str] = set()
        for items in self._agent_items.values():
            all_items.update(items)
        return len(all_items)

    def compute_dependency_depth(self, agent_id: str) -> int:
        """
        エージェントが到達した依存ツリーの最大深度を計算する。
        深い深度 = より高度な技術的進歩。
        """
        items = self._agent_items.get(agent_id, set())
        if not items:
            return 0

        def depth(item: str, visited: set[str] | None = None) -> int:
            if visited is None:
                visited = set()
            if item in visited or item not in self.DEPENDENCY_TREE:
                return 0
            visited.add(item)
            deps = self.DEPENDENCY_TREE[item]
            if not deps:
                return 0
            return 1 + max(depth(d, visited) for d in deps)

        return max(depth(item) for item in items if item in self.DEPENDENCY_TREE)

    def compute_metrics(self, agent_id: str) -> ItemCollectionMetrics:
        """エージェントの収集メトリクスを計算する。"""
        items = self._agent_items.get(agent_id, set())
        agent_snapshots = [
            s for s in self._collection_timeline if s.agent_id == agent_id
        ]

        elapsed = 0.0
        if agent_snapshots and self._start_time is not None:
            elapsed = (agent_snapshots[-1].timestamp - self._start_time) / 60.0

        rate = len(items) / elapsed if elapsed > 0 else 0.0

        # 飽和検出: 直近10分間で新規アイテムが2個以下
        saturation = self._detect_saturation(agent_snapshots)

        return ItemCollectionMetrics(
            unique_item_count=len(items),
            total_items_collected=len(agent_snapshots),
            collection_rate=rate,
            dependency_depth=self.compute_dependency_depth(agent_id),
            saturation_time=saturation,
        )

    def _detect_saturation(
        self, snapshots: list[ItemCollectionSnapshot],
        window_seconds: float = 600.0,
        threshold: int = 2,
    ) -> float | None:
        """新規アイテム取得が停滞したタイムスタンプを検出する。"""
        if len(snapshots) < 2:
            return None

        for i in range(len(snapshots) - 1, 0, -1):
            window_start = snapshots[i].timestamp - window_seconds
            items_in_window = set()
            items_before = set()
            for s in snapshots:
                if s.timestamp < window_start:
                    items_before.update(s.unique_items)
                elif s.timestamp <= snapshots[i].timestamp:
                    items_in_window.update(s.unique_items)
            new_items = items_in_window - items_before
            if len(new_items) <= threshold:
                return snapshots[i].timestamp
        return None

    def get_collection_timeline(
        self, interval: float = 60.0,
    ) -> list[tuple[float, int]]:
        """
        時間間隔ごとの累積ユニークアイテム数を返す。
        論文のFigure再現用。
        """
        if not self._collection_timeline or self._start_time is None:
            return []

        all_items: set[str] = set()
        sorted_snaps = sorted(self._collection_timeline, key=lambda s: s.timestamp)
        max_t = sorted_snaps[-1].timestamp

        timeline = []
        t = self._start_time
        snap_idx = 0
        while t <= max_t:
            while snap_idx < len(sorted_snaps) and sorted_snaps[snap_idx].timestamp <= t:
                all_items.update(sorted_snaps[snap_idx].unique_items)
                snap_idx += 1
            timeline.append((t - self._start_time, len(all_items)))
            t += interval

        return timeline
```

### 7.4 アブレーション比較

```python
class ItemCollectionAblation:
    """
    PIANOモジュール構成別のアイテム収集性能比較。
    論文では行動認識モジュールが最重要と特定。
    """

    CONFIGURATIONS = {
        "baseline": ["skill_execution", "memory", "cognitive_controller"],
        "piano_no_action_awareness": [
            "skill_execution", "memory", "cognitive_controller",
            "goal_generation", "social_awareness", "talking",
            "self_reflection", "planning",
        ],
        "full_piano": [
            "skill_execution", "memory", "cognitive_controller",
            "action_awareness", "goal_generation", "social_awareness",
            "talking", "self_reflection", "planning",
        ],
    }

    @staticmethod
    def compare_configurations(
        results: dict[str, list[int]],  # {config_name: [unique_item_counts]}
    ) -> dict:
        """構成間のアイテム収集性能を比較する。"""
        summary = {}
        for config, counts in results.items():
            summary[config] = {
                "mean": np.mean(counts),
                "std": np.std(counts),
                "min": min(counts),
                "max": max(counts),
                "median": np.median(counts),
            }

        # Kruskal-Wallis検定（3群以上の比較）
        groups = list(results.values())
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            h_stat, p_value = stats.kruskal(*groups)
            summary["kruskal_wallis"] = {
                "h_statistic": h_stat,
                "p_value": p_value,
            }

        return summary
```

---

## 8. データ収集パイプライン

### 8.1 全体アーキテクチャ

```
┌──────────────────────────────────────────────────────────┐
│                  Minecraft Server                        │
│  Agent 1 ─┐                                              │
│  Agent 2 ─┤  Actions, Speech, Goals, Inventory           │
│  ...      ─┤                                              │
│  Agent N ─┘                                              │
└─────────────────────┬────────────────────────────────────┘
                      │ Event Stream
                      ▼
┌──────────────────────────────────────────────────────────┐
│              Event Collection Layer                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Action  │  │ Speech   │  │ State    │                 │
│  │ Logger  │  │ Logger   │  │ Logger   │                 │
│  └────┬────┘  └────┬─────┘  └────┬─────┘                │
│       └────────────┼─────────────┘                       │
│                    ▼                                      │
│           Unified Event Bus                              │
│           (Apache Kafka / Redis Streams)                  │
└─────────────────────┬────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ Time-series  │ │ Document │ │ Graph DB     │
│ DB           │ │ Store    │ │              │
│ (InfluxDB/   │ │ (MongoDB/│ │ (Neo4j/      │
│  TimescaleDB)│ │  S3)     │ │  NetworkX)   │
└──────┬───────┘ └────┬─────┘ └──────┬───────┘
       └──────────────┼──────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│              Analysis & Visualization Layer               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Metrics  │  │ Dashboard│  │ Report   │                │
│  │ Engine   │  │ (Grafana)│  │ Generator│                │
│  └──────────┘  └──────────┘  └──────────┘                │
└──────────────────────────────────────────────────────────┘
```

### 8.2 イベントスキーマ

```python
from dataclasses import dataclass
from enum import Enum
import time
import uuid

class EventType(Enum):
    ACTION = "action"           # ブロック配置、アイテム収集等
    SPEECH = "speech"           # エージェントの発話
    GOAL_UPDATE = "goal_update" # 目標変更
    INVENTORY = "inventory"     # インベントリ変更
    MOVEMENT = "movement"       # 位置変更
    SOCIAL = "social"           # 社会的インタラクション
    TAX = "tax"                 # 税関連イベント
    VOTE = "vote"               # 投票イベント
    CONSTITUTION = "constitution"  # 憲法変更

@dataclass
class SimulationEvent:
    """統一イベントスキーマ。全てのイベントがこの形式で記録される。"""
    event_id: str              # UUID
    event_type: EventType
    timestamp: float           # シミュレーション時刻
    wall_clock: float          # 実時刻
    agent_id: str
    location: tuple[float, float, float]  # (x, y, z)
    payload: dict              # イベント固有データ
    session_id: str            # 実験セッションID

    @classmethod
    def create(cls, event_type, agent_id, location, payload, session_id):
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            wall_clock=time.time(),
            agent_id=agent_id,
            location=location,
            payload=payload,
            session_id=session_id,
        )
```

### 8.3 ストレージ戦略

| データ種別 | 推奨ストレージ | 理由 |
|---|---|---|
| 時系列メトリクス | InfluxDB / TimescaleDB | 高頻度の数値データに最適化 |
| エージェント発話・目標 | MongoDB / PostgreSQL (JSONB) | 半構造化データの柔軟なクエリ |
| 社会グラフ | Neo4j / NetworkX (インメモリ) | グラフ構造のトラバーサルに最適化 |
| 大規模ログ | S3 / MinIO + Parquet | コスト効率的な長期保存 |
| リアルタイムイベント | Redis Streams / Apache Kafka | 低レイテンシのイベント配信 |

#### スケール別推奨構成

| 規模 | 構成 | 想定データ量 |
|---|---|---|
| 小規模 (30体, 20分) | SQLite + CSV | ~100MB |
| 中規模 (500体, 2.5時間) | PostgreSQL + InfluxDB | ~10GB |
| 大規模 (1000体, 4時間超) | Kafka + TimescaleDB + S3 | ~100GB以上 |

### 8.4 リアルタイム集計

```python
from collections import defaultdict

class RealTimeAggregator:
    """
    リアルタイムでイベントを集計し、メトリクスを更新する。
    各メトリクスモジュール（エントロピー、遵守率等）へイベントを配信。
    """

    def __init__(self):
        self._subscribers: dict[EventType, list] = defaultdict(list)
        self._event_count = 0
        self._buffer: list[SimulationEvent] = []
        self._flush_interval = 100  # 100イベントごとにフラッシュ

    def subscribe(self, event_type: EventType, handler):
        """特定のイベントタイプのハンドラを登録する。"""
        self._subscribers[event_type].append(handler)

    def process_event(self, event: SimulationEvent):
        """イベントを処理し、登録されたハンドラに配信する。"""
        self._event_count += 1
        self._buffer.append(event)

        for handler in self._subscribers.get(event.event_type, []):
            handler(event)

        if len(self._buffer) >= self._flush_interval:
            self._flush()

    def _flush(self):
        """バッファされたイベントをストレージに永続化する。"""
        # ストレージへのバッチ書き込み
        self._buffer.clear()
```

---

## 9. 可視化・ダッシュボード

### 9.1 ツール選定

| ツール | 用途 | 選定理由 |
|---|---|---|
| **Grafana** | 時系列メトリクスの監視 | InfluxDB/Prometheusとの統合、アラート機能 |
| **Streamlit** | 分析ダッシュボード | Pythonネイティブ、迅速なプロトタイピング |
| **NetworkX + matplotlib** | 社会グラフの可視化 | Pythonエコシステムとの統合 |
| **Plotly / Dash** | インタラクティブチャート | ズーム・フィルタ等の高度なインタラクション |

### 9.2 Grafanaダッシュボード設計

#### 推奨パネル構成

```
┌─────────────────────────────────────────────────┐
│  Civilization Benchmark Dashboard               │
├────────────────┬────────────────┬───────────────┤
│ Role Entropy   │ Tax Compliance │ Meme Count    │
│ (Time Series)  │ (Gauge + Line) │ (Bar Chart)   │
├────────────────┴────────────────┴───────────────┤
│ Geographic Spread Heatmap                       │
│ (Canvas/Image Panel - 自動更新)                  │
├────────────────┬────────────────────────────────┤
│ Agent Count    │ Conversion Timeline            │
│ by Role        │ (Stacked Area)                 │
│ (Pie Chart)    │                                │
├────────────────┴────────────────────────────────┤
│ Social Network Graph (iframe/Streamlit embed)   │
└─────────────────────────────────────────────────┘
```

#### データソース設定

```yaml
# Grafana datasource設定例 (provisioning/datasources.yaml)
apiVersion: 1
datasources:
  - name: SimulationMetrics
    type: influxdb
    url: http://influxdb:8086
    database: project_sid
    access: proxy
  - name: PostgreSQL
    type: postgres
    url: postgres:5432
    database: project_sid
    user: grafana
```

### 9.3 Streamlit分析ダッシュボード

```python
# dashboard.py - Streamlit分析ダッシュボードの構成例
"""
streamlit run dashboard.py で起動。
以下はダッシュボードの主要コンポーネント構成。
"""

# --- ページ構成 ---
# 1. Overview: 全体サマリ（エージェント数、実行時間、主要KPI）
# 2. Specialization: 役割エントロピー推移、役割分布、行動-役割相関ヒートマップ
# 3. Governance: 税遵守率推移、投票結果、インフルエンサー影響分析
# 4. Cultural Spread: ミームヒートマップ、改宗タイムライン、町別分析
# 5. Social Network: 社会グラフの可視化、中心性指標
# 6. Experiment Comparison: 複数実験の比較ビュー

# --- 主要ウィジェット ---
# - 時間範囲スライダー（シミュレーション時刻）
# - エージェントフィルタ（ID、役割、町）
# - ミームタイプセレクタ
# - 実験条件フィルタ（PIANO full / baseline / ablation）
```

### 9.4 社会グラフの可視化

```python
import networkx as nx

class SocialGraphVisualizer:
    """
    エージェント間の社会的インタラクションをグラフとして可視化する。
    ノード = エージェント、エッジ = インタラクション（会話、取引等）。
    """

    def build_graph(
        self, interactions: list[dict],
    ) -> nx.Graph:
        """
        インタラクションログからソーシャルグラフを構築する。

        Args:
            interactions: [{"agent_a": str, "agent_b": str, "type": str, "timestamp": float}]
        """
        G = nx.Graph()
        for ix in interactions:
            a, b = ix["agent_a"], ix["agent_b"]
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1, types=[])
            G[a][b]["types"].append(ix["type"])
        return G

    def compute_centrality_metrics(self, G: nx.Graph) -> dict[str, dict]:
        """中心性指標を計算する。"""
        return {
            "degree": nx.degree_centrality(G),
            "betweenness": nx.betweenness_centrality(G),
            "closeness": nx.closeness_centrality(G),
            "eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
        }

    def detect_communities(self, G: nx.Graph) -> list[set]:
        """コミュニティ検出（Louvain法）。"""
        from networkx.algorithms.community import louvain_communities
        return louvain_communities(G)

    def render(
        self, G: nx.Graph,
        color_by: str = "role",  # "role", "community", "town"
        node_attributes: dict[str, dict] | None = None,
    ):
        """グラフを描画する。"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w = max(weights) if weights else 1

        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=0.3,
            width=[w / max_w * 3 for w in weights],
        )
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, alpha=0.8)
        ax.set_title("Social Interaction Graph")
        return fig
```

### 9.5 実験結果の比較ビュー

```python
@dataclass
class ExperimentResult:
    experiment_id: str
    condition: str  # "full_piano", "baseline", "frozen_constitution"
    influencer_type: str  # "pro_tax", "anti_tax"
    run_number: int
    metrics: dict  # 各種メトリクスの値

class ExperimentComparator:
    """
    複数の実験条件間でメトリクスを比較する。
    論文では各設定ごとに4回繰り返し実行。
    """

    def compare(
        self, results: list[ExperimentResult], metric_key: str,
    ) -> dict[str, dict]:
        """条件間のメトリクス比較統計を生成する。"""
        from scipy import stats

        grouped: dict[str, list[float]] = {}
        for r in results:
            key = f"{r.condition}_{r.influencer_type}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r.metrics.get(metric_key, 0))

        summary = {}
        for key, values in grouped.items():
            summary[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "n": len(values),
                "ci_95": stats.t.interval(
                    0.95, len(values) - 1,
                    loc=np.mean(values), scale=stats.sem(values),
                ) if len(values) > 1 else (np.mean(values), np.mean(values)),
            }

        # 条件間のペアワイズ比較（Mann-Whitney U検定）
        conditions = list(grouped.keys())
        pairwise = {}
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                a, b = conditions[i], conditions[j]
                if len(grouped[a]) > 1 and len(grouped[b]) > 1:
                    stat, p_val = stats.mannwhitneyu(grouped[a], grouped[b])
                    pairwise[f"{a}_vs_{b}"] = {"statistic": stat, "p_value": p_val}

        return {"summary": summary, "pairwise_tests": pairwise}
```

---

## 10. LLM依存評価の代替案

### 10.1 問題の概要

論文の評価手法には、LLMベースエージェントの行動をLLMで評価するという循環的構造がある。

| 評価項目 | LLM依存度 | 循環性リスク |
|---|---|---|
| 役割推論 | 高（GPT-4oで推論） | LLMバイアスが役割分類を歪める可能性 |
| ミーム分析 | 高（LLMで抽出） | LLMが生成したパターンをLLMが検出する循環 |
| 行動カウント | 低（直接計測） | 循環なし |
| 税遵守率 | 低（直接計測） | 循環なし |
| 宗教伝播 | 中（キーワードベース） | キーワード選択にバイアスの可能性 |

### 10.2 人間評価プロトコル

```python
@dataclass
class HumanEvalTask:
    task_id: str
    eval_type: str  # "role_classification", "meme_detection", "behavioral_quality"
    agent_id: str
    data: dict      # 評価対象データ
    instructions: str

@dataclass
class HumanEvalResult:
    task_id: str
    evaluator_id: str
    labels: dict
    confidence: float
    time_spent_seconds: float

class HumanEvalProtocol:
    """
    LLM推論結果を人間評価で検証するプロトコル。
    Cohen's Kappaで評価者間信頼性を測定。
    """

    # 評価タスクの種類
    TASK_TYPES = {
        "role_classification": {
            "description": "エージェントの目標リストから役割を分類",
            "evaluators_per_task": 3,
            "agreement_threshold": 0.6,  # Cohen's Kappa
        },
        "meme_detection": {
            "description": "テキスト中の文化的ミームを識別",
            "evaluators_per_task": 2,
            "agreement_threshold": 0.5,
        },
        "behavioral_quality": {
            "description": "エージェント行動の自然さ・合理性を評価",
            "evaluators_per_task": 3,
            "agreement_threshold": 0.5,
        },
    }

    def compute_inter_rater_agreement(
        self, results: list[HumanEvalResult],
    ) -> float:
        """Cohen's Kappaで評価者間一致度を計算する。"""
        from sklearn.metrics import cohen_kappa_score
        # ペアワイズKappa計算
        evaluator_labels = {}
        for r in results:
            if r.evaluator_id not in evaluator_labels:
                evaluator_labels[r.evaluator_id] = {}
            evaluator_labels[r.evaluator_id][r.task_id] = r.labels

        evaluators = list(evaluator_labels.keys())
        kappas = []
        for i in range(len(evaluators)):
            for j in range(i + 1, len(evaluators)):
                common_tasks = (
                    set(evaluator_labels[evaluators[i]].keys())
                    & set(evaluator_labels[evaluators[j]].keys())
                )
                if len(common_tasks) > 1:
                    labels_i = [evaluator_labels[evaluators[i]][t] for t in common_tasks]
                    labels_j = [evaluator_labels[evaluators[j]][t] for t in common_tasks]
                    kappas.append(cohen_kappa_score(labels_i, labels_j))

        return np.mean(kappas) if kappas else 0.0

    def compare_llm_vs_human(
        self,
        llm_results: list[RoleInferenceResult],
        human_results: list[HumanEvalResult],
    ) -> dict:
        """LLM推論と人間評価の一致度を測定する。"""
        matched = 0
        total = 0
        for llm_r in llm_results:
            human_labels = [
                h.labels.get("role")
                for h in human_results
                if h.task_id == llm_r.agent_id
            ]
            if human_labels:
                majority = max(set(human_labels), key=human_labels.count)
                total += 1
                if llm_r.inferred_role.value == majority:
                    matched += 1

        return {
            "agreement_rate": matched / total if total > 0 else 0,
            "total_compared": total,
        }
```

### 10.3 行動ベースの客観指標

LLM推論に依存しない、直接計測可能な行動指標。

| 指標 | 計測方法 | 専門化の証拠 |
|---|---|---|
| アクション頻度分布 | アクションカウントの直接集計 | 特定アクションへの偏り |
| アクションエントロピー | エージェントごとのアクション種類のエントロピー | 低エントロピー = 特定行動に特化 |
| 空間的行動パターン | 位置ログからの行動範囲分析 | 特定エリアへの集中 |
| インタラクション頻度 | 会話・取引回数の集計 | 社会的役割の指標 |
| インベントリ特化度 | 所持アイテム種類の偏り | 収集行動の専門化 |

```python
class BehavioralObjectiveMetrics:
    """LLMに依存しない行動ベースの客観指標。"""

    @staticmethod
    def action_entropy(action_counts: dict[str, int]) -> float:
        """
        エージェントのアクションエントロピーを計算する。
        低い値 = 特定行動に特化（専門化の証拠）
        高い値 = 多様な行動（ゼネラリスト）
        """
        total = sum(action_counts.values())
        if total == 0:
            return 0.0
        probs = [c / total for c in action_counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @staticmethod
    def spatial_concentration(
        positions: list[tuple[float, float]], world_size: tuple[int, int],
    ) -> float:
        """
        空間的集中度を計算する（0-1: 1が最大集中）。
        特定エリアに留まるエージェントは空間的に専門化している。
        """
        if len(positions) < 2:
            return 1.0
        xs = [p[0] for p in positions]
        zs = [p[1] for p in positions]
        range_x = max(xs) - min(xs)
        range_z = max(zs) - min(zs)
        max_range = max(world_size)
        return 1.0 - (range_x + range_z) / (2 * max_range)

    @staticmethod
    def inventory_specialization(
        inventories: dict[str, dict[str, int]],
    ) -> dict[str, float]:
        """
        各エージェントのインベントリ特化度を計算する。
        ジニ係数を使用（0 = 均等、1 = 完全特化）。
        """
        results = {}
        for agent_id, items in inventories.items():
            values = sorted(items.values())
            n = len(values)
            if n == 0:
                results[agent_id] = 0.0
                continue
            cumulative = sum((i + 1) * v for i, v in enumerate(values))
            total = sum(values)
            if total == 0:
                results[agent_id] = 0.0
            else:
                gini = (2 * cumulative) / (n * total) - (n + 1) / n
                results[agent_id] = gini
        return results
```

### 10.4 統計的検出力を高める実験設計

論文の各条件4回の繰り返しでは統計的検出力が限定的。以下の改善策を提案する。

| 改善策 | 説明 | 効果 |
|---|---|---|
| **繰り返し回数の増加** | 各条件10-20回に増加 | 効果量 d=0.8 で検出力 0.80 以上 |
| **ブートストラップ法** | 限られたデータからの信頼区間推定 | 少数サンプルでの推定精度向上 |
| **効果量の報告** | p値だけでなくCohen's d等を報告 | 実質的な差の大きさを定量化 |
| **事前登録** | 仮説・分析方法の事前登録 | p-hackingの防止 |
| **混合効果モデル** | エージェント・実験をランダム効果として扱う | 個体差・実験差を考慮した推定 |

```python
class StatisticalPowerAnalysis:
    """実験設計のための統計的検出力分析ツール。"""

    @staticmethod
    def required_sample_size(
        effect_size: float = 0.8,  # Cohen's d
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> int:
        """必要なサンプルサイズを算出する（2群比較の場合）。"""
        from scipy import stats

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))

    @staticmethod
    def bootstrap_ci(
        data: list[float], n_bootstrap: int = 10000, ci: float = 0.95,
    ) -> tuple[float, float]:
        """ブートストラップ法による信頼区間の推定。"""
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            means.append(np.mean(sample))
        lower = np.percentile(means, (1 - ci) / 2 * 100)
        upper = np.percentile(means, (1 + ci) / 2 * 100)
        return (lower, upper)
```

---

## 11. 評価パイプラインのタイミング方針

### 11.1 背景

評価パイプラインの実行タイミングについて、テストシステム（E2Eテストの最終層として評価を位置づける方針）と統合設計（Phase 2前倒しで早期にベンチマークを稼働させる方針）の間で異なる見解がある。本セクションでは、実装上の明確な方針を定義する。

### 11.2 方針: 段階的統合（Progressive Integration）

評価パイプラインは**3段階で段階的に統合**する。E2Eテストの最終層としての位置づけと、早期ベンチマーク稼働は矛盾せず、以下のように共存させる。

| 段階 | タイミング | 対象指標 | 目的 |
|---|---|---|---|
| **Stage 1: 即時計測** | Phase 0 開発中から常時 | 行動ベース客観指標（アクションエントロピー、インベントリ特化度、空間集中度） | 開発中のフィードバック。LLM不要で低コスト |
| **Stage 2: シミュレーション後バッチ** | Phase 0 完了後、各実験終了時 | 役割エントロピー、税遵守率、ミーム検出、アイテム収集、社会的認知精度 | 文明的ベンチマークの本計測。LLM推論を含む |
| **Stage 3: E2Eテスト統合** | Phase 1 以降のCI/CDパイプライン | 全指標のリグレッション検出 | アーキテクチャ変更が文明的指標に悪影響を与えないことを検証 |

### 11.3 各段階の詳細

**Stage 1（即時計測）**: Phase 0 開発中からイベント収集パイプライン（セクション8）と連動して稼働する。`BehavioralObjectiveMetrics`（セクション10.3）の指標群は LLM を使用せず直接計測可能なため、開発サイクル内で即座にフィードバックを得られる。Grafanaダッシュボード（セクション9.2）に接続し、開発者がリアルタイムでエージェント行動を監視できる。

**Stage 2（シミュレーション後バッチ）**: 実験（20分〜4時間）の終了後にバッチ処理で実行する。LLMベースの役割推論パイプライン（セクション2.2）やミーム分析（セクション4.1）を含むため、コスト管理の観点から常時実行ではなく実験完了後に一括実行する。結果は Streamlit ダッシュボード（セクション9.3）で分析する。

**Stage 3（E2Eテスト統合）**: CI/CDパイプラインの最終段階として、事前定義されたシナリオ（30体/5分の短縮版実験）を自動実行し、主要指標がベースラインから一定範囲内に収まることを検証する。新しいモジュール変更がマージされる前のゲートとして機能する。

### 11.4 各メトリクスの計測頻度

| メトリクス | 計測タイミング | 頻度 | 根拠 |
|---|---|---|---|
| アクションエントロピー | Stage 1 | 60秒ごとのスナップショット | 低コスト、リアルタイムフィードバック |
| 役割エントロピー | Stage 2 | 実験終了後に一括 | LLM推論が必要 |
| 税遵守率 | Stage 1 + 2 | 課税シーズンごと（自動）+ 実験後集計 | イベント駆動で自動記録 |
| ミーム検出（キーワード） | Stage 1 | 発話ごとにリアルタイム | 低コスト |
| ミーム検出（LLM） | Stage 2 | 実験終了後に一括 | LLM推論が必要 |
| 社会的認知精度 | Stage 2 | 実験終了後に一括 | 全ペアの集計が必要 |
| アイテム収集数 | Stage 1 | インベントリ変更イベントごと | イベント駆動で自動記録 |
| インフラ発展指標 | Stage 1 + 2 | ブロック配置ごと + 実験後集計 | イベント駆動 + 構造分析 |

---

## 12. 技術スタック推奨

### 12.1 コアライブラリ

| カテゴリ | ライブラリ | バージョン | 用途 |
|---|---|---|---|
| 数値計算 | NumPy, SciPy | 最新安定版 | エントロピー計算、統計分析 |
| データ分析 | pandas | 最新安定版 | データフレーム操作 |
| 可視化 | matplotlib, Plotly | 最新安定版 | チャート、ヒートマップ |
| グラフ分析 | NetworkX | 最新安定版 | 社会グラフ |
| ダッシュボード | Streamlit / Dash | 最新安定版 | Web UI |
| 監視 | Grafana + InfluxDB | 最新安定版 | リアルタイムメトリクス |
| 統計テスト | scikit-learn, statsmodels | 最新安定版 | 検定、回帰 |
| LLMクライアント | openai / anthropic SDK | 最新安定版 | 役割推論、ミーム分析 |

### 12.2 参考フレームワーク

| フレームワーク | 用途 | 参考 |
|---|---|---|
| [MultiAgentBench](https://github.com/ulab-uiuc/MARBLE) | マルチエージェント評価のベースライン | ACL 2025, 協調・競争の両面評価 |
| [REALM-Bench](https://arxiv.org/abs/2502.18836) | 動的計画・スケジューリング評価 | マルチエージェント調整の評価 |
| [NDlib](https://ndlib.readthedocs.io/) | ネットワーク拡散モデル | SIRモデル実装、文化伝播 |
| [Crowd](https://arxiv.org/html/2412.10781v1) | 社会ネットワークシミュレーション | YAML設定、拡散タスク |
| [Mesa](https://mesa.readthedocs.io/) | エージェントベースモデリング | Python ABMフレームワーク |

### 12.3 実装優先順位

| 優先度 | コンポーネント | 依存関係 | 工数目安 |
|---|---|---|---|
| **P0** | イベント収集・統一スキーマ | なし | 基盤となるため最優先 |
| **P0** | 行動ベース客観指標 | イベント収集 | LLM不要で即座に計測可能 |
| **P1** | 役割エントロピー計算 | イベント収集 | 専門化の中核指標 |
| **P1** | 税遵守率追跡 | イベント収集 | 統治の中核指標 |
| **P1** | キーワードベースミーム検出 | イベント収集 | 文化伝播の基本指標 |
| **P1** | アイテム収集ベンチマーク | イベント収集 | 単一エージェント性能の基本指標 |
| **P1** | 社会的認知精度評価 | 社会認識モジュール | ピアソン相関・集合知効果の計測 |
| **P2** | LLM役割推論パイプライン | LLM統合 | コスト考慮が必要 |
| **P2** | 地理的伝播ヒートマップ | ミーム検出 | 可視化依存 |
| **P2** | Streamlitダッシュボード | 全メトリクス | 各指標実装後に統合 |
| **P3** | Grafanaリアルタイム監視 | InfluxDB設定 | 運用時に必要 |
| **P3** | 人間評価プロトコル | LLM推論 | LLM検証用 |
| **P3** | SIRモデルフィッティング | 伝播データ | 高度な分析 |

---

## 参考文献

- Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Lopez-Ruiz, R., Mancini, H. L., & Calbet, X. (1995). "A statistical measure of complexity"
- Altera.AL (2024). "Project Sid: Many-agent simulations toward AI civilization" [arXiv:2411.00114](https://arxiv.org/abs/2411.00114)
- Park, J. S., et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior"
- MultiAgentBench (ACL 2025). [GitHub](https://github.com/ulab-uiuc/MARBLE)
- REALM-Bench (2025). [arXiv:2502.18836](https://arxiv.org/abs/2502.18836)

---

[トップ](../index.md) | [論文の評価分析](../08-evaluation.md)
