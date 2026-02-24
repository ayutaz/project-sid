# Project Sid 再現実装 — 総合概要

> 概要: PIANOアーキテクチャ再現実装の全ドキュメントのインデックス、技術判断、読み方ガイド
> 対応論文セクション: 全セクション横断
> 最終更新: 2026-02-24

---

## エグゼクティブサマリー

本プロジェクトは、Altera.ALの論文「Project Sid: Many-agent simulations toward AI civilization」で提案された **PIANOアーキテクチャ**（Parallel Information Aggregation via Neural Orchestration）の再現実装を目指す。PIANOは10〜1000以上のAIエージェントがMinecraft環境内で社会的に振る舞うための認知アーキテクチャであり、2つの根本課題を解決する:

1. **並行性（Concurrency）**: 高速反射モジュール（非LLM）と低速熟慮モジュール（LLM）が異なる速度で並列実行
2. **一貫性（Coherence）**: 認知コントローラ（CC）がGWT情報ボトルネック+ブロードキャストで全モジュール間の一貫性を制御

**現在の状態**: **E2Eシミュレーション接続完了**。2,079テスト全通過、ruff lint clean。Phase 0コアモジュール + Phase 1全モジュール統合（10体動作） + Phase 2スケーリング基盤（50体対応） + E2Eシミュレーション接続（BridgePerception、ChatBroadcaster、BridgeManager、HealthMonitor、ActionMapper、マルチボットLauncher、Docker基盤）が実装済み。MCサーバー実動作確認済み（Paper 1.20.4、3ボット×5tick）。次のステップは言行一致改善率30%測定、大規模E2E検証（50体MC接続）、Phase 3準備。

---

## ドキュメントマップ

### コアドキュメント（実装設計）

| # | ファイル | 概要 | 行数 |
|---|---|---|---|
| 01 | [system-architecture](./01-system-architecture.md) | PIANO全体設計、GWT競合的選択、非同期モジュールスケジューラ、障害復旧 | ~1,340 |
| 02 | [llm-integration](./02-llm-integration.md) | LLM APIアブストラクション、プロンプト設計、マイクロバッチ、コスト最適化 | ~1,080 |
| 03 | [cognitive-controller](./03-cognitive-controller.md) | CC実装、点火メカニズム、出力検証、動的間隔制御、傾向検出 | ~1,430 |
| 04 | [memory-system](./04-memory-system.md) | WM/STM/LTM三層構造、Qdrantセマンティック検索、忘却関数、動的容量 | ~780 |
| 05 | [minecraft-platform](./05-minecraft-platform.md) | Mineflayer、Pufferfish/Velocity、マルチボット接続、スキル実行 | ~1,120 |
| 06 | [social-cognition](./06-social-cognition.md) | SAS設計、感情追跡、社会グラフ、好感度推定、集合知 | ~1,580 |
| 07 | [goal-planning](./07-goal-planning.md) | 目標生成、HTN計画、行動認識（ルールベース→NN移行）、自己省察 | ~1,370 |
| 08 | [infrastructure](./08-infrastructure.md) | 分散実行、シャーディング、LLMゲートウェイ、バックアップ/DR | ~1,120 |
| 09 | [evaluation](./09-evaluation.md) | 文明的ベンチマーク4軸（専門化・統治・文化・インフラ）、評価パイプライン | ~2,350 |
| 10 | [devops](./10-devops.md) | CI/CD、コンテナ化、モニタリング、実験管理、ログ戦略 | ~1,800 |

### E2E・レビュー・計画

| ファイル | 概要 |
|---|---|
| [e2e-simulation](./e2e-simulation.md) | E2Eシミュレーション接続アーキテクチャ（Perception→CC→Action→MC） |
| [nn_action_awareness](./nn_action_awareness.md) | Action Awareness NN実装ガイド |
| [review-comprehensive](./review-comprehensive.md) | 5観点の統合レビュー（セキュリティ・コスト・OSS・テスト・統合） |
| [roadmap](./roadmap.md) | Phase 0-3 実装ロードマップ（12-17ヶ月、35-55人月） |

---

## 読み方ガイド

### 全体像を把握したい場合
1. **本文書** → 2. [01 システムアーキテクチャ](./01-system-architecture.md) → 3. [roadmap](./roadmap.md)

### 各モジュールを深掘りしたい場合
```
01 (全体設計) ─┬─ 03 (認知コントローラ) ─── 02 (LLM統合)
               ├─ 04 (記憶システム)
               ├─ 06 (社会認知) ─── 07 (目標・計画)
               └─ 05 (Minecraft基盤) ─── 08 (インフラ)
                                         └── 10 (DevOps)
09 (評価) は全モジュールを横断
```

### リスク・コストを評価したい場合
[review-comprehensive](./review-comprehensive.md) → 各コアドキュメントの該当セクション

---

## アーキテクチャ概要

```
┌─────────────────── PIANO Agent ───────────────────┐
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Goal Gen │  │ Planning │  │ Talking  │  Fast    │
│  │  (LLM)   │  │  (LLM)   │  │  (LLM)   │  Modules │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │                │
│  ┌────▼──────────────▼──────────────▼────┐          │
│  │     Shared Agent State (Redis)        │          │
│  └────┬──────────────┬──────────────┬────┘          │
│       │              │              │                │
│  ┌────▼─────┐  ┌─────▼────┐  ┌─────▼────┐          │
│  │ Action   │  │ Social   │  │ Memory   │  Slow    │
│  │ Awareness│  │ Awareness│  │  Module  │  Modules │
│  │ (NN/Rule)│  │  (SAS)   │  │ (Qdrant) │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │                │
│  ┌────▼──────────────▼──────────────▼────┐          │
│  │  Cognitive Controller (CC)            │          │
│  │  GWT: 競合的選択 → 点火 → ブロードキャスト  │          │
│  └───────────────────────────────────────┘          │
│                                                     │
└──────────────┬──────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  Minecraft Server   │
    │  (Pufferfish+Velocity)│
    │  via Mineflayer      │
    └─────────────────────┘
```

**対応ドキュメント**: 01=全体設計、02=LLM呼び出し、03=CC、04=Memory、05=Minecraft、06=SAS、07=Goal/Plan、08=インフラ、09=評価、10=DevOps

---

## 技術スタック

| カテゴリ | 選定技術 | 詳細ドキュメント |
|---|---|---|
| 主言語 | Python 3.12+ (asyncio), uv | 01 |
| MCブリッジ | TypeScript (Mineflayer) | 05 |
| 共有状態 | Redis 7+ (Valkey互換) | 01, 08 |
| 分散実行 | asyncio + multiprocessing → Ray 2.x（段階的） | 08 |
| ベクトルDB | Qdrant | 04 |
| 永続化DB | PostgreSQL 16 | 08 |
| LLM抽象化 | OpenAI SDK | 02 |
| 監視 | Prometheus + Grafana | 10 |
| ログ | Grafana Loki | 10 |
| 実験管理 | MLflow + Hydra | 10 |
| CI/CD | GitHub Actions | 10 |
| コンテナ | Docker + Kubernetes | 08, 10 |
| MCサーバー | Pufferfish + Velocity | 05, 08 |

---

## 主要な技術的判断

### 確定済み

| 判断 | 内容 | 根拠 |
|---|---|---|
| ベクトルDB | Qdrant（agent_idパーティショニング前提） | 04, review |
| MCサーバー | Pufferfish（フォールバック: Paper） | 05 |
| Python | 3.12+, uv で実行 | 01 |
| CC実行間隔 | 動的 1-15秒（コスト制約付き、最大12回/分） | 03 |
| 記憶忘却 | lambda=0.5, 時間単位=時間, 半減期≈1.4h | 04 |
| モジュール実行 | 非同期独立実行（asyncio.Task） | 01 |
| API価格基準 | GPT-4o $2.50/$10.00, Haiku 4.5 $1.00/$5.00 | 02, review |

### Phase 0で解決済み

1. **Python-TypeScriptブリッジ方式**: **ZMQ**を採用（REQ-REP + PUB-SUB）。gRPC/RESTとの比較はPhase 0後半で判断予定だったが、ZMQで十分な性能が確認されたためそのまま採用
2. **CC情報圧縮アルゴリズム**: **テンプレートベース**を採用。6セクション構造の固定テンプレートで圧縮、情報保持率>0.8を達成
3. **SASスキーマのフリーズポイント**: Phase 1開始前に最終判断予定

### Phase 1で解決済み

1. **SASスキーマの最終フリーズ**: Phase 1でsas_phase1.py Mixin実装により確定

---

## コスト概要

| フェーズ | スケール | 月額コスト（インフラ+LLM） | 期間 |
|---|---|---|---|
| Phase 0 (MVP) | 1-5体 | $50-100 | 3-4ヶ月 |
| Phase 1 (基盤) | 10体 | $810-1,450 | 3-4ヶ月 |
| Phase 2 (スケール) | 50体 | $3,710-6,910 | 4-5ヶ月 |
| Phase 3 (文明) | 500体 | $25,400-46,200 | 2-3ヶ月 |

総プロジェクトコスト（人件費+予備費25%含む）: **4,700-10,600万円**（人件費が60-70%）

---

## プロジェクト状態

- **完了**: 技術調査（10ドキュメント）、レビュー（5観点統合）、ロードマップ策定
- **完了**: Phase 0 MVP実装（323テスト全通過）
  - SAS: Redis実装 + InMemorySAS（テスト用）
  - Scheduler: 3ティア並列実行（FAST/MID/SLOW）
  - CC: テンプレート圧縮 + LLM判断 + ブロードキャスト
  - Memory: WorkingMemory（容量10） + ShortTermMemory（容量100）
  - LLM: OpenAIProvider + MockLLMProvider + LLMCache（LRU+TTL）
  - Bridge: ZMQクライアント（REQ-REP + PUB-SUB）+ TypeScript Mineflayer bot
  - Skills: レジストリ + 7基本スキル + SkillExecutor
  - ActionAwareness: ルールベース期待-実際比較
  - Config: PianoSettings（pydantic-settings、環境変数対応）
  - CI: GitHub Actions（Python 3.12/3.13、pytest + ruff + mypy）
  - Docker: Redis + PostgreSQL + Pufferfish
- **完了**: Phase 1 実装（1,125テスト全通過）
  - 認知モジュール: GoalGeneration, Planning, SelfReflection, Talking
  - 社会認知: SocialAwareness, Personality(BigFive), SocialGraph, EmotionTracking
  - 記憶: LTM Store(Qdrant), LTM Search(忘却曲線), Memory Consolidation(STM→LTM)
  - LLM基盤: ModelTiering(3層), LLMGateway(priority queue+circuit breaker)
  - 行動認識: ActionAwareness NN(~100Kパラメータ), NNTrainer(SGD+momentum)
  - 評価: ItemCollectionBenchmark, SocialCognitionMetrics
  - インフラ: Checkpoint/Restore, SAS Phase1 Mixin, Orchestrator(10エージェント管理)
  - スキル拡張: Social Skills(trade/gift/vote), Advanced Skills(craft chain/farming/combat)
  - ブリッジ拡張: Protocol extensions(batch commands, validation, serialization)
- **完了**: Phase 2 実装（1,979テスト全通過）
  - スケーリング: WorkerPool, Supervisor, Sharding, ResourceLimiter
  - 可観測性: 構造化ログ, Prometheusメトリクス, トレーシング
  - 社会認知拡張: CollectiveIntelligence, InfluencerAnalysis
  - 評価拡張: GovernanceMetrics, MemeTracker, PerformanceBenchmark, RoleInference
  - LLM拡張: PromptCache(セマンティックキャッシュ)
  - ブリッジ拡張: VelocityProxy(マルチサーバー)
  - インフラ: DistributedCheckpoint, Kubernetes manifests, Grafana/Prometheus監視基盤
  - CI: Phase 2パイプライン
  - CLIランチャー: `piano` コマンド（--agents, --ticks, --mock-llm, --config, --log-level, --no-bridge）
  - TLS通信: Redis SSL, CurveZMQ(Python+TypeScript), Qdrant HTTPS/API Key
  - K8s NetworkPolicy: default-deny + コンポーネント間通信許可（11ポリシー）
  - NetworkXベンチマーク: 10/50/100/500体でのグラフ操作性能計測
  - 障害注入テスト: Redis/Bridge/LLM/Agent障害シミュレータ（chaos framework）
  - E2Eテスト基盤: --run-e2eフラグ、InMemorySAS使用の最小E2E
  - Grafanaアラート: コスト/エラー率/TPS/メモリの4種アラートルール
- **完了**: E2Eシミュレーション接続（2,079テスト全通過、MCサーバー実動作確認済み）
  - BridgePerceptionModule: Bridge PUB/SUBイベント→SAS PerceptData変換
  - ChatBroadcaster: TalkingModule発話→Bridge chat送信（asyncio.Lock二重送信防止）
  - BridgeManager: マルチBridgeClient管理（並列connect/disconnect/check）
  - BridgeHealthMonitor: connected/degraded/disconnected/stale状態監視
  - ActionMapper: CC action名→Skill名変換 + create_full_registry()
  - SkillExecutor統合: on_broadcastでmap_action()呼び出し
  - main.py統合: --no-bridgeフラグ、bridge接続時にPerception/SkillExecutor/ChatBroadcaster自動登録
  - TSハンドラ: basic/social/combat/advanced + perception拡張 + launcher.ts（マルチボット）
  - Docker: Dockerfile.agent, Dockerfile.bridge, docker-compose.sim.yml, .dockerignore, non-rootユーザー
  - MC接続実証: Paper 1.20.4サーバーで3ボット×5tickの動作確認完了
- **次のステップ**:
  1. 言行一致改善率30%測定（品質ゲート）
  2. 大規模E2E検証（50体MC接続、スケーリング動作確認）
  3. Phase 3準備（500体規模、文明シミュレーション設計）

---

## 参照

- 論文: [Project Sid: Many-agent simulations toward AI civilization (arXiv:2411.00114)](https://arxiv.org/abs/2411.00114)
- 論文分析: [docs/index.md](../index.md)
