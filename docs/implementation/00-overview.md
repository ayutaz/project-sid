# Project Sid 再現実装 — 総合概要

> 概要: PIANOアーキテクチャ再現実装の全ドキュメントのインデックス、技術判断、読み方ガイド
> 対応論文セクション: 全セクション横断
> 最終更新: 2026-02-23

---

## エグゼクティブサマリー

本プロジェクトは、Altera.ALの論文「Project Sid: Many-agent simulations toward AI civilization」で提案された **PIANOアーキテクチャ**（Parallel Information Aggregation via Neural Orchestration）の再現実装を目指す。PIANOは10〜1000以上のAIエージェントがMinecraft環境内で社会的に振る舞うための認知アーキテクチャであり、2つの根本課題を解決する:

1. **並行性（Concurrency）**: 高速反射モジュール（非LLM）と低速熟慮モジュール（LLM）が異なる速度で並列実行
2. **一貫性（Coherence）**: 認知コントローラ（CC）がGWT情報ボトルネック+ブロードキャストで全モジュール間の一貫性を制御

**現在の状態**: 技術調査・設計フェーズ完了。10本のコアドキュメント + 統合レビュー + ロードマップが整備済み。次のステップはPhase 0 MVPの実装。

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

### レビュー・計画

| ファイル | 概要 |
|---|---|
| [review-comprehensive](./review-comprehensive.md) | 5観点の統合レビュー（セキュリティ・コスト・OSS・テスト・統合） |
| [roadmap](./roadmap.md) | Phase 0-3 実装ロードマップ（9-13ヶ月、30-50人月） |

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
| 分散実行 | Ray 2.x | 08 |
| ベクトルDB | Qdrant | 04 |
| 永続化DB | PostgreSQL 16 | 08 |
| LLM抽象化 | LiteLLM | 02 |
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

### 未解決（Phase 0で決定）

1. **Python-TypeScriptブリッジ方式**: ZMQ vs gRPC vs REST
2. **CC情報圧縮アルゴリズム**: テンプレートベース vs LLMベース
3. **SASスキーマのフリーズポイント**: いつ仕様確定するか

---

## コスト概要

| フェーズ | スケール | 月額コスト（インフラ+LLM） | 期間 |
|---|---|---|---|
| Phase 0 (MVP) | 1-5体 | $50-100 | 2-3ヶ月 |
| Phase 1 (基盤) | 10体 | $1,450-2,570 | 2-3ヶ月 |
| Phase 2 (スケール) | 50体 | $6,910-12,510 | 3-4ヶ月 |
| Phase 3 (文明) | 500体 | $44,600-84,600 | 2-3ヶ月 |

総プロジェクトコスト（人件費含む）: **3,760-8,490万円**（人件費が60-70%）

---

## プロジェクト状態

- **完了**: 技術調査（10ドキュメント）、レビュー（5観点統合）、ロードマップ策定
- **次のステップ**: Phase 0 MVP実装
  1. Python-Mineflayerブリッジの技術PoC
  2. SASスキーマの初期設計とフリーズ
  3. CC MVPの実装と検証
  4. アイテム収集ベンチマーク（1体）での初回検証

---

## 参照

- 論文: [Project Sid: Many-agent simulations toward AI civilization (arXiv:2411.00114)](https://arxiv.org/abs/2411.00114)
- 論文分析: [docs/index.md](../index.md)
