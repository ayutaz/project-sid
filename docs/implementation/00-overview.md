# Project Sid 再現実装 — 全体概要

## プロジェクト概要

本プロジェクトは、Altera.ALによる研究論文「Project Sid: Many-agent simulations toward AI civilization」で提案された**PIANOアーキテクチャ**（Parallel Information Aggregation via Neural Orchestration）の再現実装を目指す技術調査・設計プロジェクトである。

### PIANOアーキテクチャとは

PIANOは、10〜1000以上のAIエージェントがMinecraft環境内で社会的に振る舞うための認知アーキテクチャである。2つの根本的課題を解決する:

1. **並行性（Concurrency）**: 高速反射モジュール（非LLM）と低速熟慮モジュール（LLM）が異なる速度で並列実行
2. **一貫性（Coherence）**: 認知コントローラ（CC）が情報ボトルネック+ブロードキャストで全モジュール間の一貫性を制御

### 論文の主要成果

| 成果 | 詳細 |
|---|---|
| 単一エージェント | 30分で平均17種のアイテム獲得 |
| 社会認知 | 50体社会で r=0.807 の認知精度 |
| 専門化 | 6種類の役割が自発的に分化 |
| 民主的統治 | 税率が20%→9%に民主的に変更 |
| 文化伝播 | 町ごとの固有ミームパターン |
| 宗教伝播 | 飽和なしの継続的改宗 |

---

## ドキュメント構成

### 技術調査ドキュメント

| # | ドキュメント | 内容 |
|---|---|---|
| [00](./00-overview.md) | **全体概要（本文書）** | プロジェクト概要、ドキュメント構成 |
| [01](./01-system-architecture.md) | システムアーキテクチャ設計 | PIANO全体設計、SAS、モジュール間通信、技術選定 |
| [02](./02-llm-integration.md) | LLM統合戦略 | APIアブストラクション、プロンプト設計、コスト最適化 |
| [03](./03-cognitive-controller.md) | 認知コントローラ実装方針 | GWT、情報ボトルネック、ブロードキャスト機構 |
| [04](./04-memory-system.md) | 記憶システム設計 | WM/STM/LTM三層構造、ベクトルDB、検索戦略 |
| [05](./05-minecraft-platform.md) | Minecraft基盤技術 | Mineflayer、マルチボット、スキル実行 |
| [06](./06-social-cognition.md) | 社会的認知モジュール | 感情追跡、社会グラフ、性格モデリング、集合知 |
| [07](./07-goal-planning.md) | 目標・計画システム | 目標生成、HTN計画、行動認識NN、自己省察 |
| [08](./08-infrastructure.md) | インフラ・スケーリング | 分散実行、LLMゲートウェイ、リソース見積もり |
| [09](./09-evaluation.md) | 評価・ベンチマーク | 文明的ベンチマーク3軸、データ収集、可視化 |
| [10](./10-devops.md) | DevOps・開発ワークフロー | CI/CD、コンテナ化、モニタリング、実験管理 |

### レビュードキュメント

| ドキュメント | 内容 |
|---|---|
| [review-security](./review-security.md) | セキュリティリスク分析（Critical 3件、High 4件、Medium 5件、Low 3件） |
| [review-cost](./review-cost.md) | コスト試算・実現可能性評価（MVP: Go、50体: Caution、1000体: Stop→再検討） |
| [review-oss](./review-oss.md) | OSSエコシステム評価（必須5件、推奨4件、参考5件、独自実装6件） |
| [review-testing](./review-testing.md) | テスト戦略設計（Unit 75%、Integration 20%、E2E 5%） |
| [review-integration](./review-integration.md) | 整合性チェック（Error 2件、Warning 10件、カバレッジ96%） |

### 統合ドキュメント

| ドキュメント | 内容 |
|---|---|
| [roadmap](./roadmap.md) | フェーズ別ロードマップ（Phase 0-3、9-13ヶ月） |

---

## 技術スタック

| カテゴリ | 選定技術 |
|---|---|
| 主言語 | Python 3.12+ (asyncio) |
| MCブリッジ | TypeScript (Mineflayer) |
| 共有状態 | Redis 7+ |
| 分散実行 | Ray 2.x |
| ベクトルDB | Qdrant |
| 永続化DB | PostgreSQL 16 |
| LLM抽象化 | LiteLLM |
| 監視 | Prometheus + Grafana |
| ログ | Grafana Loki |
| 実験管理 | MLflow + Hydra |
| CI/CD | GitHub Actions |
| コンテナ | Docker + Kubernetes |
| MCサーバー | Pufferfish + Velocity |

---

## 主要な技術的判断

### 解決済み（レビューにより確定）

1. **ベクトルDB**: Qdrantに統一（Doc04推奨、pgvectorは不採用）
2. **MCサーバー**: Pufferfishに統一（最高性能）
3. **Python最低バージョン**: 3.12に統一
4. **CC実行間隔**: 動的（1-15秒、文脈依存）に統一

### 未解決（Phase 0で決定必要）

1. **Python-TypeScriptブリッジ方式**: ZMQ vs gRPC vs REST
2. **CC情報圧縮アルゴリズム**: テンプレートベース vs LLMベース
3. **SASスキーマのフリーズポイント**: いつ仕様確定するか

---

## コスト概要

| フェーズ | スケール | 月額コスト |
|---|---|---|
| Phase 0 (MVP) | 1-5体 | $50-100 |
| Phase 1 (基盤) | 10体 | $1,450-2,570 |
| Phase 2 (スケール) | 50体 | $6,910-12,510 |
| Phase 3 (文明) | 500体 | $44,600-84,600 |

**LLMコストが全体の90%以上**を占める。モデルティアリングとプロンプトキャッシングが最重要の最適化戦略。

---

## 次のステップ

1. **Phase 0の開発チーム編成**（2-3人）
2. **Python-Mineflayerブリッジの技術PoC**
3. **SASスキーマの初期設計とフリーズ**
4. **CC MVPの実装と検証**
5. **アイテム収集ベンチマーク（1体）での初回検証**

---

## 参照

- 論文: [Project Sid: Many-agent simulations toward AI civilization (arXiv:2411.00114)](https://arxiv.org/abs/2411.00114)
- 論文分析: [docs/index.md](../index.md)
