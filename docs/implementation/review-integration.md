# 統合・整合性レビュー

[← 実装ドキュメント](../index.md)

---

## 目次

1. [レビュー概要](#1-レビュー概要)
2. [整合性チェック結果表](#2-整合性チェック結果表)
3. [矛盾・不整合の一覧と修正提案](#3-矛盾不整合の一覧と修正提案)
4. [論文要件のカバレッジマトリクス](#4-論文要件のカバレッジマトリクス)
5. [クリティカルパス図](#5-クリティカルパス図)
6. [統合リスク一覧](#6-統合リスク一覧)
7. [用語の統一性チェック](#7-用語の統一性チェック)
8. [実装優先順位の妥当性評価](#8-実装優先順位の妥当性評価)
9. [総合評価と推奨事項](#9-総合評価と推奨事項)

---

## 1. レビュー概要

### 1.1 レビュー対象

| # | ドキュメント | ファイル |
|---|---|---|
| 01 | システムアーキテクチャ設計 | `01-system-architecture.md` |
| 02 | LLM統合戦略 | `02-llm-integration.md` |
| 03 | 認知コントローラ実装方針 | `03-cognitive-controller.md` |
| 04 | 記憶システム設計 | `04-memory-system.md` |
| 05 | Minecraft基盤技術 | `05-minecraft-platform.md` |
| 06 | 社会的認知モジュール | `06-social-cognition.md` |
| 07 | 目標・計画システム | `07-goal-planning.md` |
| 08 | インフラ・スケーリング | `08-infrastructure.md` |
| 09 | 評価・ベンチマークシステム | `09-evaluation.md` |
| 10 | DevOps・開発ワークフロー | `10-devops.md` |

### 1.2 レビュー観点

1. **技術的整合性**: 言語・フレームワーク・DB選定の一貫性、データ構造互換性、処理フロー連続性
2. **用語の統一性**: ドキュメント間、および論文との用語一致
3. **依存関係の矛盾**: バージョン互換性、ライブラリ競合、前提条件の矛盾
4. **論文要件の網羅性**: 論文で記述された機能・実験の実装カバレッジ
5. **実装優先順位の妥当性**: 段階的実装の順序・依存関係の適切性
6. **見落とされた技術的リスク**: 複数ドキュメント横断でのみ顕在化するリスク

---

## 2. 整合性チェック結果表

### 2.1 技術スタック整合性

| チェック項目 | 状態 | 詳細 |
|---|---|---|
| プログラミング言語（Python） | **Warning** | Doc01はPython 3.12+、Doc10はPython 3.11/3.12。最低バージョンの統一が必要 |
| プログラミング言語（TypeScript/JS） | **Error** | Doc02はTypeScript、Doc05はJavaScript/Node.js、Doc07はTypeScript。Pythonメインとの統合方針が不明確 |
| Redis | OK | 全ドキュメントでRedis 7+を一貫して採用 |
| PostgreSQL | OK | 全ドキュメントでPostgreSQL 16を一貫して採用 |
| ベクトルDB | **Warning** | Doc04はQdrant推奨、Doc08はpgvector言及、Doc10はPostgreSQL+pgvectorをCI/CDに組込み |
| 非同期フレームワーク | OK | Doc01/08でasyncioを一貫して採用 |
| 分散計算 | OK | Doc01/08でRay 2.xを一貫して採用 |
| Minecraftサーバー | **Warning** | Doc05はPufferfish推奨、Doc10はPaper/Fabricと記載。統一が必要 |
| パッケージ管理 | OK | Doc10でuv採用を一貫して記載 |
| LLM抽象化 | OK | Doc02でLiteLLM採用を明記 |

### 2.2 アーキテクチャ設計整合性

| チェック項目 | 状態 | 詳細 |
|---|---|---|
| SAS（共有エージェント状態）設計 | OK | Doc01の定義がDoc03/04/06/07で参照されている |
| モジュールスケジューリング | **Error** | Doc01はtick-based（50ms）、Doc03はCC interval 1-15秒、Doc08はCC interval 2.0秒。定義不統一 |
| モジュール速度分類 | **Warning** | Doc01の速度分類（Fast/Mid/Slow/Selective）とDoc03のCC動的スケジューリング間で整合性が不十分 |
| 記憶三層構造 | OK | Doc01/04で一貫（WM/STM/LTM） |
| Python-Node.jsブリッジ | **Warning** | Doc05はZMQ/gRPCを提案、Doc08はREST/gRPCに言及。通信方式の最終決定なし |
| 社会グラフ実装 | OK | Doc06でNetworkX DiGraphを明確に定義 |
| 性格モデル | OK | Doc06でBig Five一貫採用 |
| 目標階層構造 | OK | Doc07でLong-term/Mid-term/Social/Immediateを明確に定義 |

### 2.3 データフロー整合性

| チェック項目 | 状態 | 詳細 |
|---|---|---|
| 環境→SAS | OK | Doc01/05で定義済み |
| SAS→各モジュール | OK | Doc01の`read_sections`/`write_sections`で制御 |
| CC→ブロードキャスト | OK | Doc03で詳細設計済み |
| LLMリクエストフロー | OK | Doc02→Doc08 LLM Gatewayで一貫 |
| メモリ遷移（WM→STM→LTM） | OK | Doc04で遷移ルール明確 |
| エージェント間通信 | OK | Doc08でRedis Pub/Subを一貫採用 |
| イベント収集→分析パイプライン | **Warning** | Doc09でTimescaleDB/MongoDB/Neo4jを追加提案。Doc01/08の基本スタックとの統合方針不明 |

### 2.4 インフラ・運用整合性

| チェック項目 | 状態 | 詳細 |
|---|---|---|
| コンテナ化 | OK | Doc10で詳細定義、Doc08のk8s構成と整合 |
| モニタリング | OK | Doc08/10でPrometheus+Grafana一貫 |
| ログ管理 | OK | Doc10でLoki採用を明確に定義 |
| 実験管理 | OK | Doc10でHydra+MLflow採用 |
| スケーリング戦略 | OK | Doc08/10で4段階スケーリング一貫 |
| コスト見積もり | **Warning** | Doc02とDoc10でLLMコスト試算の前提条件が微妙に異なる |

---

## 3. 矛盾・不整合の一覧と修正提案

### 3.1 [Error] コード言語の混在

**問題**: 実装言語が統一されていない。

| ドキュメント | 使用言語 |
|---|---|
| Doc01 (System Architecture) | Python（コード例全てPython） |
| Doc02 (LLM Integration) | **TypeScript**（インタフェース定義全てTS） |
| Doc05 (Minecraft Platform) | **JavaScript/TypeScript**（Mineflayer API例） |
| Doc07 (Goal Planning) | **TypeScript**（インタフェース定義全てTS） |
| Doc03/04/06/08/09/10 | Python |

**影響**: 実装者が各ドキュメントの設計をどの言語で実装すべきか不明。特にDoc02のLLM抽象化層とDoc07の目標システムがTypeScriptで記述されているが、Doc01のPythonメインアーキテクチャとの統合方法が未定義。

**修正提案**:
1. Python側の設計として統一的なインタフェース定義を追加
2. TypeScriptはMinecraftブリッジ（Doc05）のみに限定する方針を明記
3. Doc02/07のTypeScriptインタフェースに対応するPython型定義（dataclass/Pydantic）を併記

### 3.2 [Error] モジュール実行間隔の矛盾

**問題**: 認知コントローラ（CC）の実行間隔が3つのドキュメントで異なる。

| ドキュメント | CC実行間隔 |
|---|---|
| Doc01 | 4 tick = 200ms（tick_interval=4, 1tick=50ms） |
| Doc03 | BASE_INTERVAL=5.0秒、MIN=1.0秒、MAX=15.0秒（動的調整） |
| Doc08 | 2.0秒 |
| Doc10 | 2000ms = 2.0秒（Helm values） |

**影響**: CCの実行頻度はシステム全体の応答性・LLMコスト・一貫性に直接影響する中核パラメータ。3箇所で異なる値は実装時の混乱を招く。

**修正提案**:
1. Doc03の動的スケジューリング設計（1-15秒範囲、基準5秒）を正式な仕様とする
2. Doc01の200ms定義はtick-based schedulerの粒度として再定義（CCの実行間隔ではなくスケジューラのポーリング間隔）
3. Doc08/10の2.0秒は開発初期のデフォルト値として位置づけ
4. 全ドキュメントにCC間隔の正式定義への参照リンクを追加

### 3.3 [Warning] ベクトルDB選定の不一致

**問題**: 長期記憶のベクトルDBが複数の選択肢で揺れている。

| ドキュメント | 推奨 | 根拠 |
|---|---|---|
| Doc04 | **Qdrant** | payload-aware HNSW、バイナリ量子化、Raftクラスタリング |
| Doc08 | pgvector（代替として言及） | PostgreSQL統合の簡素さ |
| Doc10 | pgvector/pgvector:pg16（CIイメージ） | テスト環境での利便性 |

**影響**: 記憶システムは全モジュールが依存する基盤。DB選定の揺れは開発の初期段階で技術的負債となる。

**修正提案**:
1. **本番環境**: Doc04の通りQdrantを正式採用（性能・スケーラビリティの根拠が充実）
2. **開発・テスト環境**: pgvectorを許容（Docker Compose簡素化のため）
3. 環境に応じたDB切替の抽象化レイヤーをDoc01のアーキテクチャに追加
4. Doc10のCIサービス定義にQdrantコンテナも追加

### 3.4 [Warning] Minecraftサーバー種別の不一致

**問題**: Minecraftサーバー実装が統一されていない。

| ドキュメント | 推奨サーバー |
|---|---|
| Doc05 | Pufferfish（性能最適化、フォーク階CraftBukkit→Spigot→Paper→Pufferfish） |
| Doc10 | Paper/Fabric（DockerfileでTYPE=PAPER指定） |

**影響**: サーバー種別はプラグイン互換性とパフォーマンスに影響。Pufferfishはマルチスレッド対応でエージェント大量接続に有利だが、Docker公式イメージはPaperベース。

**修正提案**:
1. 開発環境: Paper（Docker公式イメージ`itzg/minecraft-server`の安定性優先）
2. 本番環境: Pufferfish（大規模エージェント対応、カスタムDockerイメージ作成）
3. Doc10にサーバー種別の環境別使い分け方針を追記

### 3.5 [Warning] Pythonバージョンの揺れ

**問題**: 最低Pythonバージョンが統一されていない。

| ドキュメント | バージョン |
|---|---|
| Doc01 | Python 3.12+ |
| Doc10 | Python 3.11/3.12（マトリクスビルド） |

**修正提案**: Doc01の通りPython 3.12+に統一。3.11サポートは不要（3.12のtypeパラメータや性能改善を活用するため）。Doc10のマトリクスビルドから3.11を削除。

### 3.6 [Warning] LLMコスト見積もりの前提差異

**問題**: LLMコスト試算が複数のドキュメントで異なる前提に基づく。

| ドキュメント | 対象 | 見積もり |
|---|---|---|
| Doc02 | 10エージェント/時 | ~$8-15（最適化後） |
| Doc02 | 500エージェント/時 | ~$250-500（最適化後） |
| Doc04 | 500エージェント/2.5時間 | ~$2.21（記憶システムのみ） |
| Doc10 | 500エージェント/2.5時間 | ~$292.50（全LLM呼び出し、gpt-4o-mini） |

**影響**: 予算計画とスケーリング判断に影響。

**修正提案**: 統合的なコスト見積もりドキュメントを別途作成し、以下を整理:
1. モジュール別LLM呼び出し頻度と使用モデル
2. スケール別（10/50/500/1000エージェント）の総コスト
3. 最適化前/後の比較

### 3.7 [Warning] 評価パイプラインのデータストア追加

**問題**: Doc09の評価パイプラインがTimescaleDB、MongoDB、Neo4jを追加提案しているが、Doc01/08の基本技術スタック（Redis + PostgreSQL）との統合方針が不明。

**修正提案**:
1. 評価・分析専用のデータストアとして位置づけ、エージェントランタイムとは分離
2. PostgreSQL + TimescaleDB拡張で時系列データを統合（追加DBを削減）
3. Neo4jの代わりにNetworkX + PostgreSQLで社会グラフ分析を実施（Doc06と整合）
4. MongoDBの代わりにPostgreSQL JSONB型で非構造化データを管理

### 3.8 [Warning] メモリモジュールの速度分類

**問題**: メモリモジュールの速度分類が不統一。

| ドキュメント | 分類 |
|---|---|
| Doc01 | "Variable"（可変） |
| Doc04 | WM=毎tick（高速）、STM=非同期（中速）、LTM=低頻度（低速） |
| Doc08 | Memory Manager interval=3.0秒 |

**修正提案**: Doc04の三層別速度分類を正式定義とする:
- WM更新: 毎tick（50ms）= Fast
- STM操作: 非同期（~500ms）= Mid
- LTM検索/保存: 低頻度（~3秒）= Slow
- Doc01/08を上記に合わせて修正

---

## 4. 論文要件のカバレッジマトリクス

### 4.1 PIANOアーキテクチャ要件

| 論文要件 | 実装Doc | カバレッジ | 備考 |
|---|---|---|---|
| 並行モジュール実行 | Doc01 | **Full** | asyncio + tick-based scheduler |
| 共有エージェント状態（SAS） | Doc01 | **Full** | Redis + optimistic concurrency control |
| 認知コントローラ（CC） | Doc03 | **Full** | GWT + 情報ボトルネック + ブロードキャスト |
| 情報ボトルネック | Doc03 | **Full** | 3軸優先度 + 圧縮レベル |
| ブロードキャスト機構 | Doc03 | **Full** | 条件付け強度設計済み |
| ステートレスモジュール設計 | Doc01 | **Full** | PianoModule基底クラス |
| 高速反射モジュール（non-LLM NN） | Doc07 | **Full** | Action Awareness NN設計済み |
| 低速熟慮モジュール | Doc07 | **Full** | Goal Generation + Planning |
| 選択的起動モジュール | Doc06 | **Full** | Social Awareness trigger条件 |

### 4.2 個別モジュール要件

| 論文モジュール | 実装Doc | カバレッジ | 備考 |
|---|---|---|---|
| Memory（WM/STM/LTM） | Doc04 | **Full** | 三層構造 + 遷移ルール + Qdrant |
| Action Awareness | Doc07 | **Full** | non-LLM NN + discrepancy検出 |
| Goal Generation | Doc07 | **Full** | 知識グラフ + HTN + DEPS |
| Social Awareness | Doc06 | **Full** | 感情追跡 + 社会グラフ + Big Five |
| Talking | Doc02 | **Partial** | LLMプロンプト設計あるが、モジュール単独設計なし |
| Skill Execution | Doc05 | **Full** | Voyager-style スキルライブラリ |
| Cognitive Controller | Doc03 | **Full** | GWT + LIDA実装 |
| Self-Reflection | Doc07 | **Full** | Reflexion-inspired 3-stage |
| Planning | Doc07 | **Full** | HTN + LLM hybrid |

### 4.3 実験再現要件

| 論文実験 | エージェント数 | 実装Doc | カバレッジ | 備考 |
|---|---|---|---|---|
| 単一エージェント（30分） | 25 | Doc05/07/09 | **Full** | アイテム収集ベンチマーク設計済み |
| 長期アイテム収集（4時間） | 49 | Doc05/09 | **Full** | 飽和検出メカニズム設計済み |
| 感情追跡 | 1+3 | Doc06 | **Full** | 感情追跡 + アブレーション |
| 食料配分 | 1+4 | Doc06 | **Partial** | 感情→行動マッピングの具体的実装未詳細 |
| 社会認知（50体） | 50 | Doc06/09 | **Full** | Pearson相関 + observer threshold |
| 専門化（30体） | 30 | Doc07/09 | **Full** | 役割エントロピー + GPT-4o推論 |
| 集団ルール（29体） | 29 | Doc06/09 | **Full** | 税遵守 + 憲法修正 + 投票 |
| 文化ミーム（500体） | 500 | Doc08/09 | **Full** | ミーム検出 + 地理的伝播 |
| 宗教伝播（500体） | 500 | Doc06/09 | **Full** | SIRモデル + キーワード追跡 |

### 4.4 カバレッジサマリー

| カテゴリ | Full | Partial | Missing | カバー率 |
|---|---|---|---|---|
| PIANOアーキテクチャ | 9 | 0 | 0 | **100%** |
| 個別モジュール | 8 | 1 | 0 | **94%** |
| 実験再現 | 8 | 1 | 0 | **94%** |
| **合計** | **25** | **2** | **0** | **96%** |

**Partial項目の詳細**:
1. **Talkingモジュール**: LLMプロンプト設計はDoc02にあるが、独立したモジュール設計書が存在しない。CCブロードキャスト後の発話生成ロジック、会話管理（ターン制御、割り込み処理）の詳細が不足。
2. **食料配分実験**: 感情推測から行動選択への具体的な意思決定アルゴリズムが未定義。Doc06の感情追跡→Doc07の目標生成の連携フローとして補完可能だが、明示的な設計が必要。

---

## 5. クリティカルパス図

### 5.1 実装依存関係

```
Phase 0: 基盤
  ├── [P0-1] Python プロジェクト構造・CI/CD (Doc01/10)
  ├── [P0-2] Redis + PostgreSQL セットアップ (Doc01/08)
  └── [P0-3] Docker Compose ローカル環境 (Doc10)

Phase 1: コアランタイム
  ├── [P1-1] SAS (共有エージェント状態) ← P0-2
  ├── [P1-2] Module Scheduler (tick-based) ← P0-1
  ├── [P1-3] LLM Gateway (LiteLLM統合) ← P0-1
  └── [P1-4] Minecraft Bridge (Mineflayer) ← P0-3

Phase 2: 基本モジュール (ベースライン構成)
  ├── [P2-1] Memory System (WM/STM/LTM) ← P1-1
  ├── [P2-2] Skill Execution ← P1-4
  └── [P2-3] Cognitive Controller (基本版) ← P1-1, P1-3

Phase 3: 拡張モジュール (完全PIANO)
  ├── [P3-1] Action Awareness (non-LLM NN) ← P2-1, P2-2
  ├── [P3-2] Goal Generation ← P2-1, P1-3
  ├── [P3-3] Social Awareness ← P2-1, P1-3
  ├── [P3-4] Talking ← P2-3, P1-3
  ├── [P3-5] Planning ← P3-2
  ├── [P3-6] Self-Reflection ← P2-1, P3-1
  └── [P3-7] CC強化版 (動的スケジューリング) ← P2-3, P3-1〜P3-6

Phase 4: スケーリング
  ├── [P4-1] マルチエージェント通信 (Redis Pub/Sub) ← P2全完了
  ├── [P4-2] エージェントプロセス分散 (Ray) ← P4-1
  ├── [P4-3] マルチMinecraftサーバー (Velocity) ← P1-4
  └── [P4-4] Kubernetes デプロイ ← P4-1〜P4-3

Phase 5: 評価・実験
  ├── [P5-1] 単一エージェントベンチマーク ← P3全完了
  ├── [P5-2] マルチエージェント実験 (50体) ← P4-1
  ├── [P5-3] 文明実験 (500体) ← P4-4
  └── [P5-4] 評価パイプライン (メトリクス収集・分析) ← P5-1〜P5-3
```

### 5.2 クリティカルパス

最長パス（ボトルネック）:

```
P0-2 → P1-1 → P2-1 → P3-1 → P3-7 → P4-1 → P4-4 → P5-3
 (基盤DB) (SAS) (Memory) (AA) (CC強化) (通信) (k8s) (500体実験)
```

**ボトルネック分析**:
- **P1-1 SAS**: 全モジュールが依存。設計品質がシステム全体の性能を決定
- **P2-1 Memory**: WM/STM/LTM全層の実装が必要。Qdrant統合も含む
- **P3-7 CC強化版**: 全拡張モジュールの完了が前提。最も依存関係が多い
- **P4-4 k8s**: 大規模実験の前提。インフラ知識も必要

### 5.3 並列化可能な作業

以下は独立して並列開発可能:
- P1-3 (LLM Gateway) と P1-4 (Minecraft Bridge) は独立
- P3-1/P3-2/P3-3 (AA/Goal/Social) は互いに独立（全てP2-1に依存）
- P3-4 (Talking) はP3-1〜P3-3と並列可能
- P5-4 (評価パイプライン) はP5-1〜P5-3と並列開発可能（データモック使用）

---

## 6. 統合リスク一覧

### 6.1 技術的リスク

| ID | リスク | 重大度 | 影響範囲 | 関連Doc | 軽減策 |
|---|---|---|---|---|---|
| TR-01 | Python-Node.jsブリッジの性能ボトルネック | **高** | Doc01/05/08 | Minecraft操作の遅延がエージェント応答性に直結 | gRPC採用、バッチ処理、接続プール |
| TR-02 | LLMレート制限によるシステムスローダウン | **高** | Doc02/08 | 500エージェントで秒間数百のLLM呼び出し。レート制限で全体停止の恐れ | Doc02のフォールバック戦略、セマンティックキャッシュ、モデルティアリング |
| TR-03 | Redis単一障害点 | **中** | Doc01/04/08 | SAS + STM + Pub/Sub + キャッシュ全てRedis依存 | Redis Cluster構成、機能別Redis分離 |
| TR-04 | Action Awareness NNの学習データ不足 | **中** | Doc07 | 非LLM NNの学習に大量のMinecraft操作ログが必要だが、初期段階ではデータなし | ルールベースのフォールバック実装、段階的データ収集 |
| TR-05 | CC動的スケジューリングの安定性 | **中** | Doc03 | 動的間隔調整が発散/振動する可能性 | 変化率制限、ヒステリシスの導入 |
| TR-06 | Qdrantクラスタの運用複雑性 | **低** | Doc04/08 | 1000エージェント x 10,000 LTMレコードで大規模クラスタ必要 | 初期はシングルノード、段階的スケーリング |
| TR-07 | LLM評価の循環性 | **中** | Doc09 | LLMエージェントの行動をLLMで評価する循環構造 | 人間評価プロトコル（Cohen's Kappa）の並行実施 |

### 6.2 統合リスク（複数ドキュメント横断）

| ID | リスク | 重大度 | 関連Doc | 詳細 |
|---|---|---|---|---|
| IR-01 | SASスキーマと各モジュールの読み書きセクションの不整合 | **高** | Doc01 vs Doc03/04/06/07 | Doc01でSASスキーマを定義しているが、各モジュールDoc(03/04/06/07)が想定するフィールドとの完全なマッピングが未検証 |
| IR-02 | CCブロードキャストとTalkingモジュールの連携仕様不足 | **中** | Doc03 vs Doc02 | CCが決定をブロードキャストした後、Talkingモジュールが具体的にどのフィールドを参照してどう発話を生成するかの詳細仕様がない |
| IR-03 | Memory遷移とGoal Generationの知識グラフ同期 | **中** | Doc04 vs Doc07 | LTMに保存された経験とGoal Generationの知識グラフ間のデータ同期方法が未定義 |
| IR-04 | 評価パイプラインとランタイムのデータ形式統一 | **中** | Doc09 vs Doc01/08 | 評価パイプラインが期待するイベント形式と、ランタイムが出力する形式の統一スキーマが未定義 |
| IR-05 | Social AwarenessのNetworkX社会グラフとインフラのスケーリング | **高** | Doc06 vs Doc08 | NetworkXはインメモリグラフライブラリ。1000エージェントの社会グラフを各エージェントプロセスで保持するとメモリ爆発の恐れ |
| IR-06 | DevOps環境変数とHydra設定の二重管理 | **低** | Doc10 | Docker Compose環境変数とHydra YAML設定の間に重複・矛盾が発生する可能性 |

### 6.3 リスク重大度マトリクス

```
影響度
  高  │ TR-01    IR-01    TR-02
      │ IR-05
  中  │ TR-04    TR-03    TR-05
      │          IR-02    IR-03
      │          IR-04    TR-07
  低  │ IR-06    TR-06
      └──────────────────────────
        低       中       高
                    発生確率
```

---

## 7. 用語の統一性チェック

### 7.1 ドキュメント間の用語揺れ

| 概念 | Doc01 | Doc03 | Doc04 | Doc06 | Doc07 | 論文 | 推奨統一名 |
|---|---|---|---|---|---|---|---|
| 共有状態 | Shared Agent State (SAS) | 共有エージェント状態 | - | - | - | Shared Agent State | **SAS (Shared Agent State)** |
| 認知コントローラ | CC | Cognitive Controller (CC) | - | - | - | Cognitive Controller | **CC (Cognitive Controller)** |
| 行動認識 | Action Awareness | - | - | - | Action Awareness | Action Awareness | **Action Awareness** |
| 感情スコア | - | - | - | sentiment score (0-10) | - | likeability score | **sentiment score** (論文のlikeabilityに対応) |
| 実行間隔 | tick_interval | interval / schedule_interval | - | - | - | - | **tick_interval**（統一） |
| 目標 | - | - | - | social goal | goal hierarchy | social goals | **social goal**（社会的目標） |

### 7.2 論文用語と実装用語の対応

| 論文での表現 | 実装ドキュメントでの表現 | 整合性 |
|---|---|---|
| PIANO | PIANO | OK |
| Global Workspace Theory | GWT / Global Workspace Theory | OK |
| Information Bottleneck | 情報ボトルネック / information bottleneck | OK |
| Broadcast | ブロードキャスト / broadcast | OK |
| Action Awareness | Action Awareness / 行動認識 | OK（日英混在は許容） |
| Social Awareness | Social Awareness / 社会認識 / 社会的認知 | **Warning** |
| Wisdom of Crowds | 集合知 / Wisdom of Crowds | OK |
| Role Entropy | 役割エントロピー / Role Entropy | OK |

**修正提案**: "Social Awareness" と "社会認識" を統一し、"社会的認知" は実験結果の文脈（social cognition accuracy）のみで使用する。

---

## 8. 実装優先順位の妥当性評価

### 8.1 各ドキュメントの優先順位提案

| ドキュメント | 提案する実装順序 | 根拠 |
|---|---|---|
| Doc01 | Phase 1: コアランタイム | 全モジュールの基盤 |
| Doc02 | Phase 1: LLM統合 | LLM依存モジュール全てに先行 |
| Doc03 | Phase 2: CC基本 → Phase 3: CC強化 | 一貫性制御の段階的実装 |
| Doc04 | Phase 2: WM/STM → Phase 3: LTM/Qdrant | メモリの段階的実装 |
| Doc05 | Phase 1: Mineflayer → Phase 4: マルチサーバー | 環境接続の段階的拡張 |
| Doc06 | Phase 3: 社会認識 | 拡張モジュール |
| Doc07 | Phase 3: Goal/Planning/AA | 拡張モジュール（一部並列可能） |
| Doc08 | Phase 1: ローカル → Phase 4: 分散 | インフラの段階的構築 |
| Doc09 | Phase 5: 評価 | 実験実行後 |
| Doc10 | Phase 0〜1: CI/CD/Docker | 開発基盤として最初期 |

### 8.2 評価結果

| 評価項目 | 判定 | 詳細 |
|---|---|---|
| 依存順序の妥当性 | **OK** | 基盤→コアランタイム→基本モジュール→拡張モジュール→スケーリング→実験の流れは論理的 |
| ベースライン先行の妥当性 | **OK** | 論文のベースライン構成（Memory+SkillExec+CC）を先に実装しアブレーション可能にする設計は適切 |
| 並列化の十分性 | **Warning** | Phase 3の各モジュールは並列開発可能だが、SASスキーマのフリーズタイミングが未定義。スキーマが頻繁に変更されると全モジュールに影響 |
| 評価の後回しリスク | **Warning** | Doc09の評価パイプラインがPhase 5と遅い。評価メトリクスの定義はPhase 1で、収集基盤はPhase 2で開始すべき |
| DevOpsの先行は適切 | **OK** | Doc10のCI/CD/Docker設定をPhase 0で行うのは適切。開発効率に直結 |

### 8.3 推奨する優先順位調整

1. **評価メトリクスの定義を前倒し**: Doc09の定量的指標定義（Role Entropy、Social Cognition Accuracy等）をPhase 2で確定し、各モジュールに計測ポイントを組込む
2. **SASスキーマのフリーズポイント設定**: Phase 2完了時点でSASスキーマv1.0をフリーズし、Phase 3はこのスキーマに準拠して開発
3. **Talkingモジュール設計書の追加**: 現在独立した設計書がないため、Phase 2の作業として追加

---

## 9. 総合評価と推奨事項

### 9.1 全体評価

全10ドキュメントは、Project Sid論文の要件を**96%のカバレッジ**で網羅しており、個々のドキュメントの技術的深度は高い。PIANOアーキテクチャの中核概念（並行性、一貫性、SAS、CC、GWT）は全ドキュメントを通じて一貫して反映されている。

ただし、複数ドキュメントにまたがる**統合レベルの設計**には以下の課題が残る:

### 9.2 対応必須（Error）

| # | 項目 | 対応策 |
|---|---|---|
| 1 | コード言語の混在（Python vs TypeScript） | 各ドキュメントにPythonインタフェース定義を追加。TypeScriptはMCブリッジのみに限定する方針を明記 |
| 2 | CC実行間隔の矛盾（200ms vs 2秒 vs 5秒） | Doc03の動的スケジューリングを正式仕様とし、他ドキュメントを更新 |

### 9.3 対応推奨（Warning）

| # | 項目 | 対応策 |
|---|---|---|
| 1 | ベクトルDB選定の不一致 | 本番Qdrant + 開発pgvectorの二環境戦略を明記 |
| 2 | MCサーバー種別の不一致 | 環境別使い分け（開発Paper / 本番Pufferfish）を明記 |
| 3 | Pythonバージョン揺れ | 3.12+に統一 |
| 4 | LLMコスト見積もりの前提差異 | 統合コスト見積もりドキュメントの作成 |
| 5 | 評価パイプラインの追加DB | PostgreSQL拡張で統一（TimescaleDB拡張 + JSONB） |
| 6 | メモリモジュール速度分類の不統一 | Doc04の三層別定義を正式仕様に |
| 7 | NetworkX社会グラフのスケーラビリティ | 中央管理サービス化 or Redis Graph検討 |
| 8 | Talkingモジュールの設計書不足 | 独立した設計ドキュメントの追加 |
| 9 | SASスキーマのフリーズポイント未定義 | Phase 2完了時にv1.0フリーズ |
| 10 | 評価メトリクス定義の前倒し | Phase 2で指標定義確定 |

### 9.4 追加作成が推奨されるドキュメント

| ドキュメント | 内容 | 優先度 |
|---|---|---|
| Talkingモジュール設計書 | CC→Talking連携、会話管理、ターン制御 | 高 |
| 統合コスト見積もり | 全モジュール・全スケールの統合コスト分析 | 中 |
| SASスキーマ定義書 | 全フィールドの型定義、各モジュールのR/Wマッピング | 高 |
| Python-Node.jsブリッジ仕様書 | 通信プロトコル、シリアライゼーション、エラーハンドリング | 中 |
| 統合テストシナリオ | 複数モジュール連携のE2Eテストケース定義 | 中 |

---

## 参照元ドキュメント

- [01-system-architecture.md](./01-system-architecture.md)
- [02-llm-integration.md](./02-llm-integration.md)
- [03-cognitive-controller.md](./03-cognitive-controller.md)
- [04-memory-system.md](./04-memory-system.md)
- [05-minecraft-platform.md](./05-minecraft-platform.md)
- [06-social-cognition.md](./06-social-cognition.md)
- [07-goal-planning.md](./07-goal-planning.md)
- [08-infrastructure.md](./08-infrastructure.md)
- [09-evaluation.md](./09-evaluation.md)
- [10-devops.md](./10-devops.md)
- [論文分析ドキュメント](../index.md)

---

[← 実装ドキュメント](../index.md)
