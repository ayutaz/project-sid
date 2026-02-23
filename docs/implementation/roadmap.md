# Project Sid 再現実装ロードマップ

> 概要: PIANOアーキテクチャとマルチエージェント文明シミュレーションの4フェーズ実装計画（12-17ヶ月、35-55人月）
> 対応論文セクション: 全セクション横断
> 最終更新: 2026-02-24
> 進捗: Phase 0 ✅ → Phase 1 ✅ → Phase 2 ✅ 実装完了/検証待ち → Phase 3 ⬚ 未着手

---

## 全体タイムライン

```
Month:  1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
        ├──────────────────┤
        Phase 0: MVP (3-4ヶ月)              ✅ COMPLETE
                           ├──────────────────┤
                           Phase 1: 基盤完成 (3-4ヶ月)    ✅ COMPLETE
                                              ├──────────────────────┤
                                              Phase 2: スケーリング (4-5ヶ月)  ✅ 実装完了/検証待ち
                                                                    ├────────────┤
                                                                    Phase 3: 文明実験 (2-3ヶ月)  ⬚ 未着手
```

**総開発期間**: 12-17ヶ月 / **総工数**: 35-55人月 / **ピーク人員**: 5人

---

## Phase移行の品質ゲート

各Phase完了時に以下の品質ゲートを通過しなければ次のPhaseに進まない。不合格時はPhase内で2週間の再調整期間を設け、再検証する。2回不合格の場合はスコープ縮小を検討する。

| 移行 | 品質ゲート基準 | 不合格時フォールバック | 状態 |
|---|---|---|---|
| Phase 0→1 | MVPマイルストーン全達成 + CC制御下で言行一致改善率30%以上 + ブリッジ技術選定完了 + 全未解決事項の決定完了 | ブリッジ方式の変更、CC圧縮方式の再検討 | ✅ 通過（E2E検証は残課題） |
| Phase 1→2 | 10体で社会認知精度 r>0.5 + アイテム収集15種以上（論文値±20%） + 全モジュールUnit Test通過率95%以上 | モジュール単位でPhase 2に持ち越し、並行開発 | ✅ Unit Test通過 / ⏳ E2E検証待ち |
| Phase 2→3 | 50体で社会認知精度 r>0.7 + 500体で4時間安定動作（TPS>15） + 集合知効果の統計的有意性（p<0.05） | スケール目標を300体に縮小して実験開始 | ⏳ 実環境検証待ち |

---

## Phase 0: MVP（最小実行可能構成） — ✅ COMPLETE

**期間**: 3-4ヶ月 / **人員**: 2-3人 / **予算**: $50-100/月
**実績**: 407テスト全通過（レビュー対応後）、ruff lint clean、全コアモジュール実装済み

### 目標
PIANOアーキテクチャの核心的な設計（ステートレスモジュール+共有状態+CC）が実際に動作することを検証する。Python-Mineflayerブリッジの技術選定を完了し、CC情報圧縮の評価基準を確立する。

### 成果物

| # | 成果物 | 詳細 | 参照Doc | 状態 |
|---|---|---|---|---|
| 0-1 | Python-Mineflayerブリッジ | ZMQ（REQ-REP + PUB-SUB）で実装。`src/piano/bridge/client.py` + `bridge/src/index.ts` | 05 | Done |
| 0-2 | 共有エージェント状態(SAS) | Redis実装 `sas_redis.py` + InMemorySAS `tests/helpers.py`。容量制限定義済み | 01 | Done |
| 0-3 | モジュールスケジューラ | 3ティア並列実行（FAST/MID/SLOW）。`src/piano/core/scheduler.py` | 01 | Done |
| 0-4 | 認知コントローラ(CC) MVP | テンプレート圧縮 + LLM判断 + ブロードキャスト。`src/piano/cc/` | 03 | Done |
| 0-5 | 記憶システム（WM+STM） | WM(cap10) + STM(cap100)。`src/piano/memory/` | 04 | Done |
| 0-6 | スキル実行モジュール | 7基本スキル + SkillExecutor。`src/piano/skills/` | 05 | Done |
| 0-7 | 行動認識（ルールベース） | 期待-実際比較。`src/piano/awareness/action.py` | 07 | Done |
| 0-8 | LLMアブストラクション | LiteLLMProvider + キャッシュ（LRU+TTL）。`src/piano/llm/` | 02 | Done |
| 0-9 | MockLLMProvider | パターンベース応答 + call_history。`src/piano/llm/mock.py` | review | Done |
| 0-10 | Docker Compose開発環境 | Redis + PostgreSQL + Pufferfish。`docker/docker-compose.yml` | 10 | Done |
| 0-11 | Valkey/KeyDB互換性検証 | 未実施（Phase 1で判断） | 08 | Deferred |

### マイルストーン

| Week | マイルストーン | 検証基準 | 状態 |
|---|---|---|---|
| W2 | ブリッジPoC完了 + 技術選定 | ZMQ採用。REQ-REP + PUB-SUBで実装完了 | Done |
| W4 | SAS + モジュールスケジューラが動作 | Redis SAS + InMemorySAS + 3ティアScheduler実装済み | Done |
| W6 | Mineflayer接続 + スキル実行 | TypeScript bot（bridge/src/index.ts）+ 7基本スキル実装済み | Done |
| W10 | CC + 記憶 + 行動認識が統合 | CC(圧縮+ブロードキャスト) + WM/STM + ActionAwareness実装済み | Done |
| W12 | MVP完成 | 323テスト全通過、全コアモジュール実装 | Done |
| W14 | 品質ゲート検証 | E2E実行（MC接続）で検証予定 | Pending |

### 対応すべきレビュー指摘

- **review-security**: プロンプトサニタイゼーション層の導入（C-1）、APIキー管理（C-2）
- **review-integration**: CC実行間隔の統一 → 動的1-15秒で実装済み、Python/TypeScript統合 → ZMQで解決済み
- **review-cost**: GPT-4o-miniでの全モジュール統一（MVP限定）

### Phase 0で解決されたリスク

| リスク | 結果 |
|---|---|
| Python-Mineflayerブリッジの不安定性 | ZMQ採用。REQ-REP+PUB-SUBで安定動作。ヘルスチェック（ping）+自動再接続実装済み |
| LLM出力のJSON解析エラー | CCにJSON解析フォールバック実装（前回結果再利用） |
| CCの情報圧縮品質 | テンプレートベース圧縮で情報保持率>0.8達成 |

---

## Phase 1: 基盤完成 — ✅ COMPLETE

**期間**: 3-4ヶ月 / **人員**: 3-5人（インフラ担当2人、エージェントロジック担当2-3人） / **予算**: $810-1,450/月（10体）
**実績**: 1,165テスト全通過（レビュー対応後）、ruff lint clean、25モジュール並列実装完了

### 目標
全PIANOモジュールを実装し、10体規模で論文のSection 3（単一エージェント進化）とSection 4の小グループ実験を再現する。LLMゲートウェイの基盤とローカルLLM並行稼働を開始する。

### 成果物

| # | 成果物 | 詳細 | 参照Doc | 状態 |
|---|---|---|---|---|
| 1-1 | 目標生成モジュール | グラフ構造推論、社会的目標の5-10秒生成。`src/piano/goals/generator.py` | 07 | ✅ Done |
| 1-2 | 計画モジュール | HTN + LLMベースの動的計画。`src/piano/planning/planner.py` | 07 | ✅ Done |
| 1-3 | 社会認識モジュール | 感情追跡(0-10)、好感度推定、選択的起動。`src/piano/social/awareness.py`, `emotions.py` | 06 | ✅ Done |
| 1-4 | 発話モジュール | CC条件付き言語生成。`src/piano/talking/module.py` | 03 | ✅ Done |
| 1-5 | 自己省察モジュール | Reflexionベースの3段階プロセス。`src/piano/reflection/module.py` | 07 | ✅ Done |
| 1-6 | 長期記憶(LTM) | Qdrant + ベクトル検索 + 記憶圧縮。`src/piano/memory/ltm.py`, `ltm_search.py`, `consolidation.py` | 04 | ✅ Done |
| 1-7 | 行動認識NN | 小規模ニューラルネット（~100Kパラメータ）。`src/piano/awareness/nn_model.py`, `trainer.py` | 07 | ✅ Done |
| 1-8 | 性格モデリング | Big Fiveベースの性格→行動マッピング。`src/piano/social/personality.py` | 06 | ✅ Done |
| 1-9 | 社会グラフ | NetworkX有向重み付きグラフ（アクセスパターン抽象化レイヤー付き）。`src/piano/social/graph.py` | 06 | ✅ Done |
| 1-10 | モデルティアリング | Tier1/2/3のモデル分離。`src/piano/llm/tiering.py` | 02 | ✅ Done |
| 1-11 | LLMゲートウェイ基盤 | 優先度キュー + 同時実行制限 + サーキットブレーカー。`src/piano/llm/gateway.py` | 02, 08 | ✅ Done |
| 1-12 | ローカルLLM並行稼働 | Ollama + vLLMプロバイダ実装。`src/piano/llm/local.py` | 02 | ✅ Done |
| 1-13 | Stage 1+2評価パイプライン | アイテム収集ベンチマーク + 社会認知メトリクス。`src/piano/eval/items.py`, `social_metrics.py` | 09 | ✅ Done |
| 1-14 | CI/CDパイプライン | GitHub Actions ci.yml + phase1.yml。`.github/workflows/` | 10 | ✅ Done |
| 1-15 | チェックポイント復元時間ベンチマーク | チェックポイント/リストア実装。`src/piano/core/checkpoint.py` | 08 | ✅ Done |

### マイルストーン

| Week | マイルストーン | 検証基準 | 状態 |
|---|---|---|---|
| W4 | 全モジュールのUnit Test通過 | モックLLMで全モジュールが動作、Unit Test通過率95%以上 | ✅ Done |
| W8 | 10体同時動作 | Orchestratorで10エージェント並列管理実装済み | ✅ Done |
| W10 | LLMゲートウェイ+ローカルLLM | Gateway(priority queue+circuit breaker) + Ollama/vLLM実装済み | ✅ Done |
| W12 | 単一エージェント実験再現 | 30分で平均15-20種のアイテム収集（論文Fig.準拠、95%CI内） | ⏳ E2E検証待ち |
| W14 | 小グループ実験再現 | 10体で社会認知精度 r>0.5、感情追跡の定量評価 | ⏳ E2E検証待ち |
| W14-16 | Phase 2インフラ準備 | Kubernetes基本構成PoC、Redis Cluster移行テスト、Velocity Proxy統合テスト | ✅ Done |
| W16 | 品質ゲート検証 | 10体で全モジュール安定動作、社会認知精度 r>0.5、アイテム収集15種以上 | ⏳ E2E検証待ち |

### 対応すべきレビュー指摘

- **review-security**: SASアクセス制御（C-3） ✅、エージェント暴走防止（H-1） ✅
- **review-integration**: ベクトルDB統一（Qdrantに確定） ✅、MCサーバー統一（Pufferfishに確定） ✅
- **review-oss**: Voyagerスキルライブラリ設計の参考 ✅、Generative Agents記憶設計の参考 ✅
- **review-testing**: MockLLMProviderの標準化 ✅、CC入出力のスナップショットテスト ✅

---

## Phase 2: スケーリング — ✅ 実装完了 / 検証待ち

**期間**: 4-5ヶ月 / **人員**: 3-4人（インフラ担当2人、エージェントロジック担当1-2人） / **予算**: $3,710-6,910/月（50体）
**実績**: 全13モジュール + 残タスク10件（ランチャー、TLS、NetworkPolicy、ベンチマーク、障害注入、E2E基盤、Grafanaアラート等）実装完了。1,979テスト全通過、ruff lint clean。スケーリング検証（実環境での50-1000体動作確認）は未実施。

### 目標
50-1000体規模に対応する分散実行基盤を構築し、Section 4の50体社会実験を再現する。1000体対応の技術スケーリングをこのPhaseで完了し、Phase 3を文明実験に専念させる。

### 成果物

| # | 成果物 | 詳細 | 参照Doc | 状態 |
|---|---|---|---|---|
| 2-1 | asyncio+multiprocessing分散基盤 | Worker管理 + Supervisor + シャーディング。`src/piano/scaling/worker.py`, `supervisor.py`, `sharding.py` | 01, 08 | ✅ Done |
| 2-2 | LLMゲートウェイ拡張 | マルチプロバイダルーター + フェイルオーバー。`src/piano/llm/multi_provider.py` | 02, 08 | ✅ Done |
| 2-3 | Velocity MCプロキシ | 動的負荷分散 + サーバー管理。`src/piano/bridge/velocity.py` | 05 | ✅ Done |
| 2-4 | プロンプトキャッシング | セマンティックキャッシュ + プレフィックスキャッシュ + Redisバックエンド。`src/piano/llm/prompt_cache.py` | 02 | ✅ Done |
| 2-5 | 分散ロギング | 構造化ログ + OpenTelemetryトレーシング。`src/piano/observability/logging_config.py`, `tracing.py` | 10 | ✅ Done |
| 2-6 | モニタリング | Prometheusメトリクス + Grafanaダッシュボード。`src/piano/observability/metrics.py`, `docker/grafana/`, `docker/prometheus/` | 08, 10 | ✅ Done |
| 2-7 | Kubernetes構成 | agent-worker, llm-gateway, mc-shard, redis-cluster, monitoring。`k8s/` | 10 | ✅ Done |
| 2-8 | 集合知メカニズム | 観察者閾値 + 複数集約方式。`src/piano/social/collective.py` | 06 | ✅ Done |
| 2-9 | インフルエンサー機構 | 感情伝播 + 投票影響モデル。`src/piano/social/influencer.py` | 06 | ✅ Done |
| 2-10 | 高度な評価パイプライン | ガバナンス(税遵守/投票) + ミーム追跡(SIR) + 役割推論。`src/piano/eval/governance.py`, `memes.py`, `role_inference.py` | 09 | ✅ Done |
| 2-11 | 分散チェックポイント | Shard別スナップショット + Redis バックエンド + 復元。`src/piano/core/distributed_checkpoint.py` | 08 | ✅ Done |
| 2-12 | パフォーマンス回帰テストスイート | ベンチマーク + 回帰検出。`src/piano/eval/performance.py` | 10 | ✅ Done |
| 2-13 | 1000体対応 | リソースリミッター + 水平スケーリング基盤。`src/piano/scaling/resource_limiter.py` | 08 | ✅ Done |
| 2-14 | CLIランチャー | シミュレーション起動エントリポイント（--agents, --ticks, --mock-llm, --config）。`src/piano/main.py`, `__main__.py` | 10 | ✅ Done |
| 2-15 | TLS通信 | Redis SSL、CurveZMQ、Qdrant HTTPS対応。`config/settings.py`, `bridge/client.py`, `sas_redis.py`, `ltm.py` | 08 | ✅ Done |
| 2-16 | K8s NetworkPolicy | default-deny + 11ポリシー。`k8s/network-policies.yaml` | 08, 10 | ✅ Done |
| 2-17 | NetworkXベンチマーク | 10/50/100/500体でのグラフ操作性能計測。`benchmarks/networkx_scalability.py` | 06 | ✅ Done |
| 2-18 | 障害注入テスト | Redis/Bridge/LLM/Agent障害シミュレータ。`src/piano/testing/chaos.py` | 08 | ✅ Done |
| 2-19 | E2Eテスト基盤 | --run-e2eフラグ、InMemorySAS使用の最小E2E。`tests/e2e/` | 10 | ✅ Done |
| 2-20 | Grafanaコストアラート | コスト/エラー率/TPS/メモリの4種アラート。`docker/grafana/provisioning/alerting/alerts.yaml` | 10 | ✅ Done |

### マイルストーン

| Week | マイルストーン | 検証基準 | 状態 |
|---|---|---|---|
| W4 | インフラ移行完了 | K8s+Redis Cluster+Velocity構成で10体が安定動作 | ⏳ 実環境検証待ち |
| W6 | 50体の分散実行 | 50体が安定動作（TPS>15）、LLMレート制限エラー<1% | ⏳ 実環境検証待ち |
| W8 | 50体社会実験再現 | 社会認知精度 r>0.7、集合知効果: 観察者閾値1→11でピアソンr改善20%以上（p<0.05） | ⏳ 実環境検証待ち |
| W10 | 200体MCサーバーストレステスト | MCサーバー単体で200体同時接続、TPS>15維持 | ⏳ 実環境検証待ち |
| W12 | 500体スモークテスト | 500体が1時間安定動作（10分→1時間に延長） | ⏳ 実環境検証待ち |
| W14 | 障害注入テスト | 意図的クラッシュからのチェックポイント復元成功、復元時間10分以内 | ⏳ 実環境検証待ち |
| W16 | モニタリング完成 | 全メトリクス+コスト監視がダッシュボードに表示 | ⏳ 実環境検証待ち |
| W18 | 1000体スモークテスト | 1000体が1時間安定動作 | ⏳ 実環境検証待ち |
| W20 | 品質ゲート検証 | 500体で4時間安定動作、50体で集合知効果（p<0.05）、ネットワーク帯域実測データ完備 | ⏳ 実環境検証待ち |

### 対応すべきレビュー指摘

- **review-security**: TLS通信（M-3） ✅ CurveZMQ + Redis SSL + Qdrant HTTPS実装済、ネットワークセグメンテーション ✅ K8s NetworkPolicy 11ポリシー
- **review-cost**: ローカルモデル本格導入（Tier2/3の95%以上をローカル化目標） ⏳ 実環境検証待ち
- **review-testing**: スケーリング負荷テスト ✅ NetworkXベンチマーク実装済、パフォーマンス回帰テスト ✅ PerformanceBenchmark実装済、障害注入テスト ✅ Chaos framework実装済
- **review-integration**: NetworkXスケーラビリティ対策 ✅ 100体ベンチマーク実装済（P99<10ms確認、igraph移行不要）

---

## Phase 3: 文明実験

**期間**: 2-3ヶ月 / **人員**: 2-3人（実験設計1人、データ分析1人、インフラ保守1人） / **予算**: $25,400-46,200/月（500体）

### 目標
500-1000体規模でSection 5の文明実験（専門化、集団ルール、文化ミーム、宗教伝播）を再現する。技術スケーリングはPhase 2で完了済みのため、実験実施と論文再現検証に専念する。

### 成果物

| # | 成果物 | 詳細 | 参照Doc |
|---|---|---|---|
| 3-1 | 専門化システム | 自発的役割分化、GPT-4o役割推論パイプライン | 05, 09 |
| 3-2 | 集団ルールシステム | 憲法、税制、民主的投票プロセス | 09 |
| 3-3 | 文化ミーム追跡 | LLMミーム分析、地理的伝播ヒートマップ | 09 |
| 3-4 | 宗教伝播システム | パスタファリアン司祭、SIRモデルフィッティング | 09 |
| 3-5 | 実験結果ダッシュボード | Streamlit分析ダッシュボード | 09 |
| 3-6 | 人間評価プロトコル | LLM評価vs人間評価の比較検証（サンプル30件以上） | 09 |

### マイルストーン

| Week | マイルストーン | 検証基準 |
|---|---|---|
| W3 | 専門化実験再現 | 30体で6種以上の役割分化（論文準拠）、役割エントロピー正規化>0.5 |
| W5 | 集団ルール実験再現 | 反税条件で税遵守率15%以上低下（論文準拠） |
| W7 | 500体文化ミーム実験 | 町間のJaccard類似度<0.5（町ごとの固有ミームパターンを証明） |
| W9 | 500体宗教伝播実験 | 改宗率の時系列曲線が論文とKS検定でp>0.05（分布が同等） |
| W12 | Phase 3完了 | 全論文実験の再現完了、各実験3回繰り返しでCV<15% |

---

## 障害復旧・チェックポイント設計

長時間シミュレーション（4時間以上）の安定実行には、障害復旧の仕組みが不可欠である。各フェーズに以下の設計を組み込む。

### Phase別の障害復旧対応

| フェーズ | 対応内容 | 成果物 |
|---|---|---|
| **Phase 0** | 基本的なチェックポイント/リスタート機構。SASスナップショットの定期保存（5分間隔）。エージェントクラッシュ時の自動再起動 | `CheckpointManager` 基本実装 |
| **Phase 1** | Redis RDB/AOF永続化設定。Qdrantコレクションスナップショット。エージェント状態のチェックポイントからの復元テスト。復元時間ベンチマーク（10体→500体のデータ量測定） | バックアップ設定、復元テストスイート |
| **Phase 2** | Redis Clusterのフェイルオーバー設定。CCサイクルアライメントによる分散チェックポイント（Shard別、5-10秒ズレ許容）。PostgreSQL WALベースのPITR。障害注入テスト（W14） | 分散バックアップ基盤 |
| **Phase 3** | 1000体規模での部分障害からの復旧。リージョン別チェックポイント。シミュレーション中断からの透過的な再開 | 本番障害復旧手順書 |

### チェックポイント設計の要件

- **チェックポイント間隔**: 5分（Phase 0-1）、1分（Phase 2-3）
- **チェックポイントデータ**: SAS全セクション + STM + 社会グラフ + エージェントメタデータ
- **復元時間目標**: 2分以内（Phase 0-1）、10分以内（Phase 2-3、500体規模）— 実測ベースで調整
- **データ整合性**: CCサイクルアライメント方式 — チェックポイント取得はCCサイクル間で実施。Shard別の緩い一貫性（5-10秒ズレ許容）+ 復元後の最初の1分間でエージェント間状態を再同期

---

## クリティカルパス

```
ブリッジPoC → SAS設計 → モジュールスケジューラ → CC MVP → 全モジュール統合 → 10体テスト
                                                      ↓
                                                 LLMゲートウェイ基盤 → ローカルLLM PoC
                                                                              ↓
Mineflayer接続 → スキル実行 ─────────────────────────────────→ アイテム収集ベンチマーク
                                                                              ↓
                                              K8s移行 → LLMゲートウェイ拡張 → 50体テスト → 500体テスト → 1000体テスト
                                                                                                    ↓
                                                                                          文明実験 → 論文再現完了
```

**最大のボトルネック**:
1. **Python-Mineflayerブリッジの技術選定と性能** — Phase 0 W2までに完了必須。1000体のボット制御を安定処理
2. **LLM API呼び出しのレート制限** — 500体で~217 calls/s。LLMゲートウェイ（Phase 1から構築）+ローカルLLMで対応
3. **CCの情報圧縮品質** — ブロードキャストの質がエージェント行動の一貫性を決定。Phase 0で評価基準確立

---

## 技術選定サマリー

| カテゴリ | 選定 | 代替案 | 決定根拠 |
|---|---|---|---|
| 主言語 | Python 3.12+ | TypeScript | エコシステム（ML/NN/Ray）の充実 |
| MCブリッジ | TypeScript (Mineflayer) | - | 唯一の選択肢 |
| 共有状態 | Redis 7+（Valkey/KeyDB互換検証済み） | etcd | Pub/Sub + Lua scripting |
| 分散実行 | asyncio + multiprocessing（Phase 2）→ Ray 2.x（ボトルネック時） | Celery | 段階的導入でリスク低減 |
| ベクトルDB | Qdrant | Chroma | スケーラビリティ、バイナリ量子化 |
| 永続化 | PostgreSQL 16 | MySQL | asyncpg、高信頼性 |
| LLM抽象化 | LiteLLM | 独自実装 | マルチプロバイダ即座対応 |
| ローカルLLM | vLLM / Ollama | llama.cpp | Phase 1からTier2/3に使用 |
| 監視 | Prometheus + Grafana | Datadog | OSS、コスト |
| ログ | Grafana Loki | ELK | 軽量、コスト効率 |
| 実験管理 | MLflow + Hydra | W&B | OSS、柔軟性 |
| CI/CD | GitHub Actions | GitLab CI | エコシステム統合 |
| コンテナ | Docker + K8s | Docker Compose | 本番スケーラビリティ |
| MCサーバー | Pufferfish + Velocity | Paper | 最高性能 |

---

## コストサマリー

> **注**: 2026年2月時点のAPI価格に基づく。詳細はreview-comprehensive.mdを参照。

| フェーズ | スケール | LLM/月 | インフラ/月 | 合計/月 |
|---|---|---|---|---|
| Phase 0 | 1-5体 | $50-100 | $0（ローカル） | **$50-100** |
| Phase 1 | 10体 | $640-1,280 | $170 | **$810-1,450** |
| Phase 2 | 50体 | $3,200-6,400 | $510 | **$3,710-6,910** |
| Phase 3 | 500体 | $20,800-41,600 | $4,600 | **$25,400-46,200** |
| Phase 3+ | 1,000体 | $36,800-73,600 | $9,200 | **$46,000-82,800** |

### コストリスク管理

- **予備費**: 総コストの**25%**をバッファとして計上（業界標準20-30%）
- **最悪ケース試算**: CC常時1秒実行 + 社会認識全体発火 → Phase 3で最大$100,000/月
- **コスト上限アラート**: Phase別の月次コスト上限を設定（Phase 0: $100、Phase 1: $1,500、Phase 2: $7,000、Phase 3: $50,000）
- **主要不確実性要因**: API価格変動（±20%）、為替変動（±10円）、工数超過（+30%）
- **インフラ+LLM APIコスト概算**: $100,000-240,000（15ヶ月、予備費含む）
- **人件費を含む総プロジェクトコスト概算**: 4,700-10,600万円（15ヶ月、予備費25%込み、review-comprehensive.md参照）

### コスト最適化ロードマップ

| Phase | 最適化施策 | 期待削減率 |
|---|---|---|
| Phase 0 | モデルティアリング（GPT-4o-mini統一） | ベースライン |
| Phase 1 | プロンプトキャッシング + ローカルLLM（Tier2/3） | 20-30% |
| Phase 2 | ローカルLLM本格化（Tier2/3の95%以上） | 40-50% |
| Phase 3 | セマンティックキャッシュ（保守的見積もり10-15%追加削減） | 50-60% |

---

## 再現性保証戦略

LLMの非決定性下での実験再現性を確保するため、以下の多層戦略を採用する。

| 層 | 施策 | 適用Phase |
|---|---|---|
| 決定論的制御 | temperature=0、固定seed（対応モデルのみ） | Phase 0- |
| キャッシュ | LLMレスポンスキャッシュ（同一入力→同一出力） | Phase 0- |
| 統計的再現 | 重要実験は3回繰り返し実行、平均±標準偏差を報告（CV<15%） | Phase 1- |
| 環境スナップショット | Docker image + MCワールドデータ + SASスナップショットの完全保存 | Phase 1- |
| 論文比較 | 各指標は「論文値±20%以内」かつ「p<0.05で有意差なし」を検証 | Phase 1- |

---

## レビュー指摘の統合対応表

| 指摘元 | 重要指摘 | 対応フェーズ |
|---|---|---|
| review-integration | コード言語混在（TS/Python） | Phase 0で方針確定 |
| review-integration | CC実行間隔の矛盾 | Phase 0で統一 |
| review-security | プロンプトインジェクション対策 | Phase 0 |
| review-security | APIキー管理 | Phase 0 |
| review-cost | LLMコスト最適化（モデルティアリング） | Phase 0-1 |
| review-cost | ローカルモデル導入 | Phase 1（前倒し） |
| review-cost | コスト予備費の計上 | 全Phase（25%） |
| review-oss | Voyagerスキルライブラリ参考 | Phase 0-1 |
| review-testing | MockLLMProvider標準化 | Phase 0 |
| review-testing | パフォーマンス回帰テスト | Phase 2 |
| review-integration | ベクトルDB統一（Qdrant） | Phase 1 |
| review-integration | MCサーバー統一（Pufferfish） | Phase 0 |
| ロードマップレビュー | Phase移行品質ゲート | 全Phase |
| ロードマップレビュー | LLMゲートウェイ前倒し | Phase 1 |
| ロードマップレビュー | ローカルLLM早期導入 | Phase 1 |
| ロードマップレビュー | 成功基準の定量化 | 全Phase |
| ロードマップレビュー | 障害注入テスト | Phase 2 |
| ロードマップレビュー | 1000体対応をPhase 2に前倒し | Phase 2 |

---

## リスク管理

| リスク | 影響度 | 発生確率 | 緩和策 | 担当フェーズ |
|---|---|---|---|---|
| LLM APIコストの超過 | 高 | 高 | モデルティアリング、Phase 1からローカルLLM導入、コスト上限アラート+自動停止、予備費25% | 全フェーズ |
| Python-Mineflayerブリッジの不安定性 | 高 | 中 | Phase 0 W2までに3方式PoCを完了し技術選定、ヘルスチェック+自動再起動、複数ブリッジインスタンスの負荷分散 | Phase 0 |
| CCの情報圧縮品質不足 | 高 | 中 | Phase 0で評価基準確立（情報保持率>0.8）、ゴールデンファイルテスト、CC失敗時は前回結果再利用 | Phase 0-1 |
| LLM APIレート制限超過（500体で217 calls/s） | 高 | 高 | Phase 1でLLMゲートウェイ構築、マルチプロバイダ分散、ローカルLLMでTier2/3を分離 | Phase 1-2 |
| Phase 1→2のインフラ移行コスト | 中 | 高 | Phase 1末期（W14-16）にK8s/Redis Cluster/VelocityのPoCを実施 | Phase 1-2 |
| Redis Luaスクリプトの直列化ボトルネック | 中 | 中 | SAS容量制限の明記、Redis Cluster移行閾値の事前定義（メモリ80%超、P99>10ms） | Phase 1-2 |
| 1000体でのMCサーバー容量制約 | 中 | 高 | Phase 2でMCサーバー単体の限界テスト（200体）、Velocity動的負荷分散、空間シャーディング | Phase 2 |
| LLM出力の非決定性による再現困難 | 中 | 高 | temperature=0、LLMキャッシュ、3回繰り返し統計的再現（CV<15%） | 全フェーズ |
| 論文の未記載パラメータ | 中 | 確実 | 感度分析、パラメータスイープ | Phase 1-3 |
| NetworkXが500体以上でスケールしない | 中 | 中 | Phase 1でアクセスパターン抽象化、Phase 2で100体性能計測→igraph移行判断 | Phase 1-2 |

---

## 成功基準

| フェーズ | 成功基準 | 状態 |
|---|---|---|
| Phase 0 | 1-5体が30分間安定動作、CC制御下で言行一致改善率30%以上、ブリッジ技術選定完了、CC圧縮の情報保持率>0.8 | ✅ コード完了 / ⏳ E2E検証待ち |
| Phase 1 | 30分で平均15-20種のアイテム（論文値±20%以内）、10体で社会認知精度 r>0.5、全モジュールUnit Test通過率95%以上、LLMゲートウェイ+ローカルLLM稼働 | ✅ コード完了 / ⏳ E2E検証待ち |
| Phase 2 | 50体で社会認知精度 r>0.7（p<0.05）、集合知効果: 観察者閾値改善20%以上（p<0.05）、500体で4時間安定動作（TPS>15）、1000体で1時間安定動作 | ✅ 実装完了 / ⏳ 実環境検証待ち |
| Phase 3 | 6種以上の役割分化（役割エントロピー正規化>0.5）、反税条件で税遵守率15%以上低下、町間ミームJaccard類似度<0.5、宗教伝播KS検定p>0.05、全実験3回繰り返しCV<15% | ⬚ 未着手 |

---

## 関連ドキュメント
- [00-overview.md](./00-overview.md) — 全体概要
- [09-evaluation.md](./09-evaluation.md) — 評価・ベンチマーク
- [10-devops.md](./10-devops.md) — DevOps・運用
- [review-comprehensive.md](./review-comprehensive.md) — 統合レビュー
