# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**Project Sid: Many-agent simulations toward AI civilization** の再現実装プロジェクト。Altera.ALの論文で提案された **PIANO（Parallel Information Aggregation via Neural Orchestration）** アーキテクチャを実装し、10〜1000以上のAIエージェントがMinecraft環境内で社会的に振る舞うシミュレーションの再現を目指す。

**現在の状態**: Phase 1 実装完了（1,125テスト全通過、ruff lint clean）

## リポジトリ構成

```
src/piano/                  # PIANOアーキテクチャ実装
  core/                     # 型定義、Module ABC、SAS ABC/Redis実装、Scheduler、Agent、Checkpoint、Orchestrator
  cc/                       # 認知コントローラ（圧縮、ブロードキャスト、コントローラ）
  memory/                   # 記憶システム（WM、STM、Manager、LTM、LTM Search、Consolidation）
  llm/                      # LLM抽象化（LiteLLM、Mock、Cache、Tiering、Gateway、Local）
  bridge/                   # ZMQブリッジクライアント、プロトコル拡張
  skills/                   # スキルレジストリ、基本/社会/高度スキル、エグゼキュータ
  awareness/                # 行動認識モジュール、NN予測モデル、トレーナー
  goals/                    # 目標生成モジュール
  planning/                 # 計画モジュール
  talking/                  # 発話モジュール
  reflection/               # 自己省察モジュール
  social/                   # 社会認知（認識、性格、社会グラフ、感情）
  eval/                     # 評価（アイテム収集ベンチマーク、社会認知メトリクス）
  config/                   # PianoSettings（pydantic-settings）

tests/                      # テストスイート（1,125テスト）
  unit/                     # ユニットテスト（全モジュール）
  integration/              # 統合テスト（agent lifecycle、Phase 1 smoke）
  helpers.py                # InMemorySAS、DummyModule

bridge/                     # TypeScript Mineflayer bot（ZMQ REP+PUB）
docker/                     # docker-compose.yml（Redis + PostgreSQL + Pufferfish）
docs/implementation/        # 技術調査・設計ドキュメント（17ファイル）
.github/workflows/          # GitHub Actions CI（ci.yml + phase1.yml）
```

## 開発コマンド

```bash
# セットアップ
uv sync --dev

# パッケージ追加
uv add <package>            # 本体依存
uv add --dev <package>      # 開発依存

# テスト実行
uv run pytest tests/                    # 全テスト
uv run pytest tests/unit/               # ユニットテストのみ
uv run pytest tests/unit/core/          # 特定モジュール
uv run pytest -x -q                     # 失敗時即停止、簡潔出力

# リント・フォーマット
uv run ruff check src/ tests/           # lint
uv run ruff check --fix src/ tests/     # auto-fix
uv run ruff format src/ tests/          # format

# 型チェック
uv run mypy src/piano/

# Docker環境（Redis + PostgreSQL + MC）
docker compose -f docker/docker-compose.yml up -d
```

## 技術スタック

- **Python**: 3.12+、パッケージマネージャは **uv**（`uv add` でパッケージ追加）
- **フレームワーク**: asyncio、Pydantic 2.0、pydantic-settings
- **共有状態**: Redis 7+（fakeredisでテスト）
- **LLM**: LiteLLM（マルチプロバイダ対応）+ Local LLM（Ollama/vLLM）
- **記憶**: Qdrant（LTM ベクトル検索）
- **社会グラフ**: NetworkX（DiGraph）
- **ブリッジ**: ZMQ（REQ-REP + PUB-SUB）
- **MCサーバー**: Pufferfish + Velocity
- **テスト**: pytest + pytest-asyncio（asyncio_mode = "auto"）
- **lint**: ruff（E/W/F/I/N/UP/B/SIM/TCH/RUF）
- **CI**: GitHub Actions

## アーキテクチャの要点

- **モジュールはステートレス**: SAS（Shared Agent State）を介して読み書き
- **3ティア実行**: FAST（毎tick）、MID（3tick毎）、SLOW（10tick毎）
- **認知コントローラ（CC）**: GWT情報ボトルネック → 圧縮 → LLM判断 → ブロードキャスト
- **ZMQブリッジ**: Python（制御）↔ TypeScript（Mineflayer）間のIPC
- **LLMゲートウェイ**: 優先度キュー + 同時実行制限 + サーキットブレーカー
- **記憶三層**: WM（即座）→ STM（短期100件）→ LTM（Qdrant永続化、忘却曲線付き検索）
- **社会認知**: Big Five性格 + 感情追跡（valence-arousal）+ 社会グラフ（PageRank影響度）
- **マルチエージェント**: Orchestrator による10エージェント並列管理

## 参考リンク

- 論文: [arXiv:2411.00114](https://arxiv.org/abs/2411.00114)
- 技術調査: [docs/implementation/00-overview.md](docs/implementation/00-overview.md)
- ロードマップ: [docs/implementation/roadmap.md](docs/implementation/roadmap.md)

## 注意事項

- 論文PDFは21MBと大きいため、内容確認にはページ指定での読み込みを推奨（`pages: "1-5"` など）
- ブリッジテスト（ZMQ）はWindows上ではモックベースで実行（実ZMQソケットはハングする場合あり）
