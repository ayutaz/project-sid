# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**Project Sid: Many-agent simulations toward AI civilization** の再現実装プロジェクト。Altera.ALの論文で提案された **PIANO（Parallel Information Aggregation via Neural Orchestration）** アーキテクチャを実装し、10〜1000以上のAIエージェントがMinecraft環境内で社会的に振る舞うシミュレーションの再現を目指す。

**現在の状態**: Phase 0 MVP実装完了（323テスト全通過、ruff lint clean）

## リポジトリ構成

```
src/piano/                  # PIANOアーキテクチャ実装
  core/                     # 型定義、Module ABC、SAS ABC/Redis実装、Scheduler、Agent
  cc/                       # 認知コントローラ（圧縮、ブロードキャスト、コントローラ）
  memory/                   # 記憶システム（WorkingMemory、ShortTermMemory、Manager）
  llm/                      # LLM抽象化（LiteLLM、Mock、Cache）
  bridge/                   # ZMQブリッジクライアント
  skills/                   # スキルレジストリ、基本スキル、エグゼキュータ
  awareness/                # 行動認識モジュール
  config/                   # PianoSettings（pydantic-settings）

tests/                      # テストスイート（323テスト）
  unit/                     # ユニットテスト（core, cc, memory, llm, bridge, skills, awareness）
  integration/              # 統合テスト（agent lifecycle）
  helpers.py                # InMemorySAS、DummyModule

bridge/                     # TypeScript Mineflayer bot（ZMQ REP+PUB）
docker/                     # docker-compose.yml（Redis + PostgreSQL + Pufferfish）
docs/implementation/        # 技術調査・設計ドキュメント（13ファイル）
.github/workflows/ci.yml   # GitHub Actions CI（Python 3.12/3.13）
```

## 開発コマンド

```bash
# セットアップ
uv sync --dev

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

- **Python**: 3.12+、パッケージマネージャは **uv**
- **フレームワーク**: asyncio、Pydantic 2.0、pydantic-settings
- **共有状態**: Redis 7+（fakeredisでテスト）
- **LLM**: LiteLLM（マルチプロバイダ対応）
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

## 参考リンク

- 論文: [arXiv:2411.00114](https://arxiv.org/abs/2411.00114)
- 技術調査: [docs/implementation/00-overview.md](docs/implementation/00-overview.md)
- ロードマップ: [docs/implementation/roadmap.md](docs/implementation/roadmap.md)

## 注意事項

- 論文PDFは21MBと大きいため、内容確認にはページ指定での読み込みを推奨（`pages: "1-5"` など）
- ブリッジテスト（ZMQ）はWindows上ではモックベースで実行（実ZMQソケットはハングする場合あり）
