# 10. DevOps・開発ワークフロー

[← 実装ドキュメント](../index.md)

---

## 目次

1. [CI/CDパイプライン](#1-cicdパイプライン)
2. [コンテナ化](#2-コンテナ化)
3. [モニタリング](#3-モニタリング)
4. [分散ロギング](#4-分散ロギング)
5. [実験管理](#5-実験管理)
6. [開発環境](#6-開発環境)
7. [デバッグツール](#7-デバッグツール)
8. [推奨技術スタック一覧](#8-推奨技術スタック一覧)

---

## 1. CI/CDパイプライン

### 1.1 設計方針

Project Sidの再現実装は、500-1000エージェントの大規模シミュレーションを支えるため、以下の方針でCI/CDを設計する。

- **段階的ゲート**: ユニットテスト → 統合テスト → E2Eテストの順に自動実行し、前段が失敗したら後段を実行しない
- **マトリクスビルド**: Python 3.11/3.12、各モジュール単位で並列テスト
- **環境分離**: dev / staging / prod を明確に分離し、各環境へのデプロイを自動化
- **コスト意識**: LLM API呼び出しを伴うテストはモックを使用し、実APIテストはnightlyに限定

### 1.2 GitHub Actionsワークフロー設計

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.12"
  REGISTRY: ghcr.io
  IMAGE_PREFIX: ghcr.io/${{ github.repository }}

jobs:
  # ===== Stage 1: 静的解析・リンティング =====
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install ruff mypy
      - name: Ruff lint
        run: ruff check src/
      - name: Ruff format check
        run: ruff format --check src/
      - name: Type check
        run: mypy src/ --ignore-missing-imports

  # ===== Stage 2: ユニットテスト =====
  unit-test:
    needs: lint
    runs-on: ubuntu-latest
    strategy:
      matrix:
        module:
          - cognitive-controller
          - memory-system
          - goal-generation
          - social-awareness
          - action-awareness
          - skill-execution
          - talking
          - planning
          - self-reflection
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: "pip"
      - name: Install dependencies
        run: pip install -e ".[test]"
      - name: Run unit tests
        run: pytest tests/unit/${{ matrix.module }}/ -v --cov=src/${{ matrix.module }}
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with:
          name: coverage-${{ matrix.module }}
          path: .coverage

  # ===== Stage 3: 統合テスト =====
  integration-test:
    needs: unit-test
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_DB: projectsid_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: "pip"
      - name: Install dependencies
        run: pip install -e ".[test]"
      - name: Run integration tests (mocked LLM)
        run: pytest tests/integration/ -v --timeout=300
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/projectsid_test
          REDIS_URL: redis://localhost:6379
          LLM_MOCK: "true"

  # ===== Stage 4: Dockerイメージビルド =====
  build:
    needs: integration-test
    runs-on: ubuntu-latest
    strategy:
      matrix:
        component:
          - agent-runtime
          - minecraft-bridge
          - llm-gateway
          - experiment-manager
          - monitoring-stack
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/${{ matrix.component }}/Dockerfile
          push: ${{ github.ref == 'refs/heads/main' }}
          tags: |
            ${{ env.IMAGE_PREFIX }}/${{ matrix.component }}:${{ github.sha }}
            ${{ env.IMAGE_PREFIX }}/${{ matrix.component }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ===== Stage 5: ステージングデプロイ =====
  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to staging
        run: |
          helm upgrade --install project-sid-staging ./helm/project-sid \
            --namespace staging \
            --values helm/project-sid/values-staging.yaml \
            --set global.image.tag=${{ github.sha }} \
            --set agentRuntime.replicas=5 \
            --wait --timeout=10m
```

```yaml
# .github/workflows/nightly.yml
name: Nightly E2E Tests

on:
  schedule:
    - cron: "0 2 * * *"  # 毎日AM2:00 UTC

jobs:
  e2e-small:
    runs-on: self-hosted  # GPU付きランナー
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
      - name: Start Minecraft + 5 agents
        run: docker compose -f docker-compose.e2e.yml up -d
      - name: Wait for environment ready
        run: ./scripts/wait-for-ready.sh --timeout=120
      - name: Run E2E tests (5 agents, 10 min sim)
        run: pytest tests/e2e/ -v --timeout=900
        env:
          AGENT_COUNT: 5
          SIM_DURATION: 600
          LLM_PROVIDER: openai
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      - name: Collect logs and metrics
        if: always()
        run: ./scripts/collect-e2e-artifacts.sh
      - name: Upload artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: e2e-results-${{ github.run_id }}
          path: artifacts/
```

### 1.3 環境別デプロイ戦略

| 環境 | トリガー | エージェント数 | LLM | 用途 |
|---|---|---|---|---|
| **dev** | PRプッシュ | 1-3 | モック | 機能開発・ユニットテスト |
| **staging** | main マージ | 5-50 | 実API（レート制限付き） | 統合テスト・パフォーマンス確認 |
| **prod** | 手動承認 + タグ | 500-1000 | 実API（フルスケール） | 本番実験 |

デプロイフロー:

```
PR作成 → lint + unit test → PRレビュー → mainマージ
  → integration test → Dockerビルド → stagingデプロイ
  → staging E2Eテスト合格 → 手動承認 → prodデプロイ
```

### 1.4 ブランチ戦略

```
main ─────────────────────────────────────────── (常にデプロイ可能)
  │
  ├── develop ────────────────────────────────── (統合ブランチ)
  │     ├── feature/piano-memory-system
  │     ├── feature/social-awareness-module
  │     └── feature/cognitive-controller-v2
  │
  └── experiment/ ────────────────────────────── (実験設定ブランチ)
        ├── experiment/specialization-30agents
        ├── experiment/culture-500agents
        └── experiment/religion-propagation
```

- `experiment/` ブランチは実験設定（Hydra config）の変更のみを含む
- 実験実行後、結果はMLflowに記録され、ブランチは設定のバージョン管理に使用

---

## 2. コンテナ化

### 2.1 Docker構成

Project Sidの再現実装は以下のコンテナで構成される。

```
project-sid/
├── docker/
│   ├── agent-runtime/
│   │   └── Dockerfile          # PIANOエージェントランタイム
│   ├── minecraft-bridge/
│   │   └── Dockerfile          # Minecraft接続ブリッジ
│   ├── minecraft-server/
│   │   └── Dockerfile          # Minecraftサーバー（Paper/Fabric）
│   ├── llm-gateway/
│   │   └── Dockerfile          # LLM APIゲートウェイ/プロキシ
│   ├── experiment-manager/
│   │   └── Dockerfile          # 実験管理サービス
│   └── monitoring-stack/
│       └── Dockerfile          # カスタムモニタリングエクスポーター
├── docker-compose.yml          # ローカル開発用
├── docker-compose.e2e.yml      # E2Eテスト用
└── docker-compose.staging.yml  # ステージング用
```

#### Dockerfileの例: agent-runtime

```dockerfile
# docker/agent-runtime/Dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# 依存関係のインストール（キャッシュ活用）
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

# アプリケーションコードのコピー
COPY src/ ./src/
COPY configs/ ./configs/

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

# 非rootユーザーで実行
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["python", "-m", "project_sid.agent.runtime"]
```

#### Dockerfileの例: minecraft-server

```dockerfile
# docker/minecraft-server/Dockerfile
FROM itzg/minecraft-server:java21

# Pufferfishサーバー設定（05-minecraft-platform.mdの選定に準拠）
ENV EULA=TRUE
ENV TYPE=PUFFERFISH
ENV VERSION=1.20.4
ENV MEMORY=4G
ENV MAX_PLAYERS=600
ENV VIEW_DISTANCE=4
ENV SIMULATION_DISTANCE=4
ENV ONLINE_MODE=FALSE

# カスタムプラグイン（エージェントブリッジ用）
COPY plugins/ /data/plugins/
COPY server.properties /data/server.properties

EXPOSE 25565
EXPOSE 25575
```

### 2.2 Docker Compose（ローカル開発環境）

```yaml
# docker-compose.yml
services:
  # ----- Minecraftサーバー -----
  minecraft:
    build: ./docker/minecraft-server
    ports:
      - "25565:25565"   # Minecraft
      - "25575:25575"   # RCON
    volumes:
      - minecraft-data:/data
    environment:
      MEMORY: "2G"
      MAX_PLAYERS: 10
    healthcheck:
      test: ["CMD", "mc-health"]
      interval: 30s
      timeout: 10s
      retries: 5

  # ----- Minecraft接続ブリッジ -----
  minecraft-bridge:
    build: ./docker/minecraft-bridge
    depends_on:
      minecraft:
        condition: service_healthy
    environment:
      MINECRAFT_HOST: minecraft
      MINECRAFT_PORT: 25565
      RCON_HOST: minecraft
      RCON_PORT: 25575
      RCON_PASSWORD: ${RCON_PASSWORD:-changeme}
      BRIDGE_PORT: 8081
    ports:
      - "8081:8081"

  # ----- LLMゲートウェイ -----
  llm-gateway:
    build: ./docker/llm-gateway
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      RATE_LIMIT_RPM: 500
      CACHE_ENABLED: "true"
      CACHE_TTL: 3600
    ports:
      - "8082:8082"
    volumes:
      - llm-cache:/app/cache

  # ----- PIANOエージェントランタイム -----
  agent-runtime:
    build: ./docker/agent-runtime
    depends_on:
      - minecraft-bridge
      - llm-gateway
      - redis
      - postgres
    environment:
      MINECRAFT_BRIDGE_URL: http://minecraft-bridge:8081
      LLM_GATEWAY_URL: http://llm-gateway:8082
      REDIS_URL: redis://redis:6379
      DATABASE_URL: postgresql://projectsid:projectsid@postgres:5432/projectsid
      AGENT_COUNT: ${AGENT_COUNT:-3}
      LOG_LEVEL: DEBUG
    deploy:
      resources:
        limits:
          cpus: "4.0"
          memory: 8G

  # ----- データストア -----
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: projectsid
      POSTGRES_USER: projectsid
      POSTGRES_PASSWORD: projectsid
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  # ----- モニタリング（開発用簡易構成） -----
  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus-dev.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:11.0.0
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards

  loki:
    image: grafana/loki:3.0.0
    ports:
      - "3100:3100"
    volumes:
      - ./monitoring/loki/loki-config.yml:/etc/loki/config.yaml

volumes:
  minecraft-data:
  redis-data:
  postgres-data:
  llm-cache:
```

### 2.3 Kubernetes構成（本番環境）

本番環境では Kubernetes + Helm で管理し、500-1000エージェントの大規模シミュレーションに対応する。

#### クラスタ構成概要

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                             │
│                                                                   │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ Namespace:   │  │ Namespace:       │  │ Namespace:       │   │
│  │ monitoring   │  │ project-sid-prod │  │ experiment-mgmt  │   │
│  │              │  │                  │  │                  │   │
│  │ Prometheus   │  │ Minecraft x3-5   │  │ MLflow           │   │
│  │ Grafana      │  │ Agent Pools      │  │ Experiment       │   │
│  │ Loki         │  │ LLM Gateway      │  │ Controller       │   │
│  │ Alertmanager │  │ Redis Cluster    │  │ Hydra Config     │   │
│  │              │  │ PostgreSQL HA    │  │ Store            │   │
│  └──────────────┘  │ Minecraft Bridge │  └──────────────────┘   │
│                    └──────────────────┘                           │
│                                                                   │
│  Node Pool: general (CPU)    ── 管理系・ブリッジ                  │
│  Node Pool: agent (CPU/高メモリ) ── エージェントランタイム        │
│  Node Pool: gpu (GPU)        ── 非LLMニューラルネット推論         │
└─────────────────────────────────────────────────────────────────┘
```

#### ノードプール設計

| ノードプール | マシンタイプ | 台数 | 用途 |
|---|---|---|---|
| **general** | 4 vCPU / 16GB | 3-5 | Minecraft、ブリッジ、管理系 |
| **agent** | 8 vCPU / 32GB | 10-20 | エージェントランタイム（50-100 agent/node） |
| **gpu** | 4 vCPU / 16GB + T4 | 1-3 | Action Awareness等の非LLM推論 |

### 2.4 Helmチャート設計

```
helm/
├── project-sid/                    # Umbrella chart
│   ├── Chart.yaml
│   ├── values.yaml                 # デフォルト値
│   ├── values-dev.yaml
│   ├── values-staging.yaml
│   ├── values-prod.yaml
│   ├── templates/
│   │   ├── _helpers.tpl
│   │   └── namespace.yaml
│   └── charts/                     # サブチャート
│       ├── minecraft-server/
│       │   ├── Chart.yaml
│       │   ├── templates/
│       │   │   ├── statefulset.yaml
│       │   │   ├── service.yaml
│       │   │   └── configmap.yaml
│       │   └── values.yaml
│       ├── agent-runtime/
│       │   ├── Chart.yaml
│       │   ├── templates/
│       │   │   ├── deployment.yaml
│       │   │   ├── hpa.yaml
│       │   │   ├── service.yaml
│       │   │   └── configmap.yaml
│       │   └── values.yaml
│       ├── llm-gateway/
│       │   ├── Chart.yaml
│       │   ├── templates/
│       │   │   ├── deployment.yaml
│       │   │   ├── service.yaml
│       │   │   ├── configmap.yaml
│       │   │   └── pdb.yaml
│       │   └── values.yaml
│       ├── minecraft-bridge/
│       │   ├── Chart.yaml
│       │   ├── templates/
│       │   │   ├── deployment.yaml
│       │   │   └── service.yaml
│       │   └── values.yaml
│       └── experiment-manager/
│           ├── Chart.yaml
│           ├── templates/
│           │   ├── deployment.yaml
│           │   ├── cronjob.yaml
│           │   └── service.yaml
│           └── values.yaml
```

#### values-prod.yaml（本番環境の例）

```yaml
# helm/project-sid/values-prod.yaml
global:
  image:
    registry: ghcr.io
    tag: ""  # CI/CDでオーバーライド
  environment: production

minecraft-server:
  replicas: 3  # 500エージェント: 3サーバー（~170体/サーバー）
  resources:
    requests:
      cpu: "4"
      memory: "8Gi"
    limits:
      cpu: "6"
      memory: "12Gi"
  persistence:
    size: 50Gi
  config:
    maxPlayers: 600
    viewDistance: 4
    simulationDistance: 4

agent-runtime:
  # 各Minecraftサーバーに対応するエージェントプール
  pools:
    - name: pool-a
      minecraftServer: minecraft-0
      agentCount: 170
      replicas: 4  # ~43 agent/pod
    - name: pool-b
      minecraftServer: minecraft-1
      agentCount: 170
      replicas: 4
    - name: pool-c
      minecraftServer: minecraft-2
      agentCount: 160
      replicas: 4
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
  modules:
    cognitiveController:
      tickInterval: 2000  # ms
    memory:
      stmCapacity: 100
      ltmCapacity: 10000
    goalGeneration:
      interval: 10000  # ms
    socialAwareness:
      enabled: true
      triggerOnInteraction: true

llm-gateway:
  replicas: 3
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
  config:
    providers:
      - name: openai
        models: ["gpt-4o", "gpt-4o-mini"]
        rateLimitRPM: 10000
        maxConcurrent: 500
      - name: anthropic
        models: ["claude-sonnet-4-5-20250929"]
        rateLimitRPM: 4000
        maxConcurrent: 200
    cache:
      enabled: true
      backend: redis
      ttl: 3600
    routing:
      strategy: least-latency
      fallback: true

minecraft-bridge:
  replicas: 3  # Minecraftサーバーと1:1
  resources:
    requests:
      cpu: "1"
      memory: "2Gi"
```

#### スケジューラ設定

500-1000エージェント規模では、KubernetesのデフォルトスケジューラではなくVolcanoまたはKueueを使用し、エージェントプールのギャングスケジューリング（全Pod同時起動）を実現する。

```yaml
# エージェントプールのVolcano Job定義例
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: agent-pool-a
spec:
  minAvailable: 4  # 全4 Podが起動するまで待機
  schedulerName: volcano
  tasks:
    - replicas: 4
      name: agent-runtime
      template:
        spec:
          nodeSelector:
            node-pool: agent
          containers:
            - name: agent-runtime
              image: ghcr.io/project-sid/agent-runtime:latest
              resources:
                requests:
                  cpu: "2"
                  memory: "4Gi"
```

---

## 3. モニタリング

### 3.1 モニタリングアーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                     データソース                                  │
│                                                                   │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │ Agent    │  │ Minecraft  │  │ LLM        │  │ K8s Nodes  │  │
│  │ Runtime  │  │ Server     │  │ Gateway    │  │            │  │
│  │ /metrics │  │ /metrics   │  │ /metrics   │  │ node-exp   │  │
│  └────┬─────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  │
│       └───────────────┴───────────────┴───────────────┘          │
│                           │                                       │
│                    ┌──────▼──────┐                                │
│                    │ Prometheus  │                                │
│                    │ (時系列DB)  │                                │
│                    └──────┬──────┘                                │
│                           │                                       │
│              ┌────────────┼────────────┐                         │
│              │            │            │                          │
│       ┌──────▼──────┐  ┌─▼──────┐  ┌──▼───────────┐            │
│       │  Grafana    │  │ Alert  │  │ Cost         │            │
│       │  Dashboard  │  │ manager│  │ Exporter     │            │
│       └─────────────┘  └────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 メトリクス設計

#### エージェントランタイムメトリクス

```python
# src/project_sid/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# ----- エージェント状態 -----
agent_state = Gauge(
    "agent_state",
    "Current state of each agent",
    ["agent_id", "module"],
)
agent_active_count = Gauge(
    "agent_active_count",
    "Number of currently active agents",
)

# ----- モジュール実行 -----
module_execution_duration = Histogram(
    "module_execution_duration_seconds",
    "Duration of module execution",
    ["module_name", "agent_id"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
module_execution_total = Counter(
    "module_execution_total",
    "Total number of module executions",
    ["module_name", "status"],  # status: success, error, timeout
)

# ----- 認知コントローラ -----
cc_decision_duration = Histogram(
    "cc_decision_duration_seconds",
    "Cognitive Controller decision latency",
    ["agent_id"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
cc_broadcast_count = Counter(
    "cc_broadcast_count",
    "Number of CC broadcasts",
    ["decision_type"],  # action, speech, goal_update
)

# ----- 記憶システム -----
memory_operations = Counter(
    "memory_operations_total",
    "Memory system operations",
    ["operation", "memory_tier"],  # operation: read/write, tier: wm/stm/ltm
)
memory_store_size = Gauge(
    "memory_store_size",
    "Number of entries in memory store",
    ["memory_tier"],
)

# ----- 社会的認知 -----
social_interaction_count = Counter(
    "social_interaction_count",
    "Number of social interactions",
    ["interaction_type"],  # conversation, observation, cooperation
)
# 注: エージェントペア単位（agent_id x target_agent_id）のラベルは
# 1000体環境で最大~100万の組み合わせとなり、Prometheusのカーディナリティ爆発を
# 引き起こすため採用しない。代わりに以下の集約方式を使用する。
social_sentiment_histogram = Histogram(
    "social_sentiment_score",
    "Distribution of sentiment scores across all agent pairs",
    ["sentiment_bucket"],  # positive / neutral / negative
    buckets=[-10, -5, -2, 0, 2, 5, 10],
)
social_sentiment_summary = Gauge(
    "social_sentiment_summary",
    "Aggregated sentiment statistics",
    ["stat"],  # mean, median, std, min, max
)
# 個別ペアの感情スコアはPostgreSQLに記録し、
# Grafana + PostgreSQLデータソースで必要時に詳細分析する
```

#### LLM Gatewayメトリクス

```python
# LLMコスト・パフォーマンス追跡
llm_request_duration = Histogram(
    "llm_request_duration_seconds",
    "LLM API request duration",
    ["provider", "model", "module"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total tokens consumed",
    ["provider", "model", "direction"],  # direction: input, output
)
llm_request_total = Counter(
    "llm_request_total",
    "Total LLM requests",
    ["provider", "model", "status"],  # status: success, error, rate_limited, cached
)
llm_cost_dollars = Counter(
    "llm_cost_dollars",
    "Estimated cost in USD",
    ["provider", "model"],
)
llm_cache_hit_total = Counter(
    "llm_cache_hit_total",
    "LLM response cache hits",
    ["provider", "model"],
)
```

#### Minecraftサーバーメトリクス

```python
# Minecraftサーバー健全性
mc_tps = Gauge(
    "minecraft_tps",
    "Minecraft server TPS (ticks per second)",
    ["server_id"],
)
mc_player_count = Gauge(
    "minecraft_player_count",
    "Current player (agent) count",
    ["server_id"],
)
mc_memory_usage = Gauge(
    "minecraft_memory_usage_bytes",
    "Minecraft server memory usage",
    ["server_id"],
)
mc_chunk_loaded = Gauge(
    "minecraft_chunks_loaded",
    "Number of loaded chunks",
    ["server_id"],
)
```

### 3.3 Grafanaダッシュボード設計

| ダッシュボード | パネル例 | 更新頻度 |
|---|---|---|
| **Simulation Overview** | アクティブエージェント数、シミュレーション進行状況、全体TPS | 5秒 |
| **Agent Performance** | モジュール実行時間分布、CC決定レイテンシ、エラー率 | 10秒 |
| **LLM Usage & Cost** | リクエスト/秒、トークン消費、コスト累計、キャッシュヒット率 | 30秒 |
| **Minecraft Health** | TPS、メモリ使用、ロードチャンク数、プレイヤー接続 | 10秒 |
| **Social Dynamics** | 感情スコア分布、社会的インタラクション頻度、ネットワーク密度 | 60秒 |
| **Memory System** | WM/STM/LTM使用量、読み書き操作レート、検索レイテンシ | 30秒 |
| **Cost Tracker** | 時間あたりコスト、モジュール別コスト、プロバイダ別コスト | 60秒 |

### 3.4 アラート設定

```yaml
# monitoring/prometheus/alerts.yml
groups:
  - name: project-sid-critical
    rules:
      # Minecraftサーバーの性能低下
      - alert: MinecraftTPSLow
        expr: minecraft_tps < 15
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Minecraft TPS {{ $value }} on {{ $labels.server_id }}"
          description: "TPS dropped below 15 for 2+ minutes. Agents may desync."

      # エージェントの大量エラー
      - alert: AgentModuleErrorRate
        expr: |
          rate(module_execution_total{status="error"}[5m])
          / rate(module_execution_total[5m]) > 0.1
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Module {{ $labels.module_name }} error rate > 10%"

      # LLM APIレート制限
      - alert: LLMRateLimited
        expr: |
          rate(llm_request_total{status="rate_limited"}[5m]) > 10
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "LLM rate limiting on {{ $labels.provider }}/{{ $labels.model }}"

      # コスト超過
      - alert: LLMCostExceeded
        expr: |
          increase(llm_cost_dollars[1h]) > 50
        labels:
          severity: warning
        annotations:
          summary: "LLM cost exceeded $50/hour"

      # エージェント切断
      - alert: AgentDisconnected
        expr: |
          agent_active_count < (agent_active_count offset 5m) * 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "10%+ agents disconnected in last 5 minutes"

  - name: project-sid-performance
    rules:
      # CC決定レイテンシの増大
      - alert: CCDecisionSlow
        expr: |
          histogram_quantile(0.95, rate(cc_decision_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "CC p95 latency > 5s"

      # メモリストアの肥大化
      - alert: MemoryStoreOverflow
        expr: memory_store_size{memory_tier="stm"} > 500
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "STM store exceeding 500 entries for agent"
```

### 3.5 コスト監視

LLM APIコストはProject Sidの主要コストドライバーであるため、専用のコスト監視を設計する。

```python
# src/project_sid/cost_tracker.py
from dataclasses import dataclass

@dataclass
class LLMPricing:
    """LLMプロバイダの料金テーブル（2025年基準、USD per 1M tokens）"""
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    }

# コスト見積もり（500エージェント、2.5時間の文化伝達実験）
# 各エージェント約10モジュール、各モジュール平均30秒に1回LLM呼び出し
# = 500 * 10 * (9000/30) = 1,500,000 LLM呼び出し
# 平均500 input tokens + 200 output tokens per call
# gpt-4o-mini使用時:
#   Input: 1.5M * 500 / 1M * $0.15 = $112.50
#   Output: 1.5M * 200 / 1M * $0.60 = $180.00
#   合計: ~$292.50 per experiment run
```

---

## 4. 分散ロギング

### 4.1 ロギングアーキテクチャ

500-1000エージェントがそれぞれ10モジュールを並行実行する環境では、秒間数千〜数万のログエントリが生成される。コスト効率と検索性を両立するため、**Grafana Loki**を採用する。

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Agent Pod 1  │  │ Agent Pod 2  │  │ Agent Pod N  │
│  stdout/json │  │  stdout/json │  │  stdout/json │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                  ┌──────▼──────┐
                  │  Promtail   │  (DaemonSet: 各ノードで収集)
                  │  (ラベル付与)│
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Grafana    │
                  │  Loki       │
                  │  (S3/GCS    │
                  │   バックエンド)│
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Grafana    │  ← LogQL で検索・可視化
                  └─────────────┘
```

**Lokiを選択する理由**:
- ELK Stackと比較して運用コストが大幅に低い（フルテキストインデックスを作成しないため）
- ラベルベースのインデックスでエージェントID・モジュール名による絞り込みが高速
- Prometheusと同じラベル体系で統一的な可観測性を実現
- S3/GCSバックエンドで大容量ログを低コストに保存（1TB/日で$5以下）

### 4.2 構造化ログフォーマット

すべてのコンポーネントでJSON構造化ログを採用する。

```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "level": "INFO",
  "service": "agent-runtime",
  "agent_id": "agent-042",
  "module": "cognitive_controller",
  "event": "decision_broadcast",
  "trace_id": "abc123def456",
  "span_id": "span-789",
  "experiment_id": "exp-culture-500-v3",
  "sim_tick": 15420,
  "data": {
    "decision_type": "action",
    "action": "give_item",
    "target_agent": "agent-117",
    "item": "bread",
    "confidence": 0.87,
    "reasoning": "Agent-117 expressed hunger and has been cooperative"
  },
  "duration_ms": 1250,
  "llm_call": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "input_tokens": 480,
    "output_tokens": 120,
    "latency_ms": 890
  }
}
```

### 4.3 ログレベル設計

| レベル | 用途 | 出力頻度（500エージェント時） | 例 |
|---|---|---|---|
| **ERROR** | 回復不能なエラー | ~10/min | モジュール例外、接続断 |
| **WARNING** | 回復可能な異常 | ~100/min | LLMリトライ、タイムアウト |
| **INFO** | 重要イベント | ~15,000-30,000/min | CC決定、モジュール実行完了、社会的インタラクション、状態変化（※1） |
| **DEBUG** | 開発用詳細情報 | ~200,000/min | モジュール入出力、キャッシュ操作 |
| **TRACE** | 最詳細（開発時のみ） | ~1,000,000/min | LLMプロンプト全文、共有状態の差分 |

> **※1 INFOログ量の算出根拠**: CC実行が2秒間隔の場合、500体 x 30回/min = 15,000回/minのCC実行ログだけで15,000エントリ。他モジュール（Goal Generation 6回/min、Social Awareness 12回/min等）のINFOログも加算すると、合計で15,000-30,000/min程度となる。Lokiのインジェスト容量を計画する際はこの値を基準にする。

**本番環境**: INFO以上のみ収集（DEBUG/TRACEはサンプリングまたは無効）

### 4.4 エージェント行動のトレーシング

OpenTelemetryを使用し、エージェントの認知サイクル全体を分散トレースとして追跡する。

```python
# src/project_sid/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

tracer = trace.get_tracer("project-sid.agent")

async def agent_cognitive_cycle(agent_id: str, shared_state: dict):
    """1回の認知サイクル全体をトレース"""
    with tracer.start_as_current_span(
        "cognitive_cycle",
        attributes={
            "agent.id": agent_id,
            "sim.tick": shared_state["tick"],
        }
    ) as cycle_span:
        # 並列モジュール実行をそれぞれ子スパンとして記録
        with tracer.start_as_current_span("module.action_awareness"):
            await action_awareness.execute(shared_state)

        with tracer.start_as_current_span("module.goal_generation"):
            await goal_generation.execute(shared_state)

        with tracer.start_as_current_span("module.social_awareness"):
            await social_awareness.execute(shared_state)

        # CC決定
        with tracer.start_as_current_span("module.cognitive_controller"):
            decision = await cognitive_controller.decide(shared_state)
            cycle_span.set_attribute("cc.decision_type", decision.type)

        # ブロードキャスト実行
        with tracer.start_as_current_span("cc.broadcast"):
            await broadcast_decision(decision, shared_state)
```

トレースは Tempo (Grafanaエコシステム) または Jaeger で収集・可視化し、以下の分析を可能にする:
- 特定エージェントの認知サイクルのボトルネック特定
- LLM呼び出しのレイテンシ分析
- モジュール間の依存関係の可視化
- 異常な行動パターンの根因分析

### 4.5 ログ保持ポリシー

| データ種別 | 保持期間 | ストレージ |
|---|---|---|
| ERROR/WARNING ログ | 90日 | S3 Standard |
| INFO ログ | 30日 | S3 Standard → S3 IA |
| DEBUG ログ | 7日 | S3 Standard |
| トレースデータ | 14日 | Tempo / S3 |
| 実験結果ログ | 永続 | S3 Glacier |
| エージェント行動履歴 | 永続 | PostgreSQL + S3 |

---

## 5. 実験管理

### 5.1 実験管理の全体像

Project Sidの実験は、複数の条件・パラメータの組み合わせを複数回繰り返し実行する必要がある（例: 集団ルール実験は各設定4回繰り返し）。再現性の保証と効率的な実験管理のため、**Hydra**（設定管理）+ **MLflow**（実験追跡）の組み合わせを採用する。

```
┌─────────────────────────────────────────────────────────┐
│                    実験管理フロー                          │
│                                                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │  Hydra   │───▶│ Experiment│───▶│  MLflow Tracking │   │
│  │  Config  │    │ Launcher │    │  Server          │   │
│  │          │    │          │    │                  │   │
│  │ 階層型   │    │ K8sジョブ │    │ パラメータ記録   │   │
│  │ 設定管理 │    │ 生成・投入│    │ メトリクス記録   │   │
│  │          │    │          │    │ アーティファクト  │   │
│  └──────────┘    └──────────┘    └──────────────────┘   │
│       │                                    │             │
│       ▼                                    ▼             │
│  ┌──────────┐                    ┌──────────────────┐   │
│  │ Git管理  │                    │ 実験比較          │   │
│  │ 設定の   │                    │ ダッシュボード    │   │
│  │ バージョン│                    └──────────────────┘   │
│  └──────────┘                                            │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Hydra設定構造

```
configs/
├── config.yaml                    # メインエントリポイント
├── experiment/
│   ├── specialization.yaml        # 専門化実験（30体、20分）
│   ├── collective_rules.yaml      # 集団ルール実験（29体、20分）
│   ├── culture_memes.yaml         # 文化ミーム実験（500体、2.5時間）
│   ├── religion_propagation.yaml  # 宗教伝播実験（500体、2時間）
│   ├── single_agent_short.yaml    # 単一エージェント短期（25体、30分）
│   └── single_agent_long.yaml     # 単一エージェント長期（49体、4時間）
├── agent/
│   ├── full_piano.yaml            # 完全PIANOアーキテクチャ
│   ├── baseline.yaml              # ベースライン（3モジュール）
│   ├── ablation_no_social.yaml    # 社会認識除去
│   ├── ablation_no_action.yaml    # 行動認識除去
│   └── ablation_no_goal.yaml      # 目標生成除去
├── llm/
│   ├── gpt4o.yaml
│   ├── gpt4o_mini.yaml
│   └── claude_sonnet.yaml
├── environment/
│   ├── default_map.yaml
│   ├── culture_map_6towns.yaml    # 6町配置（1000x1200ブロック）
│   └── small_village.yaml
└── infrastructure/
    ├── local.yaml                 # ローカル開発
    ├── staging.yaml               # ステージング
    └── production.yaml            # 本番
```

#### 設定ファイルの例

```yaml
# configs/experiment/culture_memes.yaml
defaults:
  - /agent: full_piano
  - /llm: gpt4o_mini
  - /environment: culture_map_6towns

experiment:
  name: culture_memes
  description: "Cultural meme propagation in 6-town environment"

  agents:
    total_count: 500
    town_agents: 200       # 6町 x 33体 = ~200体
    rural_agents: 300
    towns:
      - name: "Sunny Glade"
        agent_count: 33
        position: [100, 200]
      - name: "Woodhaven"
        agent_count: 33
        position: [300, 200]
      - name: "Clearwater"
        agent_count: 33
        position: [500, 200]
      - name: "Meadowbrook"
        agent_count: 33
        position: [100, 600]
        special_agents:
          pastafarian_priests: 20  # 宗教伝播実験用
      - name: "Hilltop"
        agent_count: 33
        position: [300, 600]
      - name: "Riverbend"
        agent_count: 33
        position: [500, 600]

  simulation:
    duration_seconds: 9000   # 2.5時間
    tick_rate: 20             # ticks/sec
    analysis_interval: 7200  # 2時間ごとにミーム分析

  personality:
    generation: procedural   # 手続き的に生成
    seed: ${random_seed}

  evaluation:
    meme_keywords: ["eco", "dance", "meditation", "prank", "art"]
    religion_keywords:
      direct: ["Pastafarian", "Spaghetti Monster"]
      indirect: ["Pasta", "Spaghetti"]
    snapshot_interval: 600   # 10分ごとにスナップショット

  repetitions: 4  # 各条件4回繰り返し
```

### 5.3 MLflow統合

```python
# src/project_sid/experiment/tracker.py
import mlflow
from hydra import compose, initialize
from omegaconf import OmegaConf

class ExperimentTracker:
    """MLflowによる実験追跡"""

    def __init__(self, config):
        self.config = config
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.experiment.name)

    def start_run(self, run_name: str, repetition: int):
        """実験ランの開始"""
        with mlflow.start_run(run_name=f"{run_name}_rep{repetition}") as run:
            # Hydra設定をパラメータとして記録
            flat_config = OmegaConf.to_container(self.config, resolve=True)
            mlflow.log_params(self._flatten_dict(flat_config))

            # 実験設定をアーティファクトとして保存
            config_path = "experiment_config.yaml"
            OmegaConf.save(self.config, config_path)
            mlflow.log_artifact(config_path)

            return run.info.run_id

    def log_simulation_metrics(self, tick: int, metrics: dict):
        """シミュレーション中のメトリクス記録"""
        mlflow.log_metrics(metrics, step=tick)

    def log_experiment_results(self, results: dict):
        """実験結果の記録"""
        # 定量的結果
        mlflow.log_metrics({
            "unique_items_avg": results.get("unique_items_avg", 0),
            "role_entropy": results.get("role_entropy", 0),
            "social_cognition_r": results.get("social_cognition_r", 0),
            "tax_compliance_rate": results.get("tax_compliance_rate", 0),
            "meme_diversity": results.get("meme_diversity", 0),
            "conversion_count": results.get("conversion_count", 0),
            "total_llm_cost_usd": results.get("total_llm_cost_usd", 0),
        })

        # エージェント行動ログをアーティファクトとして保存
        mlflow.log_artifact("agent_behaviors.jsonl")
        mlflow.log_artifact("social_network.graphml")
        mlflow.log_artifact("simulation_snapshots/")

    @staticmethod
    def _flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    ExperimentTracker._flatten_dict(v, new_key, sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)
```

### 5.4 パラメータスイープ

Hydraのmultirun機能を活用し、複数条件のパラメータスイープを効率的に実行する。

```bash
# 集団ルール実験: 全条件 x 4回繰り返し = 12ラン
python -m project_sid.experiment.run \
  --multirun \
  experiment=collective_rules \
  agent=full_piano,baseline \
  experiment.influencer_type=pro_tax,anti_tax \
  experiment.repetition=0,1,2,3 \
  hydra/launcher=kubernetes  # K8sジョブとして並列実行
```

```yaml
# configs/hydra/launcher/kubernetes.yaml
defaults:
  - override /hydra/launcher: submitit_slurm  # または独自K8sランチャー

hydra:
  launcher:
    _target_: project_sid.launcher.KubernetesLauncher
    namespace: experiment-runs
    image: ghcr.io/project-sid/agent-runtime:latest
    resources:
      cpu: "4"
      memory: "8Gi"
    node_selector:
      node-pool: agent
```

### 5.5 再現性の保証

LLMの非決定性がProject Sid再現の最大の課題であるため、以下の戦略で対応する。

| 戦略 | 実装方法 | 効果 |
|---|---|---|
| **ランダムシード固定** | Python random, numpy, PyTorch のシード固定 | 環境生成・エージェント配置の再現 |
| **LLMレスポンスキャッシュ** | プロンプトのハッシュでRedisにキャッシュ | 同一入力に対する同一出力の保証 |
| **LLM temperature=0** | 可能な限り temperature=0 を使用 | 出力のばらつき低減（完全な決定性は保証されない） |
| **設定のバージョン管理** | Hydra設定をGit管理 + MLflowにアーティファクトとして記録 | 設定の完全な再現 |
| **環境スナップショット** | Dockerイメージタグとマップシードの記録 | 実行環境の再現 |
| **統計的再現** | 各条件4回以上の繰り返し + 信頼区間の報告 | LLM非決定性の統計的処理 |

```python
# src/project_sid/reproducibility.py
import hashlib
import json

class LLMCache:
    """LLMレスポンスのキャッシュ（再現性向上）"""

    def __init__(self, redis_client, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl

    def _make_key(self, provider: str, model: str, messages: list, params: dict) -> str:
        """プロンプトと設定からキャッシュキーを生成"""
        content = json.dumps({
            "provider": provider,
            "model": model,
            "messages": messages,
            "temperature": params.get("temperature", 0),
            "max_tokens": params.get("max_tokens"),
        }, sort_keys=True)
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()}"

    async def get_or_call(self, provider, model, messages, params, llm_client):
        key = self._make_key(provider, model, messages, params)
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)

        response = await llm_client.complete(provider, model, messages, params)
        await self.redis.setex(key, self.ttl, json.dumps(response))
        return response
```

---

## 6. 開発環境

### 6.1 環境ティア

| ティア | エージェント数 | 必要リソース | LLM | 用途 |
|---|---|---|---|---|
| **ローカル開発** | 1-5 | 16GB RAM, 4 CPU | モック / ローカルLLM | 個別モジュール開発・デバッグ |
| **ステージング** | 5-50 | K8sクラスタ（小） | 実API（レート制限付き） | 統合テスト・パフォーマンス確認 |
| **本番** | 500-1000 | K8sクラスタ（大） | 実API（フルスケール） | 研究実験の実行 |

### 6.2 ローカル開発セットアップ

```bash
# 前提条件のインストール
# Python 3.12+, Docker, Docker Compose

# リポジトリのクローン
git clone https://github.com/your-org/project-sid.git
cd project-sid

# Python仮想環境のセットアップ
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install uv
uv sync

# 環境変数の設定
cp .env.example .env
# .envを編集: OPENAI_API_KEY等を設定（モック使用時は不要）

# ローカルインフラの起動
docker compose up -d minecraft redis postgres

# Minecraftサーバーの起動完了を待機
docker compose logs -f minecraft  # "Done (XXs)!" が表示されるまで

# 3エージェントでローカル実行
python -m project_sid.agent.runtime \
  --config-name=config \
  experiment=specialization \
  agent=full_piano \
  infrastructure=local \
  experiment.agents.total_count=3 \
  experiment.simulation.duration_seconds=300

# テストの実行
pytest tests/unit/ -v                    # ユニットテスト
pytest tests/integration/ -v --timeout=60  # 統合テスト（ローカルDB必要）
```

### 6.3 ステージング環境

```bash
# ステージングクラスタへの接続
kubectl config use-context project-sid-staging

# ステージングデプロイ（50エージェント）
helm upgrade --install project-sid ./helm/project-sid \
  --namespace staging \
  --values helm/project-sid/values-staging.yaml \
  --set agentRuntime.pools[0].agentCount=50

# ログの確認
kubectl logs -n staging -l app=agent-runtime --follow

# Grafanaダッシュボードへのアクセス
kubectl port-forward -n monitoring svc/grafana 3000:3000
# http://localhost:3000
```

### 6.4 開発者ドキュメント体系

```
docs/
├── getting-started.md           # クイックスタートガイド
├── architecture/
│   ├── system-overview.md       # システム全体像
│   ├── piano-architecture.md    # PIANOアーキテクチャ詳細
│   └── data-flow.md             # データフロー図
├── development/
│   ├── local-setup.md           # ローカル環境セットアップ
│   ├── coding-standards.md      # コーディング規約
│   ├── testing-guide.md         # テストガイド
│   └── module-development.md    # 新モジュールの追加方法
├── operations/
│   ├── deployment.md            # デプロイ手順
│   ├── monitoring.md            # モニタリングガイド
│   ├── troubleshooting.md       # トラブルシューティング
│   └── runbook.md               # オペレーションランブック
└── experiments/
    ├── experiment-guide.md      # 実験実行ガイド
    ├── config-reference.md      # 設定リファレンス
    └── analysis-guide.md        # 結果分析ガイド
```

---

## 7. デバッグツール

### 7.1 エージェント行動のリプレイ

エージェントの行動履歴をイベントソーシングパターンで記録し、任意の時点からリプレイできるシステムを構築する。

```python
# src/project_sid/debug/replay.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentEvent:
    """エージェントの行動イベント"""
    timestamp: float
    sim_tick: int
    agent_id: str
    event_type: str          # module_output, cc_decision, action, speech, state_change
    module: str | None
    data: dict[str, Any]
    shared_state_snapshot: dict[str, Any] | None = None  # オプショナル

class EventStore:
    """イベントストア: 全エージェントイベントの永続化"""

    def __init__(self, db_url: str):
        self.db_url = db_url

    async def append(self, event: AgentEvent):
        """イベントの追加"""
        # PostgreSQLにJSONBとして保存
        pass

    async def get_events(
        self,
        agent_id: str | None = None,
        start_tick: int = 0,
        end_tick: int | None = None,
        event_types: list[str] | None = None,
    ) -> list[AgentEvent]:
        """条件に基づくイベント取得"""
        pass

class SimulationReplay:
    """シミュレーションリプレイエンジン"""

    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    async def replay_agent(
        self,
        agent_id: str,
        from_tick: int,
        to_tick: int,
        speed: float = 1.0,
    ):
        """特定エージェントの行動をリプレイ"""
        events = await self.event_store.get_events(
            agent_id=agent_id,
            start_tick=from_tick,
            end_tick=to_tick,
        )
        for event in events:
            yield event
            # speed倍速で再生

    async def replay_interaction(
        self,
        agent_ids: list[str],
        from_tick: int,
        to_tick: int,
    ):
        """複数エージェント間のインタラクションをリプレイ"""
        events = []
        for agent_id in agent_ids:
            agent_events = await self.event_store.get_events(
                agent_id=agent_id,
                start_tick=from_tick,
                end_tick=to_tick,
            )
            events.extend(agent_events)
        events.sort(key=lambda e: (e.sim_tick, e.timestamp))
        for event in events:
            yield event
```

### 7.2 タイムトラベルデバッグ

シミュレーション状態のスナップショットを定期的に保存し、任意の時点に巻き戻して再実行できる機構を提供する。

```python
# src/project_sid/debug/time_travel.py
import json
from pathlib import Path

class StateSnapshot:
    """シミュレーション状態のスナップショット"""

    def __init__(self, snapshot_dir: str):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    async def save_snapshot(self, sim_tick: int, state: dict):
        """スナップショットの保存"""
        snapshot = {
            "sim_tick": sim_tick,
            "agents": {},            # 全エージェントの状態
            "shared_states": {},     # 共有状態
            "minecraft_state": {},   # Minecraftワールド状態
            "social_graph": {},      # 社会的ネットワーク
            "memory_stores": {},     # 記憶システム
        }
        # 各エージェントの状態をシリアライズ
        for agent_id, agent_state in state["agents"].items():
            snapshot["agents"][agent_id] = {
                "working_memory": agent_state.working_memory,
                "goals": agent_state.current_goals,
                "sentiment_scores": agent_state.sentiment_scores,
                "position": agent_state.position,
                "inventory": agent_state.inventory,
                "active_modules": agent_state.active_modules,
            }

        path = self.snapshot_dir / f"snapshot_{sim_tick:08d}.json"
        path.write_text(json.dumps(snapshot, indent=2))
        return path

    async def load_snapshot(self, sim_tick: int) -> dict:
        """スナップショットの読み込み"""
        path = self.snapshot_dir / f"snapshot_{sim_tick:08d}.json"
        return json.loads(path.read_text())

    async def list_snapshots(self) -> list[int]:
        """利用可能なスナップショット一覧"""
        return sorted([
            int(p.stem.split("_")[1])
            for p in self.snapshot_dir.glob("snapshot_*.json")
        ])


class TimeTravelDebugger:
    """タイムトラベルデバッガー"""

    def __init__(self, snapshot_manager: StateSnapshot, event_store):
        self.snapshots = snapshot_manager
        self.event_store = event_store

    async def goto_tick(self, target_tick: int):
        """指定tickまで巻き戻し"""
        available = await self.snapshots.list_snapshots()
        # target_tick以前の最も近いスナップショットを選択
        snapshot_tick = max(t for t in available if t <= target_tick)
        state = await self.snapshots.load_snapshot(snapshot_tick)

        if snapshot_tick < target_tick:
            # スナップショットからtarget_tickまでイベントをリプレイ
            events = await self.event_store.get_events(
                start_tick=snapshot_tick,
                end_tick=target_tick,
            )
            state = await self._apply_events(state, events)

        return state

    async def fork_from(self, tick: int, modified_config: dict):
        """指定時点から設定を変更して分岐実行"""
        state = await self.goto_tick(tick)
        # modified_configを適用して新しいシミュレーションを開始
        # 例: 社会認識モジュールを無効にして実行し、差異を観察
        return state

    async def _apply_events(self, state: dict, events: list) -> dict:
        """イベントを順に適用して状態を復元"""
        for event in events:
            state = self._apply_single_event(state, event)
        return state

    def _apply_single_event(self, state: dict, event) -> dict:
        """単一イベントの適用"""
        # イベントタイプに応じた状態更新
        return state
```

### 7.3 モジュール単体テスト支援

各PIANOモジュールを独立してテストするためのテストハーネスを提供する。

```python
# tests/conftest.py
import pytest

@pytest.fixture
def mock_shared_state():
    """テスト用のモック共有エージェント状態"""
    return {
        "agent_id": "test-agent-001",
        "tick": 100,
        "position": {"x": 100, "y": 64, "z": 200},
        "inventory": {"bread": 5, "wood": 20},
        "working_memory": {
            "current_goal": "gather_resources",
            "last_action": "mine_stone",
            "last_action_result": "success",
        },
        "nearby_agents": [
            {"id": "agent-002", "distance": 5.0, "name": "Alice"},
            {"id": "agent-003", "distance": 12.0, "name": "Bob"},
        ],
        "conversations": [],
        "sentiment_scores": {"agent-002": 7, "agent-003": 4},
    }

@pytest.fixture
def mock_llm_client():
    """LLMクライアントのモック"""
    class MockLLM:
        def __init__(self):
            self.call_log = []
            self.responses = {}

        def set_response(self, prompt_contains: str, response: str):
            self.responses[prompt_contains] = response

        async def complete(self, messages, **kwargs):
            self.call_log.append({"messages": messages, **kwargs})
            for key, response in self.responses.items():
                if any(key in msg.get("content", "") for msg in messages):
                    return {"content": response}
            return {"content": "default mock response"}

    return MockLLM()

# テスト例: 社会認識モジュールの単体テスト
class TestSocialAwarenessModule:
    async def test_sentiment_tracking(self, mock_shared_state, mock_llm_client):
        """感情追跡が正しく動作するか"""
        module = SocialAwarenessModule(llm_client=mock_llm_client)
        mock_llm_client.set_response(
            "sentiment",
            '{"agent-002": 8, "agent-003": 3}'
        )

        result = await module.execute(mock_shared_state)
        assert "sentiment_scores" in result
        assert result["sentiment_scores"]["agent-002"] >= 7

    async def test_no_nearby_agents(self, mock_shared_state, mock_llm_client):
        """近くにエージェントがいない場合はスキップ"""
        mock_shared_state["nearby_agents"] = []
        module = SocialAwarenessModule(llm_client=mock_llm_client)

        result = await module.execute(mock_shared_state)
        assert len(mock_llm_client.call_log) == 0  # LLM呼び出しなし
```

### 7.4 デバッグCLIツール

```bash
# エージェントの現在状態を確認
sid debug agent agent-042 --state

# 特定エージェントの行動履歴を表示
sid debug agent agent-042 --history --from-tick=1000 --to-tick=2000

# 2エージェント間のインタラクションを表示
sid debug interaction agent-042 agent-117 --from-tick=1000

# シミュレーション状態のスナップショットを取得
sid debug snapshot --tick=5000

# タイムトラベル: 指定tickから再実行
sid debug time-travel --from-tick=5000 --config-override="agent.social_awareness.enabled=false"

# LLM呼び出しの詳細を表示
sid debug llm-calls --agent=agent-042 --module=cognitive_controller --last=10

# 社会ネットワークの可視化
sid debug social-graph --tick=5000 --output=graph.html

# モジュール実行のプロファイリング
sid debug profile --agent=agent-042 --duration=60
```

### 7.5 Webベースデバッグダッシュボード

リアルタイムでシミュレーションの状態を可視化するWebダッシュボードを提供する。

| 機能 | 説明 |
|---|---|
| **エージェントマップ** | Minecraftマップ上にエージェント位置をリアルタイム表示 |
| **社会ネットワーク** | エージェント間の関係性をグラフとして可視化 |
| **エージェント詳細** | 特定エージェントのモジュール状態、記憶、目標を表示 |
| **タイムライン** | シミュレーション全体のイベントタイムライン |
| **比較ビュー** | 異なる実験条件の結果を並べて比較 |
| **ログエクスプローラー** | フィルタリング可能なログビューワー |

---

## 8. 推奨技術スタック一覧

| カテゴリ | ツール | 用途 |
|---|---|---|
| **CI/CD** | GitHub Actions | パイプライン自動化 |
| **コンテナ** | Docker + Docker Compose | ローカル開発・テスト |
| **オーケストレーション** | Kubernetes (GKE/EKS) | 本番環境の管理 |
| **パッケージ管理** | Helm | K8sマニフェスト管理 |
| **スケジューラ** | Volcano / Kueue | ギャングスケジューリング |
| **メトリクス** | Prometheus | 時系列メトリクス収集 |
| **ダッシュボード** | Grafana | 可視化・アラート |
| **ログ集約** | Grafana Loki + Promtail | 分散ログ収集・検索 |
| **トレーシング** | OpenTelemetry + Tempo | 分散トレース |
| **設定管理** | Hydra | 階層型実験設定 |
| **実験追跡** | MLflow | パラメータ・メトリクス・アーティファクト記録 |
| **キャッシュ** | Redis | LLMレスポンスキャッシュ・共有状態 |
| **データベース** | PostgreSQL + pgvector | エージェント状態・記憶・実験結果 |
| **IaC** | Terraform | クラウドインフラのプロビジョニング |
| **シークレット** | External Secrets Operator | K8sシークレット管理 |
| **リンター** | Ruff | Python静的解析・フォーマット |
| **型チェック** | mypy | 型安全性の確保 |
| **テスト** | pytest | テスト実行 |

---

## 参考資料

- [Kubernetes for Agentic Apps](https://platformengineering.org/blog/kubernetes-for-agentic-apps-a-platform-engineering-perspective) - K8sにおけるAIエージェントワークロード管理
- [Google: Agent Execution on Kubernetes](https://opensource.googleblog.com/2025/11/unleashing-autonomous-ai-agents-why-kubernetes-needs-a-new-standard-for-agent-execution.html) - Agent Sandbox等の新しいK8sプリミティブ
- [Docker Compose for Agents](https://github.com/docker/compose-for-agents) - Docker ComposeによるAIエージェント構築
- [HydraFlow](https://github.com/daizutabi/hydraflow) - Hydra + MLflow統合
- [Grafana Loki](https://grafana.com/oss/loki/) - コスト効率の高いログ集約
- [Prometheus + Grafana Complete Guide](https://devtoolbox.dedyn.io/blog/prometheus-grafana-complete-guide) - メトリクス・モニタリング
- [AgentOps](https://www.agentops.ai/) - エージェント監視・デバッグプラットフォーム
- [Volcano Scheduler](https://volcano.sh/) - K8sバッチスケジューリング

---

[← 実装ドキュメント](../index.md)
