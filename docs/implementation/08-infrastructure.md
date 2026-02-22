# 8. インフラストラクチャ・スケーリング設計

[トップ](../index.md)

---

## 8.1 全体インフラアーキテクチャ

### システム構成図

```
                            ┌─────────────────────────────────────────┐
                            │            ロードバランサー               │
                            │        (Nginx / Envoy Proxy)            │
                            └──────────┬──────────┬───────────────────┘
                                       │          │
             ┌─────────────────────────┤          ├─────────────────────────┐
             │                         │          │                         │
    ┌────────▼────────┐   ┌────────────▼──┐  ┌───▼──────────────┐  ┌──────▼───────┐
    │  APIゲートウェイ  │   │ Webダッシュボード│  │  管理API           │  │ メトリクス     │
    │  (LLM Gateway)  │   │ (Monitoring)  │  │  (Admin REST)    │  │ (Prometheus) │
    └────────┬────────┘   └───────────────┘  └──────────────────┘  └──────────────┘
             │
    ┌────────▼─────────────────────────────────────────────────────────┐
    │                    メッセージブローカー                             │
    │              (Redis Pub/Sub + Redis Streams)                     │
    │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
    │   │ agent-cmd│  │ agent-evt│  │ mc-events│  │ llm-queue│       │
    │   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
    └────────┬─────────────┬───────────────┬───────────────────────────┘
             │             │               │
    ┌────────▼──┐  ┌───────▼───┐   ┌──────▼──────────────────────────┐
    │ Agent     │  │ Agent     │   │       Minecraft サーバー群        │
    │ Worker    │  │ Worker    │   │  ┌──────┐ ┌──────┐ ┌──────┐    │
    │ Node 1    │  │ Node 2    │   │  │Shard1│ │Shard2│ │Shard3│    │
    │ (250体)   │  │ (250体)   │   │  │(~333)│ │(~333)│ │(~334)│    │
    │           │  │ ...       │   │  └──────┘ └──────┘ └──────┘    │
    └─────┬─────┘  └─────┬─────┘   │      Velocity Proxy            │
          │              │         └─────────────────────────────────┘
          │              │
    ┌─────▼──────────────▼─────────────────────────────────────────────┐
    │                    永続化レイヤー                                  │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
    │  │ PostgreSQL   │  │ Redis        │  │ Object Storage     │     │
    │  │ (Agent State │  │ (Cache/      │  │ (S3 / MinIO)       │     │
    │  │  + History)  │  │  Session)    │  │ (Logs / Snapshots) │     │
    │  └──────────────┘  └──────────────┘  └────────────────────┘     │
    └──────────────────────────────────────────────────────────────────┘
```

### 各コンポーネントの役割

| コンポーネント | 役割 | 技術候補 |
|---|---|---|
| **LLM APIゲートウェイ** | LLM呼び出しの一元管理、レート制限、フォールバック、キャッシュ | LiteLLM Proxy / Portkey / 自作Gateway |
| **メッセージブローカー** | エージェント間通信、イベント配信、コマンドキューイング | Redis 7+ (Pub/Sub + Streams) |
| **Agent Workerノード** | エージェントプロセスの実行・管理（各ノード最大250体） | Python 3.12+ / Ray Actor |
| **Minecraftサーバー群** | ゲーム環境の提供、Velocity Proxyで複数サーバーを統合 | Pufferfish + Velocity + Mineflayer |
| **PostgreSQL** | エージェント状態、記憶、実験結果の永続化 | PostgreSQL 16 + pgvector |
| **Redis** | セッションキャッシュ、共有状態、Pub/Sub | Redis 7+ (Cluster mode) |
| **Object Storage** | ログ、スナップショット、実験データの長期保存 | MinIO (self-hosted) / S3 |
| **Prometheus + Grafana** | メトリクス収集、ダッシュボード、アラート | Prometheus + Grafana |

### 通信フロー

```
1. Agent Worker → Redis Pub/Sub → 他のAgent Worker  (エージェント間対話)
2. Agent Worker → LLM Gateway → OpenAI/Claude API    (LLM推論リクエスト)
3. Agent Worker → Mineflayer → Minecraft Server       (環境操作)
4. Minecraft Server → Redis → Agent Worker            (環境イベント通知)
5. Agent Worker → PostgreSQL                          (状態永続化)
6. 全コンポーネント → Prometheus → Grafana             (モニタリング)
```

### デプロイメントトポロジー

**推奨構成**: Kubernetes (k8s) クラスター

```
┌─────────────────────────────────────────────────────────────┐
│  Kubernetes Cluster                                         │
│                                                             │
│  Namespace: project-sid                                     │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │ StatefulSet:     │  │ Deployment:       │                │
│  │ minecraft-shard  │  │ agent-worker      │                │
│  │ replicas: 1-6    │  │ replicas: 4-16    │                │
│  └─────────────────┘  └──────────────────┘                 │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │ Deployment:      │  │ StatefulSet:      │                │
│  │ llm-gateway      │  │ redis-cluster     │                │
│  │ replicas: 2-8    │  │ replicas: 6       │                │
│  └─────────────────┘  └──────────────────┘                 │
│  ┌─────────────────┐  ┌──────────────────┐                 │
│  │ StatefulSet:     │  │ Deployment:       │                │
│  │ postgresql       │  │ monitoring        │                │
│  │ replicas: 2 (HA) │  │ (Prom+Grafana)   │                │
│  └─────────────────┘  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

**小規模開発環境**: Docker Compose で全コンポーネントを1台で起動可能にし、開発体験を確保する。

---

## 8.2 エージェントプロセス管理

### 1000エージェントの実行モデル

PIANOアーキテクチャでは各エージェントが約10個のモジュールを並行実行するため、1000エージェント = 最大10,000の並行タスクが発生する。これを効率的に管理するために、**asyncio + マルチプロセス**のハイブリッドアプローチを採用する。

#### アプローチ比較

| 方式 | 利点 | 欠点 | 推奨用途 |
|---|---|---|---|
| **asyncio (単一プロセス)** | 低オーバーヘッド、メモリ効率 | GILの制約、CPU集約処理に不向き | LLM API待ち等のI/Oバウンド処理 |
| **マルチスレッド** | 共有メモリ | GIL制約、デバッグ困難 | 非推奨 |
| **マルチプロセス** | 真の並列性、GIL回避 | IPC オーバーヘッド、メモリ消費大 | CPU集約処理 |
| **Ray Actor** | 分散対応、自動スケーリング | 依存ライブラリ、学習コスト | 大規模分散 |

#### 推奨: ハイブリッド構成

```python
# 概念的なアーキテクチャ構造

# --- Worker Process (1プロセス = 最大250エージェント、暫定値) ---
class AgentWorkerProcess:
    """
    1つのOSプロセス内で最大250体のエージェントをasyncioで並行管理。
    各エージェントのモジュールはasyncio.Taskとして非同期実行。

    250体/プロセスの根拠（暫定見積もり）:
    - メモリ: 250体 x 64MB = 16GB（32GBノードの50%以内）
    - asyncioタスク数: 250体 x 9モジュール = 2,250タスク
      （大半がI/O待ちのためCPU負荷は低い）
    - LLM API待ちが支配的なI/Oバウンド処理のため、
      asyncioのイベントループで効率的にスケジュール可能

    注意: この値はPhase 2-3で実測ベンチマークを行い調整する。
    特にasyncio 2,250タスクのスケジューリングオーバーヘッドと
    実効メモリ消費を検証し、最適値を決定する。
    """

    async def run_agent(self, agent_id: str):
        """1エージェントの全モジュールを並行実行"""
        agent_state = SharedAgentState(agent_id)

        # 各モジュールを並行タスクとして起動
        tasks = [
            asyncio.create_task(self.run_module(
                "action_awareness", agent_state, interval=0.5
            )),
            asyncio.create_task(self.run_module(
                "goal_generation", agent_state, interval=10.0
            )),
            asyncio.create_task(self.run_module(
                "social_awareness", agent_state, interval=5.0
            )),
            asyncio.create_task(self.run_module(
                "cognitive_controller", agent_state, interval=2.0
            )),
            asyncio.create_task(self.run_module(
                "talking", agent_state, interval=1.0
            )),
            asyncio.create_task(self.run_module(
                "skill_execution", agent_state, interval=0.5
            )),
            asyncio.create_task(self.run_module(
                "memory_manager", agent_state, interval=3.0
            )),
            asyncio.create_task(self.run_module(
                "planning", agent_state, interval=15.0
            )),
            asyncio.create_task(self.run_module(
                "self_reflection", agent_state, interval=30.0
            )),
        ]
        await asyncio.gather(*tasks)

    async def run_module(self, module_name, state, interval):
        """モジュールを指定間隔で繰り返し実行"""
        while True:
            await module_registry[module_name].execute(state)
            await asyncio.sleep(interval)


# --- Supervisor (全Worker Processを管理) ---
class AgentSupervisor:
    """
    マルチプロセスでWorkerを起動し、エージェントを分配。
    1000体 = 4 Worker Process x 250体/Worker
    """

    def start(self, total_agents: int, agents_per_worker: int = 250):
        num_workers = math.ceil(total_agents / agents_per_worker)
        for i in range(num_workers):
            agent_ids = self.assign_agents(i, agents_per_worker)
            process = multiprocessing.Process(
                target=run_worker, args=(agent_ids,)
            )
            process.start()
```

### モジュール実行頻度の設計

論文の記述から推定される各モジュールの実行頻度:

| モジュール | 実行間隔 | LLM呼び出し | 推定トークン/回 |
|---|---|---|---|
| Action Awareness | 0.5秒 | なし (小規模NN) | 0 |
| Skill Execution | 0.5秒 | 条件付き | ~200 |
| Talking | 1.0秒 | あり | ~500 |
| Cognitive Controller | 2.0秒 | あり | ~800 |
| Memory Manager | 3.0秒 | 条件付き | ~300 |
| Social Awareness | 5.0秒 (対話時) | あり | ~600 |
| Goal Generation | 5-10秒 | あり | ~1,000 |
| Planning | 15.0秒 | あり | ~1,200 |
| Self-Reflection | 30.0秒 | あり | ~800 |

### リソース制限と公平なスケジューリング

```python
class AgentResourceLimiter:
    """エージェントごとのリソース制限"""

    def __init__(self):
        self.max_concurrent_llm_calls_per_agent = 3
        self.max_memory_per_agent_mb = 64
        self.llm_semaphore_per_worker = asyncio.Semaphore(50)

    async def throttled_llm_call(self, agent_id, prompt):
        """Worker単位でLLM同時呼び出し数を制限"""
        async with self.llm_semaphore_per_worker:
            return await llm_gateway.call(prompt)
```

**公平性の確保**:
- Worker単位のセマフォで同時LLM呼び出し数を制限（Worker内50並行）
- エージェントごとの最大同時LLM呼び出し数を3に制限
- 優先度キュー: 高速モジュール（Action Awareness）を低速モジュール（Planning）より優先
- メモリ使用量のモニタリングとOOMキラー回避のためのバックプレッシャー

---

## 8.3 LLM APIゲートウェイ

### アーキテクチャ

```
┌──────────────────────────────────────────────────────────────┐
│                    LLM API Gateway                           │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │ Request   │  │ Token-Aware  │  │ Provider       │        │
│  │ Queue     │→ │ Rate Limiter │→ │ Router         │        │
│  │ (Redis)   │  │              │  │                │        │
│  └──────────┘  └──────────────┘  └───────┬────────┘        │
│                                          │                  │
│                    ┌─────────────────────┼──────────┐       │
│                    │                     │          │       │
│              ┌─────▼─────┐  ┌────────────▼┐  ┌─────▼─────┐ │
│              │ OpenAI    │  │ Anthropic   │  │ Local     │ │
│              │ GPT-4o    │  │ Claude      │  │ vLLM      │ │
│              │ (Primary) │  │ (Fallback)  │  │ (高速用)   │ │
│              └───────────┘  └─────────────┘  └───────────┘ │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │ Prompt   │  │ Response     │  │ Cost           │        │
│  │ Cache    │  │ Cache        │  │ Tracker        │        │
│  │ (Redis)  │  │ (Semantic)   │  │                │        │
│  └──────────┘  └──────────────┘  └────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

### レート制限管理

LLM APIのレート制限は**トークンベース**と**リクエストベース**の両方を考慮する必要がある。

```python
class TokenAwareRateLimiter:
    """
    トークン数ベースのレート制限。
    OpenAI GPT-4oの制限例: 800K TPM (tokens per minute)
    """

    def __init__(self, redis_client):
        self.redis = redis_client
        self.limits = {
            "openai": {
                "tpm": 800_000,     # tokens per minute
                "rpm": 10_000,      # requests per minute
            },
            "anthropic": {
                "tpm": 400_000,
                "rpm": 4_000,
            },
        }

    async def acquire(self, provider: str, estimated_tokens: int) -> bool:
        """トークン消費を記録し、制限内かチェック"""
        key = f"ratelimit:{provider}:{current_minute()}"
        current = await self.redis.incrby(key, estimated_tokens)
        await self.redis.expire(key, 120)

        if current > self.limits[provider]["tpm"]:
            return False  # 制限超過 → キューで待機
        return True
```

### リクエストキューイング

```python
class LLMRequestQueue:
    """
    優先度付きリクエストキュー。
    高速モジュール（Action Awareness後のCC判断等）を優先。
    """

    PRIORITY_HIGH = 0     # 即時応答が必要（Talking, CC）
    PRIORITY_NORMAL = 1   # 通常処理（Social Awareness）
    PRIORITY_LOW = 2      # バックグラウンド処理（Planning, Self-Reflection）

    async def enqueue(self, request, priority=PRIORITY_NORMAL):
        score = time.time() + (priority * 1000)
        await self.redis.zadd("llm:queue", {request.id: score})

    async def dequeue_batch(self, batch_size=10):
        """バッチ処理でスループットを向上"""
        return await self.redis.zpopmin("llm:queue", batch_size)
```

### 複数プロバイダー負荷分散

| 戦略 | 説明 | 用途 |
|---|---|---|
| **Primary-Fallback** | メインプロバイダー障害時に自動切替 | 高可用性 |
| **Weighted Round Robin** | 重み付きで複数プロバイダーに分散 | コスト最適化 |
| **Latency-Based** | レスポンス時間が最短のプロバイダーを選択 | 低レイテンシ |
| **Cost-Based** | 最も安価なプロバイダーを優先（品質が同等の場合） | コスト削減 |

```python
class ProviderRouter:
    """
    複数LLMプロバイダーのインテリジェントルーティング。
    サーキットブレーカーパターンでの障害隔離を含む。
    """

    def __init__(self):
        self.providers = {
            "openai_gpt4o": ProviderConfig(
                weight=0.6, max_latency_ms=5000
            ),
            "anthropic_claude": ProviderConfig(
                weight=0.3, max_latency_ms=8000
            ),
            "local_vllm": ProviderConfig(
                weight=0.1, max_latency_ms=2000
            ),
        }
        self.circuit_breakers = {
            name: CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60
            )
            for name in self.providers
        }

    async def route(self, request: LLMRequest) -> LLMResponse:
        for provider_name in self.select_providers(request):
            cb = self.circuit_breakers[provider_name]
            if cb.is_open:
                continue
            try:
                response = await self.call_provider(provider_name, request)
                cb.record_success()
                return response
            except (RateLimitError, TimeoutError) as e:
                cb.record_failure()
                continue
        raise AllProvidersUnavailable()
```

### コネクションプーリング

```python
# httpx の AsyncClient を利用したコネクションプーリング
class LLMConnectionPool:
    def __init__(self):
        self.pools = {
            "openai": httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=200,
                    max_keepalive_connections=100,
                    keepalive_expiry=30,
                ),
                timeout=httpx.Timeout(30.0, connect=5.0),
            ),
            "anthropic": httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=50,
                ),
                timeout=httpx.Timeout(30.0, connect=5.0),
            ),
        }
```

### プロンプトキャッシュ

同一のシステムプロンプトやコンテキストが多くのエージェントで共有されるため、キャッシュの効果が大きい。

| キャッシュ層 | 対象 | TTL | 効果 |
|---|---|---|---|
| **プロンプトプレフィックス** | 共通システムプロンプト | 無期限 | AnthropicのPrompt Cachingを活用 |
| **セマンティックキャッシュ** | 類似リクエストの応答 | 5分 | 類似状況の重複呼び出しを削減 |
| **完全一致キャッシュ** | 同一リクエストの応答 | 1分 | 同時刻の同一判断を共有 |

---

## 8.4 水平スケーリング

### エージェントシャーディング戦略

1000体のエージェントを複数のWorkerノードに分散する。シャーディングの鍵は**空間局所性（spatial locality）**である。

```
┌─────────────────────────────────────────────────────┐
│         Shard Manager (Consistent Hashing)           │
│                                                      │
│   町 / 地理的領域に基づくシャーディング                    │
│   （論文の6町配置: 各33体、農村302体に準拠）:             │
│                                                      │
│   Shard 1: Sunny Glade (33体) + 近隣農村 (~50体)     │
│   Shard 2: Woodhaven (33体) + 近隣農村 (~50体)       │
│   Shard 3: Clearwater (33体) + 近隣農村 (~50体)      │
│   Shard 4: Meadowbrook (33体) + 近隣農村 (~50体)     │
│   Shard 5: Hilltop (33体) + 近隣農村 (~51体)         │
│   Shard 6: Riverbend (33体) + 近隣農村 (~51体)       │
│                                                      │
│   合計: 198 町エージェント + 302 農村エージェント        │
│        = 500体 (論文の文化伝達実験構成)                  │
└─────────────────────────────────────────────────────┘
```

**シャーディング原則**:
- **空間近接性**: 同じ町のエージェントは同一Shardに配置（対話頻度が高い）
- **負荷均等**: 各Shardのエージェント数を均等化
- **動的リバランス**: エージェントの移動に伴うShard間の再配置

### クラスター間通信

```
┌──────────┐    gRPC / Redis Pub/Sub    ┌──────────┐
│ Worker 1 │ ◄──────────────────────►  │ Worker 2 │
│ (Shard1) │                            │ (Shard2) │
└────┬─────┘                            └────┬─────┘
     │              ┌──────────┐             │
     └──────────►   │  Redis   │  ◄──────────┘
                    │  Cluster │
                    └──────────┘
```

| 通信パターン | 方式 | レイテンシ目標 |
|---|---|---|
| **同一Shard内対話** | インプロセス直接呼び出し | < 1ms |
| **異なるShard間対話** | Redis Pub/Sub | < 10ms |
| **ブロードキャスト** | Redis Pub/Sub (チャンネル) | < 50ms |
| **状態同期** | Redis Streams (永続) | < 100ms |
| **大量データ転送** | gRPC (直接) | 可変 |

### Redis Pub/Sub の選択理由

論文の500-1000体規模では、Redis Pub/SubがKafkaより適切と判断する:

| 観点 | Redis Pub/Sub | Apache Kafka |
|---|---|---|
| **レイテンシ** | ミリ秒レベル | 数十ミリ秒 |
| **スループット** | ハードウェア依存（数万〜数十万msg/sec/インスタンス） | 100万+ msg/sec |
| **必要スループット** | 1000体 x 10 msg/sec = 10K/sec | 同左 |
| **永続性** | Redis Streams で補完可能 | ネイティブ対応 |
| **運用複雑性** | 低 | 高（ZooKeeper/KRaft） |
| **メモリ消費** | 低 | 中～高 |

**結論**: 1000体規模ではRedis Cluster (6ノード) のPub/Sub + Streamsで十分。将来的に10,000体超を目指す場合はKafkaへの移行を検討する。Redis Pub/Subのスループットはハードウェア依存であり固定の上限はないが、単一チャンネルに集中するとパブリッシャー側がボトルネックになりうるため、チャンネルを地域ごとに分割して並列化する。実環境でのスループットはPhase 2-3で実測し、ボトルネックが確認された場合はチャンネル分割の粒度を調整する。

### ロードバランシング

```python
class ShardLoadBalancer:
    """
    エージェントの動的再配置によるShard間負荷均等化。
    """

    async def rebalance(self):
        shard_loads = await self.get_shard_metrics()
        avg_load = sum(s.cpu_percent for s in shard_loads) / len(shard_loads)

        for shard in shard_loads:
            if shard.cpu_percent > avg_load * 1.3:  # 30%超過
                agents_to_migrate = self.select_migration_candidates(shard)
                target_shard = self.find_least_loaded_shard(shard_loads)
                await self.migrate_agents(agents_to_migrate, target_shard)
```

### 状態の分散管理

| 状態種別 | 保存先 | 整合性モデル | 説明 |
|---|---|---|---|
| **Working Memory** | インプロセス (dict) | 強整合性 | 各エージェント固有、高頻度アクセス |
| **Short-Term Memory** | Redis (hash) | 結果整合性 | 最近の対話・観察、Shard間で共有可能 |
| **Long-Term Memory** | PostgreSQL + pgvector | 結果整合性 | ベクトル検索対応、永続化 |
| **Agent Profile** | PostgreSQL | 強整合性 | 名前、性格、初期設定 |
| **Simulation State** | Redis (shared) | 結果整合性 | グローバルな時間、環境状態 |

---

## 8.5 リソース要件見積もり

### エージェント1体あたりのリソース消費推定

| リソース | 推定値 | 根拠 |
|---|---|---|
| **メモリ** | 50-100 MB | Agent State + モジュールコンテキスト + バッファ |
| **CPU** | 0.05-0.1 コア | asyncio で大半がI/O待ち |
| **LLM API呼び出し** | 6-12回/分 | CC(30/min) + Goal(6-12/min) + Talk(条件付き) + 他 |
| **推定トークン消費** | 5,000-10,000 トークン/分 | 入力+出力の合計 |
| **ネットワーク** | 10-50 KB/sec | API呼び出し + Redis通信 + MC通信 |

### スケール別リソース要件

#### 10体 (開発・テスト環境)

| リソース | 要件 |
|---|---|
| **CPU** | 4 vCPU |
| **メモリ** | 8 GB |
| **構成** | 単一プロセス、Docker Compose |
| **Minecraft** | 単一サーバー (Pufferfish, view-distance=4) |
| **Redis** | 単一インスタンス |
| **DB** | SQLite or PostgreSQL (単一) |
| **LLM API** | 60-120呼び出し/分, 50K-100Kトークン/分 |
| **ネットワーク** | 1 Mbps |
| **推定月額** | $50-100 (LLM API) + $20 (コンピュート) |

#### 50体 (小規模実験)

| リソース | 要件 |
|---|---|
| **CPU** | 8-16 vCPU |
| **メモリ** | 32 GB |
| **構成** | 2 Worker Process, Docker Compose or k8s |
| **Minecraft** | 単一サーバー (Pufferfish, view-distance=4, 高性能設定) |
| **Redis** | 単一インスタンス (8GB) |
| **DB** | PostgreSQL (単一、16GB RAM) |
| **LLM API** | 300-600呼び出し/分, 250K-500Kトークン/分 |
| **ネットワーク** | 10 Mbps |
| **推定月額** | $250-500 (LLM API) + $100-200 (コンピュート) |

#### 500体 (文化伝達実験規模)

| リソース | 要件 |
|---|---|
| **CPU** | 64-128 vCPU |
| **メモリ** | 256-512 GB |
| **構成** | 8-16 Worker Process (k8s, 4-8ノード) |
| **Minecraft** | 3-6 Sharded Server (Velocity Proxy) |
| **Redis** | Redis Cluster (6ノード、各8GB) |
| **DB** | PostgreSQL (HA構成、64GB RAM) |
| **LLM API** | 3,000-6,000呼び出し/分, 2.5M-5Mトークン/分 |
| **ネットワーク** | 100 Mbps |
| **推定月額** | $2,500-5,000 (LLM API) + $800-1,600 (コンピュート) |

#### 1000体 (最大規模実験)

| リソース | 要件 |
|---|---|
| **CPU** | 128-256 vCPU |
| **メモリ** | 512 GB - 1 TB |
| **構成** | 16-32 Worker Process (k8s, 8-16ノード) |
| **Minecraft** | 6-10 Sharded Server (Velocity Proxy) |
| **Redis** | Redis Cluster (6ノード、各16GB) |
| **DB** | PostgreSQL (HA構成、128GB RAM、リードレプリカ) |
| **LLM API** | 6,000-12,000呼び出し/分, 5M-10Mトークン/分 |
| **ネットワーク** | 500 Mbps |
| **推定月額** | $5,000-10,000 (LLM API) + $2,000-4,000 (コンピュート) |

### ネットワーク帯域幅の内訳 (1000体時)

| 通信種別 | 帯域幅推定 | 備考 |
|---|---|---|
| LLM API (HTTPS) | 200-400 Mbps | 主要な帯域消費源 |
| Redis Pub/Sub | 10-50 Mbps | エージェント間メッセージ |
| Minecraft通信 | 50-100 Mbps | Mineflayerのワールドデータ |
| DB永続化 | 10-30 Mbps | 定期的なState書き込み |
| モニタリング | 5-10 Mbps | メトリクス送信 |

### ストレージ要件

| 用途 | 容量/時間 | 4時間実験時 |
|---|---|---|
| Agent State (PostgreSQL) | 100 MB/h (1000体) | 400 MB |
| 対話ログ | 500 MB/h (1000体) | 2 GB |
| メモリ (LTM, ベクトル) | 200 MB/h (1000体) | 800 MB |
| Minecraftワールド | 固定 ~500 MB | 500 MB |
| メトリクス (Prometheus) | 50 MB/h | 200 MB |
| **合計** | **~850 MB/h** | **~4 GB** |

### クラウドコスト概算

#### AWS (us-east-1)

| 規模 | インスタンス構成 | コンピュート月額 | LLM API月額 | 合計月額 |
|---|---|---|---|---|
| **10体** | 1x t3.xlarge | ~$120 | $50-100 | **$170-220** |
| **50体** | 1x m6i.4xlarge | ~$550 | $250-500 | **$800-1,050** |
| **500体** | 4x m6i.8xlarge + 3x m6i.4xlarge | ~$4,500 | $2,500-5,000 | **$7,000-9,500** |
| **1000体** | 8x m6i.8xlarge + 6x m6i.4xlarge | ~$9,000 | $5,000-10,000 | **$14,000-19,000** |

**注**: LLM APIコストはGPT-4o ($2.50/M入力, $10.00/M出力)ベース。Claude Sonnetなどの安価なモデルの活用やプロンプトキャッシュにより30-50%のコスト削減が可能。

#### GCP (us-central1)

| 規模 | インスタンス構成 | コンピュート月額 |
|---|---|---|
| **10体** | 1x n2-standard-4 | ~$100 |
| **50体** | 1x n2-standard-16 | ~$500 |
| **500体** | 4x n2-standard-32 + 3x n2-standard-16 | ~$4,200 |
| **1000体** | 8x n2-standard-32 + 6x n2-standard-16 | ~$8,500 |

#### コスト最適化戦略

| 戦略 | 削減率 | 詳細 |
|---|---|---|
| **スポット/プリエンプティブル** | 60-70% | Worker ノードにスポットインスタンスを使用 |
| **プロンプトキャッシュ** | 20-40% | Anthropic Prompt Caching / OpenAI batch API |
| **モデルティアリング** | 30-50% | 高速モジュールに安価なモデル (GPT-4o-mini)、低速モジュールにGPT-4o |
| **セマンティックキャッシュ** | 10-20% | 類似プロンプトの応答を再利用 |
| **リザーブドインスタンス** | 30-40% | 1年/3年のコミットメント |

**モデルティアリング詳細**:

| モジュール | 推奨モデル | 1Mトークン単価 | 理由 |
|---|---|---|---|
| Cognitive Controller | GPT-4o / Claude Sonnet | $2.50-3.00 入力 | 高品質な意思決定が必要 |
| Goal Generation | GPT-4o / Claude Sonnet | $2.50-3.00 入力 | 複雑な推論 |
| Planning | GPT-4o / Claude Sonnet | $2.50-3.00 入力 | 複雑な推論 |
| Talking | GPT-4o-mini / Claude Haiku | $0.15-0.25 入力 | 高頻度だが比較的単純 |
| Social Awareness | GPT-4o-mini / Claude Haiku | $0.15-0.25 入力 | パターン認識中心 |
| Skill Execution | GPT-4o-mini / Claude Haiku | $0.15-0.25 入力 | コマンド生成 |
| Memory Manager | GPT-4o-mini / Claude Haiku | $0.15-0.25 入力 | 要約・検索 |

---

## 8.6 メッセージングシステム

### エージェント間通信アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    Messaging Layer                           │
│                                                             │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Redis Pub/Sub Channels              │       │
│  │                                                  │       │
│  │  agent:{id}:inbox     - 個別メッセージ受信         │       │
│  │  shard:{id}:broadcast - Shard内ブロードキャスト    │       │
│  │  town:{name}:chat     - 町単位のチャットチャンネル   │       │
│  │  global:events        - グローバルイベント          │       │
│  │  sim:control          - シミュレーション制御        │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Redis Streams (永続)                │       │
│  │                                                  │       │
│  │  stream:dialogue:{session_id}  - 対話ログ         │       │
│  │  stream:events:{shard_id}      - イベントログ      │       │
│  │  stream:metrics                - メトリクスログ     │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### イベント駆動アーキテクチャ

```python
# イベントの型定義
class SimulationEvent:
    """シミュレーション内のイベント基底クラス"""
    event_type: str
    timestamp: float
    source_agent_id: str

class ChatEvent(SimulationEvent):
    """エージェント間の対話イベント"""
    event_type = "chat"
    target_agent_id: str
    message: str
    location: Tuple[int, int, int]

class MovementEvent(SimulationEvent):
    """エージェントの移動イベント"""
    event_type = "movement"
    from_position: Tuple[int, int, int]
    to_position: Tuple[int, int, int]

class EnvironmentEvent(SimulationEvent):
    """環境変化イベント"""
    event_type = "environment"
    change_type: str  # "block_break", "block_place", "mob_spawn"
    details: dict


# イベントバスの実装
class EventBus:
    """
    Redis Pub/Sub ベースのイベントバス。
    近接エージェントへの選択的配信を実現。
    """

    async def publish(self, event: SimulationEvent):
        channel = self.resolve_channel(event)
        await self.redis.publish(channel, event.serialize())

    def resolve_channel(self, event: SimulationEvent) -> str:
        """イベントの種類と空間的位置に基づいてチャンネルを決定"""
        if isinstance(event, ChatEvent):
            return f"agent:{event.target_agent_id}:inbox"
        elif isinstance(event, MovementEvent):
            shard = self.get_shard_for_position(event.to_position)
            return f"shard:{shard}:broadcast"
        else:
            return "global:events"
```

### チャット/対話のルーティング

エージェント間の対話は以下のフローで処理される:

```
Agent A (Talking Module)
    │
    ▼
[1] 対話相手の決定 (Social Awareness)
    │
    ▼
[2] メッセージ生成 (Talking Module + CC条件付け)
    │
    ▼
[3] Redis Pub/Sub で配信
    │  channel: agent:{B_id}:inbox
    ▼
Agent B (受信)
    │
    ▼
[4] Social Awareness が感情・意図を解析
    │
    ▼
[5] CC が応答方針を決定
    │
    ▼
[6] Talking Module が応答を生成
    │
    ▼
[7] Redis Pub/Sub で返信
       channel: agent:{A_id}:inbox
```

**近接性ベースのフィルタリング**: 対話はMinecraft環境内の距離に基づいてフィルタリングされる。一定距離以内のエージェントのみが対話可能で、これにより通信量を自然に制限する。

```python
class ProximityRouter:
    """空間近接性に基づく対話ルーティング"""

    CHAT_RANGE = 32        # ブロック (通常会話)
    SHOUT_RANGE = 64       # ブロック (叫び声)
    WHISPER_RANGE = 8      # ブロック (ささやき)

    async def get_nearby_agents(
        self, agent_id: str, range_type: str = "chat"
    ) -> list[str]:
        pos = await self.get_agent_position(agent_id)
        r = getattr(self, f"{range_type.upper()}_RANGE")

        # Redis GEO を使用した空間検索
        nearby = await self.redis.georadius(
            "agent:positions", pos.x, pos.z, r, unit="m"
        )
        return [a for a in nearby if a != agent_id]
```

---

## 8.7 障害耐性とリカバリ

### 障害シナリオと対策

| 障害 | 影響 | 対策 |
|---|---|---|
| **Worker Process クラッシュ** | 該当Shardのエージェントが停止 | k8s自動再起動 + Redis/DBからの状態復元 |
| **LLM API障害** | 全エージェントの認知処理が停止 | フォールバックプロバイダー + 応答キャッシュ |
| **Redis障害** | 通信・キャッシュが停止 | Redis Cluster HA + センチネル |
| **Minecraft Server障害** | 該当Shardの環境操作が停止 | 自動再起動 + ワールド永続化 |
| **ネットワーク分断** | Shard間通信途絶 | 独立動作モード + 再接続時の状態マージ |

### チェックポイントとスナップショット

```python
class SimulationCheckpointer:
    """
    定期的にシミュレーション全体のスナップショットを保存。
    障害時のロールバック復元を可能にする。
    """

    CHECKPOINT_INTERVAL = 300  # 5分ごと

    async def create_checkpoint(self):
        checkpoint = {
            "timestamp": time.time(),
            "simulation_tick": self.sim.current_tick,
            "agent_states": await self.dump_all_agent_states(),
            "redis_snapshot": await self.dump_redis_state(),
            "minecraft_state": await self.dump_minecraft_state(),
        }
        await self.object_storage.put(
            f"checkpoints/{checkpoint['timestamp']}.json.gz",
            gzip.compress(json.dumps(checkpoint).encode())
        )

    async def restore_from_checkpoint(self, checkpoint_id: str):
        data = json.loads(gzip.decompress(
            await self.object_storage.get(f"checkpoints/{checkpoint_id}.json.gz")
        ))
        await self.restore_agent_states(data["agent_states"])
        await self.restore_redis_state(data["redis_snapshot"])
        # Minecraft stateは手動復元（ワールドファイルのリストア）
```

---

## 8.8 データバックアップ・災害復旧（DR）計画

長時間シミュレーション（2.5-4時間）の途中でデータ損失が発生した場合、実験全体をやり直す必要があるため、堅牢なバックアップ戦略と復旧手順が不可欠である。

### バックアップ戦略

#### PostgreSQL（エージェント状態・記憶・実験結果）

| 方式 | 頻度 | RPO | 用途 |
|---|---|---|---|
| **WAL連続アーカイブ** | リアルタイム | ~0（直近のWALセグメント分のみ損失） | ポイントインタイムリカバリ（PITR） |
| **pg_basebackup（フルバックアップ）** | 日次（+ 実験開始前） | 最大24時間 | ベースラインリストア |
| **論理バックアップ（pg_dump）** | 実験終了時 | 実験単位 | 実験データのアーカイブ・移行 |

```bash
# WAL連続アーカイブの設定例（postgresql.conf）
archive_mode = on
archive_command = 'aws s3 cp %p s3://project-sid-backup/wal/%f'

# 実験開始前のフルバックアップ
pg_basebackup -h localhost -D /backup/base -Ft -z -P

# 実験終了時の論理バックアップ
pg_dump -h localhost -d projectsid -F c -f /backup/experiment_$(date +%Y%m%d_%H%M%S).dump
```

#### Redis（キャッシュ・セッション・共有状態）

| 方式 | 頻度 | データ損失許容 | 用途 |
|---|---|---|---|
| **RDBスナップショット** | 5分ごと（実験中） | 最大5分分のキャッシュデータ | 定期バックアップ |
| **AOF（Append Only File）** | `everysec`（毎秒fsync） | 最大1秒 | 高頻度書き込み対応 |
| **RDB + AOF併用** | 上記の組み合わせ | 最大1秒 | 推奨構成 |

```redis
# redis.conf
save 300 100        # 5分間に100回以上の変更でスナップショット
appendonly yes
appendfsync everysec
```

**注意**: Redisのキャッシュ・Pub/Subデータは本質的に一時的であり、完全な復元より再構築を優先する。エージェントの永続状態はPostgreSQLが正とする。

#### Qdrant（ベクトルDB：スキル・記憶の埋め込み）

| 方式 | 頻度 | 用途 |
|---|---|---|
| **スナップショットAPI** | 実験開始前 + 1時間ごと | コレクション単位のバックアップ |
| **コレクションのエクスポート** | 実験終了時 | アーカイブ・別環境への移行 |

```bash
# Qdrantスナップショット取得（REST API）
curl -X POST "http://localhost:6333/collections/skills/snapshots"
curl -X POST "http://localhost:6333/collections/agent_memory/snapshots"

# スナップショットをS3に転送
aws s3 sync /qdrant/snapshots/ s3://project-sid-backup/qdrant/
```

#### Minecraftワールドデータ

| 方式 | 頻度 | 用途 |
|---|---|---|
| **ワールドフォルダのコピー** | 実験開始前 + チェックポイント時（5分ごと） | ワールド状態の復元 |
| **サーバー側auto-save** | Paper/Pufferfishのデフォルト | 通常運用 |

```bash
# RCON経由でsave-allを実行してからコピー
mcrcon -H localhost -P 25575 -p $RCON_PASSWORD "save-all flush"
sleep 5
tar czf /backup/world_$(date +%s).tar.gz /data/world/
```

### 災害復旧（DR）手順

#### シナリオ1: シミュレーション中のWorkerプロセスクラッシュ

```
1. Kubernetes自動再起動（RestartPolicy: Always）
2. Worker起動時にRedisから直近のエージェント状態を復元
3. PostgreSQLから永続状態（LTM、プロファイル）を読み込み
4. Mineflayerボットを再接続
5. 直前のチェックポイントから処理を再開
   - 復旧時間目標（RTO）: 2分以内
```

#### シナリオ2: PostgreSQL障害

```
1. HA構成のスタンバイにフェイルオーバー（自動、Patroni/CloudNativePG）
2. フェイルオーバー不可の場合:
   a. WALアーカイブから最新のベースバックアップ + WALリプレイでPITR
   b. pg_basebackupのリストア + pg_wal_replayでポイントインタイムリカバリ
   c. 復旧時間目標（RTO）: 15分以内
3. エージェントWorkerを一時停止し、DB復旧後に再接続
```

#### シナリオ3: Redis Cluster障害

```
1. Redis ClusterのHA機能で自動フェイルオーバー（Sentinel/Cluster内レプリカ昇格）
2. 全ノード障害の場合:
   a. RDB + AOFファイルからデータ復元
   b. キャッシュデータは再構築（エージェントWorkerが自動的にキャッシュを再投入）
   c. 復旧時間目標（RTO）: 5分以内
3. Pub/Subの一時的な断絶はエージェント側でリトライにより吸収
```

#### シナリオ4: 長時間シミュレーション（4時間+）の中断・復旧

```
1. 5分間隔のシミュレーションチェックポイント（8.7節）を利用
2. 復旧手順:
   a. 最新の正常なチェックポイントを特定
   b. PostgreSQLをチェックポイント時点にPITR
   c. RedisにチェックポイントのAgent状態を復元
   d. Minecraftワールドをチェックポイント時点のバックアップに差し替え
   e. 全Workerを再起動し、チェックポイントから実験を再開
3. 復旧時間目標（RTO）: 20分以内
4. データ損失（RPO）: 最大5分（チェックポイント間隔）
```

### バックアップの保持ポリシー

| データ種別 | 保持期間 | ストレージ階層 |
|---|---|---|
| WALアーカイブ | 7日 | S3 Standard |
| フルバックアップ（日次） | 30日 | S3 Standard → S3 IA |
| 実験終了時バックアップ | 永続 | S3 Glacier |
| Redisスナップショット | 7日 | S3 Standard |
| Qdrantスナップショット | 実験終了後30日 | S3 Standard |
| Minecraftワールド | 実験終了後30日 | S3 Standard |

---

## 8.9 モニタリングとオブザーバビリティ

### メトリクス体系

```yaml
# Prometheusメトリクス定義

# エージェント関連
agent_active_count:           # アクティブエージェント数
agent_module_execution_seconds: # モジュール実行時間 (histogram)
agent_state_memory_bytes:     # エージェントあたりメモリ使用量

# LLM関連
llm_request_total:            # LLMリクエスト総数 (counter)
llm_request_duration_seconds: # LLMリクエスト所要時間 (histogram)
llm_tokens_consumed_total:    # 消費トークン数 (counter)
llm_cache_hit_ratio:          # キャッシュヒット率 (gauge)
llm_rate_limit_exceeded_total: # レート制限超過回数 (counter)
llm_cost_dollars_total:       # 累積API費用 (counter)

# 通信関連
messaging_published_total:    # 配信メッセージ数 (counter)
messaging_latency_seconds:    # メッセージ遅延 (histogram)
dialogue_sessions_active:     # アクティブ対話セッション数

# インフラ関連
worker_cpu_usage_percent:     # Worker CPU使用率
worker_memory_usage_percent:  # Worker メモリ使用率
minecraft_tps:                # Minecraft TPS (ticks per second)
redis_connected_clients:      # Redis接続クライアント数
```

### Grafanaダッシュボード構成

| パネル | 表示内容 |
|---|---|
| **Overview** | 総エージェント数、稼働率、シミュレーション時間 |
| **LLM Performance** | リクエスト/秒、レイテンシp50/p95/p99、キャッシュヒット率 |
| **LLM Cost** | 累積コスト、プロバイダー別内訳、トークン消費推移 |
| **Agent Health** | モジュール実行時間分布、エラー率、メモリ使用量 |
| **Communication** | メッセージスループット、対話セッション数、Shard間通信量 |
| **Minecraft** | TPS推移、プレイヤー(Bot)数、チャンク負荷 |
| **Infrastructure** | CPU/メモリ使用率、ネットワークI/O、ディスクI/O |

---

## 8.10 段階的スケーリング戦略

### Phase 1: 開発環境 (1-10体)

```yaml
# docker-compose.yml (概念)
services:
  minecraft:
    image: itzg/minecraft-server:pufferfish
    mem_limit: 2g

  redis:
    image: redis:7-alpine

  postgres:
    image: postgres:16-alpine

  agent-worker:
    build: .
    environment:
      AGENT_COUNT: 10
      LLM_PROVIDER: openai
    depends_on: [minecraft, redis, postgres]
```

**ゴール**: 単一マシンで全コンポーネントが動作し、開発・デバッグが容易。

### Phase 2: 小規模実験 (10-50体)

- Docker Composeの `deploy.replicas` でWorkerをスケール
- Redis / PostgreSQL は依然として単一インスタンス
- LLM APIの呼び出し最適化（バッチ処理、キャッシュ）を実装

### Phase 3: 中規模実験 (50-500体)

- Kubernetesへの移行
- Redis Clusterの導入
- Minecraftサーバーのシャーディング (Velocity Proxy)
- LLM APIゲートウェイの本格運用（マルチプロバイダー）
- モニタリング/アラートの整備

### Phase 4: 大規模実験 (500-1000+体)

- 複数ノードへの完全分散
- HPA (Horizontal Pod Autoscaler) によるWorkerの自動スケーリング
- Redis ClusterとPostgreSQLのリードレプリカによる読み取りスケーリング
- コスト最適化（スポットインスタンス、モデルティアリング）
- チェックポイント/リカバリの本格運用

---

## 8.11 技術選定サマリー

| カテゴリ | 推奨技術 | 代替候補 | 選定理由 |
|---|---|---|---|
| **エージェント実行** | asyncio + multiprocessing | Ray Actor | シンプル、低オーバーヘッド、段階的にRayへ移行可能 |
| **メッセージング** | Redis Pub/Sub + Streams | Kafka, RabbitMQ | 低レイテンシ、運用が容易、1000体規模で十分 |
| **LLMゲートウェイ** | LiteLLM Proxy | Portkey, 自作 | OSS、多プロバイダー対応、プロキシモード |
| **状態永続化** | PostgreSQL + pgvector | MongoDB | ベクトル検索対応、トランザクション、成熟したエコシステム |
| **キャッシュ** | Redis 7 (Cluster) | Valkey, KeyDB, Memcached | Pub/Sub/Streams/GEO等の多機能活用。**注**: Redis 7.4以降はSSPL+RSALv2ライセンスに変更されたため、クラウドサービスとしての再配布に制約がある。セルフホストでの利用には影響しないが、将来的なリスク軽減のためValkey（Linux Foundation管理のRedisフォーク、BSD-3ライセンス）またはKeyDBを代替候補として評価する |
| **コンテナオーケストレーション** | Kubernetes (k8s) | Docker Swarm, Nomad | 自動スケーリング、セルフヒーリング、エコシステム |
| **Minecraftプロキシ** | Velocity | BungeeCord | 高性能、モダン、Paper MC公式推奨 |
| **モニタリング** | Prometheus + Grafana | Datadog, CloudWatch | OSS、高カスタマイズ性、コスト効率 |
| **オブジェクトストレージ** | MinIO (self-hosted) / S3 | GCS, Azure Blob | S3互換API、開発環境はMinIO |

---

## 8.12 論文との対応関係

| 論文の記述 | 本設計での対応 |
|---|---|
| 1000体以上のエージェント同時実行 | 8.2 asyncio + multiprocessing ハイブリッド、8.4 水平シャーディング |
| 各エージェント約10モジュールの並列実行 | 8.2 asyncio.create_task による並行実行 |
| 5-10秒間隔の社会的目標生成 | 8.2 モジュール実行頻度テーブル、8.3 LLM APIゲートウェイ |
| 500体: 1000x1200ブロック、6つの町 | 8.4 空間局所性ベースのシャーディング |
| 計算コストにより独立再現が困難 | 8.5 コスト見積もり + 8.9 段階的スケーリング |
| GPT-4oへの依存 | 8.3 マルチプロバイダー対応、8.5 モデルティアリング |

---

## 参考文献

- [AgentScope: Very Large-Scale Multi-Agent Simulation](https://arxiv.org/abs/2407.17789) - 100万エージェント規模のアクターモデルベース分散シミュレーション
- [LLM Gateway Patterns: Rate Limiting and Load Balancing Guide](https://collabnix.com/llm-gateway-patterns-rate-limiting-and-load-balancing-guide/) - LLMゲートウェイの設計パターン
- [Redis Pub/Sub vs Apache Kafka](https://thenewstack.io/redis-pub-sub-vs-apache-kafka/) - メッセージングシステムの比較
- [Ray: Scale Machine Learning & AI Computing](https://www.ray.io/) - 分散Pythonフレームワーク
- [Kubernetes Horizontal Pod Autoscaler](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/) - k8s自動スケーリング
- [LLM API Pricing Comparison 2025](https://intuitionlabs.ai/articles/llm-api-pricing-comparison-2025) - LLM APIの料金比較
- [Rate Limiting in AI Gateway: The Ultimate Guide](https://www.truefoundry.com/blog/rate-limiting-in-llm-gateway) - トークンベースレート制限

---

[トップ](../index.md)
