# MC Readiness Report — 15エージェント総合レビュー

**日付**: 2026-02-24
**目的**: Java版Minecraft購入後に本プロジェクトを動かせるかの網羅的レビュー
**レビュアー数**: 15 Explore エージェント（並列実行）

---

## 総合判定: NOT READY

15レビュアー中の判定:
| 判定 | エージェント数 | レビュアー |
|------|-------------|-----------|
| NOT READY | 7 | prereqs, basic-skills, mineflayer-behavior, advanced-skills, social-skills, redis-sas, error-handling |
| READY WITH CAVEATS | 7 | mc-server, bridge-connection, zmq-communication, perception, llm-integration, multi-agent, docker-deployment |
| READY | 1 | windows-compat |

---

## Severity別集計

| Severity | 件数 | 内容概要 |
|----------|------|---------|
| **CRITICAL** | 10 | スキルパラメータ不一致、ハンドラ欠損、SAS結合度、再接続なし、リスポーンなし |
| **HIGH** | 20 | no-opスキル、タイムアウト不整合、MockLLM idle固定、ドキュメント不足 |
| **MEDIUM** | 19 | 知覚制限、Docker設定、アクション名不一致、社会グラフ未連携 |
| **LOW** | 15 | コードスタイル、デフォルト値、エッジケース |

---

## CRITICAL問題一覧（修正必須）

### C1. craft_item パラメータ名不一致 (`item_name` vs `item`)
- **レビュアー**: basic-skills, mineflayer-behavior, advanced-skills
- **場所**: `basic.py:74`, `advanced.py:166`, `bridge/src/index.ts:106`
- **問題**: Python側は`item_name`キーで送信、TS側は`item`キーで読み取り → `mcData.itemsByName[undefined]` で必ず失敗
- **影響**: 全てのクラフト操作が機能しない

### C2. look_at 座標型不一致 (`{x,y,z}` vs `{yaw,pitch}`)
- **レビュアー**: basic-skills, mineflayer-behavior
- **場所**: `basic.py:104`, `bridge/src/index.ts:126`
- **問題**: Python側は`{x,y,z}`座標を送信、TS側は`{yaw,pitch}`を期待 → `bot.look(undefined, undefined)` で失敗
- **影響**: 全ての視線操作が機能しない

### C3. farm_plant/farm_harvest アクション名ルーティング不一致
- **レビュアー**: advanced-skills, mineflayer-behavior
- **場所**: `advanced.py:229,249`, `bridge/src/handlers/advanced.ts:49`
- **問題**: Python側は`action="plant"`/`"harvest"`を送信、TS側は`"farm"`ハンドラのみ登録 → `Unknown action` エラー
- **影響**: 農業スキルが全て失敗

### C4. flee/deposit/withdraw のTSハンドラ完全欠損
- **レビュアー**: advanced-skills, mineflayer-behavior
- **場所**: `advanced.py:339,363,386`
- **問題**: Python側がBridgeCommand生成するが、TS側にハンドラなし → `Unknown action` エラー
- **影響**: 逃走・チェスト操作が機能しない

### C5. ソーシャルスキル 5/9 がTS側未実装
- **レビュアー**: social-skills
- **場所**: `social.py` vs `bridge/src/handlers/social.ts`
- **問題**: unfollow, request_help, form_group, leave_group, send_message がTS側に未実装
- **影響**: 社会的相互作用の大部分が動作不能

### C6. ACTION_TO_SKILL マッピングの大幅欠落
- **レビュアー**: social-skills
- **場所**: `action_mapper.py`
- **問題**: vote, unfollow, request_help等がマッピング未登録。CCが生成してもスキル実行に到達しない
- **影響**: CCの判断がスキル実行に変換されない

### C7. `--mock-llm` が InMemorySAS を強制（SAS/LLM結合度）
- **レビュアー**: redis-sas
- **場所**: `main.py:153-164`
- **問題**: `--mock-llm`指定でRedis SASが使えず、Docker Redisが無意味に。LLM選択とSASバックエンドが密結合
- **影響**: Docker E2E環境で`--mock-llm`使用時にエージェント間状態共有が不可能

### C8. Bridge自動再接続機構の欠如
- **レビュアー**: error-handling
- **場所**: `bridge/client.py:164-168`, `main.py`
- **問題**: 3リトライ後DISCONNECTED→自動復帰なし。実行中のBridge切断に対するリカバリなし
- **影響**: MCサーバーやBridgeプロセスの一時的再起動でエージェントが永久停止

### C9. BridgeHealthMonitor がシミュレーションループに未接続
- **レビュアー**: error-handling
- **場所**: `bridge/health.py`, `main.py`
- **問題**: HealthMonitorが実装済みだがmain.pyからインスタンス化・呼び出しされていない
- **影響**: Bridge障害が検知されず黙って進行

### C10. ボット死亡→リスポーン機構なし
- **レビュアー**: error-handling
- **場所**: `bridge/src/index.ts:305-316`, `bridge/perception.py:134-136`
- **問題**: death時にPUBイベント発行+health=0.0設定のみ。pathfinder goalリセット、進行中スキルキャンセルなし
- **影響**: 死亡後にボットが異常動作を継続

---

## HIGH問題一覧（強く推奨）

### H1. equip/use/drop/eat が `_async_generic` (no-op) に委任
- **レビュアー**: basic-skills, mineflayer-behavior, advanced-skills
- **場所**: `advanced.py:648-671`
- **問題**: TS側にハンドラ実装済みなのに、Python側がBridgeCommandを生成せず"成功"を返す
- **影響**: 装備・使用・ドロップ・食事がMCワールドに反映されない

### H2. REQタイムアウト(5s) vs ハンドラタイムアウト(30s) 不整合
- **レビュアー**: zmq-communication
- **場所**: `client.py:22,37`, `index.ts:242`
- **問題**: move/explore等の長時間コマンドがPython REQの5sタイムアウトで失敗
- **影響**: 移動・探索コマンドが高確率でタイムアウト

### H3. offlineモード認証パラメータ未設定
- **レビュアー**: bridge-connection
- **場所**: `bridge/src/index.ts:167-172`
- **問題**: `auth: "offline"`が未設定。Mineflayerデフォルト`auth: "microsoft"`でMicrosoft認証フローに入る
- **影響**: offlineモードMCサーバーへの接続が失敗またはハング

### H4. MockLLMが常にidle応答 → デモ不能
- **レビュアー**: llm-integration, multi-agent
- **場所**: `llm/mock.py:21`, `main.py:139-140`
- **問題**: デフォルト応答`{"action": "idle"}`固定。`--mock-llm`でのデモが「何もしないエージェント」に
- **影響**: APIキーなしのデモが意味をなさない

### H5. CC SystemプロンプトにAction Paramsスキーマなし
- **レビュアー**: llm-integration
- **場所**: `cc/controller.py:46-63`
- **問題**: 各アクションの必須パラメータ仕様がプロンプトに含まれていない
- **影響**: LLMがaction_paramsを正しく返せず、スキル実行が高頻度で失敗

### H6. 15エージェント×MIDティアでのOpenAIレート制限
- **レビュアー**: llm-integration
- **場所**: `provider.py:94`, `main.py`
- **問題**: LLMGateway未使用で直接OpenAIProvider使用。100 RPM制限を容易に超過
- **影響**: RateLimitExceededError頻発

### H7. インベントリ同一アイテム複数スロット集計バグ
- **レビュアー**: perception
- **場所**: `bridge/perception.py:116-121`
- **問題**: dict内包表記で後のスロットが前のスロットを上書き。oak_log 32+16=48あるのに16と記録
- **影響**: エージェントのリソース認識が不正確

### H8. IMPORTANT_BLOCKSに基本ブロック（木/石/土）なし
- **レビュアー**: perception
- **場所**: `bridge/src/handlers/perception.ts:11-21`
- **問題**: 鉱石・ユーティリティのみ。基本資源が知覚できない
- **影響**: 「木を切る」「石を掘る」の基本行動で対象を特定不能

### H9. server.propertiesがDockerにマウントされていない
- **レビュアー**: docker-deployment
- **場所**: `docker-compose.sim.yml`
- **問題**: bukkit.ymlのみマウント。pvp, allow-flight, player-idle-timeout等の設定が反映されない

### H10. agentコンテナ512MB制限で15エージェント不足
- **レビュアー**: docker-deployment
- **場所**: `docker-compose.sim.yml:114`
- **問題**: 15エージェント×64MB/agent=960MB必要だが512MB制限。OOM Killer発動リスク

### H11. explore→move_to パラメータ変換未実装
- **レビュアー**: basic-skills
- **場所**: `action_mapper.py:27`
- **問題**: direction/distanceからx,y,z座標への変換ロジックなし

### H12. trade ハンドラが一方的アイテム投下のみ
- **レビュアー**: social-skills
- **場所**: `bridge/src/handlers/social.ts:6-41`
- **問題**: request_itemsパラメータを無視。交渉プロトコルなし

### H13. vote のACTION_TO_SKILLマッピング欠落 + 集計バックエンドなし
- **レビュアー**: social-skills
- **場所**: `action_mapper.py`, `social.ts:95-99`
- **問題**: CCがvoteを生成してもActionMapperでドロップ。投票はchatに文字列出力するだけ

### H14. LLM障害時の最終決定無限反復
- **レビュアー**: error-handling
- **場所**: `cc/controller.py:243-268`
- **問題**: _last_decisionが無限に反復ブロードキャスト。回数上限・エスカレーションなし

### H15. InMemorySASでのエージェント間状態共有不可
- **レビュアー**: redis-sas, multi-agent
- **場所**: `main.py:180-290`
- **問題**: 各_LocalSASインスタンスが完全独立。エージェント間の認識が不可能

### H16. ポートバインド失敗時にゾンビボット生成
- **レビュアー**: bridge-connection
- **場所**: `bridge/src/index.ts:196-199`
- **問題**: botがMCに接続済みだがZMQソケットなし → 制御不能

### H17. REQ再接続後のREP側デッドロック可能性
- **レビュアー**: zmq-communication
- **場所**: `client.py:158-163`, `index.ts:207-266`
- **問題**: REQソケット再作成後、REP側の状態マシンが壊れる可能性

### H18. smelt完了後にtakeOutput()未呼出
- **レビュアー**: mineflayer-behavior
- **場所**: `bridge/src/handlers/advanced.ts:22-41`
- **問題**: smelted結果がfurnaceに残ったまま

### H19. 前提条件インストール手順がe2e-setup.mdに欠落
- **レビュアー**: prereqs
- **場所**: `docs/e2e-setup.md`
- **問題**: uv, Docker Desktop, Node.jsの確認コマンドやインストールリンクなし

### H20. Python初期セットアップ（`uv sync --dev`）がe2e-setup.mdに未記載
- **レビュアー**: prereqs
- **場所**: `docs/e2e-setup.md`
- **問題**: Quick Startの前に必要なgit clone + uv sync --devが記載されていない

---

## 「Java版MC購入→動かせるまで」のステップごとの障壁一覧

### Step 0: MC Java Editionの購入
| 障壁 | Severity | 詳細 |
|------|----------|------|
| 購入が不要であることが未記載 | CRITICAL | online-mode=falseのためMC購入不要だが、ドキュメントに明記なし |

### Step 1: 前提ツールのインストール
| 障壁 | Severity | 詳細 |
|------|----------|------|
| インストール手順の欠落 | HIGH | Python 3.12+, uv, Node.js 20+, Docker のインストール方法が未記載 |
| Windows固有要件の不足 | HIGH | Docker Desktop + WSL2の有効化手順が未記載 |

### Step 2: リポジトリセットアップ
| 障壁 | Severity | 詳細 |
|------|----------|------|
| `uv sync --dev` が未記載 | HIGH | Python依存関係のインストール手順がe2e-setup.mdに欠落 |
| zeromq betaのWindows安定性 | HIGH | ネイティブバイナリ依存でWindows環境でビルド失敗の可能性 |

### Step 3: MCサーバー起動（Docker）
| 障壁 | Severity | 詳細 |
|------|----------|------|
| server.properties未マウント | HIGH | 一部設定が反映されない |
| flat worldの資源欠如 | MEDIUM | 鉱石・木・石が一切存在しない |
| EULA同意の注記なし | MEDIUM | 自動同意されるが法的注記なし |

### Step 4: Bridge起動
| 障壁 | Severity | 詳細 |
|------|----------|------|
| auth: "offline" 未設定 | HIGH | Microsoft認証フローに入り接続失敗の可能性 |
| ポートバインド失敗時のクリーンアップ不足 | HIGH | ゾンビボット生成リスク |

### Step 5: Agent起動（`--mock-llm`）
| 障壁 | Severity | 詳細 |
|------|----------|------|
| MockLLMが常にidle | HIGH | 全ボットが何もしないデモ |
| --mock-llm → InMemorySAS強制 | CRITICAL | Redis利用不可、エージェント間認識不可 |

### Step 6: Agent起動（OpenAI API）
| 障壁 | Severity | 詳細 |
|------|----------|------|
| APIキー設定方法未記載 | CRITICAL | 環境変数設定手順なし |
| action_paramsスキーマ不足 | HIGH | LLMが正しいパラメータを返せない |
| レート制限対策なし | HIGH | 15エージェントで100RPM超過 |

### Step 7: スキル実行
| 障壁 | Severity | 詳細 |
|------|----------|------|
| craft パラメータ不一致 | CRITICAL | item_name vs item で全craft失敗 |
| look_at 型不一致 | CRITICAL | x,y,z vs yaw,pitch で全look失敗 |
| plant/harvest アクション名不一致 | CRITICAL | 農業スキル全失敗 |
| flee/deposit/withdraw ハンドラ欠損 | CRITICAL | Unknown actionエラー |
| equip/use/drop/eat no-op | HIGH | MCワールドに反映されない |
| REQタイムアウト5s vs 30s | HIGH | move/explore タイムアウト |

### Step 8: 長時間稼働
| 障壁 | Severity | 詳細 |
|------|----------|------|
| 自動再接続なし | CRITICAL | Bridge切断で永久停止 |
| リスポーンなし | CRITICAL | 死亡後に異常動作継続 |
| HealthMonitor未接続 | CRITICAL | 障害が黙って進行 |

---

## 修正優先度ロードマップ

### Phase A: 最低限動作（CRITICAL修正）
1. **craft パラメータ統一**: basic.py/advanced.py の `item_name` → `item` に変更
2. **look_at ハンドラ修正**: TS側で`{x,y,z}`受信時に`bot.lookAt(new Vec3(x,y,z))`を使用
3. **plant/harvest → farm ルーティング**: TS側に`plant`/`harvest`ハンドラ追加（farmハンドラに委譲）
4. **auth: "offline" 追加**: `bridge/src/index.ts` の `createBot()` に `auth: "offline"` パラメータ追加
5. **--mock-llm と SAS分離**: `--sas-backend {redis,memory}` フラグ追加
6. **ドキュメント: MC購入不要を明記**

### Phase B: デモ実行可能（HIGH修正）
7. **MockLLM デモパターン**: main.pyでデモ用の多様な応答パターンを登録
8. **CC Systemプロンプト拡張**: action_params仕様を追加
9. **equip/use/drop/eat のBridgeCommand対応**: `_async_generic`→専用wrapper
10. **REQタイムアウト調整**: BridgeCommand.timeout_msに合わせたRCVTIMEO動的設定
11. **flee/deposit/withdraw TS ハンドラ追加**
12. **explore → TSのexploreハンドラに直接ルーティング**（move_toマッピングをやめる）
13. **e2e-setup.md 大幅改善**: 前提条件インストール手順、uv sync --dev、APIキー設定

### Phase C: 安定稼働（信頼性向上）
14. **Bridge自動再接続機構**
15. **HealthMonitor → main.py統合**
16. **ボットリスポーン処理**（goalリセット、スキルキャンセル）
17. **インベントリ集計バグ修正**（defaultdict集計）
18. **IMPORTANT_BLOCKSに基本ブロック追加**
19. **LLMGateway統合**（レート制限、サーキットブレーカー）
20. **server.properties Dockerマウント追加**

---

## レビュアー別サマリー

| # | レビュアー | 判定 | CRITICAL | HIGH | MEDIUM | LOW |
|---|-----------|------|----------|------|--------|-----|
| 1 | prereqs-reviewer | NOT READY | 2 | 3 | 2 | 2 |
| 2 | mc-server-reviewer | READY/CAVEATS | 0 | 0 | 3 | 4 |
| 3 | bridge-connection-reviewer | READY/CAVEATS | 0 | 3 | 4 | 3 |
| 4 | mineflayer-behavior-reviewer | NOT READY | 1 | 5 | 4 | 2 |
| 5 | zmq-communication-reviewer | READY/CAVEATS | 0 | 3 | 4 | 3 |
| 6 | basic-skills-reviewer | NOT READY | 2 | 3 | 4 | 0 |
| 7 | social-skills-reviewer | NOT READY | 2 | 3 | 2 | 1 |
| 8 | advanced-skills-reviewer | NOT READY | 2 | 2 | 3 | 2 |
| 9 | perception-reviewer | READY/CAVEATS | 0 | 2 | 3 | 2 |
| 10 | llm-integration-reviewer | READY/CAVEATS | 0 | 3 | 2 | 2 |
| 11 | redis-sas-reviewer | NOT READY | 1 | 2 | 2 | 1 |
| 12 | multi-agent-reviewer | READY/CAVEATS | 0 | 3 | 2 | 2 |
| 13 | error-handling-reviewer | NOT READY | 3 | 2 | 2 | 1 |
| 14 | docker-deployment-reviewer | READY/CAVEATS | 0 | 2 | 2 | 3 |
| 15 | windows-compat-reviewer | READY | 0 | 0 | 0 | 3 |

---

## 結論

**現状では「Java版MC購入→動かせる」状態にはなっていない。** 主な理由:

1. **Python↔TypeScript間のインターフェース不整合が体系的** — craft, look, plant, harvest, flee, deposit, withdraw等のコマンドが実行時にエラーになる
2. **`--mock-llm`モードが実質デモ不能** — 全エージェントがidle固定、かつRedis SASも強制無効化
3. **エラーリカバリ機構の欠如** — 自動再接続なし、リスポーンなし、HealthMonitor未接続
4. **ドキュメントの不足** — MC購入要否、APIキー設定、前提条件インストール手順が欠落

**Phase A（CRITICAL修正6項目）を実施すれば、3エージェント×10tickレベルの最小デモは可能になる。**
**Phase B（HIGH修正7項目）まで実施すれば、15エージェント×100tickのデモが実行可能になる。**
**Phase C（信頼性向上7項目）まで実施すれば、長時間安定稼働のシミュレーションに耐えうる状態になる。**
