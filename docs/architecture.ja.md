# nanochat 主要フォルダ・ファイル役割まとめ

nanochat のソースコードを構成する主要フォルダ（`nanochat/`、`scripts/`、`tasks/`）について、各ファイルが担っている役割を整理した資料です。補足として `runs/`、`dev/`、`tests/` の概要も末尾に記載します。

リポジトリ全体の思想：**「`--depth`（Transformer の層数）という 1 つのダイヤルだけで、トークナイズ → 事前学習 → ファインチューニング → 評価 → 推論 → チャット UI までを通しで実行できる、最小限で読みやすい LLM 学習ハーネス」**。各フォルダはこのパイプラインの役割ごとに分かれています。

---

## 全体像（パイプラインとフォルダの対応）

```
[tasks/]      評価・学習に使うデータセット（タスク）の定義
   │
   ▼
[scripts/]    実行エントリポイント（学習・評価・推論を起動するCLI）
   │  tok_train → base_train → (base_eval) → chat_sft → chat_rl → chat_web/chat_cli
   ▼
[nanochat/]   コアライブラリ（モデル・最適化・データ・トークナイザ・推論エンジン等）
```

- **`nanochat/`** … 再利用されるコアロジック（ライブラリ層）。モデル定義、オプティマイザ、データローダ、トークナイザ、推論エンジンなど。
- **`scripts/`** … 各ステージを起動する実行スクリプト（アプリ層）。`python -m scripts.xxx` または `torchrun ... -m scripts.xxx` で実行。
- **`tasks/`** … 学習・評価で使うデータセット（タスク）の抽象化。各 LLM 評価ベンチマークやSFTデータをここで定義。

---

## `nanochat/` — コアライブラリ

LLM 学習・推論のための中核ロジック。`scripts/` や `tasks/` から import されて使われる。

| ファイル | 役割 |
|---|---|
| `__init__.py` | 空ファイル（パッケージ化のためだけに存在）。 |
| `gpt.py` | **GPT Transformer 本体**。`GPTConfig`（モデル構成）と `GPT`（`nn.Module`）を定義。特徴は RoPE（回転位置埋め込み・絶対位置埋め込みなし）、QK Norm、token embedding と lm_head の重み非共有、MLP の relu² 活性化、学習可能パラメータを持たない RMSNorm、bias なし Linear、推論効率化のための GQA（Grouped-Query Attention）、Flash Attention 3 連携。`estimate_flops()`・`num_scaling_params()`・`setup_optimizer()`・`forward()`・`generate()` などを持つ。 |
| `optim.py` | **オプティマイザ**。AdamW と Muon を組み合わせた `MuonAdamW`（単一GPU）と `DistMuonAdamW`（分散）を定義。行列パラメータは Muon、埋め込み・スカラーは AdamW に割り当てる。fused なステップ関数（`adamw_step_fused`・`muon_step_fused`）を `torch.compile` で高速化。modded-nanogpt 由来。 |
| `flash_attention.py` | **Flash Attention の統一インターフェース**。Hopper(sm90) GPU では FA3、それ以外（Ada/Blackwell/MPS/CPU）では PyTorch の SDPA に自動フォールバックする。FA3 API 互換の `flash_attn_func` 等を提供するドロップイン置き換え。 |
| `fp8.py` | **最小限の FP8 学習**。torchao の `Float8Linear`（約2000行）を約150行で置き換えた tensorwise 動的スケーリングのみの実装。`torch._scaled_mm` を直接呼び、forward/backward の3つの matmul を FP8 量子化して高速化。`Float8Linear`・`convert_to_float8_training()` を提供。 |
| `tokenizer.py` | **BPE トークナイザ**（GPT-4 スタイル）。2実装あり：(1) 学習・推論両対応だが複雑な `HuggingFaceTokenizer`、(2) 学習は自作 rustbpe、推論は tiktoken を使う `RustBPETokenizer`。BOS や `<\|user_start\|>` 等の特殊トークン管理、会話を token id 列に変換する `render_conversation()`、補完用 `render_for_completion()` などを含む。 |
| `dataset.py` | **事前学習データ（parquet シャード）**の管理。HuggingFace の `karpathy/climbmix-400b-shuffle` から必要に応じてオンデマンドダウンロードし、ドキュメントをイテレートするユーティリティ（`list_parquet_files`・`parquets_iter_batched`・`download_single_file`）。 |
| `dataloader.py` | **事前学習用の分散データローダ**。「BOS整列 best-fit」方式で、各行を BOS トークンで始め、best-fit でドキュメントを詰めてパディングなし（100%利用）で詰め込む。DDP シャーディングと近似的なレジューム（再開）に対応。 |
| `engine.py` | **効率的な推論エンジン**。トークン列を入力に次トークンを返す。`KVCache`（KVキャッシュ）、`sample_next_token()`（temperature/top-k サンプリング）、`Engine.generate()`／`generate_batch()` を提供。電卓ツール（`use_calculator`）など推論時ツール実行のヘルパも含む。 |
| `execution.py` | **LLM 生成 Python コードのサンドボックス実行**。OpenAI HumanEval 由来。プロセス分離・タイムアウト・メモリ上限(256MB)・stdout/stderr キャプチャ・一時ディレクトリ・危険関数の無効化を行う。`execute_code()` が主API。※完全なセキュリティサンドボックスではない点に注意（ネットワーク遮断やカーネルレベル分離はなし）。 |
| `core_eval.py` | **CORE 指標の評価**（DCLM 論文準拠）。多肢選択(mc)・schema・言語モデル(lm)形式のプロンプト描画、few-shot 構築、バッチ化、`evaluate_example()`／`evaluate_task()` を提供。ベースモデルの能力測定に使用。 |
| `loss_eval.py` | **bits per byte (bpb) の評価**。単なる平均損失ではなく、語彙サイズ非依存の指標 bpb を計算（特殊トークンやマスクトークンを除外）。`evaluate_bpb()` を提供。 |
| `checkpoint_manager.py` | **チェックポイントの保存/読み込み**。モデル・オプティマイザ・メタデータの保存（`save_checkpoint`/`load_checkpoint`）、構成からのモデル構築（`build_model`）、最大モデル/最終ステップの探索、フェーズ（base/sft/rl/eval）指定でのモデル読み込み（`load_model`）。古いチェックポイントの欠損キー補完も行う。 |
| `report.py` | **学習レポートカードの生成**。git 情報・GPU/システム情報・コスト見積りを収集し、各ステージのメトリクスをセクションごとに記録（`Report.log`）して Markdown レポートを生成（`Report.generate`）。`get_report()` で取得。 |
| `common.py` | **共通ユーティリティ**。計算 dtype の自動判定（`COMPUTE_DTYPE`、`NANOCHAT_DTYPE` で上書き可）、色付きログ、ベースディレクトリ取得、ロック付きファイルダウンロード、DDP（分散）初期化/後始末（`compute_init`/`compute_cleanup`）、デバイス自動判定、ピーク FLOPS 取得など。 |

---

## `scripts/` — 実行エントリポイント

各ステージを起動する CLI。`python -m scripts.xxx`（単一GPU/CPU）または `torchrun --nproc_per_node=8 -m scripts.xxx`（分散）で実行。パイプライン順におおむね以下。

| ファイル | 役割 |
|---|---|
| `tok_train.py` | **トークナイザの学習**。自作 BPE ライブラリ（rustbpe）で GPT-4 スタイルのトークナイザを学習。`--vocab-size`（デフォルト 32768）等を指定。 |
| `tok_eval.py` | **トークナイザの評価**。各種テキストでの圧縮率（compression ratio）を測定。 |
| `base_train.py` | **ベースモデルの事前学習**。中心的な学習スクリプト。`--depth` でモデルサイズを決定し、その他ハイパラは自動計算。wandb ロギング、勾配累積、FA3/FP8 対応、チェックポイント保存、定期的な CORE/bpb 評価を含む。 |
| `base_eval.py` | **ベースモデルの評価**。3モード（`core`：CORE 指標、`bpb`：bits per byte、`sample`：サンプル生成）をカンマ区切りで指定可能。nanochat モデルにも HuggingFace モデル（例：GPT-2）にも対応。`evaluate_core` は `base_train.py` からも import される。 |
| `chat_sft.py` | **教師ありファインチューニング（SFT）**。ベースモデルを会話データ（`tasks/` の `TaskMixture`）でファインチューニングしチャットモデル化。チャット評価（`run_chat_eval`）も実行。 |
| `chat_rl.py` | **強化学習（RL）**。GSM8K 上で「GRPO」を実施。実装は簡略化されており、KL正則化なし・on-policy・DAPO 風のトークンレベル正規化・アドバンテージは (r - mu) のみ、と実質 REINFORCE に近い。 |
| `chat_eval.py` | **チャットモデルの評価**。`tasks/` の各タスク（ARC・MMLU・GSM8K・HumanEval・SpellingBee 等）で生成ベースの評価を行う。汎用ループはここ、タスク固有部分は `tasks/` 側。 |
| `chat_cli.py` | **CLI でのチャット**。ターミナルからモデルと対話。`Engine` を使った効率的生成、特殊トークンによるチャット状態機械を持つ。単一GPU想定。 |
| `chat_web.py` | **Web UI / API サーバ**（FastAPI）。ChatGPT ライクな Web UI と `/chat/completions`（ストリーミング）API を単一インスタンスで提供。データ並列で複数 GPU に各 1 コピーをロードしリクエスト分配。`/health`・`/stats` エンドポイントや乱用防止（メッセージ長・温度クランプ）も実装。 |

---

## `tasks/` — データセット / タスク定義

学習・評価で使う「タスク」を抽象化。各タスクは「会話のデータセット＋メタデータ＋（多くは）評価基準」。`scripts/chat_sft.py`（学習）や `scripts/chat_eval.py`（評価）から利用される。

| ファイル | 役割 |
|---|---|
| `common.py` | **タスクの基底クラス**。`Task`（データセットへの軽量スライス的ビュー、`get_example`・`evaluate` 等）、複数タスクを混合する `TaskMixture`、順次連結する `TaskSequence`、多肢選択を描画する `render_mc()` を定義。 |
| `arc.py` | **ARC**（Allen AI の科学常識・選択式問題）。`ARC-Easy` / `ARC-Challenge` の2サブセット。`categorical`（多肢選択）評価。 |
| `mmlu.py` | **MMLU**（幅広い分野の4択問題ベンチマーク）。多数の科目グループを持つ。 |
| `gsm8k.py` | **GSM8K**（小学校レベルの算数文章題、8K問）。解答中に `<< >>` タグで電卓ツール呼び出しを含むのが特徴。RL（`chat_rl.py`）でも使用。 |
| `humaneval.py` | **HumanEval**（名称に反し人間とは無関係の、Python コーディングベンチマーク）。`nanochat/execution.py` のサンドボックスで生成コードを実行・採点。import 抽出ヘルパ等を持つ。 |
| `smoltalk.py` | **SmolTalk**（HuggingFace の汎用会話データセット、小型モデル向けの "smol" 版）。主に SFT の学習データとして使用。 |
| `spellingbee.py` | **スペル/文字数えタスク**（自作生成タスク）。例：「strawberry に r はいくつ？」→ 3。手動カウントと Python 実行を組み合わせて解かせ、小型モデルにスペリング能力を付与する狙い。`SpellingBee`（文字数え）と `SimpleSpelling`（単純スペル）の2タスク。 |
| `customjson.py` | **任意の JSONL 会話の読み込み**。各行が `{"role","content"}` メッセージ配列の JSONL から会話タスクを作成。独自データの SFT 混入などに使う（例：アイデンティティ注入）。 |

---

## 補足：その他のフォルダ

| フォルダ/ファイル | 役割 |
|---|---|
| `runs/speedrun.sh` | 約 $100 で GPT-2 相当（d20系）を学習し会話するまでの**基準パイプライン全体**（8XH100想定）。 |
| `runs/runcpu.sh` | CPU / Apple Silicon (MPS) での実行例（モデルを大幅縮小）。 |
| `runs/scaling_laws.sh` / `runs/miniseries.sh` | スケーリング則実験 / モデルのミニシリーズ学習スクリプト（研究用）。 |
| `dev/gen_synthetic_data.py` | アイデンティティ注入用の合成データ生成例。 |
| `dev/repackage_data_reference.py` | 事前学習データのシャード生成（データ準備の参考実装）。 |
| `dev/generate_logo.html` / `dev/nanochat.png` / `dev/*.ipynb` / `dev/LEADERBOARD.md` / `dev/LOG.md` | ロゴ生成、画像、分析ノートブック、リーダーボード/開発ログ等のドキュメント類。 |
| `tests/test_engine.py` | 推論エンジン（`nanochat/engine.py`）のテスト。 |
| `tests/test_attention_fallback.py` | Flash Attention の FA3↔SDPA フォールバック（`nanochat/flash_attention.py`）のテスト。 |
| `pyproject.toml` / `uv.lock` | 依存管理（`uv` を使用）。 |
