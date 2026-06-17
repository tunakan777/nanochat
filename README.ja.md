# nanochat

*この文書は [README.md](README.md) の日本語訳です / This is the Japanese translation of [README.md](README.md).*

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat は LLM を学習するための最もシンプルな実験用ハーネスです。シングル GPU ノードで動作するように設計されており、コードは最小限でハック可能、トークン化・事前学習・ファインチューニング・評価・推論・チャット UI といった LLM の主要な段階をすべて網羅しています。たとえば、2019 年に学習するのに約 43,000 ドルかかった GPT-2 相当の能力を持つ LLM を、わずか 48 ドル（8XH100 GPU ノードを約 2 時間）で自分で学習し、見慣れた ChatGPT ライクな Web UI で会話することができます。スポットインスタンスを使えば、総コストは約 15 ドルにまで近づきます。より一般的には、nanochat は `--depth`（GPT トランスフォーマーモデルのレイヤー数）というたった一つの複雑さのダイヤルを設定するだけで、計算最適（compute-optimal）なモデルのミニシリーズ全体を学習できるよう、最初から構成されています（GPT-2 相当の能力はおおよそ depth 26 で得られます）。その他のすべてのハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率の調整、学習ホライズン、ウェイトディケイなど）は、最適になるように自動で計算されます。

リポジトリに関する質問については、Devin/Cognition の [DeepWiki](https://deepwiki.com/karpathy/nanochat) を使ってリポジトリについて質問するか、[Discussions タブ](https://github.com/karpathy/nanochat/discussions)を使うか、Discord の [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) チャンネルに来ることをおすすめします。

## Time-to-GPT-2 リーダーボード

現在、開発の主な焦点は、最も多くの計算量を必要とする事前学習段階のチューニングにあります。modded-nanogpt リポジトリに触発され、また進捗とコミュニティのコラボレーションを促すために、nanochat は「GPT-2 スピードラン」のリーダーボードを維持しています。これは、DCLM CORE スコアで測定される GPT-2 グレードの能力に達するまでに nanochat モデルを学習するのに必要な実時間（wall-clock time）です。[runs/speedrun.sh](runs/speedrun.sh) スクリプトは常に、GPT-2 グレードのモデルを学習して会話するための基準（リファレンス）となる方法を反映しています。現在のリーダーボードは以下の通りです。

| # | time | val_bpb | CORE | 説明 | 日付 | Commit | 貢献者 |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | オリジナルの OpenAI GPT-2 チェックポイント | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 ベースライン、わずかに過学習 | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 わずかに学習不足 **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | 合計バッチサイズを 1M トークンに引き上げ | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | データセットを NVIDIA ClimbMix に変更 | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

私たちが最も重視する指標は「time to GPT-2」、すなわち 8XH100 GPU ノード上で GPT-2 (1.6B) の CORE 指標を上回るのに必要な実時間です。GPT-2 の CORE スコアは 0.256525 です。2019 年には GPT-2 の学習に約 43,000 ドルかかったので、7 年間にわたるスタック全体での多くの進歩により、はるかに高速かつ 100 ドルを大きく下回るコストで実現できるようになったのは驚くべきことです（たとえば現在の約 3 ドル/GPU/時で 8XH100 ノードは約 24 ドル/時なので、2 時間で約 48 ドルです）。

リーダーボードの解釈方法や貢献方法に関する詳細なドキュメントは [dev/LEADERBOARD.md](dev/LEADERBOARD.md) を参照してください。

## はじめに

### セットアップ

nanochat は依存関係の管理に [uv](https://docs.astral.sh/uv/) を使用します。インストール方法：

```bash
uv sync --extra gpu    # CUDA（A100/H100 など）の場合に使用
uv sync --extra cpu    # （または）CPU のみ / MPS の場合に使用
source .venv/bin/activate
```

開発用（pytest、matplotlib、ipykernel、transformers などを追加）：

```bash
uv sync --extra gpu --group dev
```

### GPT-2 を再現して会話する

最も楽しめるのは、自分自身の GPT-2 を学習してそれと会話することです。そのためのパイプライン全体は、単一のファイル [runs/speedrun.sh](runs/speedrun.sh) に含まれており、8XH100 GPU ノードで実行するように設計されています。お好みのプロバイダ（私は [Lambda](https://lambda.ai/service/gpu-cloud) を使っていて気に入っています）で新しい 8XH100 GPU ボックスを起動し、学習スクリプトを開始してください。

```bash
bash runs/speedrun.sh
```

実行には約 3 時間かかるため、screen セッション内で実行するとよいでしょう。完了すると、ChatGPT ライクな Web UI でモデルと会話できます。再度、ローカルの uv 仮想環境がアクティブであることを確認し（`source .venv/bin/activate` を実行）、サーブします。

```bash
python -m scripts.chat_web
```

そして表示された URL にアクセスします。正しくアクセスするように注意してください。たとえば Lambda では、使用しているノードのパブリック IP の後にポートを付けます。例：[http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/) など。あとは普段 ChatGPT と話すように、自分の LLM と話してください！ 物語や詩を書かせてみましょう。あなたが誰かを尋ねてハルシネーションを見てみましょう。空がなぜ青いのか聞いてみましょう。あるいはなぜ緑なのか聞いてみましょう。このスピードランは 4e19 FLOPs の能力のモデルなので、幼稚園児と話しているような感じです :)。

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

いくつか補足：

- このコードは Ampere の 8XA100 GPU ノードでも問題なく動作しますが、少し遅くなります。
- すべてのコードは `torchrun` を省略することでシングル GPU でも問題なく動作し、ほぼ同一の結果が得られます（コードは自動的に勾配累積（gradient accumulation）に切り替わります）が、8 倍の時間を待つ必要があります。
- GPU のメモリが 80GB 未満の場合、一部のハイパーパラメータを調整しないと OOM / VRAM 不足になります。スクリプト内の `--device-batch-size` を探し、収まるまで減らしてください。たとえばデフォルトの 32 から 16、8、4、2、あるいは 1 まで。それ以下にする場合は、もう少し知識が必要で、より創意工夫が求められます。
- ほとんどのコードはかなり一般的な PyTorch なので、それをサポートするもの（xpu、mps など）なら何でも動作するはずですが、これらすべてのコードパスを私自身が動かしたわけではないので、鋭いエッジ（不具合）があるかもしれません。

## 研究

研究者の方で nanochat の改善を手伝いたい場合、興味深いスクリプトが 2 つあります：[runs/scaling_laws.sh](runs/scaling_laws.sh) と [runs/miniseries.sh](runs/miniseries.sh) です。関連ドキュメントは [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) を参照してください。素早い実験（約 5 分の事前学習実行）に私が好む規模は、12 レイヤーのモデル（GPT-1 サイズ）を学習することです。たとえば次のようにします：

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

これは wandb（run 名 "d12"）を使用し、CORE 指標を最終ステップでのみ実行し、中間チェックポイントのサンプリングや保存を行いません。私はコードの何かを変更し、d12（あるいは d16 など）を再実行して、それが役立ったかどうかを反復ループで確認するのが好きです。実行が役立つかどうかを確認するために、私は次の wandb プロットを監視するのが好きです：

1. `val_bpb`（語彙サイズに依存しない bits per byte 単位での検証損失）を `step`、`total_training_time`、`total_training_flops` の関数として。
2. `core_metric`（DCLM CORE スコア）
3. VRAM 使用率、`train/mfu`（Model FLOPS utilization）、`train/tok_per_sec`（学習スループット）

例は[こちら](https://github.com/karpathy/nanochat/pull/498#issuecomment-3850720044)を参照してください。

重要なのは、nanochat はたった一つの複雑さのダイヤル、すなわちトランスフォーマーの深さ（depth）を中心に記述・構成されているという点です。この単一の整数が、その他すべてのハイパーパラメータ（トランスフォーマーの幅、ヘッド数、学習率の調整、学習ホライズン、ウェイトディケイなど）を自動的に決定し、学習されたモデルが計算最適になるようにします。その狙いは、ユーザーがこれらを考えたり設定したりする必要をなくすことです。ユーザーは単に `--depth` を使ってより小さい、あるいはより大きいモデルを要求するだけで、すべてが「ちゃんと動く」のです。深さをスイープすることで、さまざまなサイズの計算最適なモデルからなる nanochat ミニシリーズが得られます。GPT-2 相当の能力のモデル（現在最も関心が高い）は、現在のコードではおおよそ d24〜d26 の範囲のどこかにあります。ただし、リポジトリへの変更候補は、すべての深さの設定で機能するほど原理的（principled）でなければなりません。

## CPU / MPS での実行

スクリプト [runs/runcpu.sh](runs/runcpu.sh) は、CPU または Apple Silicon での実行のごく簡単な例を示します。学習される LLM を大幅に縮小して、数十分の学習という妥当な時間内に収まるようにしています。この方法では強力な結果は得られません。

## 精度 / dtype

nanochat は `torch.amp.autocast` を使用しません。代わりに、精度は単一のグローバルな `COMPUTE_DTYPE`（`nanochat/common.py` で定義）を通じて明示的に管理されます。デフォルトでは、ハードウェアに基づいて自動検出されます：

| ハードウェア | デフォルトの dtype | 理由 |
|----------|--------------|-----|
| CUDA SM 80+（A100、H100 など） | `bfloat16` | ネイティブの bf16 テンソルコア |
| CUDA SM < 80（V100、T4 など） | `float32` | bf16 なし。fp16 は `NANOCHAT_DTYPE=float16` で利用可能（GradScaler を使用） |
| CPU / MPS | `float32` | 低精度テンソルコアなし |

`NANOCHAT_DTYPE` 環境変数でデフォルトを上書きできます：

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # fp32 を強制
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # bf16 を強制
```

仕組み：モデルの重みは（オプティマイザの精度のために）fp32 で保存されますが、カスタムの `Linear` レイヤーがフォワードパス中にそれらを `COMPUTE_DTYPE` にキャストします。埋め込み（Embeddings）はメモリ節約のために直接 `COMPUTE_DTYPE` で保存されます。これにより、autocast と同じ混合精度の利点が得られますが、どの処理をどの精度で実行するかを完全に明示的に制御できます。

注：`float16` 学習では、勾配のアンダーフローを防ぐために `base_train.py` で `GradScaler` が自動的に有効になります。SFT もこれをサポートしますが、RL は現在サポートしていません。fp16 での推論はどこでも問題なく動作します。

## ガイド

役立つ情報を含むかもしれないガイドをいくつか公開しています（新しいものから古いものの順）：

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) は最初の nanochat モデルのミニシリーズを解説しています。
- nanochat に新しい能力を追加するには、[Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164) を参照してください。
- nanochat をカスタマイズするには、Discussions の [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) を参照してください。合成データ生成によって nanochat のパーソナリティをチューニングし、そのデータを SFT 段階に混ぜ込む方法を説明しています。
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) は nanochat を紹介していますが、現在では一部に古い情報が含まれており、モデルも現在の master よりずっと古い（結果も劣る）ものです。

## ファイル構成

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # アイデンティティ用の合成データの例
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # 事前学習データのシャード生成
├── nanochat
│   ├── __init__.py                 # 空
│   ├── checkpoint_manager.py       # モデルチェックポイントの保存/読み込み
│   ├── common.py                   # その他の小さなユーティリティ、QOL 改善
│   ├── core_eval.py                # ベースモデルの CORE スコアを評価（DCLM 論文）
│   ├── dataloader.py               # トークン化する分散データローダ
│   ├── dataset.py                  # 事前学習データのダウンロード/読み込みユーティリティ
│   ├── engine.py                   # KV Cache を用いた効率的なモデル推論
│   ├── execution.py                # LLM がツールとして Python コードを実行できるようにする
│   ├── gpt.py                      # GPT nn.Module トランスフォーマー
│   ├── logo.svg
│   ├── loss_eval.py                # （損失の代わりに）bits per byte を評価
│   ├── optim.py                    # AdamW + Muon オプティマイザ、1GPU および分散
│   ├── report.py                   # nanochat レポート作成用ユーティリティ
│   ├── tokenizer.py                # GPT-4 スタイルの BPE トークナイザラッパー
│   └── ui.html                     # nanochat フロントエンドの HTML/CSS/JS
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # ミニシリーズ学習スクリプト
│   ├── runcpu.sh                   # CPU/MPS での実行の小さな例
│   ├── scaling_laws.sh             # スケーリング則の実験
│   └── speedrun.sh                 # 約 100 ドルの nanochat d20 を学習
├── scripts
│   ├── base_eval.py                # ベースモデル：CORE スコア、bits per byte、サンプル
│   ├── base_train.py               # ベースモデル：学習
│   ├── chat_cli.py                 # チャットモデル：CLI で会話
│   ├── chat_eval.py                # チャットモデル：評価タスク
│   ├── chat_rl.py                  # チャットモデル：強化学習
│   ├── chat_sft.py                 # チャットモデル：SFT 学習
│   ├── chat_web.py                 # チャットモデル：Web UI で会話
│   ├── tok_eval.py                 # トークナイザ：圧縮率を評価
│   └── tok_train.py                # トークナイザ：学習
├── tasks
│   ├── arc.py                      # 選択式の科学問題
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 任意の jsonl 会話から Task を作成
│   ├── gsm8k.py                    # 8K の小学校算数問題
│   ├── humaneval.py                # 名前は誤解を招くが、シンプルな Python コーディングタスク
│   ├── mmlu.py                     # 幅広いトピックの選択式問題
│   ├── smoltalk.py                 # HF の SmolTalk を集約したデータセット
│   └── spellingbee.py              # モデルにスペル/文字数えを教えるタスク
├── tests
│   └── test_engine.py
└── uv.lock
```

## コントリビューション

nanochat の目標は、1000 ドル未満の予算でエンドツーエンドに扱えるマイクロモデルの最先端を改善することです。アクセシビリティとは、全体のコストだけでなく、認知的な複雑さに関するものでもあります。nanochat は徹底的に設定可能な LLM「フレームワーク」ではありません。コードベースには巨大な設定オブジェクトも、モデルファクトリも、if-then-else の怪物もありません。それは、最初から最後まで実行して会話できる ChatGPT モデルを生み出すように設計された、単一かつ一貫した、最小限で読みやすく、ハック可能で、最大限フォークしやすい「強力なベースライン」のコードベースです。現在、私が個人的に最も興味を持っているのは、GPT-2 までのレイテンシ（つまり CORE スコアを 0.256525 以上にすること）を高速化することです。現在これには約 3 時間かかりますが、事前学習段階を改善することでさらに短縮できます。

現在の AI ポリシー：開示（disclosure）。PR を提出する際は、LLM が実質的に貢献した部分、および自分で書いていない、あるいは完全には理解していない部分を申告してください。

## 謝辞

- 名前（nanochat）は、事前学習のみを扱っていた私の以前のプロジェクト [nanoGPT](https://github.com/karpathy/nanoGPT) に由来します。
- nanochat は、明確な指標とリーダーボードで nanoGPT リポジトリをゲーム化した [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) にも触発されており、そのアイデアの多くと事前学習の実装の一部を借用しています。
- fineweb と smoltalk を提供してくれた [HuggingFace](https://huggingface.co/) に感謝します。
- このプロジェクトの開発に使用した計算リソースを提供してくれた [Lambda](https://lambda.ai/service/gpu-cloud) に感謝します。
- LLM の達人 🧙‍♂️ Alec Radford 氏の助言とガイダンスに感謝します。
- nanochat の issue、プルリクエスト、ディスカッションの管理を手伝ってくれたリポジトリ czar の Sofie 氏 [@svlandeg](https://github.com/svlandeg) に感謝します。

## 引用

nanochat が研究に役立った場合は、シンプルに次のように引用してください：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## ライセンス

MIT
