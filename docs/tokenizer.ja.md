# nanochat トークナイザーの処理フロー解説

対象実装: `nanochat/tokenizer.py`、学習スクリプト `scripts/tok_train.py`、評価スクリプト `scripts/tok_eval.py`

この資料は、nanochat のトークナイザー（テキスト ⇄ トークンID の相互変換器）が「どういう処理を、どういう流れで」行っているかをまとめたものです。

---

## 1. 全体像

nanochat のトークナイザーは **GPT-4 スタイルの BPE（Byte Pair Encoding）** です。実装には 2 系統あります。

| 実装 | 学習 | 推論 | 用途 |
|---|---|---|---|
| `RustBPETokenizer` | rustbpe（自作 Rust ライブラリ） | tiktoken | **本番（デフォルト）**。学習は rustbpe、推論は高速な tiktoken。 |
| `HuggingFaceTokenizer` | HuggingFace `tokenizers` | HuggingFace `tokenizers` | 互換・比較用。学習/推論とも可能だが「really confusing」とコメントされている。 |

`get_tokenizer()`（`tokenizer.py:390`）はデフォルトで `RustBPETokenizer.from_directory(...)` を返すため、以降は主に **RustBPETokenizer** の流れを説明します。

両実装に共通する 2 つの設計上のキモ:

1. **特殊トークン `SPECIAL_TOKENS`**（`tokenizer.py:13`）
   ```
   <|bos|>                         # 文書の区切り（Beginning of Sequence）
   <|user_start|> <|user_end|>     # ユーザ発話
   <|assistant_start|> <|assistant_end|>   # アシスタント発話
   <|python_start|> <|python_end|>         # アシスタントによる Python ツール呼び出し
   <|output_start|> <|output_end|>         # Python 実行結果
   ```
   `<|bos|>` 以外はファインチューニング時に「会話」をトークン列へ変換するために使われます。

2. **分割用正規表現 `SPLIT_PATTERN`**（`tokenizer.py:30`）
   GPT-4 が BPE の前にテキストを大まかなチャンクへ分割するための正規表現。nanochat では数字部分を `\p{N}{1,3}` から **`\p{N}{1,2}`** に変更しています（小さい語彙サイズで数字にトークンを使いすぎないため。vocab=32K では 2 が最適と検証済み）。

---

## 2. 学習（training）の流れ

エントリポイント: `scripts/tok_train.py`

### 2-1. データの供給（text_iterator）
`scripts/tok_train.py:28-44`

1. `parquets_iter_batched(split="train")`（`nanochat/dataset.py`）で事前学習用 parquet シャードからドキュメントをバッチ単位で取得。
2. バッチを 1 件ずつにフラット化。
3. 各ドキュメントを `--doc-cap`（デフォルト 10,000 文字）で切り詰め。
4. 累計文字数が `--max-chars`（デフォルト 2B 文字）を超えたら停止。

```
parquet shards ──► batch ──► doc（doc_cap で切詰）──► yield ──► (max_chars で打ち切り)
```

### 2-2. BPE 学習（RustBPETokenizer.train_from_iterator）
`tokenizer.py:170-190`

1. `rustbpe.Tokenizer()` を生成。
2. 語彙サイズは `vocab_size - len(SPECIAL_TOKENS)` で学習（特殊トークンは学習せず、後で ID を付与する）。`vocab_size_no_special >= 256`（生バイトぶん）を要求。
3. `tokenizer.train_from_iterator(text_iter, vocab_size_no_special, pattern=SPLIT_PATTERN)` で実際の BPE マージ規則を学習。
4. 学習後、推論用に **tiktoken の `Encoding`** を構築:
   - rustbpe から `pattern`（分割正規表現）と `mergeable_ranks`（`bytes -> マージ優先度ランク`）を取得。
   - 特殊トークンを語彙の末尾（`tokens_offset = len(mergeable_ranks)` 以降）に連番で割り当て。
   - `tiktoken.Encoding(name="rustbpe", pat_str=..., mergeable_ranks=..., special_tokens=...)` を生成。
5. `RustBPETokenizer(enc, "<|bos|>")` を返す（`__init__` で BOS の ID をキャッシュ）。

```
rustbpe で学習 ──► mergeable_ranks / pattern を取り出す
                       │
                       ▼
       特殊トークンを末尾IDに割当 ──► tiktoken.Encoding を構築（＝推論器）
```

### 2-3. 保存とメタデータ生成
`scripts/tok_train.py:54-106`

- **保存**: `tokenizer.save(dir)` が tiktoken の `enc` を `tokenizer.pkl` に pickle 保存（`tokenizer.py:258`）。
- **サニティチェック**: 多言語・記号・絵文字を含むテスト文字列で `decode(encode(x)) == x` を確認（ラウンドトリップ）。
- **`token_bytes.pt` の生成**（`tok_train.py:72-91`）: 各トークンIDが「何バイトに相当するか」を表す配列を作成。特殊トークンは 0。これは **bits per byte (bpb)** 評価（語彙サイズに依存しない損失指標）のために `nanochat/loss_eval.py` で使われる。
- **レポート**: 学習時間・特殊トークン数・トークンバイト統計を `report.py` に記録。

---

## 3. 推論（encode / decode）の流れ

### 3-1. エンコード（テキスト → トークンID）
`RustBPETokenizer.encode`（`tokenizer.py:225-250`）

- 入力は **単一文字列**または**文字列のリスト**の両方に対応。
- 文字列: `enc.encode_ordinary(text)`（特殊トークンを通常テキストとして扱わない通常エンコード）。
- リスト: `enc.encode_ordinary_batch(text, num_threads=8)` で**マルチスレッド並列**エンコード。
- オプションで `prepend` / `append` に「特殊トークン名」または「トークンID」を指定でき、先頭/末尾に挿入（例: 文書頭への `<|bos|>` 付与）。

```
text ──► (SPLIT_PATTERN で分割) ──► 各チャンクを BPE マージ ──► token ids
                                              ▲
                                  mergeable_ranks（マージ優先度）
```

### 3-2. デコード（トークンID → テキスト）
`tokenizer.py:255` … `enc.decode(ids)`（tiktoken）。

### 3-3. その他のユーティリティ
- `encode_special(name)`（`tokenizer.py:218`）: 特殊トークン名 → ID。`lru_cache` 付き。
- `get_bos_token_id()`: BOS の ID。
- `get_vocab_size()` / `get_special_tokens()` / `id_to_token(id)`。

---

## 4. 会話のレンダリング（SFT / RL 用）

チャットモデルの学習では、構造化された「会話」を 1 本のトークン列に変換する必要があります。これを担うのが `render_conversation` です。

### 4-1. render_conversation
`tokenizer.py:266-350`

戻り値は **`ids`（トークン列）** と **`mask`（同じ長さ。アシスタントが学習対象とすべきトークンを 1、それ以外を 0）** のペア。

処理の流れ:

1. **system メッセージの吸収**: 先頭が system の場合、次の user メッセージへ `system + "\n\n" + user` として結合（元データは破壊しないよう deepcopy）。
2. 必要な特殊トークンID（bos / user_* / assistant_* / python_* / output_*）を取得。
3. 先頭に `<|bos|>` を追加（mask=0）。
4. メッセージを順に処理。`i` が偶数なら user、奇数なら assistant という**交互制約**を assert で検査。
   - **user**: `<|user_start|>` + 本文 + `<|user_end|>` を **すべて mask=0**（学習対象外）で追加。
   - **assistant**: `<|assistant_start|>` の後に content を追加し、最後に `<|assistant_end|>`（mask=1）。
     - content が文字列: 本文を **mask=1**（学習対象）で追加。
     - content がリスト（ツール呼び出しを含む）: パートごとに分岐。
       - `text`: mask=1
       - `python`（ツール呼び出し）: `<|python_start|>` 本文 `<|python_end|>` を mask=1（モデルが生成すべき）
       - `python_output`（実行結果）: `<|output_start|>` 本文 `<|output_end|>` を **mask=0**（実行時に Python から来るのでモデルは学習しない）
5. 最後に `max_tokens`（デフォルト 2048）で **切り詰め**（OOM 防止）。

```
[bos](0) [u_s](0) user本文(0) [u_e](0) [a_s](0) assistant本文(1) [a_e](1) ...
                                                  ▲ 学習対象は mask=1 のみ
```

ポイント: **損失はアシスタント出力（mask=1）にのみ掛かる**。ユーザ発話やツール出力は文脈として与えるが学習はしない。

### 4-2. render_for_completion（RL 用）
`tokenizer.py:367-385`

強化学習では「アシスタントに続きを生成させる」状態でレンダリングしたい。よって:
1. 末尾のアシスタントメッセージを取り除き、
2. `render_conversation` で途中までをトークン化し（mask は不要）、
3. 末尾に `<|assistant_start|>` を付けて**生成を促す**形にする。

### 4-3. visualize_tokenization（デバッグ用）
`tokenizer.py:352-365`。mask=1 を緑、mask=0 を赤で色付けして、どこが学習対象かを目視確認できる。

---

## 5. 評価（compression ratio）

`scripts/tok_eval.py` は、各種テキスト（ニュース・コード・数式・多言語など）でトークナイザーの**圧縮率**（文字数/トークン数など）を測定し、語彙の効率を確認する。

---

## 6. まとめ（データの流れ）

```
[学習時]
  parquet（事前学習データ）
      │ text_iterator(doc_cap, max_chars)
      ▼
  rustbpe で BPE 学習（vocab_size - 特殊トークン数）
      │ mergeable_ranks + pattern
      ▼
  tiktoken.Encoding を構築（＋特殊トークンを末尾IDに割当）
      │ save → tokenizer.pkl
      ▼
  token_bytes.pt も生成（bpb 評価用）

[推論/学習データ生成時]
  text ──encode──► token ids ──(モデル)──► token ids ──decode──► text
  会話 ──render_conversation──► (ids, mask)   # SFT は mask=1 のみ学習
  会話 ──render_for_completion──► ids（末尾に <|assistant_start|>）  # RL
```

### 主要シンボル早見表

| シンボル | 場所 | 役割 |
|---|---|---|
| `SPECIAL_TOKENS` | `tokenizer.py:13` | 特殊トークン定義 |
| `SPLIT_PATTERN` | `tokenizer.py:30` | BPE 前分割の正規表現（GPT-4 派生、数字は 1–2 桁） |
| `RustBPETokenizer.train_from_iterator` | `tokenizer.py:170` | rustbpe 学習 → tiktoken 構築 |
| `RustBPETokenizer.encode` / `decode` | `tokenizer.py:225` / `255` | エンコード/デコード（tiktoken） |
| `render_conversation` | `tokenizer.py:266` | 会話 → (ids, mask)（SFT用） |
| `render_for_completion` | `tokenizer.py:367` | 会話 → ids（RL用、補完プライミング） |
| `get_tokenizer` / `get_token_bytes` | `tokenizer.py:390` / `397` | 保存済みトークナイザー/バイト表のロード |
| `text_iterator` | `tok_train.py:28` | 学習データの供給イテレータ |
| `token_bytes.pt` 生成 | `tok_train.py:72` | bpb 評価用メタデータ |
