# AI課題（Python データサイエンス演習）

モバイルマルチメディア通信研究室向けの Python / 機械学習演習教材です。  
第1回〜第17回までの Jupyter Notebook 演習問題と，演習で使用する CSV データセット，参考テキストをまとめています。

## 配布バージョンについて

| 対象 | 備考 |
|------|------|
| **q1–q7** | 2025年度配布版。リポジトリ上は来年配布用に誤字・矛盾を修正済み |
| **q8–q17** | **理解重視フォーマット**に再設計（ボイラープレートは提供し、核心の最低限コーディング＋設計判断・実験・説明で理解を測る） |

## このリポジトリに含まれるもの

| 種類 | 内容 |
|------|------|
| 演習ノートブック | `q1.ipynb` 〜 `q17.ipynb` |
| 参考資料（基礎） | [`data/Pythonの基礎.ipynb`](data/Pythonの基礎.ipynb) |
| 参考資料（発展） | [`data/Pythonの発展.ipynb`](data/Pythonの発展.ipynb), [`data/Pythonの発展.md`](data/Pythonの発展.md) |
| データセット | [`data/`](data/) 配下の CSV |

演習ノートブックの形式は2部で異なります。

- **q1–q7**: 空欄（`# ここにあなたのコードを書いてください`）を生徒自身が埋めて完成させる形式。
- **q8–q17**: **理解重視フォーマット**。これからの時代は「コードを書くのは AI、ハイパーパラメータや特徴量の設計を人間が担う」という考えに基づき、コーディングそのものではなく **AI（機械学習）の中身の理解** を測ります。

### 理解重視フォーマット（q8–q17）

コードは **ボイラープレート（読み込み・前処理の枠組み・描画など）を完成形で提供** します（AI に書かせてもよい）。ただし q1–q7 のように全部を書かせるのではなく、**各問の「核心となる最低限の数行」だけを `# ★あなたが書く★` として空欄**にしてあり、ここは生徒が自分で書きます。これにより「要となる処理は理解して書ける」ことも確認します。生徒は次の4つの仕組みで「理解していること」を提出物で示します。各問の見出しにタグが付きます。

| タグ | 意味 | 生徒がすること |
|------|------|----------------|
| **【骨格】** | 動く骨格は与えられている | 設計上の決定点（数値・選択肢・特徴量）を変更し、**核心の最低限コーディング**（`# ★あなたが書く★`）を埋める |
| **【選択】** | 適切な手法を選ぶ問題 | 「ここで適切な処理をしないと精度が上がらない」場面で **5〜6個の候補（A〜F）** から選び、**理由**を解答欄に書く（不適切な候補も混在） |
| **【実験】** | 試行錯誤の記録 | ハイパーパラメータや特徴量を **2軸以上で最低5通り** 変え、結果を **実験ログ表** に記録し、**考察**する（q8・q12 は `settings` ループ、q14–q16 は `EXPERIMENTS` リストが標準） |
| **【説明】** | 理解の証跡 | 与えられたコードの各行に `# 説明:` で意味を自分の言葉で書く（このタイプはコードは完成形のまま） |

> **q1–q7 との違い**: q1–q7 は空欄をすべて埋めて完成させる形式ですが、q8–q17 は「ボイラープレートは提供・核心の数行だけ書く」点が異なります。空欄は `# ★あなたが書く★` とヒント（`ヒント:`）、未記入箇所は `___` で示します。

各問の最後に **✍️ 解答用コードセル**（`# === ✍️ 問題N 解答（採点対象）===`）があり、選んだ選択肢・理由・実験結果・考察を **項目ごと（例 `(2-a)` `(2-b)`…）** に記入します。これが主な採点対象です。見本は [`q8.ipynb`](q8.ipynb)（機械学習系）と [`q16.ipynb`](q16.ipynb)（深層学習系）です。

#### 解答用コードセルの命名（q8–q17 共通）

| 用途 | 変数名の例 | 備考 |
|------|------------|------|
| 選択肢（A〜F） | `design1_choice`, `design2_1_choice`, `answer_4_a` | 問題文の **(A)〜(F)** と解答セルのコメントを必ず照合 |
| 理由・考察（短文） | `design1_reason`, `answer_3_b`, `reflection2` | 複数行は `""""""` |
| 実験ログ | `experiment_log`, `experiment_log_axis1` | 実行セルが出す `experiment_log_run` を転記してもよい |
| 観察表 | `observation_1_a` | `pd.DataFrame` または文字列 |
| 説明問題のコード内 | `# 説明:` コメント | 別セルの `answer_N_a` と併用する回あり |

採点時は **markdown の選択肢定義を正**とし、解答セル先頭の `# (A) ...` コメントが一致していることを確認してください。

## 演習の構成

### 第1部：Python 基礎（q1–q7）

| 回 | ファイル | 主な内容 | 参考テキスト |
|----|----------|----------|--------------|
| 第1回 | q1.ipynb | 変数，辞書，関数，クラス | 第1章 |
| 第2回 | q2.ipynb | NumPy，Pandas，Matplotlib，COVID 時系列 | 第2章 |
| 第3回 | q3.ipynb | 記述統計，単回帰分析 | 第3章 |
| 第4回 | q4.ipynb | NumPy/Pandas 応用，時系列解析 | 第4章 |
| 第5回 | q5.ipynb | 機械学習入門（タイタニック生存予測） | 第5章 |
| 第6回 | q6.ipynb | 交差検証，グリッドサーチ，評価指標 | 第6章 |
| 第7回 | q7.ipynb | 線形回帰の自作実装，総復習 | 第3章（統計） |

### 第2部：Python 発展（q8–q17）

| 回 | ファイル | 主な内容 | 参考テキスト |
|----|----------|----------|--------------|
| 第8回 | q8.ipynb | 前処理，Pipeline | 第1章 |
| 第9回 | q9.ipynb | 正則化（Ridge / Lasso / LogisticRegression） | 第2章 |
| 第10回 | q10.ipynb | アンサンブル学習（RF / GB / XGBoost） | 第3章 |
| 第11回 | q11.ipynb | 教師なし学習（k-means，PCA） | 第4章 |
| 第12回 | q12.ipynb | 不均衡データ，ROC / PR 曲線 | 第5章 |
| 第13回 | q13.ipynb | 学習曲線，validation curve | 第6章 |
| 第14回 | q14.ipynb | PyTorch 入門（Tensor・DataLoader・MNIST 可視化） | 第7章 |
| 第15回 | q15.ipynb | 全結合1層 NN 実装・学習・損失曲線 | 第8章 |
| 第16回 | q16.ipynb | ANN・CNN・3モデル比較・混同行列 | 第9章 |
| 第17回 | q17.ipynb | Colab Canvas UI + 手書き数字推論・考察 | 第9章 |

## データセット

演習で使用する CSV は [`data/`](data/) に配置しています。  
ノートブックからは GitHub の raw URL 経由で読み込みます。

```
data/
├── covid/newly_confirmed_cases_daily.csv   # q2, q4
├── student/student-mat.csv                 # q3, q4, q7, q9
├── student/student-por.csv                 # q3
├── titanic/titanic.csv                     # q5, q8, q10
├── titanic/train.csv, test.csv             # Kaggle 形式
└── sklearn/                                # 参考用 CSV（iris, wine など）
    ├── breast_cancer.csv
    ├── digits.csv
    ├── iris.csv
    └── wine.csv
```

データ読み込みのベース URL:

```
https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data
```

使用例:

```python
import pandas as pd

df_cvd = pd.read_csv(
    "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/covid/newly_confirmed_cases_daily.csv",
    index_col=None,
)
```

q14〜q17 の MNIST はファイルとしては含めず，実行時に `torchvision` が `./data` 配下へ自動ダウンロードします。

```python
DATA_ROOT = "./data"  # q14, q15, q16, q17 で共通
```

q17 の手書き推論 UI は **Google Colab** 上での実行を推奨します。Colab では q14〜q16 を順に実行後，q17 を開いてください。ローカル Jupyter では q17 の問題1（モデル読み込み）と問題3（考察）のみ実施可能です。

## 環境構築

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name ai-kadai --display-name "Python 3 (AI課題)"
```

**macOS で XGBoost を使う場合（q10）**: OpenMP ランタイムが必要です。

```bash
brew install libomp
```

## 参考資料の使い方

- **Python の基礎** … q1–q7 を解く前に読む教材（文法，NumPy/Pandas，統計，Scikit-learn 入門）
- **Python の発展** … q8–q17 を解く前に読む教材（Pipeline，正則化，アンサンブル，深層学習入門）

## Copyright

Copyright (c) モバイルマルチメディア通信研究室

All rights reserved.
