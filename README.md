# AI課題（Python データサイエンス演習）

モバイルマルチメディア通信研究室向けの Python / 機械学習演習教材です。  
第1回〜第17回までの Jupyter Notebook 演習問題と，演習で使用する CSV データセット，参考テキストをまとめています。

## このリポジトリに含まれるもの

| 種類 | 内容 |
|------|------|
| 演習ノートブック | `q1.ipynb` 〜 `q17.ipynb` |
| 参考資料（基礎） | [`data/Pythonの基礎.ipynb`](data/Pythonの基礎.ipynb) |
| 参考資料（発展） | [`data/Pythonの発展.ipynb`](data/Pythonの発展.ipynb), [`data/Pythonの発展.md`](data/Pythonの発展.md) |
| データセット | [`data/`](data/) 配下の CSV |

各演習ノートブックは，空欄（`# ここにあなたのコードを書いてください`）を生徒自身が埋めて完成させる形式です。

## 演習の構成

### 第1部：Python 基礎（q1–q7）

| 回 | ファイル | 主な内容 |
|----|----------|----------|
| 第1回 | q1.ipynb | 変数，辞書，関数，クラス |
| 第2回 | q2.ipynb | NumPy，Pandas，Matplotlib，COVID 時系列 |
| 第3回 | q3.ipynb | 記述統計，単回帰分析 |
| 第4回 | q4.ipynb | NumPy/Pandas 応用，時系列解析 |
| 第5回 | q5.ipynb | 機械学習入門（タイタニック生存予測） |
| 第6回 | q6.ipynb | 交差検証，グリッドサーチ，評価指標 |
| 第7回 | q7.ipynb | 線形回帰の自作実装，総復習 |

### 第2部：Python 発展（q8–q17）

| 回 | ファイル | 主な内容 |
|----|----------|----------|
| 第8回 | q8.ipynb | 前処理，Pipeline |
| 第9回 | q9.ipynb | 正則化（Ridge / Lasso / LogisticRegression） |
| 第10回 | q10.ipynb | アンサンブル学習（RF / GB / XGBoost） |
| 第11回 | q11.ipynb | 教師なし学習（k-means，PCA） |
| 第12回 | q12.ipynb | 不均衡データ，ROC / PR 曲線 |
| 第13回 | q13.ipynb | 学習曲線，validation curve |
| 第14回 | q14.ipynb | ニューラルネットワーク（MLPClassifier） |
| 第15回 | q15.ipynb | PyTorch 基礎（MNIST） |
| 第16回 | q16.ipynb | CNN（Fashion-MNIST） |
| 第17回 | q17.ipynb | 総合課題（EDA + Pipeline + モデル比較） |

## データセット

演習で使用する CSV は [`data/`](data/) に配置しています。  
ノートブックからは GitHub の raw URL 経由で読み込みます。

```
data/
├── covid/newly_confirmed_cases_daily.csv   # q2, q4
├── student/student-mat.csv                 # q3, q4, q7, q9
├── student/student-por.csv                 # q3
├── titanic/titanic.csv                     # q5, q8, q10, q17
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

q15 / q16 の MNIST・Fashion-MNIST はファイルとしては含めず，実行時に `torchvision` が自動ダウンロードします。

## 環境構築

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install ipykernel jupyter pandas matplotlib numpy
python -m ipykernel install --user --name ai-kadai --display-name "Python 3 (AI課題)"
```

## 参考資料の使い方

- **Python の基礎** … q1–q7 を解く前に読む教材（文法，NumPy/Pandas，統計，Scikit-learn 入門）
- **Python の発展** … q8–q17 を解く前に読む教材（Pipeline，正則化，アンサンブル，深層学習入門）

## Copyright

Copyright (c) モバイルマルチメディア通信研究室

All rights reserved.
