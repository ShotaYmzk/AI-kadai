# Pythonの発展
***
## 目次
1. 前処理と Pipeline
2. 正則化
3. アンサンブル学習
4. 教師なし学習
5. 不均衡データと ROC
6. 学習曲線とハイパーパラメータ
7. PyTorch 入門
8. ニューラルネットワーク実装
9. ANN・CNN・推論

※ 前提: [Pythonの基礎.ipynb](Pythonの基礎.ipynb) 第5–6章
※ 演習: [q8.ipynb](../q8.ipynb) 〜 [q17.ipynb](../q17.ipynb)

本資料は，[Pythonの基礎.ipynb](Pythonの基礎.ipynb) で学んだ Scikit-learn の基礎を土台に，**前処理の自動化**，**正則化**，**アンサンブル学習**，**教師なし学習**，**深層学習** など，第8回以降の演習（q8–q17）で必要となる知識を解説する．

各章では概念の説明と，実行可能な短いコード例を示す．演習の答えそのものは書かないので，例を参考にしながら q8.ipynb 〜 q17.ipynb を自分で完成させてほしい．

# 第1章　前処理と Pipeline
***
※ 対応演習: [q8.ipynb](../q8.ipynb)
## 目次
1. データの読み込みと確認
2. 欠損値の補完
3. ColumnTransformer
4. Pipeline の構築

## 1. データの読み込みと確認
***
機械学習では，**モデルを作る前に必ずデータの状態を確認する**のが大原則である．

なぜか。型のミスマッチ（文字列が混入した数値列）や欠損値（NaN）を知らないままモデルに渡すと，`ValueError: could not convert string to float` などのエラーで止まる。しかも，エラーが出ずに「おかしな精度」のまま学習が完了してしまうケースもあり，後から原因を追うのが大変になる。

C言語で言えば，配列にアクセスする前にサイズと中身を確認するのと同じ感覚である。

確認するべき主な項目：

- `head()` … 先頭数行で値のサンプルを目視
- `info()` … 列の型（dtype）と **non-null count**（= 欠損がない行数）
- `isnull().sum()` … 列ごとの欠損数。0 以外が出たら補完が必要

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TITANIC_URL = "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/titanic/titanic.csv"

df = pd.read_csv(TITANIC_URL)
cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
df_titanic = df[cols].copy()

print(df_titanic.head())       # 先頭5行で値を目視
print()
df_titanic.info()              # 各列の型と non-null count を確認
print()
print("欠損数:")
print(df_titanic.isnull().sum())   # Age に欠損が多いことがわかる
# → 出力例: Age    177  ← 177件が NaN
```

## 2. 欠損値の補完
***
欠損値（NaN）がある列は，そのままでは多くの機械学習アルゴリズムに渡せない．

Scikit-learn の `SimpleImputer` を使うと，**中央値**（`strategy="median"`）や**最頻値**（`strategy="most_frequent"`）などで欠損を自動補完できる．

### fit と transform が分かれている理由

`SimpleImputer` には `fit()` と `transform()` の2段階がある。これは意図的な設計である。

- `fit(X_train)` … 訓練データだけを使って「補完に使う統計量（中央値など）」を計算・記憶
- `transform(X_test)` … 記憶した統計量で別データを変換

**なぜ分けるか**: テストデータで `fit` し直すと，「テストデータの統計量がモデルに漏れる（データ漏洩）」ことになる。C言語で言えば，ヘッダに定数を1か所だけ定義し，複数のソースファイルで共有するのに近い感覚である。統計量は「訓練データから決めた定数」として保存しておく。

`fit_transform(X_train)` は `fit` + `transform` を一発で行うショートカットである。

```python
from sklearn.impute import SimpleImputer

print("補完前 Age の欠損数:", df_titanic["Age"].isnull().sum())
# → 177

imputer = SimpleImputer(strategy="median")  # 中央値で補完するインスタンスを生成

# fit_transform: 訓練データでのみ中央値を計算し，同時に変換する
age_filled = imputer.fit_transform(df_titanic[["Age"]])

print("補完後 Age の欠損数:", pd.isna(age_filled).sum())   # → 0
print("補完に使った中央値:", imputer.statistics_[0])        # → 28.0 付近
```

## 3. ColumnTransformer
***
実際のデータには **数値列**（年齢・運賃など）と **カテゴリ列**（性別・乗客クラスなど）が混在する。それぞれに適した前処理が異なるため，`ColumnTransformer` で「列の種類ごとに異なる変換器を適用」する。

- **数値列** → `StandardScaler`（平均0，分散1に標準化。単位が違う列を同じスケールに揃える）
- **カテゴリ列** → `OneHotEncoder`（"male" / "female" を 1 / 0 の数値列に変換）

### `drop="first"` が必要な理由

`Sex` を OneHot 化すると `Sex_male` と `Sex_female` の2列が生まれる。しかし，`Sex_male = 1 - Sex_female` であり，片方が決まればもう片方は自動的に決まる（**多重共線性**）。線形モデルはこの冗長な列に悩まされるため，`drop="first"` で片方を落とす。

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]   # 数値列
categorical_features = ["Sex"]                                     # カテゴリ列

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),               # 数値列 → 標準化
        ("cat", OneHotEncoder(drop="first"), categorical_features),# カテゴリ列 → ダミー変数（冗長列を除去）
    ]
)

X = df_titanic.drop("Survived", axis=1)   # 説明変数
y = df_titanic["Survived"]                # 目的変数

# Age の欠損を先に補完してから ColumnTransformer に渡す（この章では単純化）
X_temp = X.copy()
X_temp["Age"] = SimpleImputer(strategy="median").fit_transform(X[["Age"]]).ravel()

X_transformed = preprocessor.fit_transform(X_temp)
print("変換後の shape:", X_transformed.shape)
# → (891, 6) ← 数値5列 + Sex_male 1列（OneHot で female が落ちた）
```

## 4. Pipeline の構築
***
前処理とモデル学習を**一つのオブジェクトにまとめる**のが `Pipeline` である。

### Pipeline を使わないとどうなるか

```python
# Pipeline なしの危険なコード例（データ漏洩が起きる）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # OK: 訓練データで fit
X_test_scaled  = scaler.fit_transform(X_test)    # ← NG! テストデータで再 fit → 漏洩
```

テストデータの統計量が前処理に混入し，「未知データに対する性能を正しく測れない」問題が起きる。

### Pipeline の利点

C言語で言えば，`main()` から `preprocess()` → `train()` → `predict()` と順に呼ぶ処理パイプラインを1つの関数ポインタ配列にまとめるようなイメージである。

- `fit` を1回呼ぶだけで全ステップが順番に実行される
- `predict` 時も同じ前処理が自動で適用される（スケーラーの `transform` を書き忘れない）
- 交差検証（`cross_val_score`）と組み合わせると，分割ごとに前処理を独立して行える

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# 欠損補完 → 標準化/ダミー変数化 → 決定木の流れをまとめる
pipe = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),  # 欠損補完
                ("scaler", StandardScaler()),                   # 標準化
            ]), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ],
    )),
    ("model", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

pipe.fit(X_train, y_train)         # 前処理 + 学習を一発で実行
y_pred = pipe.predict(X_test)      # テストデータにも同じ前処理が自動で適用される
print("テスト正解率:", accuracy_score(y_test, y_pred))
# → 0.78 付近
```

## つまずきやすいポイント（第1章）

- **`fit_transform` をテストデータに使う** → `transform` のみを使う。`fit_transform` は訓練データ専用
- **`ColumnTransformer` に渡す列名の順番** → `fit` 時と `transform` 時で同じ列リストを使わないと列がずれる
- **欠損値を `ColumnTransformer` の前に処理する場合と後に処理する場合** → Pipeline 内で `SimpleImputer` を各変換器に含めるのが最もシンプルで安全

---

# 第2章　正則化
***
※ 対応演習: [q9.ipynb](../q9.ipynb)
## 目次
1. 過学習と正則化
2. Ridge / Lasso 回帰
3. ロジスティック回帰の正則化
4. 正則化強度 C の比較

## 1. 過学習と正則化
***
**過学習**（オーバーフィッティング）とは，モデルが訓練データに過度に適合し，未知データへの汎化性能が下がる状態である。

C言語でソートアルゴリズムを書くとき「テストケースの入力をそのまま答えとして返す」コードは訓練スコア100%だが，新しい入力には無意味なのと同じである。

```
精度
  │  訓練 ──────────── 上がり続ける
  │   ╱
  │  ╱  テスト ────╮
  │ ╱              ╲___  ← ここから過学習
  └────────────────────  モデルの複雑さ
```

### 正則化とは

損失関数に「重みの大きさへのペナルティ」を足すことで，過度に複雑な係数を抑制する手法である。

| 手法 | ペナルティ | 効果 |
|:--:|:--:|:--|
| **Ridge** | L2: Σwᵢ² | 係数を小さく均す。全係数が残る |
| **Lasso** | L1: Σ\|wᵢ\| | 不要な係数を 0 にしやすい（スパース解） |
| **LogisticRegression(C)** | L2（既定） | C が小さいほど正則化が強い |

Ridge は「全ての水道を細める」，Lasso は「不要な水道を閉める」イメージである。

## 2. Ridge / Lasso 回帰
***
`alpha` は正則化の強さを制御するパラメータである。`alpha=0` はペナルティなし（通常の線形回帰と等価），`alpha` を大きくするほど係数が小さく抑えられる。

```python
import io
import zipfile
import requests

from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

STUDENT_ZIP_URL = "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/student/student-mat.csv"

response = requests.get(STUDENT_ZIP_URL, stream=True, timeout=60)
response.raise_for_status()
with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
    with archive.open("student-mat.csv") as handle:
        df_student = pd.read_csv(handle, sep=";")

X = df_student[["G1"]]   # 1学期の成績を説明変数
y = df_student["G3"]      # 3学期（最終）の成績を目的変数

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 正則化の効果を正確に比較するために標準化必須（スケールが違うと係数の「大きさ」の意味が変わる）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # ← テストデータは transform のみ

models = {
    "LinearRegression": LinearRegression(),     # alpha=0 と等価
    "Ridge(alpha=1.0)": Ridge(alpha=1.0),       # ← alpha が大きいほど正則化が強い
    "Lasso(alpha=0.1)": Lasso(alpha=0.1),       # ← L1: 一部の係数が 0 になりやすい
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    print(f"{name}: MSE={mean_squared_error(y_test, pred):.3f}, R2={r2_score(y_test, pred):.3f}")
```

## 3. ロジスティック回帰の正則化
***
分類問題では `LogisticRegression` の `C` パラメータが正則化の強さを制御する。

### C の直感（注意：逆向き）

**C は正則化の「強さ」ではなく「逆数」**である。`Ridge` の `alpha` と向きが逆なので混同しやすい。

```
C が大きい (例: C=100) → 正則化が弱い  → 複雑なモデル → 過学習しやすい
C が小さい (例: C=0.01) → 正則化が強い → シンプルモデル → 未学習になりやすい

C = 1 / alpha  （alpha は Ridge と同じ意味のペナルティ強度）
```

目的変数を2値化する例: `G3 >= 10` を「合格（1）」，それ以外を「不合格（0）」とする．

```python
from sklearn.linear_model import LogisticRegression

y_bin = (df_student["G3"] >= 10).astype(int)   # ← 2値化
X_cls = df_student[["G1", "G2", "studytime", "failures"]]

X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_bin, test_size=0.2, random_state=0, stratify=y_bin)

X_tr_s = StandardScaler().fit_transform(X_tr)
X_te_s = StandardScaler().fit_transform(X_te)  # ← 本来は fit は X_tr のみで行うべき（簡略例）

logreg = LogisticRegression(C=1.0, max_iter=1000)  # ← C=1.0 はデフォルト（中程度の正則化）
logreg.fit(X_tr_s, y_tr)
print("テスト正解率:", logreg.score(X_te_s, y_te))
```

## 4. 正則化強度 C の比較
***
`C` を変化させてテスト正解率がどう変わるか確認する。横軸は対数スケール（`plt.xscale("log")`）にすると，幅広い範囲を見やすい。

極端に小さい C では「何でも不合格（シンプルすぎる予測）」になり underfitting になる。極端に大きい C では訓練データに過適合した overfitting になる。

```python
C_values = [0.01, 0.1, 1, 10, 100]   # 対数的に変化させる
scores = []

for c in C_values:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_tr_s, y_tr)
    scores.append(model.score(X_te_s, y_te))

plt.figure(figsize=(6, 4))
plt.plot(C_values, scores, marker="o")
plt.xscale("log")                     # ← 横軸を対数スケールに
plt.xlabel("C（大きいほど正則化が弱い）")
plt.ylabel("Test accuracy")
plt.title("Regularization strength C vs accuracy")
plt.grid(True, alpha=0.3)
plt.show()

print("C と正解率:", dict(zip(C_values, scores)))
```

## つまずきやすいポイント（第2章）

- **`C` は正則化の「強さ」ではなく「逆数」** → `Ridge` の `alpha` とは逆向きなので注意
- **`StandardScaler` を忘れると正則化の効果が列ごとにバラつく** → 正則化は「係数の大きさ」にペナルティをかけるので，スケールが揃っていないと効果が不均一になる
- **`Lasso` で係数が 0 になる現象** → 意図して特徴量選択に使うこともできる（スパースモデル）

---

# 第3章　アンサンブル学習
***
※ 対応演習: [q10.ipynb](../q10.ipynb)
## 目次
1. ランダムフォレスト
2. 勾配ブースティング
3. XGBoost
4. 特徴量重要度

## 1. ランダムフォレスト
***
**アンサンブル学習**は，複数の弱いモデルを組み合わせて強いモデルを作る手法である。

### なぜ複数の木を作ると精度が上がるか

1本の決定木は「訓練データの一部の偏り」を覚えてしまいやすい（過学習）。しかし，**互いに異なるデータで学んだ木を多数決させると，各木のバラバラなエラーが打ち消し合い，系統的なバイアスが残りにくくなる**。

C言語のネットワークプロトコルで「同じデータを複数のノードに送り，多数決で整合性を取る」ように，複数の独立した判断を束ねることで信頼性が上がる。

```
木1（データAで学習） → 予測a ─╮
木2（データBで学習） → 予測b ─┼─ 多数決 → 最終予測
木3（データCで学習） → 予測c ─╯
```

`n_estimators` は木の本数，`random_state` は乱数シード（再現性のため固定）である。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_rf = pd.read_csv(TITANIC_URL)
df_rf = df_rf[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].dropna()
df_rf["Sex"] = (df_rf["Sex"] == "male").astype(int)   # ← "male"→1, "female"→0 に変換

X_rf = df_rf.drop("Survived", axis=1)
y_rf = df_rf["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=0, stratify=y_rf
)

rf = RandomForestClassifier(
    n_estimators=100,   # ← 100本の決定木を学習（多いほど安定するが遅くなる）
    random_state=0,     # ← 乱数シード固定（再現性確保）
)
rf.fit(X_train, y_train)
print("RandomForest 正解率:", accuracy_score(y_test, rf.predict(X_test)))
# → 0.82 付近
```

## 2. 勾配ブースティング
***
**勾配ブースティング**は，ランダムフォレストとは逆の発想で，**前のモデルが犯した誤差を次のモデルが学習する逐次学習方式**である。

```
Random Forest（並列）:
  木A ─╮
  木B ─┼─ 多数決 → 予測
  木C ─╯

Gradient Boosting（逐次）:
  木1 → 誤差1 → 木2（誤差1を学習） → 誤差2 → 木3（誤差2を学習） → ... → 合計
```

Random Forest は並列なので速いが，Gradient Boosting は直列なので学習は遅め。ただし，残差を次々に修正するため表形式データで高い精度を出しやすい。

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()       # 乳がん診断データセット（良性/悪性の2値分類）
X_c, y_c = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X_c, y_c, test_size=0.2, random_state=0, stratify=y_c
)

gb = GradientBoostingClassifier(random_state=0)   # デフォルトは n_estimators=100, max_depth=3
gb.fit(X_train, y_train)
print("GradientBoosting 正解率:", accuracy_score(y_test, gb.predict(X_test)))
```

## 3. XGBoost
***
**XGBoost**（eXtreme Gradient Boosting）は Gradient Boosting を高速化・高精度化したライブラリである。

### なぜ XGBoost が強いか

1. **木の刈り込み（pruning）**: 分割してもスコアが改善しない枝を削除し，過学習を防ぐ
2. **L1/L2 正則化が内蔵**: アンサンブルなのに正則化できる
3. **並列計算・キャッシュ最適化**: 勾配計算を効率化し sklearn の GradientBoosting より高速

`eval_metric="logloss"` は「内部の評価指標を logloss に設定する」パラメータで，sklearn 連携時に警告が出るのを防ぐために指定する。

```python
# %pip install -q xgboost

from xgboost import XGBClassifier

xgb = XGBClassifier(
    random_state=0,
    eval_metric="logloss",   # ← sklearn と組み合わせるときに警告抑制のため指定
)
xgb.fit(X_train, y_train)

print("GradientBoosting 正解率:", gb.score(X_test, y_test))
print("XGBoost 正解率:         ", xgb.score(X_test, y_test))
# → XGBoost がわずかに高いことが多い
```

## 4. 特徴量重要度
***
ツリーベースのモデルは `feature_importances_` 属性で，各特徴量の予測への貢献度を確認できる。

**重要度が高い = そのモデルが分岐に多く使った特徴量**，という意味である。ドメイン知識と照らし合わせることで「モデルが正しい根拠で判断しているか」を確かめられる。

合計は必ず 1.0 になる（相対的な重要度の比較）。

```python
feature_names = X_rf.columns      # 特徴量名のリスト
importances = rf.feature_importances_   # ← 合計が 1.0 になる相対的重要度

plt.figure(figsize=(8, 4))
plt.bar(feature_names, importances)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("RandomForest feature importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# → "Fare" や "Age" が上位に来ることが多い
```

## つまずきやすいポイント（第3章）

- **XGBoost は macOS で `brew install libomp` が必要な場合がある** → OpenMP が入っていないとインポートエラーになる
- **`feature_importances_` の合計は 1.0** → 0.5 が出ても「50%の確率で重要」ではなく「全特徴量の中での相対的シェア」
- **Gradient Boosting は学習率（`learning_rate`）と `n_estimators` がトレードオフ** → 学習率を下げて木の数を増やすと精度が上がりやすいが遅くなる

---

# 第4章　教師なし学習
***
※ 対応演習: [q11.ipynb](../q11.ipynb)
## 目次
1. k-means クラスタリング
2. クラスタ数の評価
3. PCA による次元削減
4. 2次元可視化

## 1. k-means クラスタリング
***
これまでの章は「正解ラベルあり（教師あり学習）」だった。**教師なし学習**は，**正解ラベルなしでデータの隠れた構造を発見する**手法である。

例えば，顧客データをグループ分けして「似た購買行動を持つ層」を発見するような用途に使う。

### k-means のアルゴリズム（ステップで理解する）

```
Step 1: k 個のセントロイド（重心）をランダムに配置
Step 2: 各データ点を「最も近いセントロイド」のクラスタに割り当て
Step 3: 各クラスタの平均座標をセントロイドとして更新
Step 4: 割り当てが変わらなくなるまで Step 2〜3 を繰り返す（収束）
```

C言語でポインタを更新しながらループする最適化アルゴリズムに近いイメージである。

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
# PCA と同様，k-means もスケールに敏感なので標準化が必要
X_iris = StandardScaler().fit_transform(iris.data)

kmeans = KMeans(
    n_clusters=3,      # ← k = 3（Iris は3種なので）
    random_state=0,
    n_init=10,         # ← 初期値を変えて10回試行し，最良の結果を採用（初期値依存を緩和）
)
labels = kmeans.fit_predict(X_iris)   # fit + predict を一発で実行

print("クラスタラベル（先頭10件）:", labels[:10])
print("シルエットスコア:", silhouette_score(X_iris, labels))
# → 0.45 付近（1.0 に近いほど良い）
```

## 2. クラスタ数の評価
***
k-means では事前に k を決める必要がある。**シルエットスコア**を使うと，「クラスタ内のまとまり」と「クラスタ間の離れ具合」のバランスを -1〜1 で評価できる。

直感的には，「自分のクラスタ内では近く，隣のクラスタとは遠い」ほどスコアが高い。

```
シルエットスコア = (隣クラスタとの距離 - 自クラスタ内の平均距離) / max(両者)

1.0  に近い → クラスタがはっきり分かれている（良い）
0.0  付近   → クラスタの境界が曖昧
-1.0 に近い → 間違ったクラスタに割り当てられている可能性
```

k を変えてスコアを比較し，最もスコアが高い k を選ぶ。

```python
k_range = range(2, 7)
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    pred = km.fit_predict(X_iris)
    sil_scores.append(silhouette_score(X_iris, pred))

best_k = list(k_range)[sil_scores.index(max(sil_scores))]
print("最もスコアが高い k:", best_k)   # → 2 か 3

plt.figure(figsize=(6, 4))
plt.plot(list(k_range), sil_scores, marker="o")
plt.xlabel("k（クラスタ数）")
plt.ylabel("Silhouette score")
plt.title("Number of clusters k vs silhouette score")
plt.grid(True, alpha=0.3)
plt.show()
```

## 3. PCA による次元削減
***
**PCA**（主成分分析, Principal Component Analysis）は，高次元データを少数の軸（主成分）に圧縮する**次元削減**手法である。

### なぜ圧縮しても情報が保てるか

「写真を真上から撮った影（射影）が元の物体の形を最もよく表す方向がある」イメージである。PCA は「データのバラつきが最大になる方向（=情報量が最大の方向）」を第1主成分として選ぶ。

`explained_variance_ratio_` は各主成分が「元のデータの情報のうち何%を保持しているか」を示す。2成分の累積が 0.6 なら「60%の情報を2次元に圧縮できた」と解釈する。

```python
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

wine = load_wine()           # ワインの化学成分データ（13次元, 3クラス）
X_wine = StandardScaler().fit_transform(wine.data)   # ← PCA 前に標準化必須

pca = PCA(n_components=2)    # ← 2次元に圧縮（可視化のため）
X_pca = pca.fit_transform(X_wine)

print("変換後 shape:", X_pca.shape)                          # → (178, 2)
print("第1主成分の寄与率:", pca.explained_variance_ratio_[0])  # → 0.36 付近（36%の情報）
print("第2主成分の寄与率:", pca.explained_variance_ratio_[1])  # → 0.19 付近
print("累積寄与率（2成分）:", pca.explained_variance_ratio_.sum())  # → 0.55 付近（55%の情報で2次元化）
```

## 4. 2次元可視化
***
PCA で2次元に圧縮したデータを散布図で可視化する。

**重要な点**: 色分けには正解ラベル（`wine.target`）を使っているが，**PCA の学習にはラベルを一切使っていない**。教師なし学習でも，事後的に正解ラベルで色づけして「クラスタが意味のある分離をしているか」を確認することは有効である。

```python
plt.figure(figsize=(7, 5))
for label in np.unique(wine.target):
    mask = wine.target == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=f"class {label}", alpha=0.7)
plt.xlabel("PC1（第1主成分）")
plt.ylabel("PC2（第2主成分）")
plt.title("Wine dataset PCA (2D)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# → 3クラスがある程度分かれて見えるはず
```

## つまずきやすいポイント（第4章）

- **k-means は初期値に依存する** → `n_init=10` で複数試行して最良の結果を採用（デフォルトは `n_init="auto"` だが明示的に指定すると安心）
- **PCA の前に `StandardScaler` が必要** → スケールが大きい特徴量が第1主成分を支配してしまう
- **k-means は「丸いクラスタ」を仮定している** → 細長い形や月型のクラスタには弱い（そういうデータには DBSCAN 等が向く）

---

# 第5章　不均衡データと ROC
***
※ 対応演習: [q12.ipynb](../q12.ipynb)
## 目次
1. 不均衡データの生成
2. class_weight
3. ROC 曲線と AUC
4. 適合率-再現率曲線

## 1. 不均衡データの生成
***
**不均衡データ**とは，クラス間のサンプル数に大きな偏りがあるデータである。

### 正解率だけでは誤魔化される問題

クラス0が 95%，クラス1が 5% のデータで，「全部クラス0と予測するだけ」のモデルを作ると，正解率（accuracy）は **95%** になる。しかしこのモデルは，本当に検出したいはずの少数クラス1を1件も正解できていない。

→ 不均衡データでは **F1スコア・Recall・Precision** を確認することが不可欠。

| 指標 | 意味 | 式 |
|:--:|:--|:--|
| **Precision（適合率）** | 陽性と予測したうち本当に陽性の割合 | TP / (TP + FP) |
| **Recall（再現率）** | 本当の陽性のうち正しく陽性と予測した割合 | TP / (TP + FN) |
| **F1** | Precision と Recall の調和平均 | 2PR / (P + R) |

```python
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(
    n_samples=2000,
    n_features=20,
    weights=[0.95, 0.05],   # ← クラス0が95%, クラス1が5%
    random_state=0,
)

X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=0, stratify=y_imb
)

print("訓練データ クラス0:", (y_train == 0).sum())   # → 1520 件
print("訓練データ クラス1:", (y_train == 1).sum())   # →   80 件 ← 圧倒的に少ない
```

## 2. class_weight
***
`LogisticRegression` に `class_weight="balanced"` を指定すると，サンプル数が少ないクラスほど**誤りの損失を大きく**扱い，少数クラスを見逃しにくくする。

### 混同行列の読み方

```
              予測 0   予測 1
正解 0  [  TN     FP  ]   ← 正解0を0と予測（正解）/ 0を1と予測（誤り）
正解 1  [  FN     TP  ]   ← 正解1を0と予測（見逃し）/ 1を1と予測（正解）

TP = True Positive  FP = False Positive
TN = True Negative  FN = False Negative（見逃し，Recall を下げる）
```

少数クラスを見逃すコスト（FN）が高い場面（医療診断，詐欺検知）では，`class_weight="balanced"` が有効である。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

for cw in [None, "balanced"]:
    label = "class_weight なし" if cw is None else "class_weight=balanced"
    model = LogisticRegression(max_iter=1000, class_weight=cw)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"=== {label} ===")
    print("混同行列:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
# → balanced にすると recall（再現率）が上がり，クラス1の見逃しが減る
```

## 3. ROC 曲線と AUC
***
**ROC 曲線**は，分類器の「閾値」を 0→1 に変化させたときの **FPR**（偽陽性率）と **TPR**（真陽性率）の軌跡を示すグラフである。

| 指標 | 意味 |
|:--:|:--|
| **TPR（真陽性率 = Recall）** | 本当の陽性のうち正しく陽性と予測した割合 |
| **FPR（偽陽性率）** | 本当の陰性のうち誤って陽性と予測した割合 |

ROC 曲線の下の面積 **AUC**（Area Under the Curve）が分類器の総合性能の指標となる。

```
AUC = 1.0  → 完全な分類（理想）
AUC = 0.5  → ランダムな予測と同等（対角線）
AUC < 0.5  → ランダムより悪い（予測が逆向き）
```

```python
from sklearn.metrics import RocCurveDisplay, auc

model_bal = LogisticRegression(max_iter=1000, class_weight="balanced")
model_bal.fit(X_train, y_train)

RocCurveDisplay.from_estimator(model_bal, X_test, y_test)
plt.title("ROC curve")
plt.show()

y_score = model_bal.decision_function(X_test)   # ← 確率ではなく決定関数値を使う
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_score)
print("AUC:", auc(fpr, tpr))
# → 0.9 以上なら良好な分類器
```

## 4. 適合率-再現率曲線
***
**PR 曲線**（Precision-Recall 曲線）は，ROC 曲線より**不均衡データに有効な評価指標**である。

クラス1が非常に少ない場合，ROC の AUC は楽観的になりがちだが，PR 曲線はクラス1の Precision と Recall を直接評価する。

```python
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision)
plt.xlabel("Recall（再現率）")
plt.ylabel("Precision（適合率）")
plt.title("Precision-Recall curve")
plt.grid(True, alpha=0.3)
plt.show()
# → 曲線が右上に張り出すほど良い分類器
```

## つまずきやすいポイント（第5章）

- **不均衡データで `accuracy_score` だけを見ると誤解を招く** → F1・Recall・AUC も合わせて確認する
- **`decision_function` がないモデル（ツリー系など）では `predict_proba` を使う** → `roc_curve(y_test, model.predict_proba(X_test)[:, 1])`
- **PR 曲線の AP（Average Precision）はクラス1の比率に依存する** → 単純比較には要注意

---

# 第6章　学習曲線とハイパーパラメータ
***
※ 対応演習: [q13.ipynb](../q13.ipynb)
## 目次
1. learning_curve
2. validation_curve
3. 過学習の判断
4. 交差検証の復習

## 1. learning_curve
***
**学習曲線**（learning curve）は，「訓練データを増やしていったとき，モデルの性能がどう変わるか」を示すグラフである。

- **横軸**: 訓練サンプル数
- **縦軸**: スコア（精度など）
- **訓練スコア** と **交差検証スコア** を重ねて描く

### 4パターンの見方

```
パターン1: 過学習（Overfitting）
  score │  訓練 ─────────── 高い
        │  検証 ──────╮
        │             ╰──── 低い（ギャップ大）

パターン2: 未学習（Underfitting）
  score │  訓練 ────╮
        │  検証 ────╯  両方低い，ギャップ小

パターン3: 良好
  score │  訓練 ──────
        │  検証 ─────  両方高く，ギャップ小

パターン4: データ不足
  score │  訓練 ─────── 高い（少データで暗記）
        │  検証 ───╮
        │          ╰────  データを増やすと改善見込み
```

```python
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(random_state=0),
    X, y,
    cv=5,                                  # ← 5分割交差検証でスコアを計算
    train_sizes=np.linspace(0.1, 1.0, 10), # ← 10〜100%の訓練データ量で評価
    scoring="accuracy",
)

train_mean = train_scores.mean(axis=1)     # ← 5分割の平均
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(7, 4))
plt.plot(train_sizes, train_mean, marker="o", label="Train score")
plt.plot(train_sizes, val_mean, marker="o", label="CV score")
plt.xlabel("Training set size")
plt.ylabel("Accuracy")
plt.title("learning_curve（横軸 = データ量）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 2. validation_curve
***
**validation_curve** は，「特定のハイパーパラメータを変化させたとき，モデルの性能がどう変わるか」を示すグラフである。

- **横軸**: ハイパーパラメータの値（`max_depth` など）
- **縦軸**: スコア

`param_name` に変化させるパラメータ名，`param_range` に試したい値の配列を渡す。

決定木の `max_depth` を変えた場合，depth が大きくなるほど訓練スコアは上がるが，検証スコアは途中で頭打ちになる（= 過学習が始まる）。

```python
from sklearn.model_selection import validation_curve

param_range = [1, 2, 3, 5, 10, 20, None]   # None = 制限なし
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=0),
    X, y,
    param_name="max_depth",   # ← 変化させるパラメータ
    param_range=param_range,
    cv=5,
    scoring="accuracy",
)

x_labels = [str(v) for v in param_range]
plt.figure(figsize=(7, 4))
plt.plot(x_labels, train_scores.mean(axis=1), marker="o", label="Train score")
plt.plot(x_labels, val_scores.mean(axis=1), marker="o", label="CV score")
plt.xlabel("max_depth（大きいほど複雑）")
plt.ylabel("Accuracy")
plt.title("validation_curve（横軸 = ハイパーパラメータ）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# → max_depth=3〜5 付近で CV スコアがピークになることが多い
```

## 3. 過学習の判断
***
訓練スコアとテストスコアの差（ギャップ）が大きいほど過学習が疑われる。`max_depth` を小さく制限することで汎化性能が改善できる。

`max_depth=3` vs `max_depth=20` を比較すると，20の訓練スコアはほぼ1.0だがテストスコアは下がる。これが「過学習」の典型的なパターンである。

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

results = []
for depth in [3, 20]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree.fit(X_train, y_train)
    results.append({
        "max_depth": depth,
        "train_acc": tree.score(X_train, y_train),
        "test_acc":  tree.score(X_test, y_test),
    })

df_result = pd.DataFrame(results)
print(df_result)
# →  max_depth=3:  train≈0.93, test≈0.93  ギャップ小（良好）
# →  max_depth=20: train≈1.00, test≈0.88  ギャップ大（過学習）
print()
print("訓練とテストの差が大きいほど過学習の可能性が高い")
```

## 4. 交差検証の復習
***
**k-fold 交差検証**（cross-validation）は，データを k 分割し「k-1 分割で訓練，残り1分割でテスト」を k 回繰り返してスコアの平均を取る手法である。

なぜ1回の train/test 分割より信頼できるか：1回の分割では「たまたま簡単なテストデータに当たった」可能性がある。k-fold では**データの全部分がテストとして使われるため，評価が安定する**。

```python
from sklearn.model_selection import cross_val_score

for depth in [3, 20]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    scores = cross_val_score(tree, X, y, cv=5)   # ← 5分割CV
    print(f"max_depth={depth}: {scores.mean():.3f} ± {scores.std():.3f}")
    # ± が大きいほどスコアがデータ分割に依存している（不安定）
```

## つまずきやすいポイント（第6章）

- **`learning_curve` と `validation_curve` の混同**: 横軸が全く違う（前者 = データ量，後者 = ハイパーパラメータ値）
- **`scoring` の選択**: 不均衡データでは `"accuracy"` より `"f1"` や `"roc_auc"` を使うべきケースが多い
- **`max_depth=None` の意味**: 制限なし = 完全に成長した木 = 訓練データに完全過学習

---

# 第7章　PyTorch 入門
***
※ 対応演習: [q14.ipynb](../q14.ipynb)
## 目次
1. Tensor と自動微分
2. MNIST の読み込み
3. MNIST の可視化
4. DataLoader の理解

## 1. Tensor と自動微分
***
**PyTorch** は深層学習のためのライブラリである。Scikit-learn との最大の違いは「**勾配を自分で計算し，重みを自分で更新する**」点である。これにより，自由にモデルを設計できる代わりに，学習ループを自分で書く必要がある。

### Tensor とは

`Tensor` は NumPy 配列に似た多次元配列だが，以下の2点が大きく異なる。

| 機能 | NumPy ndarray | PyTorch Tensor |
|:--:|:--:|:--:|
| **GPU 計算** | 不可（標準） | `.to("cuda")` で転送可 |
| **自動微分** | なし | `requires_grad=True` で有効化 |

### 自動微分（autograd）の仕組み

`requires_grad=True` は「この Tensor に対する計算履歴を記録せよ」というフラグである。

```
x = 5.0  （requires_grad=True）
    ↓ y = 2x² + 3
y = 53.0
    ↓ y.backward() で逆向きに勾配を計算
x.grad = dy/dx = 4x = 20.0
```

**backward() の前に設定しないと `.grad` が None になる**ことに注意。

```python
# %pip install -q torch torchvision

import torch

x = torch.tensor([1.0, 2.0, 3.0])
print("shape:", x.shape)   # → torch.Size([3])

# 自動微分の例
x = torch.tensor(5.0, requires_grad=True)   # ← このフラグが計算グラフを記録する
y = 2.0 * x ** 2 + 3.0                      # 計算グラフが構築される
y.backward()                                 # dy/dx を逆伝播で計算
print("y =", y.item())                       # → 53.0
print("dy/dx =", x.grad.item())              # → 20.0（= 4x = 4*5）
```

## 2. MNIST の読み込み
***
**MNIST** は機械学習の「Hello World」とも言われる手書き数字データセットである。

- 画像サイズ: 28 × 28 ピクセル（グレースケール）
- クラス数: 10（0〜9の数字）
- 訓練データ: 60,000 枚 / テストデータ: 10,000 枚

`transforms.ToTensor()` は PIL Image（0〜255 の uint8）を PyTorch Tensor（0.0〜1.0 の float32）に変換し，shape を `(H, W)` から `(C, H, W)` = `(1, 28, 28)` に変える変換である。

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = "./data"   # MNIST はここに自動ダウンロードされる

train_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=True,                          # ← True=訓練データ, False=テストデータ
    download=True,                       # ← 初回のみダウンロード
    transform=transforms.ToTensor(),     # ← PIL Image → [0,1] の Tensor に変換
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
print("サンプル数:", len(train_dataset))   # → 60000
```

## 3. MNIST の可視化
***
9枚をグリッド表示し，ラベルを確認する。

`img.squeeze()` は `(1, 28, 28)` の Tensor から**チャネル次元（1）を除去**して `(28, 28)` にする操作である。`imshow` はチャネル次元を持つとエラーになるため必要である。

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]           # ← i 番目の (image, label) を取得
    ax.imshow(img.squeeze(), cmap="gray")   # ← (1,28,28) → (28,28) に除次元してから表示
    ax.set_title(f"label: {label}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## 4. DataLoader の理解
***
`DataLoader` は「データセットをミニバッチ単位に切り出すイテレータ」である。

`batch_size=64` なら，1回の `next(iter(loader))` で 64 枚の画像とラベルが返ってくる。

shape の読み方：`(64, 1, 28, 28)` は `(バッチサイズ N, チャネル C, 高さ H, 幅 W)` を意味する。

**flatten（平坦化）**: 全結合層に入力する際は，`(N, 1, 28, 28)` を `(N, 784)` に変換する。`view(-1, 784)` の `-1` は「バッチサイズを自動計算せよ」という意味である。

```python
images, labels = next(iter(train_loader))   # ← 1バッチ分を取り出す
print("images.shape:", images.shape)        # → torch.Size([64, 1, 28, 28])
print("labels.shape:", labels.shape)        # → torch.Size([64])
print("flatten後:", images.view(-1, 784).shape)  # → torch.Size([64, 784])
#                    ↑ -1 = バッチサイズを自動計算（64）
```

## つまずきやすいポイント（第7章）

- **`requires_grad=True` を忘れると `.grad` が None** → `y.backward()` しても勾配が計算されない
- **`DataLoader(shuffle=True)` を忘れると毎 epoch 同じ順番で学習** → モデルが順序に過学習する可能性がある
- **GPU が使えない環境でも `device = "cpu"` でそのまま動く** → `torch.device("cuda" if torch.cuda.is_available() else "cpu")` で自動切替
- **`transforms.ToTensor()` を忘れると PIL Image のまま返ってくる** → Tensor に変換されないのでモデルに渡せない

---

# 第8章　ニューラルネットワーク実装
***
※ 対応演習: [q15.ipynb](../q15.ipynb)
## 目次
1. nn.Module の実装
2. 学習ループ
3. 損失曲線
4. テスト正解率

## 1. nn.Module の実装
***
PyTorch では `nn.Module` を継承してモデルクラスを定義する。第7回の `MyLinearRegression` クラスを自作したのと同じ発想である。

**役割の分担**:
- `__init__`: 「**層の定義**」（どんなレイヤーを持つかを宣言）
- `forward`: 「**データの流れ**」（入力からどう出力を計算するか）

`super().__init__()` は必ず最初に呼ぶ。PyTorch が内部で層のパラメータを登録・管理するために必要であり，これを忘れるとパラメータが `optimizer` に認識されない。

`nn.Linear(784, 128)` は「入力 784 次元 → 出力 128 次元の全結合層（重み行列 784×128 + バイアス 128）」を意味する。

```python
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()                    # ← 必須: PyTorch の内部登録機構を初期化
        self.fc1 = nn.Linear(784, 128)        # ← 入力 784（28×28 flatten）→ 隠れ層 128
        self.relu = nn.ReLU()                 # ← 非線形活性化関数（ReLU: max(0, x)）
        self.fc2 = nn.Linear(128, 10)         # ← 隠れ層 128 → 出力 10（0〜9の10クラス）

    def forward(self, x):
        x = x.view(-1, 784)                   # ← (N,1,28,28) → (N,784) に flatten
        x = self.relu(self.fc1(x))            # ← 全結合 → ReLU で非線形変換
        return self.fc2(x)                    # ← 出力層（Softmax はロス関数側で処理）
```

## 2. 学習ループ
***
PyTorch の学習ループは毎回同じ4ステップで構成される。

```
[1] forward:  y_pred = model(images)          ← 予測（順伝播）
[2] loss:     loss = criterion(y_pred, labels) ← 正解との誤差を計算
[3] backward: optimizer.zero_grad()            ← 前回の勾配をリセット（重要！）
              loss.backward()                  ← 逆伝播で勾配を計算
[4] update:   optimizer.step()                 ← 勾配方向に重みを更新
```

### `zero_grad()` を忘れると何が起きるか

PyTorch はデフォルトで **勾配を蓄積（加算）する**。`zero_grad()` を忘れると，前の mini-batch の勾配と今の勾配が足し合わされ，誤った方向に重みが更新されて発散する。

### `model.train()` / `model.eval()` の切り替え

Dropout や BatchNorm を使う場合，訓練中と推論中で**挙動が異なる**。`model.train()` で訓練モード，`model.eval()` で推論モードに切り替える。今の SimpleMLP では Dropout を使っていないが，癖として必ず書く習慣をつけるとよい。

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = "./data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = SimpleMLP().to(device)                        # ← モデルを device（GPU/CPU）に転送
criterion = nn.CrossEntropyLoss()                     # ← 多クラス分類の損失関数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # ← Adam は学習率を自動調整する

for epoch in range(3):
    model.train()    # ← 訓練モードに切り替え（Dropout 等が有効になる）
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # ← データを同じ device に

        optimizer.zero_grad()                         # ← [3a] 前回の勾配をリセット（必須）
        loss = criterion(model(images), labels)       # ← [1][2] 順伝播 + 損失計算
        loss.backward()                               # ← [3b] 逆伝播で勾配計算
        optimizer.step()                              # ← [4] 重みを更新
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
```

## 3. 損失曲線
***
第13回の学習曲線と同様，epoch ごとの損失をプロットして学習の進み具合を確認する。損失が下がり止まった場合は epoch 数の不足または学習率の調整が必要かもしれない。

`epoch_losses.append(...)` は学習ループ内で各 epoch の終わりに呼ぶ。

```python
epoch_losses = []

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)   # ← 1 epoch の平均損失
    epoch_losses.append(avg_loss)               # ← 損失を記録
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss curve")
plt.grid(True, alpha=0.3)
plt.show()
```

## 4. テスト正解率
***
学習済みモデルでテストデータを評価する。

- `model.eval()`: 推論モードに切り替え（Dropout 等を無効化）
- `torch.no_grad()`: 勾配計算を無効化してメモリと計算コストを節約。推論時は `backward()` が不要なので必ず使う

```python
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)   # ← 評価時は shuffle 不要

model.eval()   # ← 推論モード（必須）
correct, total = 0, 0
with torch.no_grad():   # ← 勾配計算を無効化（メモリ節約・高速化）
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)           # ← 最大スコアのクラスを予測ラベルとする
        correct += (preds == labels).sum().item()     # ← 正解数を累積
        total += labels.size(0)                       # ← 総サンプル数を累積
print(f"テスト正解率: {correct / total:.4f}")
# → SimpleMLP なら 0.97 付近
```

## つまずきやすいポイント（第8章）

- **`zero_grad()` 忘れ** → 勾配が蓄積して発散する。毎回 `optimizer.zero_grad()` を `backward()` の前に呼ぶ
- **`model.train()` / `model.eval()` の切り替え忘れ** → Dropout を使う場合に特に影響が出る
- **モデルとデータが別 device に** → `model.to(device)` と `images.to(device)` を両方行う
- **テスト時に `torch.no_grad()` を忘れる** → メモリ不足になりやすい（勾配計算が走るため）

---

# 第9章　ANN・CNN・推論
***
※ 対応演習: [q16.ipynb](../q16.ipynb), [q17.ipynb](../q17.ipynb)
## 目次
1. 3モデルの比較
2. CNN モデルの実装
3. 混同行列とモデル保存
4. 手書き推論（q17）のヒント

## 1. 3モデルの比較
***
第16回では，同じ MNIST データ・同じ条件で以下の3モデルを比較する。それぞれの精度の違いには明確な理由がある。

| モデル | 構成 | なぜ他と精度が違うか |
|:--:|:--|:--|
| `LinearNet` | 784 → 10 | 活性化関数なし = 線形境界しか引けない。XOR のような複雑なパターンを学べない |
| `DeepMLP` | 784 → 256 → 128 → 10 | 非線形変換ができるが，画像の「局所的な特徴（エッジなど）」を無視して全ピクセルを独立に見る |
| `SimpleCNN` | Conv→Pool×2 → FC | **局所的なパターン（エッジ・曲線）を重み共有で効率的に学習**。MNIST の数字の形状に強い |

```python
def train_one_model(model, train_loader, test_loader, epochs=3, lr=0.001):
    """3モデルを同条件（epoch/lr/batch_size）で学習・評価するヘルパー関数"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    # 学習後にテスト正解率を返す
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
```

## 2. CNN モデルの実装
***
**畳み込み（Convolution）** はフィルタを画像上でスライドさせ，局所的なパターンを検出する操作である。全結合層と比べて「重みを共有する（= パラメータが少ない）」点が特徴である。

### shape の変化を追う

```
入力:        (N, 1, 28, 28)
↓ Conv(1→16) + ReLU  ← 16 種類のフィルタでエッジ等を検出
             (N, 16, 28, 28)   ← padding=1 で空間サイズは変わらない
↓ MaxPool(2)
             (N, 16, 14, 14)   ← 2×2 の領域の最大値を取り，半分のサイズに
↓ Conv(16→32) + ReLU
             (N, 32, 14, 14)
↓ MaxPool(2)
             (N, 32,  7,  7)   ← さらに半分に
↓ Flatten（view）
             (N, 32×7×7) = (N, 1568)
↓ Linear(1568→128) + ReLU
↓ Linear(128→10)
出力:        (N, 10)            ← 10クラスのロジット
```

`32 * 7 * 7 = 1568` は，プーリング後のサイズ計算から決まる。`kernel_size=3, padding=1` にすると Conv 後の空間サイズが変わらないため，MaxPool(2) だけがサイズを半分にする。

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # --- 特徴抽出部 ---
        self.conv1 = nn.Conv2d(
            in_channels=1,      # ← 入力チャネル数（グレースケールなので 1）
            out_channels=16,    # ← 出力チャネル数（16 種類のフィルタを学習）
            kernel_size=3,      # ← フィルタサイズ 3×3
            padding=1,          # ← パディング 1 → 空間サイズを保つ
        )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)   # ← 2×2 領域の最大値を取る（サイズ 1/2）
        self.relu = nn.ReLU()

        # --- 分類部 ---
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # ← 32チャネル × 7×7 = 1568
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 特徴抽出
        x = self.pool(self.relu(self.conv1(x)))  # (N,1,28,28)→(N,16,14,14)
        x = self.pool(self.relu(self.conv2(x)))  # (N,16,14,14)→(N,32,7,7)
        # Flatten して全結合層へ
        x = x.view(x.size(0), -1)               # (N,32,7,7)→(N,1568)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

## 3. 混同行列とモデル保存
***
**混同行列**（Confusion Matrix）は「行 = 正解クラス，列 = 予測クラス」の表である。

```
        予測0  予測1  予測2 ...
正解0 [  多    少    少  ...]   ← 対角がゼロでない = その数字を別の数字と混同
正解1 [  少    多    少  ...]
...
```

対角成分が大きく，それ以外が小さいほど良い分類器である。どの行・列で間違いが多いかを見ると「3と8を混同しやすい」などの発見ができる。

`torch.save` でモデルの重みを `.pth` ファイルに保存し，`torch.load` + `load_state_dict` で復元する。**復元時にはモデルクラスの定義が必要**なので，q17 では `SimpleCNN` クラスを再度定義してから読み込む。

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

# テストデータで全予測を収集
all_preds, all_labels = [], []
best_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = best_model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("予測クラス")
plt.ylabel("正解クラス")
plt.title("Confusion matrix (MNIST)")
plt.show()
# → 対角が大きく，"3" と "5" や "4" と "9" が混同されることが多い

# モデルの重みを保存（第17回で使用）
torch.save(best_model.state_dict(), "best_mnist_model.pth")
print("best_mnist_model.pth を保存しました")
```

## 4. 手書き推論（q17）のヒント
***
第17回では Google Colab 上の Canvas UI で自作数字を推論する．UI コードは演習ノートブックに完成コードとして同梱されているので，実行するだけでよい（編集不要）．

### 取り組み方
1. 第16回で保存した `best_mnist_model.pth` を `load_state_dict` で復元
2. Canvas UI セルを実行し，0〜9 を書いて「推論する」ボタンを押す
3. 結果を表形式で記録し，誤認識の原因を考察

### 前処理のポイント（Canvas → MNIST 形式へ変換）

```
Canvas の描画:  黒背景（0.0）に白線（1.0）で描く
MNIST の訓練:   白背景（1.0）に黒文字（0.0）
                 ↓
arr = 1.0 - arr  で色反転が必要
```

モデルは「白背景・黒文字」で訓練されているため，色反転しないとほぼランダムな予測になる。

### 考察で述べるとよい観点
1. 第16回で CNN が MLP より良かった理由（局所的なエッジ・曲線を重み共有で検出できるため）
2. 自作数字で誤認識した原因（線の太さ，文字の位置ズレ，前処理の色反転）
3. 精度向上のアイデア（epoch 増加，データ拡張，Dropout の追加 など）

## つまずきやすいポイント（第9章）

- **CNN に flatten した `(N, 784)` を渡すのは NG** → CNN の入力 shape は `(N, 1, 28, 28)` のまま。`LinearNet` / `DeepMLP` と CNN では `forward` の先頭処理が違う
- **`32 * 7 * 7` はプーリング後のサイズで決まる** → MaxPool の数・サイズを変えるとこの値も変わる
- **`load_state_dict` の前にモデルクラスを定義しておく必要がある** → `.pth` は重みだけを保存しており，モデルの構造は保存されていない
- **色反転を忘れると推論が全く機能しない** → Canvas（黒背景）と MNIST（白背景）の向きが逆なので `1.0 - arr` が必須
