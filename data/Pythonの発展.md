# Pythonの発展
***
## 目次
1. 前処理と Pipeline
2. 正則化
3. アンサンブル学習
4. 教師なし学習
5. 不均衡データと ROC
6. 学習曲線とハイパーパラメータ
7. ニューラルネットワーク入門
8. PyTorch 基礎
9. CNN と総合のヒント

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
機械学習の前処理では，まずデータの**形式**（各列の型）と**欠損値**の有無を確認する．

Pandas の `read_csv` で CSV を読み込み，`info()` で列の型と非欠損数，`isnull().sum()` で列ごとの欠損数を確認するのが基本である．

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TITANIC_URL = "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/titanic/titanic.csv"

df = pd.read_csv(TITANIC_URL)
cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
df_titanic = df[cols].copy()

print(df_titanic.head())
print()
df_titanic.info()
print()
print("欠損数:")
print(df_titanic.isnull().sum())
```

## 2. 欠損値の補完
***
欠損値（NaN）がある列は，そのままでは多くの機械学習アルゴリズムに渡せない．

Scikit-learn の `SimpleImputer` を使うと，**中央値**（`strategy="median"`）や**平均値**（`strategy="mean"`），**最頻値**（`strategy="most_frequent"`）で欠損を補完できる．

`fit_transform` を呼ぶと，統計量の計算（fit）と変換（transform）を一度に行える．

```python
from sklearn.impute import SimpleImputer

print("補完前 Age の欠損数:", df_titanic["Age"].isnull().sum())

imputer = SimpleImputer(strategy="median")
age_filled = imputer.fit_transform(df_titanic[["Age"]])

print("補完後 Age の欠損数:", pd.isna(age_filled).sum())
print("補完に使った中央値:", imputer.statistics_[0])
```

## 3. ColumnTransformer
***
列の種類によって前処理を変えたい場合，`ColumnTransformer` を使う．

- **数値列** → `StandardScaler`（平均0，分散1に標準化）
- **カテゴリ列** → `OneHotEncoder`（ダミー変数化）

`ColumnTransformer` には `(名前, 変換器, 列名リスト)` のタプルを渡す．

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features),
    ]
)

X = df_titanic.drop("Survived", axis=1)
y = df_titanic["Survived"]

X_temp = X.copy()
X_temp["Age"] = SimpleImputer(strategy="median").fit_transform(X[["Age"]]).ravel()

X_transformed = preprocessor.fit_transform(X_temp)
print("変換後の shape:", X_transformed.shape)
print("先頭3行:")
print(X_transformed[:3])
```

## 4. Pipeline の構築
***
前処理とモデル学習を**一つのオブジェクト**にまとめるのが `Pipeline` である．

Pipeline を使う利点:
- 訓練データだけで前処理の統計量を学習し，テストデータへ漏洩しない
- `fit` / `predict` を1回呼ぶだけで前処理＋予測が完結する
- 交差検証やハイパーパラメータ探索と組み合わせやすい

`Pipeline` には `(名前, ステップ)` のタプルを順に渡す．

```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

pipe = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ],
    )),
    ("model", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("テスト正解率:", accuracy_score(y_test, y_pred))
```

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
**過学習**（オーバーフィッティング）とは，訓練データに過度に適合し，未知データへの汎化性能が低下する状態である．

**正則化**は，モデルの複雑さにペナルティを課すことで過学習を抑える手法である．

| 手法 | 用途 | 特徴 |
| :--: | :--: | :-- |
| Ridge | 回帰 | L2正則化。係数を小さく均す |
| Lasso | 回帰 | L1正則化。不要な係数を0にしやすい |
| LogisticRegression(C=...) | 分類 | C が小さいほど正則化が強い |

## 2. Ridge / Lasso 回帰
***
UCI の学生成績データを用いて，線形回帰と正則化付き回帰を比較する．

`LinearRegression`，`Ridge(alpha=1.0)`，`Lasso(alpha=0.1)` の流れはいずれも `fit` → `predict` → 評価指標（MSE, R2）で同じである．

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

X = df_student[["G1"]]
y = df_student["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    print(f"{name}: MSE={mean_squared_error(y_test, pred):.3f}, R2={r2_score(y_test, pred):.3f}")
```

## 3. ロジスティック回帰の正則化
***
分類問題では `LogisticRegression` の `C` パラメータが正則化の強さを制御する．

- **C が大きい** → 正則化が弱い（複雑なモデルになりやすい）
- **C が小さい** → 正則化が強い（シンプルなモデルになりやすい）

目的変数を2値化する例: `G3 >= 10` を 1，それ以外を 0 とする．

```python
from sklearn.linear_model import LogisticRegression

y_bin = (df_student["G3"] >= 10).astype(int)
X_cls = df_student[["G1", "G2", "studytime", "failures"]]

X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_bin, test_size=0.2, random_state=0, stratify=y_bin)

X_tr_s = StandardScaler().fit_transform(X_tr)
X_te_s = StandardScaler().fit_transform(X_te)

logreg = LogisticRegression(C=1.0, max_iter=1000)
logreg.fit(X_tr_s, y_tr)
print("テスト正解率:", logreg.score(X_te_s, y_te))
```

## 4. 正則化強度 C の比較
***
`C` を変化させ，テスト正解率の変化をグラフで確認する．横軸は対数スケール（`plt.xscale("log")`）が見やすい．

```python
C_values = [0.01, 0.1, 1, 10, 100]
scores = []

for c in C_values:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_tr_s, y_tr)
    scores.append(model.score(X_te_s, y_te))

plt.figure(figsize=(6, 4))
plt.plot(C_values, scores, marker="o")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Test accuracy")
plt.title("Regularization strength C vs accuracy")
plt.grid(True, alpha=0.3)
plt.show()

print("C と正解率:", dict(zip(C_values, scores)))
```

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
**アンサンブル学習**は，複数のモデルを組み合わせて予測性能を高める手法である．

**ランダムフォレスト**（Random Forest）は，多数の決定木を学習し，多数決（分類）や平均（回帰）で最終予測を行う．

`RandomForestClassifier(n_estimators=100)` の `n_estimators` は木の本数である．

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_rf = pd.read_csv(TITANIC_URL)
df_rf = df_rf[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].dropna()
df_rf["Sex"] = (df_rf["Sex"] == "male").astype(int)

X_rf = df_rf.drop("Survived", axis=1)
y_rf = df_rf["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=0, stratify=y_rf
)

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("RandomForest 正解率:", accuracy_score(y_test, rf.predict(X_test)))
```

## 2. 勾配ブースティング
***
**勾配ブースティング**は，前のモデルの誤差を次のモデルが修正していく逐次学習方式である．

Scikit-learn では `GradientBoostingClassifier` が使える．

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()
X_c, y_c = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X_c, y_c, test_size=0.2, random_state=0, stratify=y_c
)

gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)
print("GradientBoosting 正解率:", accuracy_score(y_test, gb.predict(X_test)))
```

## 3. XGBoost
***
**XGBoost** は高速かつ高性能な勾配ブースティングライブラリである．

初回のみ `pip install xgboost` が必要な場合がある．

```python
# %pip install -q xgboost

from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=0, eval_metric="logloss")
xgb.fit(X_train, y_train)

print("GradientBoosting 正解率:", gb.score(X_test, y_test))
print("XGBoost 正解率:", xgb.score(X_test, y_test))
```

## 4. 特徴量重要度
***
ツリーベースのモデルは `feature_importances_` 属性で，各特徴量の重要度を取得できる．

棒グラフで可視化すると，どの変数が予測に効いているか把握しやすい．

```python
feature_names = X_rf.columns
importances = rf.feature_importances_

plt.figure(figsize=(8, 4))
plt.bar(feature_names, importances)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("RandomForest feature importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

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
**教師なし学習**は，正解ラベルなしでデータの構造を見つける手法である．

**k-means** は，データを k 個のクラスタに分割する代表的なクラスタリング手法である．

`KMeans(n_clusters=3)` でクラスタ数を指定し，`fit_predict` で各サンプルのクラスタラベルを得る．

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X_iris = StandardScaler().fit_transform(iris.data)

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(X_iris)

print("クラスタラベル（先頭10件）:", labels[:10])
print("シルエットスコア:", silhouette_score(X_iris, labels))
```

## 2. クラスタ数の評価
***
**シルエットスコア**は，クラスタリングの良さを -1〜1 で評価する指標である．値が大きいほど，クラスタ内がまとまり，クラスタ間が離れている．

k を変えてスコアを比較し，最適な k を選ぶ．

```python
k_range = range(2, 7)
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    pred = km.fit_predict(X_iris)
    sil_scores.append(silhouette_score(X_iris, pred))

best_k = list(k_range)[sil_scores.index(max(sil_scores))]
print("最もスコアが高い k:", best_k)

plt.figure(figsize=(6, 4))
plt.plot(list(k_range), sil_scores, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.title("Number of clusters k vs silhouette score")
plt.grid(True, alpha=0.3)
plt.show()
```

## 3. PCA による次元削減
***
**PCA**（主成分分析）は，高次元データを少数の軸（主成分）に射影する**次元削減**手法である．

`PCA(n_components=2)` で2次元に圧縮し，`explained_variance_ratio_` で各主成分の**寄与率**を確認できる．

```python
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

wine = load_wine()
X_wine = StandardScaler().fit_transform(wine.data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_wine)

print("変換後 shape:", X_pca.shape)
print("第1主成分の寄与率:", pca.explained_variance_ratio_[0])
print("第2主成分の寄与率:", pca.explained_variance_ratio_[1])
print("累積寄与率:", pca.explained_variance_ratio_.sum())
```

## 4. 2次元可視化
***
PCA で2次元に圧縮したデータを散布図で可視化する．色分けに正解ラベル（`target`）を使うと，クラスタ構造が見えやすい．

```python
plt.figure(figsize=(7, 5))
for label in np.unique(wine.target):
    mask = wine.target == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"class {label}", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Wine dataset PCA (2D)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

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
**不均衡データ**とは，クラス間のサンプル数に大きな偏りがあるデータである．

正解率だけでは性能を誤評価しやすい．`make_classification` の `weights` パラメータで不均衡データを人工的に生成できる．

```python
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(
    n_samples=2000, n_features=20, weights=[0.95, 0.05], random_state=0
)

X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=0, stratify=y_imb
)

print("訓練データ クラス0:", (y_train == 0).sum())
print("訓練データ クラス1:", (y_train == 1).sum())
```

## 2. class_weight
***
`LogisticRegression` に `class_weight="balanced"` を指定すると，少数クラスに大きな重みを付けて学習する．

混同行列（`confusion_matrix`）と `classification_report` で，適合率・再現率・F1スコアを確認する．

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
```

## 3. ROC 曲線と AUC
***
**ROC 曲線**は，分類器の閾値を変えたときの偽陽性率（FPR）と真陽性率（TPR）の関係を示す．

曲線の下の面積 **AUC** が 1 に近いほど性能が良い．`RocCurveDisplay.from_estimator` で描画できる．

```python
from sklearn.metrics import RocCurveDisplay, auc

model_bal = LogisticRegression(max_iter=1000, class_weight="balanced")
model_bal.fit(X_train, y_train)

RocCurveDisplay.from_estimator(model_bal, X_test, y_test)
plt.title("ROC curve")
plt.show()

y_score = model_bal.decision_function(X_test)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_score)
print("AUC:", auc(fpr, tpr))
```

## 4. 適合率-再現率曲線
***
不均衡データでは **PR 曲線**（適合率-再現率曲線）も重要な評価指標である．

`precision_recall_curve` で描画できる．

```python
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.grid(True, alpha=0.3)
plt.show()
```

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
**学習曲線**（learning curve）は，訓練データ量を増やしたときの訓練スコアと交差検証スコアの変化を示す．

`sklearn.model_selection.learning_curve` で取得し，両者の差が大きい場合は過学習の可能性がある．

```python
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(random_state=0),
    X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy",
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(7, 4))
plt.plot(train_sizes, train_mean, marker="o", label="Train score")
plt.plot(train_sizes, val_mean, marker="o", label="CV score")
plt.xlabel("Training set size")
plt.ylabel("Accuracy")
plt.title("learning_curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 2. validation_curve
***
**validation_curve** は，特定のハイパーパラメータを変化させたときの交差検証スコアを示す．

決定木の `max_depth` を変え，過学習が起きやすい領域を確認する．

```python
from sklearn.model_selection import validation_curve

param_range = [1, 2, 3, 5, 10, 20, None]
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=0),
    X, y,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
)

x_labels = [str(v) for v in param_range]
plt.figure(figsize=(7, 4))
plt.plot(x_labels, train_scores.mean(axis=1), marker="o", label="Train score")
plt.plot(x_labels, val_scores.mean(axis=1), marker="o", label="CV score")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("validation_curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 3. 過学習の判断
***
訓練スコアとテスト（検証）スコアの差が大きい場合，**過学習**が起きている可能性が高い．

`max_depth` を小さく制限することで，汎化性能を改善できる場合がある．

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
        "test_acc": tree.score(X_test, y_test),
    })

df_result = pd.DataFrame(results)
print(df_result)
print()
print("訓練とテストの差が大きいほど過学習の可能性が高い")
```

## 4. 交差検証の復習
***
[Pythonの基礎.ipynb](Pythonの基礎.ipynb) 第6章で学んだ `cross_val_score` を使い，異なる `max_depth` のモデルを交差検証で比較する．

```python
from sklearn.model_selection import cross_val_score

for depth in [3, 20]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    scores = cross_val_score(tree, X, y, cv=5)
    print(f"max_depth={depth}: {scores.mean():.3f} +- {scores.std():.3f}")
```

# 第7章　ニューラルネットワーク入門
***
※ 対応演習: [q14.ipynb](../q14.ipynb)
## 目次
1. 手書き数字データ
2. MLPClassifier
3. 活性化関数の比較
4. 隠れ層構成の比較

## 1. 手書き数字データ
***
Scikit-learn の `load_digits()` は 8×8 ピクセルの手書き数字（0〜9）データセットである．

ニューラルネットワークでは入力を標準化してから学習するのが一般的である．

```python
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

digits = load_digits()
print("サンプル数:", digits.data.shape[0])
print("特徴量次元数:", digits.data.shape[1])
print("クラス数:", len(np.unique(digits.target)))

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=0, stratify=digits.target
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

## 2. MLPClassifier
***
**MLP**（多層パーセプトロン）は，Scikit-learn で `MLPClassifier` として使える．

- `hidden_layer_sizes=(64, 32)` … 隠れ層のユニット数
- `activation="relu"` … 活性化関数
- `max_iter=300` … 最大反復回数

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=300,
    random_state=0,
)
mlp.fit(X_train_s, y_train)
print("MLP テスト正解率:", mlp.score(X_test_s, y_test))
```

## 3. 活性化関数の比較
***
活性化関数を `relu`, `tanh`, `logistic` で変え，性能を比較する．

```python
activations = ["relu", "tanh", "logistic"]
acc_list = []

for act in activations:
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation=act, max_iter=300, random_state=0)
    model.fit(X_train_s, y_train)
    acc_list.append(model.score(X_test_s, y_test))

plt.figure(figsize=(6, 4))
plt.bar(activations, acc_list)
plt.ylabel("Test accuracy")
plt.title("Activation function comparison")
plt.ylim(0.9, 1.0)
plt.show()
```

## 4. 隠れ層構成の比較
***
隠れ層の構成 `(32,)`, `(64, 32)`, `(128, 64, 32)` を変えて比較する．

層を深くしすぎると，データ量に対して過学習しやすくなる場合がある．

```python
layer_configs = [(32,), (64, 32), (128, 64, 32)]
layer_labels = ["(32,)", "(64,32)", "(128,64,32)"]
acc_layers = []

for layers in layer_configs:
    model = MLPClassifier(hidden_layer_sizes=layers, activation="relu", max_iter=300, random_state=0)
    model.fit(X_train_s, y_train)
    acc_layers.append(model.score(X_test_s, y_test))

plt.figure(figsize=(6, 4))
plt.bar(layer_labels, acc_layers)
plt.ylabel("Test accuracy")
plt.title("Hidden layer configuration comparison")
plt.ylim(0.9, 1.0)
plt.show()
```

# 第8章　PyTorch 基礎
***
※ 対応演習: [q15.ipynb](../q15.ipynb)
## 目次
1. Tensor と自動微分
2. Dataset / DataLoader
3. MLP の実装
4. MNIST 分類

## 1. Tensor と自動微分
***
**PyTorch** は深層学習のためのライブラリである．

`Tensor` は NumPy 配列に似た多次元配列で，`requires_grad=True` を指定すると**自動微分**（autograd）が有効になる．

`backward()` で勾配を計算し，`.grad` で取得できる．

```python
# %pip install -q torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

x = torch.tensor(5.0, requires_grad=True)
y = 2.0 * x ** 2 + 3.0
y.backward()
print("x =", x.item())
print("y =", y.item())
print("dy/dx =", x.grad.item())  # d(2x^2+3)/dx = 4x = 20
```

## 2. Dataset / DataLoader
***
`torchvision.datasets.MNIST` で手書き数字データを読み込み，`DataLoader` でバッチ単位に取り出す．

`transforms.ToTensor()` で画像を Tensor（0〜1）に変換する．

```python
DATA_ROOT = "./data"  # MNIST/Fashion-MNIST は torchvision が自動ダウンロード

train_dataset = datasets.MNIST(
    root=DATA_ROOT, train=True, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

images, labels = next(iter(train_loader))
print("画像テンソル shape:", images.shape)  # (batch, channel, H, W)
print("ラベル shape:", labels.shape)
```

## 3. MLP の実装
***
PyTorch では `nn.Module` を継承し，`__init__` で層を定義，`forward` で順伝播を記述する．

MNIST（28×28 = 784 次元）を入力とし，10クラスを出力する MLP の例．

```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (batch, 784) に flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleMLP()
print(model)
```

## 4. MNIST 分類
***
学習ループの基本:
1. 順伝播 → 損失計算（`CrossEntropyLoss`）
2. `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`

3 エポック程度の短い学習例．

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)

test_dataset = datasets.MNIST(
    root=DATA_ROOT, train=False, download=True, transform=transforms.ToTensor()
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    acc = evaluate(test_loader)
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, test_acc={acc:.4f}")
```

# 第9章　CNN と総合のヒント
***
※ 対応演習: [q16.ipynb](../q16.ipynb), [q17.ipynb](../q17.ipynb)
## 目次
1. Conv2d と Pooling
2. CNN モデルの実装
3. Fashion-MNIST 学習
4. 総合課題（q17）のヒント

## 1. Conv2d と Pooling
***
**CNN**（畳み込みニューラルネットワーク）は，画像認識に広く使われる．

- `Conv2d` … 畳み込み層（局所的な特徴を抽出）
- `MaxPool2d` … プーリング層（空間サイズを縮小）
- `Flatten` … 多次元テンソルを1次元に変換

Fashion-MNIST は 28×28 グレースケールの衣類画像10クラスデータセットである．

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

fashion_train = datasets.FashionMNIST(
    root=DATA_ROOT, train=True, download=True, transform=transforms.ToTensor()
)

image, label = fashion_train[0]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"label: {CLASS_NAMES[label]}")
plt.axis("off")
plt.show()
```

## 2. CNN モデルの実装
***
2層の畳み込み + 全結合層からなる `SimpleCNN` の例．

28×28 → Conv+Pool → 14×14 → Conv+Pool → 7×7 → Flatten → Linear

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

cnn = SimpleCNN()
print(cnn)
```

## 3. Fashion-MNIST 学習
***
CNN を Fashion-MNIST で学習し，テスト正解率と混同行列を確認する．

```python
fashion_test = datasets.FashionMNIST(
    root=DATA_ROOT, train=False, download=True, transform=transforms.ToTensor()
)
fashion_train_loader = DataLoader(fashion_train, batch_size=64, shuffle=True)
fashion_test_loader = DataLoader(fashion_test, batch_size=256, shuffle=False)

cnn = SimpleCNN().to(device)
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    cnn.train()
    total_loss = 0.0
    for images, labels in fashion_train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(cnn(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    cnn.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in fashion_test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = cnn(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch+1}: loss={total_loss/len(fashion_train_loader):.4f}, test_acc={correct/total:.4f}")
```

```python
# 混同行列
cnn.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in fashion_test_loader:
        images = images.to(device)
        pred = cnn(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

## 4. 総合課題（q17）のヒント
***
第17回の総合課題では，これまで学んだ内容を組み合わせる．以下は取り組み方のヒントである（答えは書かない）．

### 探索的データ分析（EDA）
- Seaborn の `countplot`, `boxplot` で `Survived` と各変数の関係を可視化
- 女性・1等客の生存率が高い，など傾向を言語化

### Pipeline + 交差検証
- 第1章の Pipeline を拡張し，`Embarked` などカテゴリ列も `OneHotEncoder` で処理
- `cross_val_score(pipeline, X, y, cv=5)` で平均正解率を比較

### モデル比較
- Pipeline の末尾を `RandomForestClassifier` と `XGBClassifier` で差し替え
- 交差検証スコアと `classification_report` で最終評価

### 考察で述べるとよい観点
1. Pipeline を使う利点（データ漏洩防止，再利用性）
2. RandomForest と XGBoost のどちらが良かったか，その理由
3. さらに精度を上げる施策（特徴量エンジニアリング，ハイパーパラメータ探索 など）

```python
# q17 向け EDA + Pipeline パターン例（骨格のみ）
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df_q17 = pd.read_csv(TITANIC_URL)
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X_q17 = df_q17[feature_cols]
y_q17 = df_q17["Survived"]

# EDA 例
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(data=df_q17, x="Sex", hue="Survived", ax=axes[0])
sns.boxplot(data=df_q17, x="Survived", y="Fare", ax=axes[1])
plt.tight_layout()
plt.show()

num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
cat_cols = ["Sex", "Embarked"]

pipe_q17 = Pipeline(steps=[
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])),
    ("model", RandomForestClassifier(n_estimators=100, random_state=0)),
])

scores = cross_val_score(pipe_q17, X_q17, y_q17, cv=5)
print(f"RF 5-fold CV: {scores.mean():.3f} +- {scores.std():.3f}")
```
