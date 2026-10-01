# Pythonの発展
***
## 目次
0. 試行錯誤の作法（q9–q17 共通）
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

本資料は，Scikit-learn の基礎を土台に，**前処理の自動化**，**正則化**，**アンサンブル学習**，**教師なし学習**，**深層学習** など，第8回以降の演習（q8–q17）で必要となる知識を，概念の「なぜ」から実装の「どう書くか」まで徹底的に解説する．

各章では概念説明・数式・図解・コード例を組み合わせる．コードは動作確認済みのもので，コメントにより各行の意味を明記する．演習の答えそのものは書かないが，ここを読めば確実に解けるレベルの情報を提供する．

**q9–q17 は「探索型」の演習である．** 正解のパラメータは与えられず，仮説を立てて何度も試し，記録し，最適な設定を自分で見つける．そのため本資料では，

- **第0章**：探索の共通の作法（検証とテストの分離，差が偶然かの見分け方，粗→細の探索，実験前の予測，記録帳，LLM への指示書の書き方）
- **第2〜9章の各末尾**：その回の探索に必要な知識（探索軸の意味，桁の目安，落とし穴，予測を立てるための問い，指示書に書くべき知識）

を追加している．**探索の結果（最適なパラメータの値）は書いていない**．それはあなたが試行錯誤で見つけるものである．

---

# 第0章　試行錯誤の作法（q9–q17 共通）
***
※ 対応演習: [q9.ipynb](../q9.ipynb) 〜 [q17.ipynb](../q17.ipynb)

## 目次
1. なぜ「探索」を学ぶのか
2. 探索のサイクル：仮説 → 試行 → 記録
3. 検証とテストの分離（探索するほどスコアは楽観的になる）
4. 差が「偶然」かどうかを見分ける
5. 探索の設計：粗く → 細かく，端に張り付いたら広げる
6. 実験前の予測を書く理由
7. 記録帳 `Lab` の仕組み
8. LLM への指示書の書き方

---

## 1. なぜ「探索」を学ぶのか

q9–q17 では，**「正解のパラメータ」は与えられない**．`alpha` も `max_depth` も `lr` も，自分で何度も試して見つける．

コードは AI（LLM）に書かせられる時代である．しかし LLM は次のことを**あなたに代わって決められない**．

| 人が決めること | 例 |
|---|---|
| 何を最大化するか（目的・評価指標） | 正解率か，Recall を保った上での Precision か |
| どう測れば騙されないか（検証の設計） | テストデータは最後に1回だけ，CV で選ぶ |
| どこを探すか（探索範囲と刻み） | `alpha` は桁で振る，`k` は 2〜8 |
| いつ止めるか（採用基準） | 差が偶然の範囲なら単純な方を選ぶ |
| 出力が正しいかを見抜くこと | ベストが探索範囲の端にないか，テストを見て選んでいないか |

この5つは，**自分で試行錯誤した経験がないと持てない知識**である．q9–q17 の狙いは，その経験を積み，最後に「LLM に渡せる指示書」として書き出せるようにすることにある．

## 2. 探索のサイクル：仮説 → 試行 → 記録

```
①仮説を立てる → ②試す → ③記録する → ④結果を見て仮説を直す → ①へ戻る
 「alpha を上げると       1行の記録が       「外れた．なぜ？」
  テストが上がるはず」     自動で残る
```

**仮説を書く**ことが重要である．やみくもな総当たりでは，結果が出ても「なぜそうなったか」が残らない．仮説が外れたとき，そこに学びがある．

記録に残すもの：

| 項目 | 理由 |
|---|---|
| 試行番号 | 後で「試行 #7 では〜」と引用できる |
| 設定（パラメータ） | 再現するため |
| 検証スコア | 比較するため |
| ばらつき（`cv_std` など） | 差が偶然かを判断するため |
| 補助指標（訓練スコア，gap，所要時間） | 過学習やコストを読むため |
| 仮説（1行） | 何を確かめる試行だったかを残すため |

## 3. 検証とテストの分離（探索するほどスコアは楽観的になる）

データは3つの役割に分ける．

| 役割 | 使い道 | 使ってよい回数 |
|---|---|---|
| 訓練 | モデルの学習 | 何度でも |
| 検証（CV など） | 設定の比較・選択 | 何度でも（ただし後述の注意あり） |
| テスト | 最終的な性能の報告 | **1回だけ** |

**なぜ検証スコアで選ぶだけでも楽観的になるのか**．設定を何十通りも試すと，偶然スコアが高く出た設定が「最良」として選ばれやすい．次のコードは，**完全にノイズのデータ**でこれを再現する．

```python
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression

rng = np.random.RandomState(0)
X = rng.randn(200, 50)                   # 50 個の特徴量（すべてノイズ）
y = rng.randint(0, 2, 200)               # ラベルもランダム（本当は予測できない）
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)

best_cv, best_cols = -1, None
for trial in range(200):                 # ランダムな特徴量の組合せを 200 通り「探索」する
    cols = rng.choice(50, 5, replace=False)
    cv = cross_val_score(LogisticRegression(), X_tr[:, cols], y_tr, cv=5).mean()
    if cv > best_cv:
        best_cv, best_cols = cv, cols

final = LogisticRegression().fit(X_tr[:, best_cols], y_tr)
print(f"探索で見つけた最良の CV 正解率 = {best_cv:.3f}")                          # 0.5 より明らかに高く見える
print(f"テスト正解率（1回だけ）       = {final.score(X_te[:, best_cols], y_te):.3f}")  # 0.5 付近に戻る
```

探索を重ねた検証スコアは，本来の性能より高く出る．だからこそ**テストデータは探索に一切使わず，最後に1回だけ**評価する．テストを見てから設定を変えると，テストも「検証」になってしまい，この楽観バイアスから逃れられなくなる（データリーク）．

> q9–q17 の記録帳は，`final_test()` を **2回呼ぶとエラー**になる．これは「テストを見て選び直す」誘惑を仕組みで防ぐためである．

## 4. 差が「偶然」かどうかを見分ける

CV の 5 回の分割で出るスコアには**ばらつき**がある．平均だけでなく標準偏差（`cv_std`）を見る．

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
cv5 = StratifiedKFold(5, shuffle=True, random_state=0)

for depth in (2, 3, 5, None):
    s = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=0), X, y, cv=cv5)
    print(f"max_depth={str(depth):>4}  平均={s.mean():.4f}  標準偏差={s.std():.4f}")
```

判断の目安：

- 2つの設定の**平均の差が標準偏差より小さい** → 「差がある」とは言えない（偶然かもしれない）
- 僅差のときは，**乱数シードを変えて繰り返す**（`random_state` や `torch.manual_seed`）か，`RepeatedStratifiedKFold` で分割を増やして確かめる
- 差が偶然の範囲なら，**最良から標準偏差以内で最も単純なモデルを選ぶ**（**1標準偏差ルール**）．単純なモデルの方が過学習しにくく，解釈もしやすく，計算も軽いから

## 5. 探索の設計：粗く → 細かく，端に張り付いたら広げる

**① 桁で振る（粗探索）**．学習率や正則化の強さは，値が10倍違うだけで挙動が大きく変わる．線形に 0.1, 0.2, 0.3 … と振るのではなく，**対数スケール**で振る．

```python
import numpy as np
print(np.logspace(-3, 3, 7))    # 0.001, 0.01, 0.1, 1, 10, 100, 1000（桁で広く）
```

**② 良さそうな範囲を細かく（絞り込み）**．粗探索で山が見えたら，その周辺を細かく調べる．

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

X, y = load_diabetes(return_X_y=True)
def cv_r2(alpha):
    return cross_val_score(Lasso(alpha=alpha, max_iter=10000), X, y, cv=5, scoring="r2").mean()

coarse = np.logspace(-3, 3, 7)                                # 粗探索
scores = [cv_r2(a) for a in coarse]
a_best = coarse[int(np.argmax(scores))]
print("粗探索のベスト:", a_best, "／ 探索範囲の端か:", a_best in (coarse[0], coarse[-1]))   # 端なら範囲を広げて確かめる
fine = np.logspace(np.log10(a_best) - 1, np.log10(a_best) + 1, 9)   # ベスト周辺を細かく
for a in fine:
    print(f"alpha={a:8.4f}  CV R2={cv_r2(a):.4f}")
```

**③ ベストが探索範囲の端に来たら，範囲を広げる**．端がベストということは，「その先にもっと良い所がある」かもしれない．逆に，**端まで行っても頭打ちなら**，そこが答えの近くである．

**④ 頭打ちの判断**．ベスト更新の推移が横ばいになり，上位設定の差が `cv_std` 以内になったら，探索を止めてよい．探索には計算予算がある．**1試行の時間 × 試行回数**を見積もり，粗探索は少ないデータ・少ないエポックで回す，といった工夫も有効である．

**⑤ 1軸ずつか，同時か**．軸が多いときは，まず1軸ずつ動かして「効く軸」を見つけ（**効果の分解**），効く軸に予算を集中する．軸同士に相互作用があるとき（例：学習率を下げたら木の本数を増やす）は，組合せて確かめる．

## 6. 実験前の予測を書く理由

試す前に「最良は alpha≈○○，理由は△△」と**予測を書いて固定する**．結果が出てから「やっぱりそう思っていた」と後付けしてしまうのを防ぐためである．

- 予測が当たった → その知識は使える
- 予測が外れた → **どの前提が間違っていたか**を特定する．ここが最大の学習機会である

LLM に予測を聞いてもよい．そのときは LLM の回答をそのまま記録し，**LLM の予測と実験結果のズレ**も分析する．LLM が外す場所（そのデータ固有の性質）こそ，あなたが持つべき知識である．

## 7. 記録帳 `Lab` の仕組み

q9–q17 の最初のコードセルには `Lab` クラスが入っている．中身は次のような小さな仕組みである（読んで理解しておくこと）．

```python
class MiniLab:
    """試行を1行ずつ溜める最小の記録帳"""
    def __init__(self):
        self.rows = []

    def log(self, hypothesis, params, score, **extra):
        if any(r["params"] == params for r in self.rows):      # 同じ設定は重複記録しない
            print("既に試した設定です:", params); return
        self.rows.append(dict(trial=len(self.rows) + 1, hypothesis=hypothesis,
                              params=params, score=score, extra=extra))

    def best(self):
        return max(self.rows, key=lambda r: r["score"])

lab = MiniLab()
lab.log("alpha=1 を基準にする", {"alpha": 1.0}, 0.42, cv_std=0.05)
lab.log("alpha を10倍にする",   {"alpha": 10.0}, 0.47, cv_std=0.04)
print(lab.best()["params"])
```

実際の `Lab` は，これに次の機能が加わっている．

| 機能 | 内容 |
|---|---|
| `lab.df()` | 全試行を表（DataFrame）にする |
| `lab.plot(x=..., group=...)` | 設定とスコアの関係，ベスト更新の推移を描く |
| `lab.report()` | 探索の自己診断（回数・範囲・端に張り付いていないか・仮説を書いたか） |
| `lab.final_test(fn)` | テストの最終評価。**1回だけ**（2回目はエラー） |
| `lab.save()` | 全履歴を CSV に保存 |

## 8. LLM への指示書の書き方

q9・q13・q17 の最終問では，得た知識を **LLM がそのまま使える指示書**に書き起こす（負担を考えて3回に絞っている．他の回でも，自分のログを読み返して「次の人に何を伝えるか」を考えると良い）．良い指示書の条件は，「この課題を初めて見る別の人（や LLM）が，**あなたの試行錯誤をやり直さずに**最適設定へ最短で到達できる」ことである．

**指示書に入れる6項目**

| 項目 | 書くこと | 悪い例 → 良い例 |
|---|---|---|
| 目的と指標 | 何を最大化し，なぜその指標か | 「精度を上げて」→「CV 正解率を最大化．テストは最後に1回だけ．理由：探索でテストを使うと楽観的になる」 |
| データの事実 | 件数・特徴量数・クラス比・スケール | 「表データです」→「569 件・30 特徴量・良性 63%．CV の標準偏差は約 0.02」 |
| 探索計画 | 軸・範囲・刻み・順序と，その根拠 | 「いろいろ試して」→「learning_rate は 0.01〜0.5 を対数で．根拠：試行 #5 で 0.5 が端だったので広げた」 |
| 落とし穴 | 自分が実際に見た罠と見抜き方 | 「過学習に注意」→「max_depth=50 で train 1.0 / CV 0.93．gap が 0.05 を超えたら過学習と判断」 |
| 採用基準・止めどき | いつ止めるか，僅差のときの選び方 | 「良さそうなら終了」→「上位が cv_std 以内なら最も単純なものを採用．ベスト更新が5回連続で無ければ終了」 |
| 検証方法 | LLM の出力をどの数値で信じる／疑うか | 「確認して」→「テストを使っていないか，ベストが範囲の端でないか，記録した CV が再現できるかを確認」 |

**LLM の出力を採点するチェックリスト**

- [ ] テストデータを探索に使っていないか（`test` が探索ループに現れていないか）
- [ ] 探索範囲が桁で広く取られているか．ベストが端に来ていないか
- [ ] 差が `cv_std` 以内なのに「最良」と断定していないか
- [ ] 提案された数値を，**自分のログで再現・反証できるか**（試行番号で示せるか）
- [ ] 前処理（標準化・分割）が CV の内側で行われているか（リークしていないか）

LLM が正しく答えたら○，違ったら×を，**自分のログを根拠に**書く．これが「知識を得て LLM にインプットできる」ことの証拠になる．

---

# 第1章　前処理と Pipeline
***
※ 対応演習: [q8.ipynb](../q8.ipynb)

## 目次
1. データの読み込みと確認
2. 欠損値の補完
3. スケーリング手法の選択
4. ColumnTransformer
5. Pipeline の構築
6. データリーク（leakage）の本質

---

## 1. データの読み込みと確認

機械学習では，**モデルを作る前に必ずデータの状態を確認する**のが大原則である．

### なぜ確認が必要か

型のミスマッチ（文字列が混入した数値列）や欠損値（NaN）を知らないままモデルに渡すと，`ValueError: could not convert string to float` などのエラーで止まる．しかも，エラーが出ずに「おかしな精度」のまま学習が完了してしまうケースもあり，後から原因を追うのが大変になる．

C言語で言えば，配列にアクセスする前にサイズと中身を確認するのと同じ感覚である．

### 確認するべき主な項目

| メソッド | 確認内容 | 読み方 |
|:--|:--|:--|
| `head(n)` | 先頭 n 行のサンプル | 値の範囲・形式を目視 |
| `info()` | 列の型（dtype）と non-null count | 0以外のdtype（objectが数値列に混入）や non-null が行数より少ない列が危険 |
| `isnull().sum()` | 列ごとの欠損数 | 0以外が出たら補完が必要 |
| `describe()` | 数値列の統計量（min/max/mean/std） | 極端な外れ値の検出に有効 |
| `value_counts()` | カテゴリ列の頻度 | 表記ゆれ（"Male" と "male"）の発見 |

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TITANIC_URL = "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/titanic/titanic.csv"

df = pd.read_csv(TITANIC_URL)
cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
df_titanic = df[cols].copy()

print(df_titanic.head())        # 先頭5行で値を目視
print()
df_titanic.info()               # 各列の型と non-null count を確認
print()
print("欠損数:")
print(df_titanic.isnull().sum())   # Age に欠損が多いことがわかる
# → 出力例: Age    177  ← 177件が NaN
print()
print("統計サマリー:")
print(df_titanic.describe())       # Fare の最大値が512と大きい（外れ値）
```

---

## 2. 欠損値の補完

欠損値（NaN）がある列は，そのままでは多くの機械学習アルゴリズムに渡せない．

### SimpleImputer の戦略比較

`SimpleImputer` の `strategy` パラメータには4種類ある．

| strategy | 補完値 | 適した列・状況 |
|:--:|:--|:--|
| `"mean"` | 列の平均値 | 正規分布に近い数値列 |
| `"median"` | 列の中央値 | **外れ値がある数値列**（Ageなどに最適） |
| `"most_frequent"` | 最頻値 | カテゴリ列や二値列 |
| `"constant"` | `fill_value` で指定した固定値 | 「不明」という意味を持つ列 |

**なぜ Age は median が良いか**: タイタニックの Age は上位クラスの乗客（高齢者が多い）などで外れ値が発生しやすい．mean は外れ値に引っ張られるが，median は外れ値の影響を受けない．

### fit と transform が分かれている理由

`SimpleImputer` には `fit()` と `transform()` の2段階がある．これは意図的な設計である．

- `fit(X_train)` … 訓練データだけを使って「補完に使う統計量（中央値など）」を計算・記憶
- `transform(X_test)` … 記憶した統計量で別データを変換

**なぜ分けるか**: テストデータで `fit` し直すと，「テストデータの統計量がモデルに漏れる（データ漏洩）」ことになる．訓練データで計算した統計量を「定数」として保存し，テストデータにも同じ変換を適用する必要がある．

```python
from sklearn.impute import SimpleImputer

print("補完前 Age の欠損数:", df_titanic["Age"].isnull().sum())
# → 177

imputer = SimpleImputer(strategy="median")   # 中央値で補完するインスタンスを生成

# fit_transform: 訓練データでのみ中央値を計算し，同時に変換する
age_filled = imputer.fit_transform(df_titanic[["Age"]])

print("補完後 Age の欠損数:", pd.isna(age_filled).sum())    # → 0
print("補完に使った中央値:", imputer.statistics_[0])         # → 28.0 付近
```

---

## 3. スケーリング手法の選択

### なぜスケーリングが必要か

異なる単位・スケールを持つ特徴量（例：Age は 0〜80，Fare は 0〜512）をそのままモデルに渡すと，スケールの大きい特徴量がモデルの挙動を支配してしまう．**正則化**（第2章）や **距離ベースのモデル**（k-means など）では特に影響が大きい．

### 主要なスケーリング手法の比較

| スケーラー | 変換式 | 特徴 | 使いどき |
|:--|:--|:--|:--|
| `StandardScaler` | `(x - μ) / σ` → 平均0・標準偏差1 | 正規分布に近くなる | **正則化あり**のモデル全般 |
| `MinMaxScaler` | `(x - min) / (max - min)` → [0, 1] | 範囲が明確になる | NN・画像・[0,1]が必要な場面 |
| `RobustScaler` | `(x - median) / IQR` | **外れ値に頑健** | 外れ値が多いデータ |

**直感**: `StandardScaler` は「偏差値変換」のようなもの．平均を 50，標準偏差を 10 にする偏差値と同じ発想である（ただしスケールは異なる）．

```python
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# 3つのスケーラーの比較
fare_sample = np.array([[7.25], [71.28], [512.33], [0.0]])   # Fare の例（外れ値含む）

for name, scaler in [
    ("StandardScaler", StandardScaler()),
    ("MinMaxScaler",   MinMaxScaler()),
    ("RobustScaler",   RobustScaler()),
]:
    transformed = scaler.fit_transform(fare_sample)
    print(f"{name}: {transformed.ravel().round(2)}")
# StandardScaler: [-0.84 -0.18  1.84 -0.83]  ← 外れ値512が大きく突出
# MinMaxScaler:   [ 0.01  0.14  1.    0.  ]  ← 0〜1に収まる
# RobustScaler:   [-0.49  0.99  8.87 -0.52]  ← 外れ値の影響が小さい
```

---

## 4. ColumnTransformer

### なぜ ColumnTransformer が必要か

実際のデータには **数値列**（年齢・運賃など）と **カテゴリ列**（性別・乗客クラスなど）が混在する．それぞれに適した前処理が異なるため，`ColumnTransformer` で「列の種類ごとに異なる変換器を適用」する．

```
数値列 ──→ StandardScaler（スケールを揃える）
カテゴリ列 ──→ OneHotEncoder（数値に変換）
```

### OneHotEncoder vs OrdinalEncoder の選択

| エンコーダ | 変換例 | 使いどき |
|:--|:--|:--|
| `OneHotEncoder` | "male"→[1,0], "female"→[0,1] | **順序のない**カテゴリ（性別，色，都市名など） |
| `OrdinalEncoder` | "低"→0, "中"→1, "高"→2 | **順序のある**カテゴリ（低・中・高，月・火・水など） |

### `drop="first"` が必要な理由

`Sex` を OneHot 化すると `Sex_male` と `Sex_female` の2列が生まれる．しかし，`Sex_male = 1 - Sex_female` であり，片方が決まればもう片方は自動的に決まる（**多重共線性**）．線形モデルはこの冗長な列に悩まされるため，`drop="first"` で片方を落とす．ツリーベースモデルでは多重共線性の問題が小さいため `drop=None` でも大丈夫なことが多い．

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features    = ["Pclass", "Age", "SibSp", "Parch", "Fare"]   # 数値列
categorical_features = ["Sex"]                                        # カテゴリ列

preprocessor = ColumnTransformer(
    transformers=[
        # "num" という名前で数値列に欠損補完→標準化を適用
        ("num", SimpleImputer(strategy="median"), numeric_features),
        # "cat" という名前でカテゴリ列にOneHot（冗長列削除）を適用
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
    ]
)

X = df_titanic.drop("Survived", axis=1)
y = df_titanic["Survived"]

X_transformed = preprocessor.fit_transform(X)
print("変換後の shape:", X_transformed.shape)
# → (891, 6) ← 数値5列 + Sex_male 1列（female が落ちた）
```

---

## 5. Pipeline の構築

前処理とモデル学習を**一つのオブジェクトにまとめる**のが `Pipeline` である．

### Pipeline を使わないとどうなるか（データリークの例）

```python
# NG: Pipeline なしの危険なコード（データ漏洩が起きる）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # OK: 訓練データで fit
X_test_scaled  = scaler.fit_transform(X_test)    # ← NG! テストデータで再 fit → 漏洩

# この時点で scaler は X_test の平均・分散を記憶してしまっている
# → テストデータの「性質」がスケーリングを通じてモデルに漏れている
```

テストデータの統計量が前処理に混入し，「未知データに対する性能を正しく測れない」問題が起きる．

### Pipeline の内部動作

```
fit(X_train, y_train) 時:
    X_train → [前処理.fit_transform] → [モデル.fit] → 学習完了

predict(X_test) 時:
    X_test  → [前処理.transform のみ！] → [モデル.predict] → 予測

※ predict 時は前処理の fit は一切行わず，訓練時に保存した統計量で変換する
```

### Pipeline の利点まとめ

- `fit` を1回呼ぶだけで全ステップが順番に実行される
- `predict` 時も同じ前処理が自動で適用される（`transform` を書き忘れない）
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
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ],
    )),
    ("model", DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=0)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

pipe.fit(X_train, y_train)          # 前処理 + 学習を一発で実行
y_pred = pipe.predict(X_test)       # テストデータにも同じ前処理が自動で適用される
print("テスト正解率:", accuracy_score(y_test, y_pred))
# → 0.78 付近
```

---

## 6. データリーク（leakage）の本質

**データリーク**とは，「本来テスト時には知り得ない情報」が訓練プロセスに混入することである．

### よくあるリークのパターン

| パターン | 例 | 問題 |
|:--|:--|:--|
| **前処理のリーク** | テストデータで `fit_transform` を使う | テストの統計量がモデルに漏れる |
| **目的変数のリーク** | 目的変数と高相関な「未来の特徴量」を使う | 実運用では使えない情報が混入 |
| **時系列のリーク** | 時系列データをランダム分割する | 未来のデータで過去を予測する状態になる |

リークがあると「訓練では良いスコアが出るが，実運用で性能が落ちる」という典型的な失敗になる．

---

## つまずきやすいポイント（第1章）

- **`fit_transform` をテストデータに使う** → `transform` のみを使う．`fit_transform` は訓練データ専用
- **`ColumnTransformer` に渡す列名の順番** → `fit` 時と `transform` 時で同じ列リストを使わないと列がずれる
- **数値列に `SimpleImputer` → `StandardScaler` の順** → 欠損値があるままスケーリングすると NaN が伝播する

### よくある質問

**Q: `StandardScaler` を忘れても決定木は動くのに，なぜ必要？**
A: 決定木はスケール不変だが，`LogisticRegression` や `Ridge` はスケールに敏感で，正則化の効果がスケールによって不均一になる．「どのモデルでも標準化する」習慣をつけると安全．

**Q: `OneHotEncoder(sparse_output=False)` の `False` は何？**
A: デフォルトでは変換結果がメモリ効率の良いスパース行列になる．`False` にすると通常の NumPy 配列になり，後続の処理でエラーが出にくくなる．

---

# 第2章　正則化
***
※ 対応演習: [q9.ipynb](../q9.ipynb)

## 目次
1. 過学習とバイアス・バリアンストレードオフ
2. 正則化の数学的基礎
3. Ridge / Lasso 回帰
4. ロジスティック回帰の正則化
5. 正則化強度 C の比較
6. ElasticNet と正則化パス
7. 探索の手引き（q9 対応）

---

## 1. 過学習とバイアス・バリアンストレードオフ

**過学習**（オーバーフィッティング）とは，モデルが訓練データに過度に適合し，未知データへの汎化性能が下がる状態である．

### バイアス・バリアンス分解

モデルの予測誤差は3つの要因に分解できる：

```
期待予測誤差 = Bias² + Variance + Noise（不可避な誤差）

Bias²    : モデルの表現力不足（単純すぎるモデル → underfitting）
Variance : モデルの訓練データへの過敏さ（複雑すぎるモデル → overfitting）
Noise    : データ自体のランダム性（どうしようもない誤差）
```

**正則化の役割**: モデルの複雑さにペナルティを課すことで，Variance を下げる代わりに Bias を少し上げる．この**バイアスとバリアンスのトレードオフ**を制御するのが正則化パラメータ（α や C）である．

```
高複雑さ  →  低Bias + 高Variance  →  過学習
低複雑さ  →  高Bias + 低Variance  →  未学習
正則化   →  Biasを少し上げてVarianceを大幅に下げる → 汎化性能UP
```

---

## 2. 正則化の数学的基礎

### 損失関数にペナルティを追加する

通常の線形回帰の損失関数（MSE）:
```
L(w) = (1/n) Σ(yᵢ - ŷᵢ)²
```

Ridge（L2正則化）はこれに「係数の二乗和」を追加：
```
L_Ridge(w) = (1/n) Σ(yᵢ - ŷᵢ)² + α * Σwⱼ²
```

Lasso（L1正則化）は「係数の絶対値和」を追加：
```
L_Lasso(w) = (1/n) Σ(yᵢ - ŷᵢ)² + α * Σ|wⱼ|
```

`α`（アルファ）が大きいほど，正則化が強くなり係数が小さく抑えられる．

### L1 がスパース解を生む幾何的直感

最適化の観点から見ると，「損失関数の等高線（楕円）」と「正則化の制約領域」が接する点が解になる：

```
L2（Ridge）の制約領域: 球（なめらか） → 等高線が球のどこかに接触 → 係数はほぼ0だが0にはならない

       w₂
       │   ● 損失の最小点（正則化なし）
       │  /
   ○──────○  ← L2の制約球（なめらかな境界）
  /    │    \   接触点 → 係数が少し縮む
 ○     │     ○
  \    └──────  w₁
   ○──────○

L1（Lasso）の制約領域: 菱形（角がある） → 等高線が角に接触しやすい → 係数が0になりやすい

       w₂
       │
       ◆  ← 角（w₁=0 or w₂=0）に等高線が接触しやすい！
      /│\
     / │ \
────◆──┼──◆──  w₁
     \ │ /
      \│/
       ◆
```

**直感**: Ridge は「全ての水道を細める」，Lasso は「不要な水道を閉める（一部を0にする）」．Lasso は**特徴量選択**の効果を持つ．

---

## 3. Ridge / Lasso 回帰

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

# 正則化の効果を正確に比較するために標準化必須
# （スケールが違うと係数の「大きさ」の意味が変わり，正則化の効果が不均一になる）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)      # ← テストデータは transform のみ

models = {
    "LinearRegression": LinearRegression(),     # alpha=0 と等価
    "Ridge(alpha=1.0)": Ridge(alpha=1.0),        # L2: 係数を均等に縮小
    "Ridge(alpha=100)": Ridge(alpha=100),        # 強い正則化: 係数がさらに小さく
    "Lasso(alpha=0.1)": Lasso(alpha=0.1),        # L1: 一部の係数が 0 になりやすい
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    coef = model.coef_[0] if hasattr(model, "coef_") else "N/A"
    print(f"{name}: MSE={mean_squared_error(y_test, pred):.3f}, "
          f"R2={r2_score(y_test, pred):.3f}, coef={coef:.3f}")
```

---

## 4. ロジスティック回帰の正則化

分類問題では `LogisticRegression` の `C` パラメータが正則化の強さを制御する．

### C の直感（注意：逆向き）

**C は正則化の「強さ」ではなく「逆数」**である．`Ridge` の `alpha` と向きが逆なので混同しやすい．

```
C = 1 / α  （α は正則化強度）

C が大きい（例: C=100） → α が小さい → 正則化が弱い → 複雑なモデル → 過学習しやすい
C が小さい（例: C=0.01） → α が大きい → 正則化が強い → シンプルモデル → 未学習になりやすい
C = 1 （デフォルト）  → 中程度の正則化
```

なぜ C という逆数設計にしたかというと，「C = Capacity（モデルの容量）」という直感から来ているとも言われる．

```python
from sklearn.linear_model import LogisticRegression

y_bin = (df_student["G3"] >= 10).astype(int)   # G3>=10 を「合格（1）」に2値化
X_cls = df_student[["G1", "G2", "studytime", "failures"]]

X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_bin, test_size=0.2, random_state=0, stratify=y_bin)

# 重要: 訓練データで fit してテストデータには transform のみ
ss = StandardScaler()
X_tr_s = ss.fit_transform(X_tr)
X_te_s = ss.transform(X_te)    # ← fit_transform ではなく transform

logreg = LogisticRegression(C=1.0, max_iter=1000)  # C=1.0 はデフォルト（中程度の正則化）
logreg.fit(X_tr_s, y_tr)
print("テスト正解率:", logreg.score(X_te_s, y_te))
```

---

## 5. 正則化強度 C の比較

`C` を変化させてテスト正解率がどう変わるか確認する．横軸は対数スケール（`plt.xscale("log")`）にすると，幅広い範囲を見やすい．

```python
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]   # 対数的に変化させる
train_scores, test_scores = [], []

for c in C_values:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_tr_s, y_tr)
    train_scores.append(model.score(X_tr_s, y_tr))
    test_scores.append(model.score(X_te_s, y_te))

plt.figure(figsize=(8, 5))
plt.plot(C_values, train_scores, marker="o", label="Train accuracy")
plt.plot(C_values, test_scores,  marker="s", label="Test accuracy")
plt.xscale("log")                           # ← 横軸を対数スケールに
plt.xlabel("C (larger = weaker regularization)")
plt.ylabel("Accuracy")
plt.title("Regularization strength C vs accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

best_idx = test_scores.index(max(test_scores))
print(f"最良の C: {C_values[best_idx]}，テスト正解率: {test_scores[best_idx]:.4f}")
```

**グラフの読み方**:
- C が極端に小さい: 両方のスコアが低い → **underfitting（未学習）**
- C が適切な値: テストスコアが最大
- C が極端に大きい: 訓練スコアは高いがテストスコアが下がる → **overfitting（過学習）**

---

## 6. ElasticNet と正則化パス

### ElasticNet: L1 と L2 の折衷

```python
from sklearn.linear_model import ElasticNet

# l1_ratio=0.5 なら L1 と L2 を半々に使用
# l1_ratio=1.0 なら Lasso と等価，l1_ratio=0.0 なら Ridge と等価
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train_scaled, y_train)
print("ElasticNet R²:", r2_score(y_test, enet.predict(X_test_scaled)))
```

### 正則化パス: α を変えたときの係数の変化

```python
from sklearn.linear_model import lasso_path

# Lasso のパス: α を大きくすると係数が次々と 0 になっていく
X_multi = df_student[["G1", "G2", "studytime", "failures", "absences"]]
X_m_s = StandardScaler().fit_transform(X_multi)

alphas, coefs, _ = lasso_path(X_m_s, y, alphas=np.logspace(-2, 1, 50))

plt.figure(figsize=(8, 5))
for i, name in enumerate(X_multi.columns):
    plt.plot(alphas, coefs[i], label=name)
plt.xscale("log")
plt.gca().invert_xaxis()      # 左から右に「正則化が弱→強」と読む
plt.xlabel("alpha (left = weak, right = strong)")
plt.ylabel("Coefficient")
plt.title("Lasso regularization path (coefficients shrink to 0 as alpha grows)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 7. 探索の手引き（q9 対応）

q9 では `alpha`（Ridge/Lasso）と `C`（ロジスティック回帰）を，自分で探索する．答えは書かないが，**探索を設計するために必要な知識**をここにまとめる．全体の作法は[第0章](#第0章試行錯誤の作法q9q17-共通)を参照．

### 7-1. 多項式展開で特徴量はいくつになるか

`PolynomialFeatures(degree=d)` は，元の特徴量 n 個から `C(n+d, d) - 1` 個（`include_bias=False`）の特徴量を作る．n=14 で d を上げると，特徴量の数は爆発的に増える．サンプル数（q9 の回帰データは 395 件）と比べて，**特徴量数がサンプル数を超えたとき何が起きるか**を，実験前の予測に使うこと．

```python
from math import comb
n = 14
for d in (1, 2, 3):
    print(f"degree={d}: 特徴量数 = {comb(n + d, d) - 1}")
```

### 7-2. `alpha` の探索範囲の決め方

- **必ず標準化してから**正則化する（`Pipeline` の中に `StandardScaler` を入れる．CV の各分割で標準化をやり直すため）
- `alpha` は**桁で**振る（第0章）．`Ridge` と `Lasso` では `alpha` の意味するスケールが違う．`Lasso` の目的関数は `RSS/(2n) + α·Σ|w|`（RSS をサンプル数 n で割る），`Ridge` は `RSS + α·Σw²`（割らない）．**同じ数値の `alpha` を両者に使っても，強さは同じではない**ので，モデルごとに範囲を確かめる
- `Lasso` で `alpha` を大きくしすぎると**係数が全部 0**になり，予測が「平均値」だけになる．このとき R² は 0 付近に張り付く．「探索範囲が広すぎて無意味な領域に入った」サインである
- **R² が負**になるのは「平均値を出すより悪い」ということ．過学習が激しいときに起きる

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=200, n_features=30, n_informative=5, noise=10, random_state=0)
Xs = StandardScaler().fit_transform(X)
for a in (0.01, 0.1, 1, 10, 100):
    nz = np.sum(np.abs(Lasso(alpha=a, max_iter=10000).fit(Xs, y).coef_) > 1e-8)
    print(f"Lasso alpha={a:>6}: 非ゼロ係数 {nz}/30")
```

### 7-3. `LogisticRegression` の `C` と `solver`

- `C` は `alpha` の**逆向き**（`C` 大 → 正則化弱）．探索は対数スケールで
- `penalty="l1"` を使うには `solver="liblinear"` か `"saga"` が必要
- **`liblinear` は切片（バイアス項）にも正則化をかける**．そのため，`C` を非常に小さくした L1 では，係数だけでなく切片まで 0 に縮み，**全サンプルが同じクラスに予測される**ことがある．そのとき正解率は「片方のクラスの比率」になる．「正解率が急に不自然な値になったら，全員同じ予測になっていないか」を疑うこと
- L2 と L1 で，同じ `C` でも意味する強さは違う．`penalty` ごとに `C` の範囲を確かめる

### 7-4. 特徴量グループを切り替えて効きを調べる（アブレーション）

「どの情報が予測に効くか」は，**特徴量のグループを外したり加えたりして CV スコアの変化を見る**ことで調べられる．

```python
# 例：グループを切り替えて CV 正解率を比べる（列名のリストを変えるだけ）
groups = {"A": ["f0", "f1"], "B": ["f2", "f3"]}
# for names in (["A"], ["B"], ["A", "B"]):
#     cols = [c for g in names for c in groups[g]]
#     score = cross_val_score(model, X[cols], y, cv=cv5).mean()
```

注意：グループを外して精度が落ちても，それは「**その情報が予測に効く**」（相関）であって，「**その要因が結果の原因である**」（因果）ではない．例えば「過去の落第回数が予測に効く」ことから「落第させれば成績が上がる／下がる」とは言えない．

### 7-5. 実験前に立てる予測の例（問いのみ）

- 特徴量数がサンプル数に近づくと，正則化なしの回帰の CV R² はどうなるか
- 最良の `alpha` は，`degree` を上げると大きくなるか，小さくなるか．それはなぜか
- `Ridge` と `Lasso` のどちらが，非ゼロ係数の数で違いが出るか

### 7-6. LLM 指示書に入れる知識（q9）

| 項目 | q9 で書くべき知識の例（自分のログで裏付けること） |
|---|---|
| 目的 | CV R²（回帰）／CV 正解率（分類）を最大化．テストは最後に1回 |
| データ | サンプル数と特徴量数（degree ごと），クラス比，標準化が必須なこと |
| 探索計画 | `alpha`・`C` は対数．モデルごとに範囲を分ける．`degree` は1つずつ上げる |
| 落とし穴 | Lasso の全係数ゼロ，liblinear の切片の縮み，R² が負 |
| 止めどき | 上位が `cv_std` 以内なら単純なモデル（次数が低い・非ゼロ係数が少ない）を採用 |

---

## つまずきやすいポイント（第2章）

- **`C` は正則化の「強さ」ではなく「逆数」** → `Ridge` の `alpha` とは逆向きなので注意
- **`StandardScaler` を忘れると正則化の効果が列ごとにバラつく** → 正則化は「係数の大きさ」にペナルティをかけるので，スケールが揃っていないと効果が不均一になる
- **`Lasso` で係数が 0 になる現象** → 意図して特徴量選択に使うこともできる（スパースモデル）

### よくある質問

**Q: Ridge と Lasso はどちらを使えばいいか？**
A: 特徴量が多くて「どれかは不要かもしれない」場合は Lasso（特徴量選択の効果がある）．全特徴量がある程度重要だと思われる場合は Ridge．迷ったら ElasticNet を試すのも手．

**Q: `max_iter=1000` はなぜ必要？**
A: LogisticRegression はデフォルト100回の反復で収束しないことがある（特に強正則化や特徴量が多い場合）．1000に増やしておくと "ConvergenceWarning" を防げる．

---

# 第3章　アンサンブル学習
***
※ 対応演習: [q10.ipynb](../q10.ipynb)

## 目次
1. アンサンブル学習の概要
2. Bagging と Random Forest
3. Gradient Boosting の仕組み
4. XGBoost — 仕組みと実装
5. XGBoost のハイパーパラメータ詳解
6. 特徴量重要度
7. 探索の手引き（q10 対応）

---

## 1. アンサンブル学習の概要

**アンサンブル学習**は，複数の弱いモデルを組み合わせて強いモデルを作る手法である．主に2つの戦略がある：

| 戦略 | 代表手法 | 学習方式 | 特徴 |
|:--|:--|:--|:--|
| **Bagging** | Random Forest | **並列** | 各モデルが独立に異なるデータで学習 |
| **Boosting** | GradientBoosting, XGBoost | **逐次** | 前のモデルの誤りを次のモデルが補完 |

---

## 2. Bagging と Random Forest

### なぜ複数の木を作ると精度が上がるか（数学的直感）

1本の決定木の予測を `f(x) + ε` と書く（`f(x)` は真の関数，`ε` はランダムな誤差）．

**Bagging の分散削減効果**:
- 1本の木の予測誤差の分散: `Var(ε) = σ²`
- N 本の木の平均予測の分散: `Var((1/N)Σεᵢ) = σ²/N`（各木が**独立**な場合）

→ 木の数 N を増やすほど分散が減り，安定した予測になる．

**Random Forest の工夫**:
- **ブートストラップサンプリング**: 訓練データをランダムにリサンプリング（重複あり）して各木を学習させ，木の多様性を確保
- **特徴量ランダム選択**: 各分岐で全特徴量でなく「ランダムに選んだ一部の特徴量」から最良の分割を探す → 木の相関を下げる

```
木1（データAで学習，特徴量のサブセット使用） → 予測a ─╮
木2（データBで学習，特徴量のサブセット使用） → 予測b ─┼─ 多数決 → 最終予測
木3（データCで学習，特徴量のサブセット使用） → 予測c ─╯
```

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df_rf = pd.read_csv(TITANIC_URL)
df_rf = df_rf[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].dropna()
df_rf["Sex"] = (df_rf["Sex"] == "male").astype(int)

X_rf = df_rf.drop("Survived", axis=1)
y_rf = df_rf["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=0, stratify=y_rf
)

rf = RandomForestClassifier(
    n_estimators=100,    # 100本の決定木（多いほど安定するが計算コストが上がる）
    max_features="sqrt", # 各分岐でランダムに選ぶ特徴量数（sqrt(特徴量数)が経験則）
    random_state=0,
)
rf.fit(X_train, y_train)
print("RandomForest 正解率:", accuracy_score(y_test, rf.predict(X_test)))
# → 0.82 付近
```

---

## 3. Gradient Boosting の仕組み

Gradient Boosting は Random Forest とは発想が逆で，**前のモデルが犯した誤差を次のモデルが学習する逐次学習方式**である．

### 残差フィッティング → 勾配の視点

**ステップ1**: まず定数モデル `F₀(x) = ȳ`（目的変数の平均）を作る

**ステップ2**: 残差（誤差）を計算する
```
rᵢ = yᵢ - F₀(xᵢ)   ← 予測しきれなかった誤差
```

**ステップ3**: この残差を目的変数として**新しい木 h₁** を学習する

**ステップ4**: モデルを更新する
```
F₁(x) = F₀(x) + η * h₁(x)   ← η（学習率）でステップを制御
```

**ステップ5**: 2〜4 を T 回繰り返す
```
Fₜ(x) = Fₜ₋₁(x) + η * hₜ(x)
```

**なぜ「残差フィッティング = 勾配」か**:
MSE 損失 `L = (1/2)(y - F)²` の `F` に対する負の勾配は：
```
-∂L/∂F = y - F = r（残差）
```
つまり，残差に向けてモデルを更新することは，損失関数を**勾配降下法**で最小化することと等価である．Gradient Boosting の名前はここから来ている．

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()
X_c, y_c = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X_c, y_c, test_size=0.2, random_state=0, stratify=y_c
)

gb = GradientBoostingClassifier(
    n_estimators=100,     # 木の数（多いほど精度が上がるが過学習リスクも上がる）
    max_depth=3,          # 各木の最大深さ（Boosting では浅い木が定石）
    learning_rate=0.1,    # 学習率 η（小さいほど慎重に，多くの木が必要）
    subsample=1.0,        # 各木に使う訓練データの割合（1.0 = 全部使用）
    random_state=0,
)
gb.fit(X_train, y_train)
print("GradientBoosting 正解率:", accuracy_score(y_test, gb.predict(X_test)))
```

---

## 4. XGBoost — 仕組みと実装

**XGBoost**（eXtreme Gradient Boosting）は，Gradient Boosting を高速化・高精度化したライブラリである．2016年の Kaggle 競技で多数の優勝チームが使用し，一躍有名になった．

### sklearn の GradientBoosting との違い

| 比較点 | sklearn GradientBoosting | XGBoost |
|:--|:--|:--|
| **勾配の計算** | 1次微分（勾配）のみ | **1次 + 2次微分**（より精密） |
| **正則化** | なし | **L1/L2 + 葉の数ペナルティ**（内蔵） |
| **木の刈り込み** | 深さ制限のみ | **gain < γ なら分割削除**（後ろ向き刈り込み） |
| **並列化** | 非対応（逐次） | **特徴量ソートを並列化**（高速） |
| **欠損値処理** | 手動補完が必要 | **自動的に欠損を処理** |
| **メモリ効率** | 普通 | **キャッシュ最適化**済み |

### 2次テイラー展開による目的関数の近似

XGBoost の核心は，損失関数 L を2次近似することにある：

```
目的関数 Obj(t) = Σ L(yᵢ, Fₜ₋₁(xᵢ) + hₜ(xᵢ)) + Ω(hₜ)

テイラー展開で2次近似:
  Obj(t) ≈ Σ [gᵢ * hₜ(xᵢ) + (1/2) * hᵢ * hₜ(xᵢ)²] + Ω(hₜ)

  gᵢ = ∂L/∂F  （損失の1次微分 = 勾配）
  hᵢ = ∂²L/∂F² （損失の2次微分 = ヘッセ行列）
```

**なぜ2次微分が役に立つか**: 1次微分だけでは「どの方向に進むか」しかわからないが，2次微分（曲率）があると「どれくらい進むか」も推定できる．これにより最適なステップサイズを計算できるため，収束が速く精度も上がる．

### XGBoost の正則化

```
Ω(h) = γT + (λ/2)Σwⱼ²

T  : 葉の数（木の複雑さに直接ペナルティ）
wⱼ : 各葉の値（出力）
γ  : 葉追加のペナルティ強度
λ  : L2 正則化の強度（デフォルト = 1）
```

分割の gain が `γ` 未満であれば，その分割は削除される（**後ろ向き刈り込み**）．sklearn の GradientBoosting は「分割してから深さ制限」するのに対し，XGBoost は「gain が小さければ分割しない」という**事前刈り込み**もできる．

### 実装

```python
# pip install xgboost  が必要（macOS では brew install libomp も必要な場合あり）
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,        # 木の数
    max_depth=6,             # 各木の最大深さ
    learning_rate=0.1,       # 学習率（shrinkage）
    subsample=0.8,           # 各木に使う訓練サンプルの割合
    colsample_bytree=0.8,    # 各木に使う特徴量の割合（過学習防止）
    reg_alpha=0.0,           # L1 正則化（Lasso 相当）
    reg_lambda=1.0,          # L2 正則化（Ridge 相当）
    gamma=0.0,               # 分割のための最小gain（刈り込み閾値）
    random_state=0,
    eval_metric="logloss",   # sklearn と組み合わせるときに警告抑制のため指定
)
xgb.fit(X_train, y_train)

print("GradientBoosting 正解率:", gb.score(X_test, y_test))
print("XGBoost 正解率:         ", xgb.score(X_test, y_test))
```

---

## 5. XGBoost のハイパーパラメータ詳解

適切なハイパーパラメータ設定が XGBoost の性能を大きく左右する．

### 主要パラメータの役割と調整指針

| パラメータ | デフォルト | 役割 | 調整のコツ |
|:--|:--|:--|:--|
| `n_estimators` | 100 | 木の数（多いほど精度UP・過学習リスクUP） | early stopping と組み合わせて自動決定 |
| `max_depth` | 6 | 各木の最大深さ | 3〜8が一般的．深いほど複雑 |
| `learning_rate` | 0.3 | 学習率 η（小さいほど慎重） | 小さく（0.01〜0.1）して `n_estimators` を増やすと精度UP |
| `subsample` | 1.0 | 各木に使うサンプル割合 | 0.6〜0.8で過学習防止 |
| `colsample_bytree` | 1.0 | 各木に使う特徴量割合 | 0.6〜0.9で多様性UP |
| `reg_alpha` | 0 | L1正則化（スパース係数） | スパース特徴量が多い場合に有効 |
| `reg_lambda` | 1 | L2正則化 | デフォルトで有効 |
| `gamma` | 0 | 分割の最小gain | 0.1〜1で軽い刈り込み |

### Early Stopping（過学習の自動検出）

`n_estimators` を大きく設定し，検証スコアが改善しなくなった時点で自動停止する：

```python
from sklearn.model_selection import train_test_split

# XGBoost の native API を使う場合（より細かい制御が可能）
X_tr2, X_val, y_tr2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

xgb_es = XGBClassifier(
    n_estimators=1000,          # 上限を大きく設定
    learning_rate=0.05,         # 学習率を小さく
    early_stopping_rounds=20,   # 20ラウンド改善しなければ停止
    eval_metric="logloss",
    random_state=0,
)
xgb_es.fit(
    X_tr2, y_tr2,
    eval_set=[(X_val, y_val)],  # 検証セットを指定
    verbose=False,
)
print("最適な木の数:", xgb_es.best_iteration)
print("Early Stopping後の正解率:", xgb_es.score(X_test, y_test))
```

### LightGBM との比較

| 比較点 | XGBoost | LightGBM |
|:--|:--|:--|
| **木の成長方針** | depth-wise（深さ優先） | **leaf-wise**（最大利得葉を優先） |
| **大規模データ** | 普通 | **高速**（ヒストグラム手法） |
| **精度** | 同等 | 同等〜やや高い場合も |
| **デフォルト性能** | 安定 | チューニング依存 |

---

## 6. 特徴量重要度

### 重要度の3種類

XGBoost では特徴量重要度に3種類ある：

| 種類 | 計算方法 | 意味 |
|:--|:--|:--|
| `weight` | 特徴量が分割に使われた回数 | よく使われた特徴量が高い |
| `gain` | 分割による損失の改善量（平均） | **最も信頼性が高い**。実際に重要な特徴量が高い |
| `cover` | 分割で影響を受けるサンプル数 | 多くのサンプルに効く特徴量が高い |

```python
feature_names = X_rf.columns
importances = rf.feature_importances_   # ← 合計が 1.0 になる相対的重要度

# 重要度の降順で並べ替えて表示
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 4))
plt.bar(
    [feature_names[i] for i in sorted_idx],
    importances[sorted_idx]
)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("RandomForest feature importance (descending)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("最重要特徴量:", feature_names[sorted_idx[0]])
# → "Fare" や "Sex" が上位に来ることが多い
```

### 重要度の限界と注意点

- **合計は 1.0**（相対的シェア）→ 0.5 が出ても「50%の確率で重要」ではない
- **相関する特徴量は重要度が分散**する（どちらを使っても同程度なら両方が中程度の値になる）
- **重要度が低い = 不要** とは限らない → 相関によって他の特徴量に重要度が「吸われている」可能性

より信頼性の高い解釈のために SHAP（SHapley Additive exPlanations）という手法もある（発展的内容）．

## 7. 探索の手引き（q10 対応）

q10 では `GradientBoostingClassifier` の `n_estimators`・`max_depth`・`learning_rate`・`subsample` を探索する．

### 7-1. 4つのパラメータの役割と目安

| パラメータ | 役割 | 探索の目安 |
|---|---|---|
| `learning_rate` | 各木の寄与率．大きいほど一気に学習（過学習しやすい） | 対数で 0.01〜0.5 |
| `n_estimators` | 木の本数．多いほど複雑 | `learning_rate` とセットで考える（次項） |
| `max_depth` | 各木の深さ．浅い木を多数組み合わせるのが基本．ただしデータが難しいと深めが効くこともある | 1〜8 |
| `subsample` | 各木の学習に使う訓練データの割合．1.0 未満だと**確率的勾配ブースティング**になり，木ごとに見るデータが変わって過学習を抑える | 0.5〜1.0 |

### 7-2. `learning_rate` と `n_estimators` はトレードオフ

学習率を小さくすると1本あたりの寄与が減るので，同じ精度に到達するには木が多く必要になる．探索では**別々に動かすのではなく，組み合わせて**確かめる．

### 7-3. 1回の学習で `n_estimators` の効果を見る（`staged_predict`）

勾配ブースティングは木を順番に足していくので，**最大本数で1回学習すれば，途中の本数での予測も全部取り出せる**．`n_estimators` を何通りも学習し直す必要はない．

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=2, random_state=0).fit(X_tr, y_tr)
val_acc = [accuracy_score(y_val, p) for p in gb.staged_predict(X_val)]   # 1本目〜300本目までの検証正解率
for n in (10, 50, 100, 200, 300):
    print(f"n_estimators={n:>3}: 検証正解率 = {val_acc[n - 1]:.4f}")
```

これは「探索コストを減らす工夫」の例である．**探索の前に，1回の学習で何がわかるかを考える**習慣をつける．

### 7-4. 正解率が「飽和」するデータでの比べ方

乳がんデータのように分けやすいデータでは，多くの設定で正解率が 0.95 前後に集まり，**差が `cv_std` より小さく**なりやすい（q10 の問題2では，これを避けるためにラベル誤り 8% の人工データを使っている．問題3の乳がんデータでは飽和が起きる）．そのとき：

- 正解率の代わりに**対数損失**（`scoring="neg_log_loss"`）で比べると，確信度の違いまで見えて差が出やすい
- 乱数分割を増やす（`RepeatedStratifiedKFold`）と，`cv_std` が小さくなって差が判断しやすくなる
- それでも差が偶然の範囲なら，**計算が軽く単純な設定**（浅い・本数が少ない）を選ぶ

```python
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
s = cross_val_score(GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0), X, y, cv=cv, scoring="neg_log_loss")
print(f"neg_log_loss 平均={s.mean():.4f}  標準偏差={s.std():.4f}")
```

### 7-5. 過学習の見方

`train_acc` が 1.0 なのに CV が低い（gap が大きい）のは典型的な過学習．ただしブースティングでは **`train_acc` が 1.0 に張り付いても CV が下がらない**ことがある．「訓練が 1.0 だから過学習」と決めつけず，**CV の推移で判断する**．

### 7-6. 実験前に立てる予測の例（問いのみ）

- `max_depth` を 1 → 5 と上げたとき，CV スコアは単調に上がるか
- `subsample=0.7` は，`subsample=1.0` より良くなるか．どのくらいの差なら「意味がある」と言えるか
- 最も効く軸は，4つのうちどれか

### 7-7. LLM 指示書に入れる知識（q10）

| 項目 | 書くべき知識の例 |
|---|---|
| 目的 | CV 正解率（または対数損失）最大化．テストは最後に1回 |
| データ | 件数・特徴量数・クラス比，`cv_std` の大きさ |
| 探索計画 | `learning_rate` は対数．`n_estimators` は学習率とセット．`staged_predict` で本数の探索を節約 |
| 落とし穴 | 正解率の飽和，train が 1.0 でも CV が下がらない場合がある |
| 止めどき | 上位が `cv_std` 以内なら軽くて単純な設定 |

---

## つまずきやすいポイント（第3章）

- **XGBoost は macOS で `brew install libomp` が必要な場合がある** → OpenMP が入っていないとインポートエラーになる
- **`feature_importances_` の合計は 1.0** → 相対的シェア
- **Gradient Boosting は学習率（`learning_rate`）と `n_estimators` がトレードオフ** → 学習率を下げて木の数を増やすと精度が上がりやすいが遅くなる
- **XGBoost の `eval_metric` を指定しないと警告が出る** → sklearn と組み合わせるときは `eval_metric="logloss"` を指定

### よくある質問

**Q: Random Forest と XGBoost はどちらを使えばいい？**
A: 一般的には XGBoost（や LightGBM）の方が精度が高い．ただし Random Forest はハイパーパラメータが少なくチューニングが簡単なため，ベースラインとして最初に試すのに向いている．

**Q: Gradient Boosting の `learning_rate` を小さくするとなぜ精度が上がるか？**
A: 学習率を小さくすると1回の更新量が小さくなり，より細かく誤差を修正できる（ただし木の数を増やす必要がある）．過学習しにくくなる効果もある．

---

# 第4章　教師なし学習
***
※ 対応演習: [q11.ipynb](../q11.ipynb)

## 目次
1. 教師あり学習 vs 教師なし学習
2. k-means クラスタリングの仕組み
3. クラスタ数の選択（エルボー法・シルエット法）
4. k-means の限界と DBSCAN
5. PCA による次元削減
6. t-SNE との比較
7. 2次元可視化
8. 探索の手引き（q11 対応）

---

## 1. 教師あり学習 vs 教師なし学習

| | 教師あり学習 | 教師なし学習 |
|:--|:--|:--|
| **正解ラベル** | 必要 | 不要 |
| **目標** | 入力から正解を予測 | データの隠れた構造を発見 |
| **評価** | 正解率・MSE など（客観的） | シルエットスコアなど（解釈が必要） |
| **代表的手法** | ロジスティック回帰，決定木 | k-means，PCA，オートエンコーダ |
| **用途例** | スパム分類，価格予測 | 顧客セグメント，異常検知，次元削減 |

---

## 2. k-means クラスタリングの仕組み

### アルゴリズムのステップ

```
Step 1: k 個のセントロイド（重心）をランダムに配置
Step 2: 各データ点を「最も近いセントロイド」のクラスタに割り当て
Step 3: 各クラスタの平均座標をセントロイドとして更新
Step 4: 割り当てが変わらなくなるまで Step 2〜3 を繰り返す（収束）
```

### 収束保証と局所解の問題

k-means は必ず収束するが，**大域最適解**に収束する保証はない．初期値が悪いと局所解（最適でないクラスタ配置）に陥る．

```
良い初期値の例:           悪い初期値の例:
○  ●    ■               ●●  ○   ■
  ●   ●      ■         ← 両方が密集した左エリアに
○       ■              集中してしまい，右を見逃す
```

対策として `n_init=10`（10種類の初期値から最良の結果を採用）と `init="k-means++"`（賢い初期値選択アルゴリズム）を使う．

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X_iris = StandardScaler().fit_transform(iris.data)   # k-means もスケールに敏感

kmeans = KMeans(
    n_clusters=3,           # k = 3（Iris は3種なので）
    init="k-means++",       # 賢い初期値選択（ランダムよりも良い収束保証）
    n_init=10,              # 初期値を変えて10回試行し，最良の結果を採用
    random_state=0,
)
labels = kmeans.fit_predict(X_iris)

print("クラスタラベル（先頭10件）:", labels[:10])
print("シルエットスコア:", silhouette_score(X_iris, labels))
# → 0.45 付近（1.0 に近いほど良い）
```

---

## 3. クラスタ数の選択（エルボー法・シルエット法）

k を事前に決める必要がある k-means では，2つの方法でクラスタ数を選ぶ：

### エルボー法（Elbow Method）

**WCSS**（Within-Cluster Sum of Squares: クラスタ内二乗和）を k ごとに計算する．k が増えるにつれ WCSS は減少するが，「**急激な減少が止まる点**（エルボー）」が最適な k の候補になる．

```
WCSS = Σ（各データ点 - 属するクラスタの重心）²

k が小さい: クラスタが粗い → WCSS が大きい
k が大きい: クラスタが細かい → WCSS が小さい
k = データ数: WCSS = 0 → 意味がない

エルボー（肘）: WCSSの減少率が急に鈍くなる点
```

```python
k_range = range(2, 8)
wcss_scores = []
sil_scores  = []

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0)
    pred = km.fit_predict(X_iris)
    wcss_scores.append(km.inertia_)                 # WCSS（クラスタ内二乗和）
    sil_scores.append(silhouette_score(X_iris, pred))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(list(k_range), wcss_scores, marker="o")
ax1.set_xlabel("k (number of clusters)")
ax1.set_ylabel("WCSS (within-cluster sum of squares)")
ax1.set_title("Elbow method: pick k where WCSS flattens")
ax1.grid(True, alpha=0.3)

ax2.plot(list(k_range), sil_scores, marker="o")
ax2.set_xlabel("k (number of clusters)")
ax2.set_ylabel("Silhouette score")
ax2.set_title("Silhouette method: pick k with the highest score")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

best_k = list(k_range)[sil_scores.index(max(sil_scores))]
print("シルエットスコアが最大の k:", best_k)
```

### シルエットスコアの数式

```
各データ点 i のシルエット係数:
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

  a(i): i と同じクラスタ内の他の点との平均距離（内部凝集度）
  b(i): i と最も近い隣接クラスタとの平均距離（外部分離度）

s(i) が 1 に近い → i は自分のクラスタに正しく属している（良い）
s(i) が 0 付近  → i はクラスタの境界にいる（曖昧）
s(i) が -1 に近い → i は誤ったクラスタに属している可能性（悪い）
```

---

## 4. k-means の限界と DBSCAN

k-means は**円形（球形）のクラスタ**を仮定しているため，以下のような形状には弱い：

```
k-means が苦手な形状:

三日月型:   ○○○○○
            ○     ○     ← k-means は直線境界しか引けない
            ○○○○○

ノイズ混入: ● ○ ●○● ●  ← 外れ値をどこかのクラスタに強制割り当て
```

**DBSCAN**（Density-Based Spatial Clustering of Applications with Noise）:
- k を指定しなくてよい
- 密度が高い領域をクラスタとして認識
- ノイズ（孤立点）を「外れ値」として扱える
- 月型・螺旋型などの複雑な形状に対応

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X_moon, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# k-means は月型に失敗する
km_moon = KMeans(n_clusters=2, random_state=0)
labels_km = km_moon.fit_predict(X_moon)

# DBSCAN は月型を正確に捉える
dbscan = DBSCAN(eps=0.3, min_samples=5)   # eps: 近傍半径, min_samples: 密度閾値
labels_db = dbscan.fit_labels_

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.scatter(X_moon[:,0], X_moon[:,1], c=labels_km, cmap="bwr", alpha=0.7)
ax1.set_title("k-means (fails on moons)")
ax2.scatter(X_moon[:,0], X_moon[:,1], c=labels_db, cmap="bwr", alpha=0.7)
ax2.set_title("DBSCAN (finds the moons)")
plt.tight_layout()
plt.show()
```

---

## 5. PCA による次元削減

**PCA**（主成分分析, Principal Component Analysis）は，高次元データを少数の軸（主成分）に圧縮する次元削減手法である．

### 数学的基礎

PCA は「データの**分散が最大になる方向**（= 情報量が最大の方向）」を第1主成分として選ぶ．

```
1. データを標準化（平均0，分散1に）
2. 共分散行列 C = (1/n) XᵀX を計算
3. C の固有値分解: C = VΛVᵀ
   V: 固有ベクトル（主成分の方向）
   Λ: 固有値（各方向の分散の大きさ）
4. 固有値が大きい順に k 個の固有ベクトルを選ぶ
5. X を k 次元に投影: Z = X V_k
```

**直感的説明**:
「写真を真上から撮った影（射影）が元の物体の形を最もよく表す方向がある」イメージ．PCA は「データのバラつきが最大になる方向（= 情報量が最大の方向）」を第1主成分として選ぶ．

```
高次元データ（例: 3次元）
        ●
      ●   ●
    ●       ●     ← この「伸びている方向」が第1主成分
      ●   ●
        ●
          ↓ PCA
          ━━━━━━━━  ← 情報を最も保持した2次元平面に投影
```

### 寄与率の解釈

`explained_variance_ratio_` は各主成分が「元のデータの情報のうち何%を保持しているか」を示す：

```
第1主成分の寄与率 = 第1固有値 / 全固有値の和

累積寄与率が 0.95 以上になるコンポーネント数を選ぶのが経験則
```

```python
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

wine = load_wine()           # ワインの化学成分データ（13次元, 3クラス）
X_wine = StandardScaler().fit_transform(wine.data)   # ← PCA 前に標準化必須

# まず全成分で PCA して寄与率を確認
pca_all = PCA()
pca_all.fit(X_wine)

# 累積寄与率のプロット
cumvar = pca_all.explained_variance_ratio_.cumsum()
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cumvar)+1), cumvar, marker="o")
plt.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance ratio")
plt.title("Variance kept by the first n components")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 2次元に圧縮（可視化のため）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_wine)

print("変換後 shape:", X_pca.shape)                             # → (178, 2)
print("第1主成分の寄与率:", pca.explained_variance_ratio_[0])   # → 0.36 付近
print("第2主成分の寄与率:", pca.explained_variance_ratio_[1])   # → 0.19 付近
print("累積寄与率（2成分）:", pca.explained_variance_ratio_.sum())  # → 0.55 付近
```

---

## 6. t-SNE との比較

| | PCA | t-SNE |
|:--|:--|:--|
| **手法の種類** | 線形変換 | 非線形変換 |
| **目的** | 次元削減 + 可視化 | **可視化専用** |
| **大域構造** | 保持する | 保持しない（距離スケールが意味を失う） |
| **局所構造** | 中程度 | **非常によく保持** |
| **速度** | 高速 | 低速（データ数 > 10,000 では特に） |
| **再現性** | 固定 | `random_state` 固定が必要 |
| **用途** | 前処理・特徴抽出・ノイズ除去 | クラスタの可視化のみ |

**重要な注意**: t-SNE の軸（次元）は意味を持たない．PCA とは違い「第1軸が最大分散方向」という解釈はできない．可視化ツールとしてのみ使う．

---

## 7. 2次元可視化

```python
plt.figure(figsize=(7, 5))
for label in np.unique(wine.target):
    mask = wine.target == label
    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        label=f"class {label} ({wine.target_names[label]})",
        alpha=0.7,
    )
plt.xlabel(f"PC1 (explained variance: {pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 (explained variance: {pca.explained_variance_ratio_[1]:.1%})")
plt.title("Wine dataset PCA (2D)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# → 3クラスがある程度分かれて見えるはず
```

**注意**: 色分けには正解ラベルを使っているが，**PCA の学習にはラベルを一切使っていない**．教師なし学習の評価として，「PCA の結果を正解ラベルで事後的に確認する」のは正当な手続きである．

## 8. 探索の手引き（q11 対応）

q11 では，**正解ラベル無し**でクラスタリングの設定（手法・クラスタ数・標準化・PCA）を探索する．「良し悪しをどう測るか」が最大の論点である．

### 8-1. ラベル無しで使える物差し

| 指標 | 何を測るか | 注意 |
|---|---|---|
| シルエットスコア | クラスタ内が近く，他クラスタと遠いほど高い（-1〜1） | **どの空間で測るかを固定する**．標準化の有無や PCA で空間が変わると，スコアは比べられない．ただし固定した空間（q11 では標準化した空間）で最適化した設定が有利になる偏りは残る |
| inertia | クラスタ内誤差の合計（k-means のみ） | k を増やせば**必ず**下がる．エルボー（折れ曲がり）を読むために使う |
| BIC / AIC（混合ガウス） | モデルの当てはまりと複雑さのバランス | 小さいほど良い．クラスタ数の選択に使える |
| ARI | 正解ラベルとの一致度 | 現実にはラベルが無い．**「答え合わせ」であって探索の物差しではない** |

ARI（や正解ラベル）を見てから手法を選ぶと，「ラベルが無いと使えない選び方」になってしまう．q11 で ARI を最後に1回だけ見るのはこのためである．

### 8-2. k-means 以外の手法

- **階層型クラスタリング**（`AgglomerativeClustering(linkage="ward")`）：近いものから順に併合していく．`inertia_` は無い
- **混合ガウス**（`GaussianMixture`）：クラスタをガウス分布の混合で表す．楕円形のクラスタも表現でき，`bic(X)` でクラスタ数を選べる

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(load_iris().data)
for k in (2, 3, 4):
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xs)
    wd = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
    gm = GaussianMixture(n_components=k, random_state=0).fit(Xs)
    print(f"k={k}: silhouette kmeans={silhouette_score(Xs, km):.3f}  ward={silhouette_score(Xs, wd):.3f}  "
          f"GMM の BIC={gm.bic(Xs):.1f}")
```

### 8-3. 標準化・PCA の効き方

- 距離を使う手法は，**スケールの大きい特徴量に引きずられる**．標準化の有無で結果が変わるかを，「標準化なし・あり」の1つずつの比較（効果の分解）で確かめる
- PCA で次元を減らしてからクラスタリングすると，ノイズが減って安定することも，情報が落ちて悪化することもある．**どちらになるかはデータ次第**なので，実験で確かめる
- シルエットは「はっきり離れた大きなまとまり」を高く評価する傾向がある．近接した群があるデータでは，**シルエット最大の k と，人が「意味がある」と思う k が食い違う**ことがある．食い違ったときに何を優先するかが，指示書に書くべき知識である

### 8-4. 実験前に立てる予測の例（問いのみ）

- 標準化を外すと，クラスタは変わるか．どの特徴量が支配的になるか
- シルエットが最大になる k は，実際の品種数と一致するか
- 手法（k-means・階層型・混合ガウス）で，シルエットに差は出るか

### 8-5. LLM 指示書に入れる知識（q11）

| 項目 | 書くべき知識の例 |
|---|---|
| 目的 | ラベル無しの物差し（シルエット，エルボー，BIC）で選ぶ．ARI は最後に1回の答え合わせ |
| データ | 件数・特徴量数・スケールの違い |
| 探索計画 | k の範囲，手法，標準化の有無，PCA の次元．1つずつ動かして効果を分解 |
| 落とし穴 | シルエットの測る空間を揃える，ラベルを見て選ぶリーク，シルエット最大＝意味のある k とは限らない |
| 止めどき | シルエットとエルボーが食い違うとき，どちらを優先し，何を追加で確認するか |

---

## つまずきやすいポイント（第4章）

- **k-means は初期値に依存する** → `n_init=10` と `init="k-means++"` を使う
- **PCA の前に `StandardScaler` が必要** → スケールが大きい特徴量が第1主成分を支配してしまう
- **k-means は「丸いクラスタ」を仮定している** → 非球形クラスタには DBSCAN 等が向く

### よくある質問

**Q: PCA で次元を減らしてからモデルに入れるのはいつ有効か？**
A: (1) 特徴量数が多くモデルが遅い場合，(2) 特徴量間の相関が強い場合（多重共線性の解消），(3) 可視化のため2〜3次元に落としたい場合．ただし木系モデル（RF, XGBoost）はそもそも高次元に強いため，PCA は必須ではない．

---

# 第5章　不均衡データと ROC
***
※ 対応演習: [q12.ipynb](../q12.ipynb)

## 目次
1. 不均衡データとは
2. 評価指標の選択（Precision / Recall / F1）
3. class_weight による重み付け
4. ROC 曲線と AUC
5. Precision-Recall 曲線
6. 不均衡データへの対処法まとめ
7. 探索の手引き（q12 対応）

---

## 1. 不均衡データとは

**不均衡データ**とは，クラス間のサンプル数に大きな偏りがあるデータである．

### 正解率だけでは誤魔化される問題

クラス0が 95%，クラス1が 5% のデータで，「全部クラス0と予測するだけ」のモデルを作ると：
- **正解率（accuracy）**: **95%**（一見高い！）
- **クラス1の Recall**: **0%**（1件も検出できていない！）

→ 不均衡データでは正解率は意味をなさない．**F1スコア・Recall・AUC** を必ず確認する．

### 実際の不均衡データの例

| 問題 | 少数クラス | 特徴 |
|:--|:--|:--|
| 詐欺検知 | 詐欺取引 | 0.01〜0.1% |
| 医療診断 | 陽性患者 | 1〜10% |
| 工場不良品検査 | 不良品 | 0.1〜5% |
| タイタニック | 生存者（38%） | 比較的バランスが取れている |

```python
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(
    n_samples=2000,
    n_features=20,
    weights=[0.95, 0.05],   # クラス0が95%, クラス1が5%
    random_state=0,
)

X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=0, stratify=y_imb
)

print("訓練データ:")
print(f"  クラス0: {(y_train == 0).sum()} 件（{(y_train == 0).mean():.1%}）")
print(f"  クラス1: {(y_train == 1).sum()} 件（{(y_train == 1).mean():.1%}）")
```

---

## 2. 評価指標の選択

### 混同行列と各指標の関係

```
              予測 0（Negative）   予測 1（Positive）
正解 0（N）  [  TN（真陰性）        FP（偽陽性）   ]  ← 正解0を正しく0と予測 / 誤って1と予測
正解 1（P）  [  FN（偽陰性）        TP（真陽性）   ]  ← 正解1を見逃し0と予測 / 正しく1と予測

Precision（適合率） = TP / (TP + FP)   ← 「陽性」と予測したうち本当に陽性の割合
Recall（再現率）    = TP / (TP + FN)   ← 本当の陽性のうち正しく検出できた割合
F1 スコア           = 2 * Precision * Recall / (Precision + Recall)   ← 調和平均
```

### 数値例（手計算）

```
予測結果: 正解[0,0,1,1,1], 予測[0,1,1,1,0]

混同行列:
  TN=1, FP=1   （正解0を2件→1件正解，1件誤検知）
  FN=1, TP=2   （正解1を3件→2件検出，1件見逃し）

Precision = 2 / (2 + 1) = 0.667
Recall    = 2 / (2 + 1) = 0.667
F1        = 2 * 0.667 * 0.667 / (0.667 + 0.667) = 0.667
```

### どの指標を重視するか

| 重視すべき場面 | 指標 | 理由 |
|:--|:--|:--|
| 見逃しが致命的（癌の見落とし，詐欺検知） | **Recall を高く** | FN を減らすことが最優先 |
| 誤検知が問題（スパムフィルタ，重要メールを消す） | **Precision を高く** | FP を減らすことが最優先 |
| バランスが取れているべき | **F1 スコア** | Precision と Recall の調和平均 |

---

## 3. class_weight による重み付け

`LogisticRegression` に `class_weight="balanced"` を指定すると，サンプル数が少ないクラスほど**誤りの損失を大きく**扱い，少数クラスを見逃しにくくする．

### class_weight の内部計算

```python
# sklearn の内部計算式
class_weight_value = n_samples / (n_classes * np.bincount(y))
# n_samples=1600, クラス0=1520, クラス1=80, n_classes=2 の場合:
# クラス0の重み = 1600 / (2 * 1520) ≈ 0.53
# クラス1の重み = 1600 / (2 * 80)   = 10.0  ← クラス1の誤りを10倍重視！
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

for cw in [None, "balanced"]:
    label = "no class_weight" if cw is None else "class_weight=balanced"
    model = LogisticRegression(max_iter=1000, class_weight=cw)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"=== {label} ===")
    print("混同行列:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
# → balanced にすると recall（再現率）が上がり，クラス1の見逃しが減る
# → ただし precision が下がることに注意（トレードオフ）
```

---

## 4. ROC 曲線と AUC

**ROC 曲線**は，分類器の「閾値」を 0→1 に変化させたときの FPR と TPR の軌跡を示すグラフである．

### 閾値変化とROCの関係

```
閾値 = 0.5（デフォルト）: 通常の予測
閾値 = 0.1（低い）: より多くを「陽性」と予測 → TPR↑, FPR↑
閾値 = 0.9（高い）: より少なくを「陽性」と予測 → TPR↓, FPR↓

ROC 曲線はこの閾値を 0→1 に動かした全ての点を結んだ線
```

### AUC の確率的解釈

**AUC = P（ランダムに選んだ陽性サンプルのスコア > ランダムに選んだ陰性サンプルのスコア）**

つまり，AUC は「正例と負例の2件をランダムに選んだとき，モデルが正例に高いスコアを付ける確率」である．

```
AUC = 1.0  → 完全な分類（正例スコアが必ず負例スコアより高い）
AUC = 0.5  → ランダムな予測と同等（対角線）
AUC < 0.5  → ランダムより悪い（予測が逆向き→ラベルを反転すれば良くなる）
AUC = 0.8 程度 → 実用的に良好
AUC = 0.9 以上 → 優秀
```

```python
from sklearn.metrics import RocCurveDisplay, auc, roc_curve

model_bal = LogisticRegression(max_iter=1000, class_weight="balanced")
model_bal.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_estimator(model_bal, X_test, y_test, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)")
ax.set_title("ROC curve")
ax.legend()
plt.show()

y_score = model_bal.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
print("AUC:", auc(fpr, tpr))
```

---

## 5. Precision-Recall 曲線

**PR 曲線**（Precision-Recall 曲線）は，ROC 曲線より**不均衡データに有効な評価指標**である．

### PR 曲線を使うべき理由

クラス1が非常に少ない（例: 5%）場合，ROC の AUC は楽観的になりがちである．なぜなら ROC は FPR（= FP / (FP + TN)）を使うが，TN が非常に多いため FPR が低く見える．

一方，PR 曲線は**クラス1の Precision と Recall を直接評価**するため，不均衡データでより現実的な性能評価ができる．

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_score)
ap = average_precision_score(y_test, y_score)   # Average Precision（AP）

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
# ベースライン（ランダム予測）: 少数クラスの割合に等しい水平線
baseline = y_test.mean()
plt.axhline(y=baseline, color="r", linestyle="--", label=f"Baseline (random): {baseline:.2%}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 6. 不均衡データへの対処法まとめ

| 手法 | 説明 | sklearn / imblearn |
|:--|:--|:--|
| `class_weight="balanced"` | モデルの損失関数に重みを付ける | sklearn の大半のモデルで対応 |
| オーバーサンプリング（SMOTE） | 少数クラスの合成サンプルを生成 | `imblearn.over_sampling.SMOTE` |
| アンダーサンプリング | 多数クラスのサンプルを減らす | `imblearn.under_sampling.RandomUnderSampler` |
| 閾値の調整 | predict の閾値（デフォルト0.5）を変更 | `predict_proba` + 手動閾値 |

```python
# 閾値を調整する例（閾値を下げると Recall が上がる）
y_proba = model_bal.predict_proba(X_test)[:, 1]

for threshold in [0.3, 0.5, 0.7]:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    print(f"閾値={threshold}: "
          f"Precision={precision_score(y_test, y_pred_thresh):.3f}, "
          f"Recall={recall_score(y_test, y_pred_thresh):.3f}, "
          f"F1={f1_score(y_test, y_pred_thresh):.3f}")
```

## 7. 探索の手引き（q12 対応）

q12 では「**Recall（見逃さない割合）が目標以上という条件で，Precision を最大にする**」設定を探索する．目的を**制約つきの最適化**として書く練習である．

### 7-1. なぜ「見逃しを減らす」だけではだめか

見逃しを減らすだけなら「全員を陽性と予測する」のが最善になってしまう（Precision は最悪）．現実の依頼は，`Recall ≥ 0.80 のとき Precision を最大にする` のように**制約つき**で書く．探索の物差し（目的関数）は次のように作る．

```python
RECALL_TARGET = 0.80
def objective(prec, rec):
    """Recall が目標を満たせば Precision，満たさなければ（負の）不足分"""
    return prec if rec >= RECALL_TARGET else rec - RECALL_TARGET
```

制約を満たさない設定を「負の値」にしておくと，探索が制約違反の設定を選ばず，しかも「どれだけ足りないか」も比べられる．

### 7-2. 閾値を動かすときは，学習し直さない（交差検証の予測 `cross_val_predict`）

`class_weight` や `C` は学習が必要だが，**閾値は学習済みの確率を切り直すだけ**で試せる．訓練データ内で「各サンプルが検証側に回ったときの予測確率」を1回作っておけば，閾値を何通り試しても再学習は不要である．

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split

X, y = make_classification(n_samples=3000, weights=[0.95, 0.05], class_sep=0.8, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

cv5 = StratifiedKFold(5, shuffle=True, random_state=0)
proba = cross_val_predict(LogisticRegression(max_iter=1000), X_train, y_train, cv=cv5, method="predict_proba")[:, 1]
for th in (0.5, 0.3, 0.2, 0.1):                  # 閾値だけ変える（再学習なし）
    pred = (proba >= th).astype(int)
    print(f"threshold={th}: precision={precision_score(y_train, pred, zero_division=0):.3f}  recall={recall_score(y_train, pred):.3f}")
```

### 7-3. `class_weight` の数式と閾値との関係

- `class_weight="balanced"` は，クラス c の重みを `n / (クラス数 × n_c)` にする．2クラスでは，**少数クラスの重みが多数クラスの何倍か**は `n_多数 / n_少数`（陽性が 5% なら約 19 倍）
- `class_weight={0: 1, 1: w}` と書けば，重みの倍率 `w` を自由に探索できる
- ロジスティック回帰では，重みを増やすことは，おおむね**予測確率（対数オッズ）を全体にずらす**効果があり，**閾値を下げること**と似た働きをする．ただし完全に同じではない（重みは学習される係数自体も変える）．「どちらが良いか」は実験で確かめる

### 7-4. 少数クラスの件数が少ないと，Recall は大きくぶれる

陽性が 40 件しかなければ，Recall の標準誤差は `√(p(1-p)/n)` で見積もると次のようになる．

```python
import numpy as np
n_pos, p = 40, 0.8
print(f"Recall={p} の標準誤差 ≈ {np.sqrt(p * (1 - p) / n_pos):.3f}")   # 約 0.06
```

つまり探索時に「Recall 0.82」で制約を満たしていても，**別のデータ（テスト）では 0.75 に下がって制約を破る**ことが普通に起きる．目標ぎりぎりの設定を選ばず，**余裕をもたせる**（例：探索では Recall≥0.85 を狙う）のが実務の知恵である．

### 7-5. 実験前に立てる予測の例（問いのみ）

- 「何もしない」設定の Recall はいくつくらいか．Precision はどうか
- `class_weight` を上げる方法と閾値を下げる方法で，同じ Recall のときの Precision はどちらが高いか
- `C`（正則化の強さ）は結果に影響するか

### 7-6. LLM 指示書に入れる知識（q12）

| 項目 | 書くべき知識の例 |
|---|---|
| 目的 | 制約つきの目的関数の式（Recall 下限と Precision 最大） |
| データ | 陽性の比率と件数，Recall のばらつきの大きさ |
| 探索計画 | 重みと閾値を別々→組合せ．閾値は `cross_val_predict` で再学習なしに |
| 落とし穴 | 全員陽性の自明解，Accuracy の罠，テストで閾値を調整するリーク，制約ぎりぎりの選択 |
| 止めどき | 制約を満たす設定の中で Precision が頭打ち．テストで制約が破れたときは選び直さず余裕を見直す |

---

## つまずきやすいポイント（第5章）

- **不均衡データで `accuracy_score` だけを見ると誤解を招く** → F1・Recall・AUC も合わせて確認
- **`decision_function` がないモデルでは `predict_proba` を使う** → `roc_curve(y_test, model.predict_proba(X_test)[:, 1])`
- **`stratify=y` を `train_test_split` に指定** → 不均衡データでは分割後もクラス比率を保つ必要がある

---

# 第6章　学習曲線とハイパーパラメータ
***
※ 対応演習: [q13.ipynb](../q13.ipynb)

## 目次
1. バイアス・バリアンス分解の復習
2. learning_curve
3. validation_curve
4. 過学習の診断と対処
5. GridSearchCV と RandomizedSearchCV
6. 交差検証の種類
7. 探索の手引き（q13 対応）

---

## 1. バイアス・バリアンス分解の復習

モデルの予測誤差を3つの要因に分解する考え方（第2章より）：

```
期待予測誤差（MSE）= Bias² + Variance + Noise

Bias²（偏り²）  : 真の関数 f(x) と予測の期待値 E[f̂(x)] のズレ
                   モデルが表現力不足（単純すぎ）→ 訓練スコアが低い
                   対策: モデルを複雑にする，特徴量を増やす

Variance（分散）: 訓練データが変わったときに予測がどれだけ変動するか
                   モデルが複雑すぎ → 訓練スコアは高いがテストスコアが低い
                   対策: 正則化，データ増量，モデルを単純にする

Noise（ノイズ） : データ自体のランダム性（どうしようもない誤差）
```

**学習曲線と validation_curve の目的**: Bias と Variance のどちらが問題かを診断する．

---

## 2. learning_curve

**学習曲線**（learning curve）は，「訓練データを増やしていったとき，モデルの性能がどう変わるか」を示すグラフである．

### 4パターンの見方と対処法

| パターン | 訓練スコア | 検証スコア | ギャップ | 診断 | 対処法 |
|:--|:--|:--|:--|:--|:--|
| **過学習** | 高い | 低い | 大きい | Variance が高い | 正則化，データ増量，特徴量削減 |
| **未学習** | 低い | 低い | 小さい | Bias が高い | モデルを複雑に，特徴量追加 |
| **良好** | 高い | 高い | 小さい | - | このまま使える |
| **データ不足** | 高い（右は収束） | 上昇中 | 中程度 | データが足りない | データ増量 |

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(random_state=0),
    X, y,
    cv=5,                                   # 5分割交差検証でスコアを計算
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10〜100%の訓練データ量で評価
    scoring="accuracy",
    n_jobs=-1,                              # 並列実行で高速化
)

train_mean = train_scores.mean(axis=1)      # 5分割の平均
train_std  = train_scores.std(axis=1)       # 5分割の標準偏差（ばらつき）
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Train score")
plt.fill_between(train_sizes,
                 train_mean - train_std,
                 train_mean + train_std, alpha=0.2)   # ← ばらつきの範囲を塗る
plt.plot(train_sizes, val_mean, marker="o", label="CV score")
plt.fill_between(train_sizes,
                 val_mean - val_std,
                 val_mean + val_std, alpha=0.2)
plt.xlabel("Training set size")
plt.ylabel("Accuracy")
plt.title("Learning curve (overfitting: large train/validation gap)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3. validation_curve

**validation_curve** は，「特定のハイパーパラメータを変化させたとき，モデルの性能がどう変わるか」を示すグラフである．

```python
from sklearn.model_selection import validation_curve

param_range = [1, 2, 3, 5, 10, 20, None]   # None = 制限なし

train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=0),
    X, y,
    param_name="max_depth",    # 変化させるパラメータ
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

x_labels = [str(v) if v is not None else "None" for v in param_range]
train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(x_labels, train_mean, marker="o", label="Train score")
plt.plot(x_labels, val_mean, marker="o", label="CV score")
plt.fill_between(range(len(x_labels)),
                 val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
plt.xlabel("max_depth (larger = more complex)")
plt.ylabel("Accuracy")
plt.title("validation_curve (find where overfitting starts)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 最良の max_depth を出力
best_idx = val_mean.argmax()
print(f"最良の max_depth: {x_labels[best_idx]}，CV正解率: {val_mean[best_idx]:.4f}")
```

---

## 4. 過学習の診断と対処

```python
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

results = []
for depth in [3, 5, 10, 20, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    tree.fit(X_train, y_train)
    cv_score = cross_val_score(tree, X_train, y_train, cv=5).mean()
    results.append({
        "max_depth": str(depth),
        "train_acc": tree.score(X_train, y_train),
        "test_acc":  tree.score(X_test, y_test),
        "cv_acc":    cv_score,
        "gap":       tree.score(X_train, y_train) - tree.score(X_test, y_test),
    })

df_result = pd.DataFrame(results)
print(df_result.to_string(index=False))
# gap が大きい = 過学習
```

---

## 5. GridSearchCV と RandomizedSearchCV

### GridSearchCV: 全パラメータの組み合わせを試す

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}
# 4 * 3 * 2 = 24 通り × 5-fold = 120 回の学習

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=0),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,    # 並列化
)
grid_search.fit(X_train, y_train)
print("最良パラメータ:", grid_search.best_params_)
print("最良 CV スコア:", grid_search.best_score_)
print("テスト正解率:", grid_search.score(X_test, y_test))
```

### RandomizedSearchCV: ランダムサンプリング（多パラメータに有効）

パラメータ数が多い場合，全組み合わせを試すと組み合わせ爆発が起きる．`RandomizedSearchCV` は指定した `n_iter` 回だけランダムに試行する：

```python
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": randint(2, 20),    # 2〜19の整数をランダムに
    "min_samples_leaf": randint(1, 10),
    "max_features": uniform(0.3, 0.7),      # 0.3〜1.0の実数をランダムに
}

random_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=0),
    param_dist,
    n_iter=50,      # 50回だけ試行
    cv=5,
    scoring="accuracy",
    random_state=0,
    n_jobs=-1,
)
random_search.fit(X_train, y_train)
print("最良パラメータ（RandomizedSearch）:", random_search.best_params_)
print("テスト正解率:", random_search.score(X_test, y_test))
```

---

## 6. 交差検証の種類

| 手法 | 説明 | 使いどき |
|:--|:--|:--|
| **k-fold CV** | データをk分割，k回検証 | 基本的なCV |
| **Stratified k-fold** | クラス比率を保ちながら分割 | **不均衡データ・分類問題** |
| **Leave-One-Out** | 1件ずつ検証 | データが非常に少ない場合 |
| **Time Series Split** | 時系列順に分割 | 時系列データ |

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Stratified k-fold（分類問題では通常こちらを使う）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

tree = DecisionTreeClassifier(max_depth=5, random_state=0)
scores = cross_val_score(tree, X, y, cv=skf, scoring="accuracy")
print(f"Stratified 5-fold CV: {scores.mean():.4f} ± {scores.std():.4f}")
# ± が大きいほどスコアがデータ分割に依存している（不安定）
```

## 7. 探索の手引き（q13 対応）

q13 では決定木（`max_depth`・`min_samples_leaf`・`ccp_alpha`・`criterion`）を探索し，最後に `GridSearchCV` と比べる．データは**ラベルの 15% をわざと反転させた乳がんデータ**（`load_noisy_breast_cancer()`）である．ラベルに誤りがあると，深い木は誤りまで暗記するため，複雑さの選び方が結果にはっきり表れる．下の例は手軽さのため元の乳がんデータで書いているが，q13 では同じことをノイズありのデータで行う．

### 7-1. 木の複雑さを制御する4つの手段

| パラメータ | 働き |
|---|---|
| `max_depth` | 木の深さの上限．小さいほど単純．`None` は制限なし（実質的には十分大きい整数でも同じ） |
| `min_samples_leaf` | 葉に必要な最小サンプル数．大きいほど細かい分岐ができず単純になる |
| `ccp_alpha` | **コスト複雑度枝刈り**．大きいほど枝を刈って葉の数を減らす |
| `criterion` | 分岐の良さの測り方（`gini` / `entropy`）．通常は大差ない |

`ccp_alpha` の有効な範囲はデータ依存なので，`cost_complexity_pruning_path` で候補を出すと桁の見当がつく．

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X, y)
print("ccp_alpha の候補（先頭 5 つと最後の 3 つ）:", path.ccp_alphas[:5].round(4), path.ccp_alphas[-3:].round(4))
```

### 7-2. gap で過学習を測る

`gap = 訓練スコア - 検証スコア`．gap が大きい → 過学習．gap が小さいのに検証スコアも低い → 未学習．**検証スコアの山と gap の開き始めは同じ深さとは限らない**ので，両方を見る．

### 7-3. 1標準偏差ルールの実装

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
cv5 = StratifiedKFold(5, shuffle=True, random_state=0)
rows = []
for depth in (1, 2, 3, 4, 5, 8):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0)
    s = cross_val_score(tree, X, y, cv=cv5)
    rows.append((depth, s.mean(), s.std()))
best = max(rows, key=lambda r: r[1])
threshold = best[1] - best[2]                                   # 最良から標準偏差以内
simplest = min((r for r in rows if r[1] >= threshold), key=lambda r: r[0])   # その中で最も浅い（単純な）もの
print("CV 最良:", best[0], "／ 1標準偏差ルールで選ぶ:", simplest[0])
```

### 7-4. 手動探索と `GridSearchCV`

| | 手動探索 | GridSearchCV |
|---|---|---|
| 長所 | 結果を見て次の一手を変えられる（粗→細）．仮説が残る | 網羅的．漏れがない．書くのが楽 |
| 短所 | 人手がかかる | **全組合せ×分割数**の学習が必要（`24 設定 × 5 分割 = 120 回`）．軸が増えると爆発する |

軸が多いときは `RandomizedSearchCV`（ランダムに抽出）が現実的である．**自動化しても，人が決めること**は残る：探索空間（何を・どの範囲で），評価指標，止めどき，リークの防止．これらが LLM への指示書に書く内容である．

### 7-5. 実験前に立てる予測の例（問いのみ）

- `max_depth` を上げていくと gap はどこから開き始めるか
- `min_samples_leaf` と `max_depth` のどちらが過学習に効くか
- 最良設定と「1標準偏差ルールで選ぶ設定」は，同じになるか

### 7-6. LLM 指示書に入れる知識（q13）

| 項目 | 書くべき知識の例 |
|---|---|
| 目的 | CV 正解率．gap も併せて監視 |
| データ | 件数，`cv_std` の大きさ（決定木は分割でぶれやすい） |
| 探索計画 | 深さ→葉の大きさ→枝刈りの順に．自動探索と手動探索の使い分け |
| 落とし穴 | 訓練精度 1.0 を信じない，僅差を「最良」と断定する |
| 止めどき | 1標準偏差ルールで単純なモデルを採用 |

---

## つまずきやすいポイント（第6章）

- **`learning_curve` と `validation_curve` の混同**: 横軸が全く違う（前者 = データ量，後者 = ハイパーパラメータ値）
- **`GridSearchCV` の `best_estimator_` はリフィット済み** → `grid_search.best_estimator_` で直接 predict できる
- **テストデータでハイパーパラメータを選択しない** → 必ず CV（訓練データ内で分割）でハイパーパラメータを決め，テストは最後に1回だけ使う

### よくある質問

**Q: learning_curve の train_sizes はどう解釈するか？**
A: `np.linspace(0.1, 1.0, 10)` は「訓練データの 10%, 20%, ..., 100%」を使ったときのスコアを計算する．左端でギャップが大きく右端で縮まっていく場合は「データを増やせば改善できる」サインである．

---

# 第7章　PyTorch 入門
***
※ 対応演習: [q14.ipynb](../q14.ipynb)

## 目次
1. PyTorch と Scikit-learn の違い
2. Tensor と基本操作
3. 自動微分（autograd）と計算グラフ
4. MNIST の読み込みと前処理
5. MNIST の可視化
6. DataLoader の理解
7. 探索の手引き（q14 対応）

---

## 1. PyTorch と Scikit-learn の違い

| | Scikit-learn | PyTorch |
|:--|:--|:--|
| **対象** | 表形式データ | 任意（特に画像・自然言語） |
| **学習ループ** | `fit()` 1行でOK | **自分で書く**（柔軟性と引き換え） |
| **勾配計算** | 内部処理（触れない） | **`backward()` を自分で呼ぶ** |
| **GPU対応** | 基本的に非対応 | `.to("cuda")` で簡単に対応 |
| **カスタムモデル** | 難しい | `nn.Module` で自由に設計 |

**使い分け**: 表形式データ + アンサンブル → Scikit-learn + XGBoost．画像・系列・カスタム構造 → PyTorch．

---

## 2. Tensor と基本操作

`Tensor` は PyTorch の基本データ構造で，NumPy の `ndarray` に似た多次元配列である．

```python
import torch

# 基本的な Tensor の作成
t1 = torch.tensor([1.0, 2.0, 3.0])           # リストから
t2 = torch.zeros(3, 4)                        # 全ゼロ
t3 = torch.ones(2, 3, dtype=torch.float32)   # 全1（dtype指定）
t4 = torch.randn(3, 3)                        # 標準正規分布

print("shape:", t1.shape)           # → torch.Size([3])
print("dtype:", t1.dtype)           # → torch.float32
print("device:", t1.device)         # → cpu

# NumPy との変換
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t_from_np = torch.from_numpy(arr)   # NumPy → Tensor
arr_back = t_from_np.numpy()        # Tensor → NumPy（GPU Tensorはまず .cpu() が必要）

# 形状変換
t5 = torch.arange(24).float()       # [0,1,2,...,23]
print(t5.reshape(4, 6).shape)       # → torch.Size([4, 6])
print(t5.view(2, 3, 4).shape)       # → torch.Size([2, 3, 4])（連続メモリが必要）
```

---

## 3. 自動微分（autograd）と計算グラフ

### 計算グラフとは

PyTorch は `requires_grad=True` の Tensor に対する演算を**計算グラフ**として記録する．`backward()` を呼ぶと，この計算グラフを**逆順に辿って勾配を計算**する（連鎖律の自動適用）．

```
順方向（forward）:
  x=5 → y = 2x² + 3 → y=53

計算グラフ（PyTorch が内部で構築）:
  x ─[×]─ x² ─[×2]─ 2x² ─[+3]─ y
         ↑grad_fn: PowBackward   ↑grad_fn: AddBackward

逆方向（backward）:
  dy/dy = 1
      ↓
  dy/d(2x²) = 1
      ↓
  dy/d(x²) = 2
      ↓
  dy/dx = 2 * 2x = 4x = 20  （x=5 のとき）
```

```python
# 自動微分の例
x = torch.tensor(5.0, requires_grad=True)   # ← このフラグが計算グラフを記録
y = 2.0 * x ** 2 + 3.0                      # 計算グラフが構築される

print("y =", y.item())                   # → 53.0
print("grad_fn:", y.grad_fn)             # → AddBackward（加算が最後の演算）

y.backward()                             # 逆伝播で勾配を計算

print("dy/dx =", x.grad.item())          # → 20.0（= 4x = 4*5）

# 多変数の例
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
z = a**2 * b + a * b**2    # z = a²b + ab²
z.backward()
print("∂z/∂a =", a.grad.item())   # → 2ab + b² = 2*2*3 + 9 = 21
print("∂z/∂b =", b.grad.item())   # → a² + 2ab = 4 + 12 = 16
```

### `requires_grad=True` に関する注意点

```python
# 注意1: backward() を呼ぶ前に zero_grad() しないと勾配が蓄積される
x = torch.tensor(3.0, requires_grad=True)
for _ in range(3):
    y = x ** 2
    y.backward()
print("蓄積された勾配:", x.grad)  # → 18.0（= 6 + 6 + 6）← 正しくは 6.0

# 解決: optimizer.zero_grad() または x.grad.zero_() で毎回リセット

# 注意2: 推論時は torch.no_grad() で計算グラフを無効化（メモリ節約）
with torch.no_grad():
    y_infer = x ** 2
print("推論時の grad_fn:", y_infer.grad_fn)   # → None（計算グラフが作られない）

# 注意3: .detach() で計算グラフから切り離す
y_detach = (x ** 2).detach()   # 計算グラフから切り離して値だけ取り出す
```

---

## 4. MNIST の読み込みと前処理

**MNIST** は機械学習の「Hello World」とも言われる手書き数字データセット：
- 画像サイズ: 28 × 28 ピクセル（グレースケール）
- クラス数: 10（0〜9の数字）
- 訓練データ: 60,000 枚 / テストデータ: 10,000 枚

### transforms のパイプライン

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = "./data"

# 複数の変換を Compose でまとめる
transform = transforms.Compose([
    transforms.ToTensor(),        # PIL Image（0〜255 uint8）→ Tensor（0.0〜1.0 float32）
                                  # かつ shape: (H,W) → (C,H,W) = (1,28,28) に変換
    transforms.Normalize(
        mean=(0.1307,),           # MNIST の全データの平均値
        std=(0.3081,),            # MNIST の全データの標準偏差
    ),                            # → ピクセル値を平均0・標準偏差1に正規化
])

train_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=True,                   # True=訓練データ, False=テストデータ
    download=True,                # 初回のみダウンロード
    transform=transform,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,                 # エポックごとにシャッフル（学習の安定性向上）
    num_workers=0,                # データ読み込みの並列プロセス数（Colab では0が安全）
    pin_memory=True,              # GPU使用時にメモリ転送を高速化
)
print("サンプル数:", len(train_dataset))   # → 60000
print("バッチ数:", len(train_loader))       # → 60000 / 64 ≈ 938
```

---

## 5. MNIST の可視化

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]            # i 番目の (image, label) を取得
    # img.shape = (1, 28, 28) → .squeeze() で (28, 28) に変換
    # squeeze(): チャネル次元（サイズ1の次元）を除去
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"label: {label}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# ピクセル値の確認
img0, _ = train_dataset[0]
print("shape:", img0.shape)         # → torch.Size([1, 28, 28])
print("min:", img0.min().item())    # → -0.42（Normalize後はマイナスになる）
print("max:", img0.max().item())    # → 2.82
print("dtype:", img0.dtype)         # → torch.float32
```

---

## 6. DataLoader の理解

`DataLoader` は「データセットをミニバッチ単位に切り出すイテレータ」である．

```
Dataset: 全60,000枚の画像データを管理するオブジェクト
          ↓ DataLoader がミニバッチに切り出す
DataLoader: 1回のイテレーションで batch_size 枚のデータを返す

1 epoch = 60,000 / 64 ≈ 938 回のイテレーション
```

### shape の読み方

```
images.shape = (N, C, H, W)
             = (64, 1, 28, 28)
             = (バッチサイズ, チャネル数, 高さ, 幅)

N=64  : ミニバッチの画像数
C=1   : グレースケール（カラー画像なら C=3）
H=28  : 高さ（ピクセル）
W=28  : 幅（ピクセル）
```

```python
images, labels = next(iter(train_loader))   # 1バッチ分を取り出す
print("images.shape:", images.shape)        # → torch.Size([64, 1, 28, 28])
print("labels.shape:", labels.shape)        # → torch.Size([64])
print("labels[:5]:", labels[:5])            # → 先頭5件のラベル

# flatten（平坦化）: 全結合層への入力のため (N,1,28,28) → (N,784) に変換
# view(-1, 784) の -1 は「バッチサイズを自動計算せよ」という意味
print("flatten後:", images.view(-1, 784).shape)   # → torch.Size([64, 784])
```

## 7. 探索の手引き（q14 対応）

q14 の探索では，MNIST の**前処理（transform）と学習設定**を，軽い学習を何度も回して比べる．

### 7-1. 検証データの作り方（訓練データから取り分ける）

MNIST には訓練 60000 枚とテスト 10000 枚がある．**設定を選ぶための検証データは，訓練データの中から取り分ける**（テストは最後の1回のために取っておく）．

```python
# 概念コード（MNIST のダウンロードが必要）
from torch.utils.data import Subset
# train_ds = datasets.MNIST(..., train=True, transform=...)
# train_part = Subset(train_ds, range(0, 8000))          # 学習に使う 8000 枚
# val_part   = Subset(train_ds, range(50000, 52000))     # 検証に使う 2000 枚（学習と重ならない範囲）
```

`torch.utils.data.random_split` でランダムに分ける方法もある．重要なのは，**学習と検証が重ならないこと**である．

### 7-2. 訓練用と評価用で transform を分ける

- **データ拡張（回転・消去など）は訓練データだけ**にかける．検証・テスト画像に拡張をかけると，評価がぶれて比較にならない
- **正規化（Normalize）は訓練と評価で同じもの**を使う．学習時と違う統計で評価すると，モデルには「別の分布」に見える

### 7-3. 乱数（シード）でスコアがぶれる

ニューラルネットの結果は，初期値・ミニバッチの順序・拡張の乱数で毎回変わる．**同じ設定でも `torch.manual_seed` を変えると，正解率が数ポイント動くことがある**（学習が短いほど大きい）．

- 僅差の比較は，**複数シードで繰り返して**平均・ばらつきで判断する
- 「シード違いの差」を測っておけば，「設定違いの差」がそれより大きいかを判断できる

### 7-4. 学習率のスケール感（SGD）

SGD の学習率は，Adam より大きめの値が必要になることが多い（`0.001` は SGD には小さすぎることがある）．**桁で広く振って**，遅すぎ・不安定すぎの両端を見つける．学習率を上げるときは，**エポック数と組み合わせて**考える（小さい lr は多くのエポックが要る）．

### 7-5. 探索の予算

1 試行の時間 × 試行回数を見積もる．粗探索は少ない枚数（例：数千枚）・少ないエポックで回し，良さそうな範囲に絞ってから増やす．

### 7-6. 実験前に立てる予測の例（問いのみ）

- 正規化あり／なしで，学習の速さ（少ないエポックでの精度）は変わるか
- 不適切な transform（例：左右反転）は，実際に精度を下げるか．それはどのくらい
- SGD で最良の学習率の桁は，0.001 か 0.01 か 0.1 か

### 7-7. LLM 指示書に入れる知識（q14）

| 項目 | 書くべき知識の例 |
|---|---|
| 目的 | 検証精度（訓練データ内の別枚数）．テストは最後に1回 |
| データ | 画像サイズ・枚数・クラス数，シード違いのばらつきの数値 |
| 探索計画 | transform を同条件で比較 → 学習率を桁で → 僅差は別シード |
| 落とし穴 | 評価画像に拡張をかける，シード1つで判断する，学習と検証が重なる |
| 止めどき | 計算予算の中で，粗探索から本学習へ切り替える基準 |

---

## つまずきやすいポイント（第7章）

- **`requires_grad=True` を忘れると `.grad` が None** → `y.backward()` しても勾配が計算されない
- **`DataLoader(shuffle=True)` を忘れると毎 epoch 同じ順番で学習** → モデルが順序に過学習する可能性
- **`transforms.ToTensor()` を忘れると PIL Image のまま返ってくる** → Tensor に変換されないのでモデルに渡せない
- **`transforms.Normalize` の mean/std は対象データセット依存** → MNIST の値（0.1307, 0.3081）は事前計算されたもの

### よくある質問

**Q: `pin_memory=True` は何のため？**
A: CPU のメモリをページロック（ピン留め）することで，GPU へのデータ転送を高速化する．CPU のみで動かす場合は不要（デフォルト `False`）．

**Q: `num_workers` を増やすと速くなるのに，なぜ Colab では 0 を推奨するか？**
A: Colab の環境（特に forking の制限）で `num_workers > 0` だとデッドロックが発生することがある．ローカル環境では `num_workers=4` などを試せる．

---

# 第8章　ニューラルネットワーク実装
***
※ 対応演習: [q15.ipynb](../q15.ipynb)

## 目次
1. ニューラルネットワークの構造
2. 活性化関数
3. バックプロパゲーション
4. 損失関数（CrossEntropyLoss）
5. 最適化アルゴリズム（SGD・Adam）
6. nn.Module の実装
7. 学習ループ
8. Dropout と Batch Normalization
9. ハイパーパラメータの感度
10. 探索の手引き（q15 対応）

---

## 1. ニューラルネットワークの構造

### 全結合層（線形変換）

全結合層（`nn.Linear`）は，入力ベクトルに重み行列を掛けてバイアスを加える線形変換である：

```
出力 = 入力 × W + b

入力 x: (N, d_in)   ← バッチサイズN, 入力次元 d_in
重み W: (d_in, d_out)
バイアス b: (d_out,)
出力: (N, d_out)

例: nn.Linear(784, 128)
  → W の shape = (784, 128)，学習パラメータ数 = 784*128 + 128 = 100,480
```

### 多層パーセプトロン（MLP）の全体像

```
入力層: 784次元（28×28 ピクセル）
   ↓ nn.Linear(784, 128)
   ↓ ReLU（非線形変換）
隠れ層: 128次元
   ↓ nn.Linear(128, 10)
出力層: 10次元（0〜9 の各クラスのスコア = logit）
   ↓ CrossEntropyLoss の内部で Softmax 適用
確率分布: 10次元（合計=1）
```

---

## 2. 活性化関数

**なぜ活性化関数が必要か**: 線形変換（全結合層）を何層重ねても，結局は1つの線形変換と等価になる（`y = W₂(W₁x) = (W₂W₁)x`）．**非線形な活性化関数**を挟むことで，モデルが複雑な非線形パターンを学習できるようになる．

| 活性化関数 | 数式 | 特徴 | 主な用途 |
|:--|:--|:--|:--|
| **Sigmoid** | `1 / (1 + e^(-x))` | 出力は (0,1)，勾配消失問題あり | 二値分類の出力層 |
| **Tanh** | `(e^x - e^(-x)) / (e^x + e^(-x))` | 出力は (-1,1)，勾配消失問題あり | RNN の隠れ状態 |
| **ReLU** | `max(0, x)` | 計算が速い，勾配消失が起きにくい | **MLP・CNNの隠れ層の定番** |
| **LeakyReLU** | `max(αx, x)（α<1）` | ReLU の「死んだニューロン」問題を緩和 | ReLU の代替 |
| **GELU** | `x * Φ(x)` | なめらかな ReLU，近年人気 | Transformer（BERT, GPT） |
| **Softmax** | `e^xᵢ / Σe^xⱼ` | 出力の合計が1（確率分布） | **多クラス分類の出力層** |

```python
import torch
import torch.nn as nn

x = torch.linspace(-3, 3, 100)

activations = {
    "ReLU":      nn.ReLU(),
    "Sigmoid":   nn.Sigmoid(),
    "Tanh":      nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU(0.1),
    "GELU":      nn.GELU(),
}

plt.figure(figsize=(10, 4))
for name, fn in activations.items():
    plt.plot(x.numpy(), fn(x).detach().numpy(), label=name)
plt.axhline(0, color="k", linewidth=0.5)
plt.axvline(0, color="k", linewidth=0.5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Comparison of activation functions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**ReLU の直感**: 入力が正なら「そのまま通す」，負なら「0にする」．微分は x>0 で 1，x<0 で 0．勾配消失が起きにくいため深いネットワークで有効．

---

## 3. バックプロパゲーション

### 連鎖律（Chain Rule）

バックプロパゲーションは**微積分の連鎖律**をニューラルネットに適用したものである：

```
z = f(g(x))  の場合:
dz/dx = dz/dg * dg/dx   ← 連鎖律

ニューラルネット（y = W₂ * ReLU(W₁ * x)）の例:
∂L/∂W₁ = ∂L/∂y * ∂y/∂h * ∂h/∂W₁  （h = W₁xを経由して遡る）
```

### 学習ループとの対応

```python
# 1バッチの学習（4ステップ）
optimizer.zero_grad()              # ← [1] 前回の勾配をリセット（必須！）
y_pred = model(images)             # ← [2] 順伝播（forward）
loss = criterion(y_pred, labels)   # ← [3] 損失計算
loss.backward()                    # ← [4] 逆伝播（backward）で全パラメータの勾配を計算
optimizer.step()                   # ← [5] 勾配方向にパラメータを更新

# [1] zero_grad() を忘れると:
#   PyTorch はデフォルトで勾配を累積（加算）する
#   → 前回の勾配 + 今回の勾配 = 誤った方向に更新
#   → 損失が発散する
```

---

## 4. 損失関数（CrossEntropyLoss）

### Softmax と交差エントロピー

多クラス分類では，**Softmax** と **交差エントロピー損失**を組み合わせる：

```
1. モデル出力: logit（生のスコア）
   z = [2.1, 0.5, -1.3]  （3クラスの例）

2. Softmax で確率に変換:
   p_i = e^(z_i) / Σe^(z_j)
   p = [softmax(2.1), softmax(0.5), softmax(-1.3)]
     = [0.71, 0.16, 0.03]  （合計 = 1）

3. 交差エントロピー損失（正解クラスが 0 の場合）:
   L = -log(p_0) = -log(0.71) = 0.34

   → 正解クラスの確率が高いほど損失が小さい
   → 正解クラスの確率が 0 に近づくと損失が無限大
```

```
CrossEntropyLoss の直感:
   正解ラベルが 0（クラス0が正解）→ L = -log(p_0)
   モデルが確信して正解: p_0 → 1, L → 0（損失小）
   モデルが迷っている:   p_0 = 0.5, L = 0.69（損失中）
   モデルが大きく外れた: p_0 → 0, L → ∞（損失大）
```

**重要**: PyTorch の `nn.CrossEntropyLoss` は**内部で Softmax を含む**．モデルの出力層に Softmax を追加すると二重適用になるので**絶対に追加しない**こと．

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# 使い方
logits = torch.tensor([[2.1, 0.5, -1.3],   # バッチ1件目（クラス3の生スコア）
                        [0.1, 3.2,  0.8]])  # バッチ2件目
labels = torch.tensor([0, 1])              # 正解クラス（0件目はクラス0，1件目はクラス1）

loss = criterion(logits, labels)
print("CrossEntropyLoss:", loss.item())

# 手動で確認
probs = torch.softmax(logits, dim=1)
loss_manual = -torch.log(probs[0, 0]) - torch.log(probs[1, 1])
print("手動計算:", (loss_manual / 2).item())   # 平均（ほぼ一致）
```

---

## 5. 最適化アルゴリズム（SGD・Adam）

### 勾配降下法の種類

| 手法 | 特徴 | 更新式 |
|:--|:--|:--|
| **SGD** | 最もシンプル | `w ← w - η * ∂L/∂w` |
| **Momentum SGD** | 過去の勾配方向を保持（慣性） | `v ← βv + ∂L/∂w`; `w ← w - η * v` |
| **Adam** | 適応的学習率 + Momentum | 下記参照 |

### Adam の仕組み

Adam（Adaptive Moment Estimation）は現在最もよく使われる最適化アルゴリズムである：

```
1. 1次モーメント（モーメンタム，勾配の指数移動平均）:
   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t   （β₁ = 0.9）

2. 2次モーメント（勾配の二乗の指数移動平均 = 分散の推定）:
   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²   （β₂ = 0.999）

3. バイアス補正（初期値が0に偏るのを補正）:
   m̂_t = m_t / (1 - β₁^t)
   v̂_t = v_t / (1 - β₂^t)

4. パラメータ更新:
   w_t ← w_{t-1} - η * m̂_t / (√v̂_t + ε)   （η = 0.001, ε = 1e-8）

→ 勾配が大きい方向は学習率を自動的に下げる（過大なステップを防ぐ）
→ 勾配が小さい方向は学習率を自動的に上げる（進みやすくする）
→ 「各パラメータに異なる学習率を適応的に設定」
```

```python
import torch.optim as optim

model = SimpleMLP()

# SGD with Momentum
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam（ほとんどの場合 lr=0.001 で良い結果を出す）
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# AdamW（重み減衰付きAdam，正則化も同時に行う）
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

---

## 6. nn.Module の実装

```python
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()            # 必須: PyTorch の内部登録機構を初期化
                                      # これを忘れるとパラメータが optimizer に認識されない
        self.fc1 = nn.Linear(784, 128)  # 入力784（28×28 flatten）→ 隠れ層128
        self.relu = nn.ReLU()           # 非線形活性化関数
        self.fc2 = nn.Linear(128, 10)   # 隠れ層128 → 出力10（0〜9の10クラス）

    def forward(self, x):
        x = x.view(-1, 784)             # (N,1,28,28) → (N,784) に flatten
        x = self.relu(self.fc1(x))      # 全結合 → ReLU
        return self.fc2(x)              # 出力層（Softmax は CrossEntropyLoss の内部）

model = SimpleMLP()
print(model)
# SimpleMLP(
#   (fc1): Linear(in_features=784, out_features=128, bias=True)
#   (relu): ReLU()
#   (fc2): Linear(in_features=128, out_features=10, bias=True)
# )

# パラメータ数の確認
total_params = sum(p.numel() for p in model.parameters())
print(f"総パラメータ数: {total_params:,}")   # → 101,770
# = 784*128 + 128 + 128*10 + 10 = 100,480 + 1,290
```

---

## 7. 学習ループ

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用デバイス:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

model     = SimpleMLP().to(device)                          # モデルをデバイスに転送
criterion = nn.CrossEntropyLoss()                           # 多クラス分類の損失関数
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_losses = []   # 損失を記録するリスト

for epoch in range(3):
    model.train()           # 訓練モードに切り替え（Dropout等が有効になる）
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)    # データをデバイスに転送（モデルと同じデバイスでないとエラー）
        labels = labels.to(device)

        optimizer.zero_grad()                         # [1] 前回の勾配をリセット
        outputs = model(images)                       # [2] 順伝播
        loss = criterion(outputs, labels)             # [3] 損失計算
        loss.backward()                               # [4] 逆伝播
        optimizer.step()                              # [5] パラメータ更新
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/3, Loss: {avg_loss:.4f}")

# 損失曲線の描画
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training loss curve")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 8. Dropout と Batch Normalization

### Dropout

**Dropout** は，訓練中に各ニューロンをランダムに一定確率でゼロにする正則化手法である：

```
訓練中:  ランダムに p の確率でニューロンを無効化
              ↓
推論中:  全ニューロンを使う（ただし訓練時のスケール補正のため (1-p) 倍に）

例（p=0.5）: [1.0, 2.0, 3.0, 4.0] → [2.0, 0.0, 6.0, 0.0]（50%がゼロに）

なぜ過学習を防ぐか: 毎回異なるサブネットワークを学習することになり，
各ニューロンが他のニューロンに依存できない（co-adaptation を防ぐ）
→ より汎用的な特徴を学ぶ
```

**`model.train()` / `model.eval()` の切り替えが必須**：
- `model.train()`: Dropout が有効（ランダムにゼロにする）
- `model.eval()`: Dropout が無効（全ニューロンを使う，スケール補正あり）

### Batch Normalization

**Batch Normalization（BatchNorm）** は，各層の入力を「ミニバッチ内で正規化」し，学習を安定させる手法である：

```
各層の入力 x に対して:
  μ_B = (1/m) Σ xᵢ        ← ミニバッチの平均
  σ²_B = (1/m) Σ (xᵢ - μ_B)²  ← ミニバッチの分散
  x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)  ← 正規化
  yᵢ = γ x̂ᵢ + β           ← 学習可能なスケールとシフト（γ, β）

効果:
  1. 各層の入力が常に N(0,1) 付近に保たれる → 勾配消失・爆発を防ぐ
  2. 学習率を大きくできる → 学習が速くなる
  3. 軽い正則化効果 → Dropout と組み合わせることも多い
```

```python
class MLPWithDropoutBN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),   # BatchNorm: 256次元の入力を正規化
            nn.ReLU(),
            nn.Dropout(dropout_rate),   # Dropout: 30%のニューロンをランダムに無効化
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))
```

---

## 9. ハイパーパラメータの感度

| パラメータ | 効果 | 調整ガイドライン |
|:--|:--|:--|
| **学習率 lr** | 大きい: 発散リスク，小さい: 収束遅い | Adam では `1e-3`（0.001）から始め，必要なら下げる |
| **batch_size** | 大きい: 安定だが汎化性能低下傾向，小さい: ノイズが多いが汎化しやすい | 64〜256が実用的 |
| **epoch 数** | 多い: 精度向上，過学習リスク | 損失曲線を見て判断，Early Stopping を使う |
| **隠れ層のサイズ** | 大きい: 表現力UP，過学習リスク | データ量に比例させる |
| **Dropout 率** | 高い: 正則化強，低い: 弱 | 0.2〜0.5が一般的 |

```python
# テスト正解率の評価
model.eval()
correct, total = 0, 0
with torch.no_grad():   # 推論時は勾配計算が不要 → メモリ節約
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)              # 最大スコアのクラスを予測ラベルとする
        correct += (preds == labels).sum().item()  # 正解数を累積
        total += labels.size(0)                    # 総サンプル数を累積
print(f"テスト正解率: {correct / total:.4f}")     # → SimpleMLP なら 0.97 付近
```

## 10. 探索の手引き（q15 対応）

q15 では，全結合ネットワーク（MLP）の `lr`・`hidden`・`epochs`・`depth` を探索する．

### 10-1. 損失曲線から探索の次の一手を決める

学習中の損失（loss）の推移は，設定の良し悪しを教えてくれる．

| 損失曲線の形 | 意味 | 次の一手 |
|---|---|---|
| ほぼ横ばい | 学習が進んでいない | `lr` を上げる（小さすぎる），または勾配消失を疑う（深すぎ・活性化） |
| 大きく振動する | `lr` が大きすぎる | `lr` を下げる |
| 発散・NaN | `lr` が大きすぎる（さらに極端） | `lr` を桁で下げる |
| 順調に減少 | 良い | エポックや `hidden` を調整．検証精度と併せて過学習を見る |

### 10-2. 学習率の探索

Adam の学習率は，桁で 1e-5〜1e-1 を振る．**大きすぎても小さすぎても失敗する**．大きすぎると更新が谷を飛び越えて損失が振動・発散し，小さすぎると1ステップの移動が小さく，同じエポック数では損失がほとんど下がらない．

### 10-3. `hidden`・`depth`・`epochs` の関係

- `hidden`（隠れユニット数）を増やす → 表現力が増える（パラメータも計算時間も増える）
- `depth`（隠れ層の数）を増やす → さらに複雑に．ただし**活性化関数によっては勾配が層を遡るうちに小さくなり，学習が進まなくなる**（勾配消失，q15 問題4）
- `epochs` を増やす → 訓練損失は下がり続けるが，**検証精度が頭打ち・低下**したら過学習の始まり

### 10-4. 探索コストの見積り

```python
sec_per_trial = 12          # 1試行の所要時間（秒）: 記録帳の sec 列から
n_trials = 20
print(f"探索全体の見積り = {sec_per_trial * n_trials / 60:.1f} 分")
```

見積りが予算を超えるなら，**粗探索は少ないデータ・少ないエポック**で回して有望な範囲を絞り，**絞った設定だけ本番の条件で学び直す**．

### 10-5. 乱数のばらつき

MLP でもシード違いで数ポイント（学習が短いほど大きく）動くことがある．**0.003 程度の差は「差がある」と言い切れない**．僅差の設定は別シードで確かめる．

### 10-6. 実験前に立てる予測の例（問いのみ）

- 最良の学習率は，Adam で 0.0001 / 0.001 / 0.01 のどの桁か．なぜそう考えるか
- `hidden` を 16 → 512 と増やすと，検証精度はどこで頭打ちになるか
- `depth` を 1 → 4 に増やすと，ReLU と Sigmoid でどう違うか

### 10-7. LLM 指示書に入れる知識（q15）

| 項目 | 書くべき知識の例 |
|---|---|
| 目的 | 検証精度（訓練データ内の別 2000 枚）．テストは最後に1回 |
| データ | 入力次元 784・クラス数 10・枚数，別シードでの差の数値，1試行の時間 |
| 探索計画 | 学習率を桁で → 有望な lr で `hidden`・`epochs`・`depth` |
| 落とし穴 | lr の桁の外し方，損失の横ばい／振動／NaN の見抜き方，シード1つで判断 |
| 止めどき | 計算予算の中で粗探索から本学習へ切り替える基準 |

---

## つまずきやすいポイント（第8章）

- **`zero_grad()` 忘れ** → 勾配が蓄積して発散する．毎回 `optimizer.zero_grad()` を `backward()` の前に呼ぶ
- **`model.train()` / `model.eval()` の切り替え忘れ** → Dropout や BatchNorm の挙動が変わる
- **モデルとデータが別 device に** → `RuntimeError: Expected all tensors to be on the same device`
- **テスト時に `torch.no_grad()` を忘れる** → メモリ不足になりやすい
- **出力層に Softmax を追加してから CrossEntropyLoss に渡す** → 二重適用でスコアがおかしくなる

### よくある質問

**Q: `loss.item()` の `.item()` は何をするか？**
A: PyTorch Tensor（スカラー）を Python の `float` に変換する．損失を記録・表示するときに使う．`.item()` なしで `total_loss += loss` とすると，Tensor が蓄積されてメモリリークの原因になる．

**Q: `argmax(dim=1)` の `dim=1` はなぜ？**
A: `outputs.shape = (N, 10)` のとき，`dim=1`（クラス方向）で最大値のインデックスを取る．`dim=0` だとバッチ方向で最大を取ることになり意味が変わる．

---

# 第9章　ANN・CNN・推論
***
※ 対応演習: [q16.ipynb](../q16.ipynb), [q17.ipynb](../q17.ipynb)

## 目次
1. 3モデルの比較（LinearNet・DeepMLP・SimpleCNN）
2. 畳み込み演算の仕組み
3. CNN の構造詳解
4. 受容野と重み共有のメリット
5. Pooling の役割
6. CNN アーキテクチャの歴史的発展
7. SimpleCNN の実装
8. 混同行列とモデル保存
9. Distribution Shift と手書き推論（q17）
10. 探索の手引き（q16・q17 対応）

---

## 1. 3モデルの比較

第16回では，同じ MNIST データ・同じ条件で以下の3モデルを比較する：

| モデル | 構成 | パラメータ数（概算） | 特徴 |
|:--|:--|:--|:--|
| `LinearNet` | 784 → 10 | 7,850 | 活性化関数なし = **線形境界のみ** |
| `DeepMLP` | 784 → 256 → 128 → 10 | 234,954 | 非線形変換可能だが画像の局所パターンを無視 |
| `SimpleCNN` | Conv→Pool×2 → FC | 約420,000 | **局所パターンを重み共有で効率学習** |

**なぜ CNN が強いか**: 手書き数字の「8」の左上のカーブと右下のカーブは「同じ種類の特徴（エッジ）」である．MLP は全ピクセルを独立に見るため，左上と右下の「カーブ」に別々の重みを割り当てる．CNN は同じフィルタを画像全体に適用することで，**位置に関わらず同じ特徴を検出できる**．

```python
import torch.optim as optim

def train_one_model(model, train_loader, test_loader, epochs=3, lr=0.001):
    """3モデルを同条件（epoch/lr/batch_size）で学習・評価するヘルパー関数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

---

## 2. 畳み込み演算の仕組み

### カーネル（フィルタ）とは

畳み込みカーネルは，小さな重み行列（例: 3×3）である．これを入力画像上でスライドさせ，各位置でカーネルと画像の「内積（要素ごとの積の和）」を計算する：

```
入力画像の一部（3×3）:   カーネル（3×3）:    出力（1要素）:
  1  0  1               0  1  0         1*0 + 0*1 + 1*0
  0  1  0        ×      1  1  1    =  + 0*1 + 1*1 + 0*1  = 4
  1  0  1               0  1  0        + 1*0 + 0*1 + 1*0

（カーネルが「垂直エッジ検出」「水平エッジ検出」「コーナー検出」など
  様々なフィルタの役割を学習する）
```

### 手動での畳み込み計算例

```
5×5 の入力画像:
  1  2  3  4  5
  5  4  3  2  1
  1  1  1  1  1
  2  2  2  2  2
  3  3  3  3  3

3×3 のカーネル（垂直エッジ検出）:
  -1  0  1
  -1  0  1
  -1  0  1

出力（3×3）の左上要素（stride=1, no padding）:
  = (-1)*1 + 0*2 + 1*3    ← 入力の1行目1〜3列
  + (-1)*5 + 0*4 + 1*3    ← 入力の2行目1〜3列
  + (-1)*1 + 0*1 + 1*1    ← 入力の3行目1〜3列
  = (-1+0+3) + (-5+0+3) + (-1+0+1) = 2 + (-2) + 0 = 0
```

### 出力サイズの公式

```
出力の空間サイズ = ⌊(H + 2P - K) / S⌋ + 1

H: 入力サイズ（高さ or 幅）
K: カーネルサイズ
P: パディング（padding）
S: ストライド（stride）

例1: H=28, K=3, P=1, S=1（padding=1 で空間サイズを保つ）
  出力 = ⌊(28 + 2*1 - 3) / 1⌋ + 1 = ⌊27/1⌋ + 1 = 28  ← 変わらない！

例2: H=28, K=3, P=0, S=1（padding なし）
  出力 = ⌊(28 + 0 - 3) / 1⌋ + 1 = ⌊25/1⌋ + 1 = 26  ← 2 小さくなる

例3: MaxPool(2)（K=2, P=0, S=2）
  出力 = ⌊(28 + 0 - 2) / 2⌋ + 1 = ⌊26/2⌋ + 1 = 14  ← 半分になる
```

### padding の効果

```
padding=0（valid padding）: 境界情報が失われ，出力が小さくなる
padding=1（same padding）: 入力の周囲に0を追加し，出力サイズを保つ

padding なし（valid）:          padding=1（same）:
  入力 5×5:                      入力 5×5 に周囲0追加 → 7×7:
  ■ ■ ■ ■ ■                    0 0 0 0 0 0 0
  ■ ■ ■ ■ ■   → 出力 3×3      0 ■ ■ ■ ■ ■ 0
  ■ ■ ■ ■ ■                    0 ■ ■ ■ ■ ■ 0
  ■ ■ ■ ■ ■                    0 ■ ■ ■ ■ ■ 0
  ■ ■ ■ ■ ■                    0 ■ ■ ■ ■ ■ 0
                                 0 ■ ■ ■ ■ ■ 0  → 出力 5×5
                                 0 0 0 0 0 0 0
```

---

## 3. CNN の構造詳解

### SimpleCNN の shape の変化を段階的に追う

```
入力:                  (N,  1, 28, 28)  ← (バッチ, チャネル, 高さ, 幅)
                                         グレースケール1チャネル

↓ Conv2d(1→16, K=3, P=1)              ← 16種類の3×3フィルタを1チャネルに適用
                       (N, 16, 28, 28)  ← 16種類の「特徴マップ」が生成される
                                         （エッジ検出，コーナー検出など各フィルタが学習）
↓ ReLU                                 ← 負の値を0にする
                       (N, 16, 28, 28)

↓ MaxPool2d(2)                         ← 2×2領域の最大値を取る（ダウンサンプリング）
                       (N, 16, 14, 14)  ← 空間サイズが半分に

↓ Conv2d(16→32, K=3, P=1)             ← 16チャネルの入力から32種類の特徴マップを生成
                       (N, 32, 14, 14)  ← より複雑な特徴（エッジの組み合わせ，形状など）

↓ ReLU
                       (N, 32, 14, 14)

↓ MaxPool2d(2)
                       (N, 32,  7,  7)  ← さらに半分に

↓ Flatten（view）                      ← 32 * 7 * 7 = 1568
                       (N,      1568)

↓ Linear(1568→128) + ReLU
                       (N,       128)

↓ Linear(128→10)
出力:                  (N,        10)   ← 10クラスの logit
```

### Conv2d のパラメータ数計算

```
Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=K, bias=True)
  パラメータ数 = C_out * (C_in * K * K + 1)   ← +1 はバイアス項

例: Conv2d(1, 16, 3)
  = 16 * (1 * 3 * 3 + 1) = 16 * 10 = 160 パラメータ

例: Conv2d(16, 32, 3)
  = 32 * (16 * 3 * 3 + 1) = 32 * 145 = 4,640 パラメータ

対比: 全結合層 Linear(1*28*28, 16) の場合
  = 784 * 16 + 16 = 12,560 パラメータ  ← CNN の 80倍！
```

**重み共有のメリット**: CNN は同じ3×3フィルタを画像全体でスライドさせて使う（重みを共有）ため，全結合層に比べてパラメータ数が圧倒的に少ない．少ないパラメータで効率的に学習できる．

---

## 4. 受容野と重み共有のメリット

### 受容野（Receptive Field）

**受容野**とは，ある特徴マップの1要素が「元の入力画像のどの範囲の情報を持っているか」である：

```
第1層 Conv（K=3, P=1, S=1）後の特徴マップの1要素:
  → 入力の 3×3 の範囲を見ている（受容野 = 3×3）

MaxPool(2) 後:
  → 元の入力の 4×4 の範囲を見ている（受容野 = 4×4）

第2層 Conv（K=3, P=1, S=1）後:
  → 元の入力の 8×8 の範囲を見ている（受容野 = 8×8 → 14×14程度）

MaxPool(2) 後の特徴マップの1要素:
  → 元の入力の 16×16 付近の範囲（受容野が広がる）

最終的な全結合層の1ニューロンは28×28全体を「見ている」
```

**直感**: 初期層は「エッジ・点」などの局所的特徴を検出し，深い層になるにつれて「数字の形状」などの大局的特徴を検出する．これはヒトの視覚野の働き方と類似している．

### 重み共有の具体的なメリット

```
全結合層で「8の字の左上カーブ」を検出する場合:
  入力が (1, 784) → 左上カーブ用の重みは「左上の座標に対応する」特定の重みのみ
  右下に同じカーブがあっても，別の重みが担当 → パラメータ数が膨大

CNN で「カーブ」を検出する場合:
  1つのカーブ検出フィルタ（3×3 = 9パラメータ）を画像全体でスライド
  左上でも右下でも同じフィルタが「カーブ検出」を担当
  → 「平行移動不変性」: 特徴が画像のどこにあっても検出できる
  → パラメータが少なくて済む
  → データが少なくても学習できる
```

---

## 5. Pooling の役割

### Max Pooling の動作

```
2×2 の MaxPooling（stride=2）:

入力 4×4:          出力 2×2:
  3  1  2  5       max(3,1,0,2)=3   max(2,5,8,1)=8
  0  2  8  1     →      3  8
  4  7  3  2       max(4,7,2,9)=9   max(3,2,6,5)=6
  2  9  6  5            9  6
```

### Max Pooling の3つの役割

1. **ダウンサンプリング**: 空間サイズを縮小 → 計算コスト削減・後続の Conv のパラメータ数削減
2. **平行移動不変性**: 特徴が少しずれていても最大値は変わらない → 位置のズレに頑健
3. **最も顕著な特徴の保持**: その領域で最も強く反応した（= その特徴が最もある）場所の値だけを残す

### Average Pooling との比較

| | Max Pooling | Average Pooling |
|:--|:--|:--|
| **計算** | 領域内の最大値 | 領域内の平均値 |
| **特徴** | 最も強い特徴を保持 | 全ての特徴の平均 |
| **用途** | 分類（一般的） | 特定のアーキテクチャ（ResNet最終層など） |
| **平行移動不変性** | 強い | 中程度 |

---

## 6. CNN アーキテクチャの歴史的発展

| 年 | アーキテクチャ | 革新点 |
|:--|:--|:--|
| 1998 | **LeNet-5** | 最初の実用的CNN（手書き文字認識） |
| 2012 | **AlexNet** | 深いCNN + ReLU + Dropout + GPU学習でImageNet制覇 |
| 2014 | **VGGNet** | 3×3 Conv を積み重ねることで深くシンプルな設計 |
| 2014 | **GoogLeNet** | Inception モジュール（複数サイズのフィルタを並列） |
| 2015 | **ResNet** | **残差接続（スキップ接続）**で超深いネット（152層）を実現 |
| 2017 | **DenseNet** | 全層が後続の全層に直接接続 |
| 2020〜 | **ViT** | Transformer を画像に適用（パッチ分割） |

### ResNet のスキップ接続（残差ブロック）

ResNet で最も重要な革新は**残差接続**である：

```
通常の CNN 層:               残差ブロック（ResNet）:
                               ┌──── identity ────┐
x → [Conv-BN-ReLU-Conv] → y   x → [Conv-BN-ReLU-Conv] → (+) → F(x) + x

勾配消失問題:                  勾配消失の解決:
  深い層の勾配が 0 に近くなる    スキップ経路を通じて勾配が直接伝わる
  → 深いネットが学習困難         → 100層以上でも安定して学習できる

y = F(x, {Wᵢ}) + x   （F は残差関数，x はスキップ）
```

**なぜスキップ接続が有効か**: 恒等写像（何もしない変換）は `F(x) = 0` のとき実現される．層が「0を学習する」方が，「完全な変換を学習する」より簡単なため，学習が安定する．

---

## 7. SimpleCNN の実装

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # --- 特徴抽出部（Convolutional Block）---
        self.conv1 = nn.Conv2d(
            in_channels=1,      # 入力チャネル数（グレースケールなので 1）
            out_channels=16,    # 出力チャネル数（16 種類のフィルタを学習）
            kernel_size=3,      # 3×3 フィルタ
            padding=1,          # padding=1 → 出力サイズを入力と同じに保つ
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,     # 前の Conv 出力チャネル数と一致させる
            out_channels=32,    # さらに多い種類のフィルタ（より複雑な特徴を検出）
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   # 2×2 Max Pooling
        self.relu = nn.ReLU()

        # --- 分類部（Fully Connected Block）---
        # MaxPool を2回通ると: 28 → 14 → 7
        # チャネル数 32, 空間サイズ 7×7 → 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 入力: (N, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))  # → (N, 16, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # → (N, 32,  7,  7)
        x = x.view(x.size(0), -1)               # → (N, 1568) Flatten
        x = self.relu(self.fc1(x))               # → (N, 128)
        return self.fc2(x)                       # → (N, 10)  logit を返す（Softmax は損失関数側）

# パラメータ数の確認
cnn = SimpleCNN()
total_params = sum(p.numel() for p in cnn.parameters())
print(f"SimpleCNN パラメータ数: {total_params:,}")
# → 約 421,130
# Conv1: 16*(1*3*3+1) = 160
# Conv2: 32*(16*3*3+1) = 4,640
# FC1: 1568*128+128 = 200,832
# FC2: 128*10+10 = 1,290
# 合計: 206,922（実際の出力で確認すること）
```

### 3モデルの比較実装

```python
import torch.nn as nn

class LinearNet(nn.Module):
    """活性化関数なし: 線形境界のみ → 非線形パターンを学べない"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 784))


class DeepMLP(nn.Module):
    """複数の全結合層: 非線形変換可能だが画像の位置情報を利用できない"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))
```

---

## 8. 混同行列とモデル保存

### 混同行列の読み方

10クラス分類の混同行列：

```
行 = 正解クラス（正解は何番だったか）
列 = 予測クラス（何番と予測したか）

        予測0  予測1  予測2 ... 予測9
正解0  [  大    少    少  ...  少  ]   ← 対角が大 = 正しく分類
正解1  [  少    大    少  ...  少  ]
正解3  [  少    少    少  ... 多め ]   ← 3行目の予測9が多い = 3 を 9 と誤認識
...

対角成分が大きく，それ以外が小さいほど良い分類器
よく混同されるペア: (3と8), (4と9), (5と6) など
```

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

all_preds, all_labels = [], []
best_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = best_model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion matrix (MNIST, 10 classes)")
plt.show()

# 最も混同されるペアを探す（対角を除いた最大値）
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)   # 対角を0にして他を探す
max_idx = np.unravel_index(cm_off_diag.argmax(), cm_off_diag.shape)
print(f"最も混同されるペア: 正解{max_idx[0]} → 予測{max_idx[1]}（{cm_off_diag[max_idx]}件）")
```

### モデルの保存と復元（state_dict の仕組み）

```
state_dict とは:
  モデルの全パラメータ（重みとバイアス）を辞書形式で保存したもの
  {"fc1.weight": tensor([[...]]), "fc1.bias": tensor([...]), ...}

.pth ファイルに保存されるのは「重みの値のみ」
モデルの「構造（クラス定義）」は保存されない

→ 復元時にはクラス定義が必要！
```

```python
# 保存
torch.save(best_model.state_dict(), "best_mnist_model.pth")
print("保存完了: best_mnist_model.pth")

# 復元（q17 での使い方）
# Step 1: SimpleCNN クラスを定義（または import）
# Step 2: インスタンスを作成
loaded_model = SimpleCNN()
# Step 3: 保存した重みを読み込む
loaded_model.load_state_dict(torch.load("best_mnist_model.pth", map_location="cpu"))
# Step 4: 推論モードに切り替え
loaded_model.eval()
print("モデル復元完了")
```

---

## 9. Distribution Shift と手書き推論（q17）

### Distribution Shift とは

**Distribution Shift**（分布シフト）とは，「モデルを訓練したデータの分布」と「実運用時のデータの分布」が異なる現象である．

```
訓練データ（torchvision の MNIST）の分布:
  ・黒背景（ピクセル値 ≈ 0）
  ・白い数字（ピクセル値 ≈ 1）
  ・28×28 ピクセル
  ・20×20 の枠に収めてセンタリングされた数字

Colab Canvas で描いた場合:
  ・黒背景・白線（向きは MNIST と同じ → 反転は不要）
  ・任意のサイズ → 28×28 にリサイズ
  ・位置ずれ・大きさ・線の太さは書き方しだい

白い紙に黒ペンで書いた写真・スキャンの場合:
  ・白背景・黒線（向きが MNIST と逆 → 反転が必要）
```

よくある誤解：「MNIST は白い紙に黒い数字」ではない．torchvision の MNIST は**黒背景に白い数字**である．

**分布シフトがあるとどうなるか**: モデルは MNIST の「黒背景・白い数字・中央寄せ」で学習している．向きが逆の入力（白背景・黒線）や，大きくずれた入力を与えると，モデルにとっては「見たことのないデータ」になり，予測が大きく崩れる．

### Canvas → MNIST 形式への前処理

```python
# Canvas から取得した画像の前処理（q17 の UI コード内に含まれる）

# arr: Canvas から取得した NumPy 配列（黒背景=0, 白線=255, shape=(H,W) or (H,W,4)）

# Step 1: グレースケール化（RGBA → グレー）
if arr.ndim == 3:
    arr = arr[:, :, 0]   # アルファチャネルがある場合，最初のチャネルを使用

# Step 2: float32 に変換して [0,1] に正規化
arr = arr.astype(np.float32) / 255.0

# Step 3: 色の向きをそろえる
# Canvas（黒背景・白線）は MNIST と同じ向きなので反転しない．
# 白い紙に黒ペンの画像を使うときだけ反転する：arr = 1.0 - arr

# Step 4: 28×28 にリサイズ
from PIL import Image
img = Image.fromarray((arr * 255).astype(np.uint8))
img = img.resize((28, 28), Image.LANCZOS)
arr = np.array(img).astype(np.float32) / 255.0

# Step 5: 訓練時と同じ正規化を必ず適用する
# q16 のモデルは ToTensor() だけ（0〜1）で学習しているので，ここでは追加の正規化はしない．
# もし訓練時に Normalize((0.1307,), (0.3081,)) を使っていたら，ここでも arr = (arr - 0.1307) / 0.3081 とする．

# Step 6: PyTorch Tensor に変換して次元を追加
tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # → (1, 1, 28, 28)

# Step 7: 推論
loaded_model.eval()
with torch.no_grad():
    output = loaded_model(tensor)
    probs = torch.softmax(output, dim=1)
    pred_class = probs.argmax(dim=1).item()
    confidence = probs.max(dim=1).values.item()
print(f"予測: {pred_class}（信頼度: {confidence:.1%}）")
```

### Distribution Shift の対処法

| 対処法 | 説明 |
|:--|:--|
| **前処理の統一** | 訓練時と同じ前処理を推論時にも必ず適用する |
| **データ拡張** | 訓練時に回転・ノイズ・反転などを加えることで，様々な分布に対応させる |
| **Domain Adaptation** | テストドメインのデータも使って訓練を適応させる（発展的内容） |

### q17 の考察ポイント

1. **CNN が MLP より高精度な理由**（2〜4文）
   - 畳み込みの**重み共有**によりパラメータ効率が高い
   - **局所的な特徴**（エッジ・曲線）を位置に関わらず検出できる（平行移動不変性）
   - **受容野**が深くなるにつれ大域的な形状特徴も捉えられる

2. **自作数字の誤認識の原因**（2〜4文）
   - 前処理の向きの取り違え（反転が不要な Canvas 入力を反転してしまう／白い紙の画像を反転し忘れる）
   - 線の太さや位置ずれ（MNIST の数字はセンタリングされている）
   - 個人の筆跡スタイル（訓練データにない書き方）

3. **精度向上のアイデア**（2つ + 理由）
   - **Epoch 数を増やす**: まだ損失が下がっている場合は学習が不十分
   - **データ拡張（回転・ノイズ）**: 様々な位置・向きの文字を学習させて Distribution Shift に強くする
   - **Dropout の追加**: 過学習を防ぎ汎化性能を向上
   - **BatchNorm の追加**: 学習を安定させより深いネットで高精度化

## 10. 探索の手引き（q16・q17 対応）

### A. q16：3モデルの比較

- `LinearNet`（線形）・`DeepMLP`・`SimpleCNN` を，**同じ条件・同じ検証データ**で比べる．探索は「モデル選択」と「そのモデルの設定」の二段構えになる
- モデルの**パラメータ数と所要時間**（コスト）も記録する．精度が僅かに上がるだけのために，コストが何倍にもなっていないかを見る．**精度とコストの兼ね合い**が実務の判断である
- 「複雑なモデルほど良い」とは限らない．データ量・エポック数・学習率次第で順位は入れ替わりうる．**予想と違ったときは，学習率が合っていないか，エポックが足りないか**を疑う
- `gap = 訓練精度 - 検証精度` で，過学習か未学習かを判断する

### B. q17：現場の入力を，学習時の形に戻す

学習したモデルは，**学習時の入力の形式**（黒地に白い数字・28×28・数字が中央・線の太さ・画素値の範囲）を暗黙の前提にしている．現場の入力がこの形式とずれていると，精度は大きく落ちる（分布シフト）．これを**前処理で学習時の形に戻す**のが q17 の探索である．

#### B-1. まず「学習時の入力仕様」を書き出す

前処理を設計する前に，モデルが何を仮定しているかを箇条書きにする．「白地に黒線か，黒地に白線か」「画素値は 0〜1 か」「数字は中央か」「線の太さはどのくらいか」——この仕様が，指示書の「データの事実」になる．

#### B-2. 色の反転・二値化

- 反転：`x = 1.0 - x`．**向きが逆だと精度は大きく崩れる**（学習時と反転した入力になる）
- 二値化：`x = (x > threshold).float()`．ノイズを消せるが，**閾値が高すぎると細い線が消え，低すぎるとノイズが残る**．中間に山があるかを探索で確かめる

#### B-3. 線の太さを変える（最大値／最小値プーリング）

`max_pool2d` は明るい画素を周囲に広げるので線が**太く**なる（膨張）．逆に，符号を反転してから `max_pool2d` して戻すと，暗い側が広がって線が**細く**なる（収縮）．

```python
import torch
import torch.nn.functional as F

img = torch.zeros(1, 1, 7, 7)
img[0, 0, 3, 1:6] = 1.0                        # 細い横線
thick = F.max_pool2d(img, kernel_size=3, stride=1, padding=1)          # 太くする
thin = -F.max_pool2d(-thick, kernel_size=3, stride=1, padding=1)       # 細くする
print("元の線の画素数 :", int(img.sum()))
print("太くした画素数 :", int(thick.sum()))
print("細くした画素数 :", int(thin.sum()))
```

カーネルを大きくしすぎると，**必要な線まで消えてしまう**．強さも探索の軸である．

#### B-4. 重心を中央に寄せる

数字の位置ずれは，**画素値の重み付き平均位置（重心）**を求め，画像の中心に平行移動して直せる．

```python
import torch

def recenter_one(img):                      # img: (28, 28)
    ys = torch.arange(28, dtype=torch.float32).view(28, 1)
    xs = torch.arange(28, dtype=torch.float32).view(1, 28)
    m = img.sum() + 1e-6
    dy = 14 - int(round(((img * ys).sum() / m).item()))   # 重心の y を 14 に
    dx = 14 - int(round(((img * xs).sum() / m).item()))   # 重心の x を 14 に
    return torch.roll(img, shifts=(dy, dx), dims=(0, 1))

img = torch.zeros(28, 28)
img[20:26, 20:26] = 1.0                     # 右下に寄った四角
moved = recenter_one(img)
ys, xs = torch.nonzero(moved, as_tuple=True)
print("移動後の重心 y, x =", ys.float().mean().item(), xs.float().mean().item())   # 14 付近
```

#### B-5. 効果の分解（一つずつ足して測る）

前処理が複数あるときは，**何もしない基準 → 1つ足す → もう1つ足す**の順に試して，「どれがどれだけ回復させたか」を数値にする．最初から全部入れると，どれが効いたか，どれが邪魔をしているか分からない．**基準（何もしない）を必ず試行に含める**こと．

#### B-6. 開発用と holdout に分ける

現場入力を模した画像を「探索に使う開発用」と「最後に1回だけ見る holdout」に分ける．holdout を見て設定を変えるのはリークである．

### C. 実験前に立てる予測の例（問いのみ）

- q16：`SimpleCNN` は `DeepMLP` より必ず良いか．差が偶然の範囲でないと言えるか
- q17：何もしない場合の精度はどのくらいか．どの前処理が最も回復させると思うか
- q17：二値化の閾値は，高すぎるとどうなるか．低すぎるとどうなるか

### D. LLM 指示書に入れる知識（q16・q17）

| 項目 | q16 | q17 |
|---|---|---|
| 目的 | 検証精度．コスト（パラメータ数・時間）も | 開発用の正解率．holdout は最後に1回 |
| データ | 画像形状・枚数・クラス数・シードのばらつき | **学習時の入力仕様**と，現場入力との違い |
| 探索計画 | モデル → lr（桁）→ epochs・hidden・batch | 基準 → 1つずつ足す → 強さ・閾値を調整 |
| 落とし穴 | テストの使い回し，複雑さ＝精度の思い込み | 反転の要否の間違い，閾値の上げすぎ／下げすぎ，holdout を見て選ぶ |
| 止めどき | 精度とコストの兼ね合い | 開発用が頭打ち．現場データが変わったら再点検 |

---

## つまずきやすいポイント（第9章）

- **CNN に flatten した `(N, 784)` を渡すのは NG** → CNN の入力 shape は `(N, 1, 28, 28)` のまま
- **`32 * 7 * 7` はプーリング後のサイズで決まる** → MaxPool の数・サイズを変えるとこの値も変わる
- **`load_state_dict` の前にモデルクラスを定義しておく必要がある** → `.pth` は重みだけを保存しており，モデルの構造は保存されていない
- **色の向きを学習時と合わせる** → MNIST も Canvas も「黒背景・白い数字」なので Canvas 入力は反転しない．白い紙に黒ペンの画像のときだけ `1.0 - arr` で反転する（反転の要否を間違えると精度が大きく落ちる）
- **`map_location="cpu"` を指定** → GPU で保存したモデルを CPU 環境で読み込む場合に必要

### よくある質問

**Q: `x.view(x.size(0), -1)` の `-1` はなぜ安全か？**
A: `-1` は「残りの次元を自動計算せよ」という意味．`x.size(0)`（バッチサイズ）を固定しているので，残りの次元 `32*7*7=1568` が自動計算される．`view(-1, 1568)` と書いても同じ結果だが，`x.size(0)` を使う方がバッチサイズへの依存を明示できる．

**Q: `model.eval()` を呼び忘れると推論結果が変わるか？**
A: SimpleCNN（Dropout/BatchNorm なし）では変わらない．しかし Dropout や BatchNorm を持つモデルでは結果が変わる（推論ごとに異なる結果が出たり，BatchNorm の統計がおかしくなる）．習慣として必ず `model.eval()` を呼ぶこと．

**Q: CNN のフィルタ（重み）は何を学習しているか？**
A: 第1層はエッジ・勾配・色のパターンを検出するフィルタを学習する（可視化すると斜め線・水平線などに見える）．深い層になるほど，より複雑な形状（曲線・閉じた形・数字の部品）を検出するフィルタになる．

---

# 付録: よく使うコードスニペット集

## PyTorch の学習ループ（完全版テンプレート）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def run_training(model, epochs=5, lr=0.001, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        # --- 訓練フェーズ ---
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # --- 評価フェーズ ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        acc = correct / total
        history["test_acc"].append(acc)
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, TestAcc={acc:.4f}")

    return history

# 使用例
# history = run_training(SimpleCNN(), epochs=5)
```

## sklearn の Pipeline 完全版テンプレート

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_pipeline(num_cols, cat_cols, model=None):
    """数値列と カテゴリ列を自動処理する汎用パイプライン"""
    if model is None:
        model = LogisticRegression(max_iter=1000)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

# 使用例
# pipe = build_pipeline(
#     num_cols=["Age", "Fare", "Pclass"],
#     cat_cols=["Sex"],
# )
# pipe.fit(X_train, y_train)
# print(pipe.score(X_test, y_test))
```
