# Python の基礎 — 解説ガイド（q1〜q7 対応）

このドキュメントは，**q1〜q7 の各課題に直接対応した解説**です。  
各章の冒頭に「どの問題と対応しているか」を明示しているので，課題を解くときに参照してください。

---

## 目次

1. [第1回 — Pythonの基礎（q1対応）](#第1回)
2. [第2回 — ライブラリの基礎（q2対応）](#第2回)
3. [第3回 — 統計と単回帰分析（q3対応）](#第3回)
4. [第4回 — ライブラリの応用（q4対応）](#第4回)
5. [第5回 — 機械学習（q5対応）](#第5回)
6. [第6回 — モデル評価とチューニング（q6対応）](#第6回)
7. [第7回 — 最終課題（q7対応）](#第7回)

---

<a id="第1回"></a>

# 第1回 — Pythonの基礎

> この解説は **q1（第1回課題）** に対応しています。

## 概要

第1回の課題では，Python の基本的な構文（変数・演算・リスト・辞書型・条件分岐・ループ・関数・クラス）を使います。

---

## 解説1：変数と辞書型（問題1 に対応）

### 問題1のテーマ
辞書型変数で自己紹介を出力する。

### 変数とは
変数は「値を入れる箱」です。Python では `=` で値を代入するだけで変数を作れます（型宣言不要）。

```python
a = 12        # 整数（int）
b = 3.14      # 小数（float）
c = 'K'       # 1文字（str）
d = "Python"  # 文字列（str）

print('a:', a)   # → a: 12
print('d:', d)   # → d: Python
```

### 辞書型とは
辞書型（dict）は「キーと値をペアで管理するデータ構造」です。  
リンゴ200円・オレンジ150円のような「名前 → 値」の対応を表すのに使います。

```python
# 辞書の作成：{ キー: 値, キー: 値, ... }
data_dict = {'Apple': 200, 'Orange': 150, 'Melon': 400}
print(data_dict)          # → {'Apple': 200, 'Orange': 150, 'Melon': 400}

# キーを指定して値を取り出す
print(data_dict['Orange'])  # → 150
```

### 問題1 の解き方ヒント

```python
# 辞書型で自己紹介データを作る
profile = {
    'Name': '芝浦太郎',
    'ID number': 'AF19000',
    'Age': 20,
    'Birthplace': 'Tokyo'
}

# for文で出力する
for key, value in profile.items():
    print(key, ':', value)
```

> **ポイント**：辞書は `{キー: 値}` 形式で作り，`.items()` で全ペアを取り出せる。

---

## 解説2：演算子（問題2 に対応）

### 問題2のテーマ
A・B・C 3人が集めた飴の合計を3等分し，均等分配後の個数と余りを求める。

### 数値演算子

| 演算子 | 例 | 説明 |
|:---:|:---:|:---|
| `+` | `a + b` | 足し算 |
| `-` | `a - b` | 引き算 |
| `*` | `a * b` | 掛け算 |
| `/` | `a / b` | 割り算（小数の結果） |
| `//` | `a // b` | 商（整数部分のみ） |
| `%` | `a % b` | 余り |
| `**` | `a ** b` | aのb乗 |

### 問題2 の解き方ヒント

```python
candy = {'a': 121, 'b': 77, 'c': 109}

# 合計
total = candy['a'] + candy['b'] + candy['c']

# 1人分（整数除算）と余り
per_person = total // 3
remainder  = total % 3

print('1人分:', per_person)
print('余り:', remainder)
```

> **ポイント**：`//` は商の整数部分，`%` は余りを返す。問題2はこの2つが核心。

---

## 解説3：if 文・for 文・関数（問題3 に対応）

### 問題3のテーマ
2以上の自然数 N が与えられたとき，2〜N の範囲で素数だけを表示する関数 `primality_test` を実装する。

### if 文（条件分岐）

```python
data_list = [1, 2, 3, 4, 5]
value = 5

if value in data_list:
    print(f"{value} はリストに含まれている")
else:
    print(f"{value} はリストに含まれていない")
```

- `if (条件):` が True なら `if` ブロックを実行
- `else:` は条件が False のとき実行
- `elif (別の条件):` で複数条件を連鎖できる

### for 文（繰り返し処理）

```python
# リストから1つずつ取り出して繰り返す
for num in [1, 2, 3, 4]:
    print('num:', num)

# range(N) → 0, 1, 2, ..., N-1 の整数を生成
for i in range(6):
    print(i)

# range(start, stop, step) も使える
for i in range(1, 11, 2):  # 1, 3, 5, 7, 9
    print(i)
```

### 関数

```python
# def 関数名(引数):  という形で定義する
def calc_multi(a, b):
    return a * b   # return で値を返す

print(calc_multi(3, 10))  # → 30
```

### 問題3 の解き方ヒント

ある自然数 N が素数かどうかを判定するには「2 〜 N-1 の数でひとつも割り切れなければ素数」という考え方を使います。

```python
def primality_test(n: int):
    for num in range(2, n + 1):        # 2 〜 n まで順番に確認
        is_prime = True
        for i in range(2, num):        # 2 〜 num-1 で割り切れるか確認
            if num % i == 0:           # 割り切れたら素数ではない
                is_prime = False
                break
        if is_prime:
            print(num)

primality_test(10)
# → 2, 3, 5, 7
```

> **ポイント**：二重の for 文 + if 文の組み合わせ。`break` で内側のループを抜け出せる。

---

## 解説4：クラスとインスタンス（問題4 に対応）

### 問題4のテーマ
4つの数値に対して加算・減算・乗算・除算を行うクラス `Calculator` を作る。

### クラスとは
クラスは「オブジェクトの設計図（型）」です。  
「たい焼きの型」がクラスで，そこから作った「実物のたい焼き」がインスタンスです。

### コンストラクタ（`__init__`）
インスタンスを生成するときに自動で呼ばれる特別なメソッドです。  
`self` は「自分自身（このインスタンス）」を指します。

```python
class MyCalcClass:

    def __init__(self, x, y):
        # self.x = x は「このインスタンスの x 属性に x を代入」
        self.x = x
        self.y = y

    def calc_add(self):
        return self.x + self.y   # self.x で自分の属性を参照

    def calc_multi(self, a, b):
        return a * b

# インスタンスを生成（__init__ が自動で呼ばれる）
instance_1 = MyCalcClass(1, 2)   # self.x = 1, self.y = 2

print(instance_1.calc_add())     # → 3
print(instance_1.calc_multi(5, 2))  # → 10
```

### 問題4 の解き方ヒント

```python
class Calculator:

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def add(self):
        return self.a + self.b + self.c + self.d

    def sub(self):
        return self.a - self.b - self.c - self.d

    def multi(self):
        return self.a * self.b * self.c * self.d

    def div(self):
        return self.a / self.b / self.c / self.d

calc_1 = Calculator(64, 4, 4, 2)
print('加算:', calc_1.add())
print('減算:', calc_1.sub())
print('乗算:', calc_1.multi())
print('除算:', calc_1.div())
```

> **ポイント**：コンストラクタの第1引数は必ず `self`。  
> `self.変数名` で「このインスタンスの属性」として値を保持できる。

---

<a id="第2回"></a>

# 第2回 — ライブラリの基礎

> この解説は **q2（第2回課題）** に対応しています。

## 概要

第2回では NumPy・Pandas・Matplotlib という3大データサイエンスライブラリを使います。

### インポートの書き方

```python
import numpy as np             # NumPy を np として使う
import pandas as pd            # Pandas を pd として使う
import matplotlib.pyplot as plt  # Matplotlib を plt として使う
```

---

## 解説1：乱数生成（問題1 に対応）

### 問題1のテーマ
標準正規分布に従う乱数を10個生成し，最大値・最小値・合計を出力する。

### 乱数シードとは
**シード（seed）**を固定すると，何度実行しても同じ乱数が得られます。  
→ 実験の再現性を保つために使います。

```python
import numpy as np

np.random.seed(0)   # シードを0に固定
```

### 代表的な乱数生成関数

| 関数 | 説明 |
|:---|:---|
| `np.random.randn(N)` | 標準正規分布（平均0，標準偏差1）の乱数を N 個 |
| `np.random.rand(N)` | 一様分布（0〜1）の乱数を N 個 |
| `np.random.randint(low, high, N)` | 整数の乱数を N 個 |

### 配列の統計メソッド

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])

print(arr.max())   # 最大値
print(arr.min())   # 最小値
print(arr.sum())   # 合計
```

### 問題1 の解き方ヒント

```python
np.random.seed(0)

# 標準正規分布に従う乱数を10個生成
rand_data = np.random.randn(10)
print(rand_data)

print('最大値:', rand_data.max())
print('最小値:', rand_data.min())
print('合計:', rand_data.sum())
```

---

## 解説2：行列生成と演算（問題2 に対応）

### 問題2のテーマ
要素が全て 2 の 4×4 行列を作り，その行列の2乗を出力する。

### 行列の作り方

```python
# np.zeros：全要素が 0 の行列
print(np.zeros((2, 3), dtype=np.int64))

# np.ones：全要素が 1 の行列
print(np.ones((2, 3), dtype=np.int64))

# np.full：全要素が指定した値の行列
mat = np.full((4, 4), 2)   # 全要素が 2 の 4×4 行列
print(mat)
```

### 行列の演算

```python
arr1 = np.arange(9).reshape(3, 3)   # 0〜8 を 3×3 に変形
arr2 = np.arange(9, 18).reshape(3, 3)

# 行列の積（np.dot）：掛け算
print(np.dot(arr1, arr2))

# * は「要素ごとの積」なので行列の掛け算ではないことに注意！
print(arr1 * arr2)  # これは各要素の積
```

### 問題2 の解き方ヒント

```python
# 全要素が 2 の 4×4 行列
mat = np.full((4, 4), 2)
print(mat)

# 行列の2乗 = 自分自身と行列積
mat_squared = np.dot(mat, mat)
print(mat_squared)
```

---

## 解説3：Pandas DataFrame とデータ抽出（問題3 に対応）

### 問題3のテーマ
東京都の家賃相場データから，2LDK が 18万円以上の区だけを抽出する。

### DataFrame とは
DataFrame は「行と列を持つ表形式のデータ構造」です。

```python
import pandas as pd

# 辞書からDataFrameを作る
data = {
    'ワンルーム': [7.5, 7.0, 7.8],
    '2LDK':      [24.5, 14.9, 15.6]
}
idx = ['港区', '世田谷区', '台東区']

df = pd.DataFrame(data, index=idx)
print(df)
```

### 条件によるデータ抽出

```python
# 特定の列を取り出す
print(df['2LDK'])

# 条件で絞り込む（条件が True の行だけ残る）
filtered = df[df['2LDK'] >= 18]
print(filtered)
```

### 問題3 の解き方ヒント

```python
dic = {
    "ワンルーム": [7.5, 7.0, 7.8, 7.5, 7.8, 7.7],
    "1k":        [9.6, 8.0, 8.7, 8.9, 9.5, 8.9],
    "1LDK":      [17.5, 11.6, 12.4, 14.3, 16.2, 13.5],
    "2LDK":      [24.5, 14.9, 15.6, 19.1, 22.1, 17.5],
    "3LDK":      [31.9, 18.3, 19.0, 24.2, 28.4, 21.8]
}
idx = ["港区", "世田谷区", "台東区", "新宿区", "渋谷区", "目黒区"]
df_tky = pd.DataFrame(dic, index=idx)

# 2LDK が 18万円以上の行を抽出
result = df_tky[df_tky['2LDK'] >= 18]
print(result)
```

---

## 解説4：Matplotlib でグラフ描画（問題4 に対応）

### 問題4のテーマ
新型コロナウィルス新規陽性者数の日別推移を，グラフとして出力する。

### 基本的なグラフの描き方

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]

plt.figure(figsize=(16, 9))  # グラフのサイズ（幅×高さ インチ）
plt.plot(x, y)               # 折れ線グラフ
plt.xlabel('X軸のラベル')
plt.ylabel('Y軸のラベル')
plt.grid(True)               # グリッド（格子）を表示
plt.show()
```

### 問題4 の解き方ヒント

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSVをURLから読み込む（pandas の read_csv）
df_cvd = pd.read_csv(
    "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/covid/newly_confirmed_cases_daily.csv",
    index_col=None
)

x = pd.to_datetime(df_cvd["Date"])  # 日付列を日付型に変換
y = df_cvd["ALL"]                   # 全国の陽性者数

plt.figure(figsize=(16, 9))
plt.plot(x, y)
plt.xlabel("Date")
plt.ylabel("ALL")
plt.grid(True)
plt.show()
```

> **ポイント**：`pd.to_datetime()` で文字列の日付を日付型に変換するのを忘れずに。

---

<a id="第3回"></a>

# 第3回 — 統計と単回帰分析

> この解説は **q3（第3回課題）** に対応しています。

## 概要

第3回では，Pandas の統計メソッド・相関係数・Seaborn・Scikit-learn の線形回帰を使います。

---

## 解説1：記述統計量（問題1 に対応）

### 問題1のテーマ
ポルトガル語の成績データ（G1, G2, G3）の平均値・標準偏差・中央値を求める。

### 記述統計とは
データの特徴を数値で表す手法です。

| 統計量 | メソッド | 意味 |
|:---|:---|:---|
| 平均値 | `.mean()` | 全データの合計 ÷ データ数 |
| 標準偏差 | `.std()` | データのばらつきの大きさ |
| 中央値 | `.median()` | 全データを並べたときの真ん中の値 |

```python
import pandas as pd

df = pd.read_csv("student-por.csv", sep=";")

# 特定の列に対して統計量を求める
print('G1の平均値:', df['G1'].mean())
print('G1の標準偏差:', df['G1'].std())
print('G1の中央値:', df['G1'].median())
```

### 問題1 の解き方ヒント

```python
for col in ['G1', 'G2', 'G3']:
    print(f"--- {col} ---")
    print(f"平均値: {df_student_por[col].mean():.3f}")
    print(f"標準偏差: {df_student_por[col].std():.3f}")
    print(f"中央値: {df_student_por[col].median():.3f}")
```

---

## 解説2：相関係数と散布図（問題2 に対応）

### 問題2のテーマ
G1, G2, G3 の相関係数を求め，散布図とヒストグラムを Seaborn で描画する。

### 相関係数とは
2つの変数の**関係の強さ**を -1〜1 の値で表したものです。

- **1 に近い** → 正の相関（片方が大きいと，もう片方も大きい）
- **-1 に近い** → 負の相関（片方が大きいと，もう片方は小さい）
- **0 に近い** → 無相関（関係が弱い）

```python
# DataFrame全体の相関係数行列
print(df[['G1', 'G2', 'G3']].corr())
```

### Seaborn の pairplot

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 散布図＋ヒストグラムを一度に描画
sns.pairplot(df[['G1', 'G2', 'G3']])
plt.show()
```

> **pairplot** は，変数の組み合わせ全ての散布図と，対角線上に各変数のヒストグラムを自動で描いてくれます。

### 問題2 の解き方ヒント

```python
# 相関係数
print(df_student_por[['G1', 'G2', 'G3']].corr())

# 散布図とヒストグラム
sns.pairplot(df_student_por[['G1', 'G2', 'G3']])
plt.show()
```

---

## 解説3：線形単回帰分析（問題3 に対応）

### 問題3のテーマ
G2（説明変数）から G3（目的変数）を予測する線形回帰モデルを構築し，回帰係数・切片・決定係数を求める。

### 単回帰分析とは
「y = ax + b」という直線でデータを近似する手法です。

- **a（回帰係数）**：直線の傾き
- **b（切片）**：直線が y 軸と交わる点
- 最小二乗法によって，実測値と予測値の差の二乗和が最小となる a, b を求める

### Scikit-learn での線形回帰

```python
from sklearn import linear_model

# 説明変数（2次元配列である必要がある）
x = df_student_mat.loc[:, ['G1']].values
# 目的変数
y = df_student_mat['G3'].values

# モデルを作成して学習（fit）
reg = linear_model.LinearRegression()
reg.fit(x, y)

print('回帰係数:', reg.coef_)     # a
print('切片:', reg.intercept_)    # b
print('決定係数:', reg.score(x, y))  # R²
```

### 決定係数（R²）とは
予測モデルの当てはまりの良さを示す指標（0〜1）。  
→ **1 に近いほど当てはまりが良い**（完璧な予測で R²=1）。

### 散布図に回帰直線を重ねる

```python
import matplotlib.pyplot as plt

plt.scatter(x, y, label='実測値')
plt.plot(x, reg.predict(x), color='red', label='回帰直線')
plt.xlabel('G2')
plt.ylabel('G3')
plt.legend()
plt.grid(True)
plt.show()
```

### 問題3 の解き方ヒント

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 説明変数・目的変数
x = df_student_por[['G2']].values   # 2次元配列が必要
y = df_student_por['G3'].values

# モデルの学習
reg = LinearRegression()
reg.fit(x, y)

print('回帰係数:', reg.coef_[0])
print('切片:', reg.intercept_)
print('決定係数:', reg.score(x, y))

# 散布図＋回帰直線
plt.scatter(x, y)
plt.plot(x, reg.predict(x), color='red')
plt.xlabel('G2')
plt.ylabel('G3')
plt.grid(True)
plt.show()
```

---

<a id="第4回"></a>

# 第4回 — ライブラリの応用

> この解説は **q4（第4回課題）** に対応しています。

## 概要

第4回では NumPy の高度な操作（インデックス参照・演算・配列結合）と Pandas の高度なデータ操作（groupby・欠損値・時系列）を扱います。

---

## 解説1：ブールインデックスと演算（問題1 に対応）

### 問題1のテーマ
標準正規分布に従う 4×4 行列を生成し，値が 0 以上の要素を抽出してその数を出力する。

### ブールインデックス参照
「True/False の配列」を使ってデータを絞り込む方法です。

```python
import numpy as np

sample_matrix = np.random.randn(4, 4)
print(sample_matrix)

# 0 以上の要素だけを抽出する
mask = sample_matrix >= 0           # True/False の行列を作る
print(mask)

result = sample_matrix[mask]        # True の位置の要素だけ取り出す
print(result)

# True の数 = 0 以上の要素数
print('0以上の要素数:', mask.sum())  # True は 1, False は 0 なのでsumで数える
```

### 問題1 の解き方ヒント

```python
np.random.seed(0)

# 4×4 の乱数行列を生成
mat = np.random.randn(4, 4)

# 0 以上の要素を抽出
positive_elements = mat[mat >= 0]

print('0以上の要素:', positive_elements)
print('0以上の要素数:', (mat >= 0).sum())
```

---

## 解説2：配列操作とブロードキャスト（問題2 に対応）

### 問題2のテーマ
2つの行列を行方向（縦方向）に結合し，全要素に 10 を加えた行列を出力する。

### 配列の結合（concatenate）

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])

# axis=0：行方向（縦）に結合
result_vertical = np.concatenate([a, b], axis=0)
print(result_vertical)
# → [[ 1  2  3]
#    [ 4  5  6]
#    [ 7  8  9]
#    [10 11 12]]

# axis=1：列方向（横）に結合
result_horizontal = np.concatenate([a, b], axis=1)
print(result_horizontal)
```

### ブロードキャスト
配列と単一の数値を演算するとき，NumPy は自動的にその数値を配列の形に合わせてくれます。

```python
arr = np.arange(10)
print(arr + 3)   # [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

### 問題2 の解き方ヒント

```python
sample_array1 = np.arange(1, 11).reshape(2, 5)
sample_array2 = np.arange(11, 21).reshape(2, 5)

# 行方向（axis=0）に結合
combined = np.concatenate([sample_array1, sample_array2], axis=0)
print(combined)

# 全要素に 10 を加える（ブロードキャスト）
result = combined + 10
print(result)
```

---

## 解説3：groupby と unstack と dropna（問題3 に対応）

### 問題3のテーマ
student-mat.csv を用いて，年齢（age）と性別（sex）を軸とした G1 の平均点を算出し，縦軸が年齢・横軸が性別の DataFrame を作る。

### groupby（グループ集計）
ある列を軸にしてデータをグループ化し，集計します。

```python
import pandas as pd

df = pd.read_csv("student-mat.csv", sep=";")

# age と sex でグループ化して G1 の平均値を求める
grouped = df.groupby(['age', 'sex'])['G1'].mean()
print(grouped)
```

### unstack（行を列に変換）
階層型インデックスの一番下の行を列へ転換します。

```python
# sex（性別）を行から列へ移動
result = grouped.unstack()
print(result)
```

### dropna（欠損値の削除）

```python
# NaN を含む行をすべて削除
result_clean = result.dropna()
print(result_clean)
```

### 問題3 の解き方ヒント

```python
df_student_mat = pd.read_csv("student-mat.csv", sep=";")

# age・sex を軸に G1 の平均値を算出
grouped = df_student_mat.groupby(['age', 'sex'])['G1'].mean()

# sex を列方向へ
result = grouped.unstack()
print(result)

# NaN 行を削除
result_clean = result.dropna()
print(result_clean)
```

---

## 解説4：移動平均と時系列データ（問題4 に対応）

### 問題4のテーマ
コロナ陽性者数のグラフに **7日間移動平均** を重ね，**実効再生産数** を算出してグラフ化する。

### 移動平均（rolling）
過去 N 日分のデータの平均を取り続けることで，短期的な変動を滑らかにする手法です。

```python
import pandas as pd

# rolling(N)：直近 N 件のデータを対象にする
# .mean()：その N 件の平均値を計算する
moving_avg = df['ALL'].rolling(7).mean()
```

### shift（データをずらす）
データを N 行分前後にずらします。「前週のデータと比較」などに使います。

```python
# shift(7)：7行分だけデータを後ろへずらす（7日前のデータを現在の行に持ってくる）
prev_week = df['ALL'].rolling(7).sum().shift(7)
```

### 実効再生産数の計算式

$$R_t = \left(\frac{\text{直近7日合計}}{\text{前7日合計}}\right)^{\frac{2}{7}}$$

- **直近7日合計**：`y.rolling(7).sum()`
- **前7日合計**：`y.rolling(7).sum().shift(7)`
- **世代時間**：2（2022年以降の簡易計算）

### 問題4 の解き方ヒント

```python
import pandas as pd
import matplotlib.pyplot as plt

df_cvd = pd.read_csv(
    "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/covid/newly_confirmed_cases_daily.csv",
    index_col=None
)

x = pd.to_datetime(df_cvd["Date"])
y = df_cvd["ALL"]

# 7日間移動平均
moving_avg = y.rolling(7).mean()

# グラフ（移動平均を重ねる）
plt.figure(figsize=(16, 9))
plt.plot(x, y, label='陽性者数')
plt.plot(x, moving_avg, label='7日間移動平均', color='red')
plt.legend()
plt.grid(True)
plt.show()

# 実効再生産数
recent_7 = y.rolling(7).sum()
prev_7   = y.rolling(7).sum().shift(7)
Rt = (recent_7 / prev_7) ** (2 / 7)

plt.figure(figsize=(16, 9))
plt.plot(x, Rt)
plt.xlabel("Date")
plt.ylabel("実効再生産数")
plt.grid(True)
plt.show()
```

---

<a id="第5回"></a>

# 第5回 — 機械学習

> この解説は **q5（第5回課題）** に対応しています。

## 概要

第5回では，タイタニック号乗客データを使って **決定木** による生存予測モデルを構築します。  
機械学習の基本的な流れ（6ステップ）に沿って学びます。

---

## 機械学習の6ステップ

```
STEP 1. データ読み込み
STEP 2. データ前処理
STEP 3. データ分割（目的変数・説明変数 / 訓練・テスト）
STEP 4. スケーリング（今回は省略）
STEP 5. モデル学習
STEP 6. モデル評価
```

---

## 解説1：データ確認（欠損値・データ型）

### データの形式と欠損の確認

```python
import pandas as pd

df = pd.read_csv("titanic.csv")

# データの形式（行数，列数）
print(df.shape)

# 欠損値の数（各列）
print(df.isnull().sum())

# データ型と欠損数を一度に確認
df.info()
```

### データ型の確認

```python
print(df.dtypes)
```

---

## 解説2：データ前処理（問題の核心）

### 欠損値の削除（dropna）

```python
# NaN を含む行を削除
df_clean = df.dropna()
print(df_clean.shape)  # 削除後の形状を確認
```

### ダミー変数化（get_dummies）
「male / female」のようなカテゴリ変数を 0/1 の数値に変換します。  
→ 機械学習モデルは**数値しか扱えない**ため，この変換が必要です。

```python
# Sex 列のダミー変数化（male列，female列に分かれる）
sex_dummies = pd.get_dummies(df_clean['Sex'])
print(sex_dummies)

# Sex を除いた元のDataFrame と結合する
df_final = df_clean.drop('Sex', axis=1).join(sex_dummies)
```

---

## 解説3：データ分割

### 説明変数と目的変数

- **目的変数（y）**：予測したい値 → `Survived`（0: 死亡，1: 生存）
- **説明変数（X）**：予測に使う値 → `Pclass, Age, SibSp, Parch, Fare, male, female`

```python
# 説明変数
X = df_final[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'female']]

# 目的変数
y = df_final['Survived']
```

### 訓練データとテストデータの分割

```python
from sklearn.model_selection import train_test_split

# 80% 訓練，20% テスト
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
```

---

## 解説4：決定木モデルの学習と評価

### 決定木とは
データの各特徴を使って「Yes/No」の条件分岐を繰り返し，予測を行うモデルです。  
→ 直感的に理解しやすい（木構造で可視化できる）

### Scikit-learn での決定木

```python
from sklearn.tree import DecisionTreeClassifier

# モデルの生成
model = DecisionTreeClassifier(criterion='entropy', max_depth=5)

# 訓練データで学習
model.fit(X_train, y_train)
```

### モデルの評価（正解率）

```python
# 訓練データとテストデータそれぞれの正解率
print('正解率(train):', model.score(X_train, y_train))
print('正解率(test):', model.score(X_test, y_test))
```

> **ポイント**：訓練データとテストデータの正解率が大きく異なる場合は**過学習（オーバーフィッティング）**が起きている可能性があります。

---

<a id="第6回"></a>

# 第6回 — モデル評価とチューニング

> この解説は **q6（第6回課題）** に対応しています。

## 概要

第6回では，機械学習モデルの精度をより正確に評価・改善するための手法を学びます。

---

## 解説1：グリッドサーチと交差検証（問題1 に対応）

### 問題1のテーマ
乳がんデータセットに対して，決定木を使ってグリッドサーチと5分割交差検証を行う。

### 交差検証法（k-fold cross validation）
データを k 個のブロックに分割し，1 ブロックを検証用・残りを訓練用として k 回繰り返す方法です。  
→ ホールドアウト法（単純な2分割）より信頼性の高い評価ができます。

```
k=4 の場合のイメージ：

1回目: [検証] [訓練] [訓練] [訓練]
2回目: [訓練] [検証] [訓練] [訓練]
3回目: [訓練] [訓練] [検証] [訓練]
4回目: [訓練] [訓練] [訓練] [検証]
```

### グリッドサーチ（GridSearchCV）
「木の深さ」などの**ハイパーパラメータ**の全組み合わせを試し，最も良い組み合わせを自動で探します。

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# 探索するパラメータの範囲を指定
param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [2, 3, 4, 5]
}

# GridSearchCV：グリッドサーチ + 5分割交差検証
tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
gs = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5)

# 訓練データで探索
gs.fit(X_train, y_train)

# 結果を確認
print('最良スコア:', gs.best_score_)
print('最良パラメータ:', gs.best_params_)
print('テストスコア:', gs.score(X_test, y_test))
```

### 問題1 の解き方ヒント

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0
)

tree = DecisionTreeClassifier(criterion='entropy', random_state=0)

param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [2, 3, 4, 5]
}

gs = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5)
gs.fit(x_train, y_train)

print('最良スコア:', gs.best_score_)
print('最良パラメータ:', gs.best_params_)
```

---

## 解説2：混同行列と評価指標（問題2 に対応）

### 問題2のテーマ
ロジスティック回帰モデルを構築し，混同行列と評価指標（正解率・適合率・再現率・F1スコア）を求める。

### 混同行列（Confusion Matrix）
予測値と正解の対応を 2×2 の表で表したものです。

|  | 予測: 正例 | 予測: 負例 |
|:---:|:---:|:---:|
| **正解: 正例** | TP（正解） | FN（見逃し） |
| **正解: 負例** | FP（誤検知） | TN（正解） |

### 評価指標の意味

| 指標 | 計算式 | 意味 |
|:---|:---|:---|
| **正解率** (Accuracy) | (TP + TN) / 全体 | 全体の中で正しく予測できた割合 |
| **適合率** (Precision) | TP / (TP + FP) | 「正例」と予測した中で本当に正例だった割合 |
| **再現率** (Recall) | TP / (TP + FN) | 実際の正例のうち正しく検出できた割合 |
| **F1スコア** | 2×Precision×Recall / (Precision+Recall) | 適合率と再現率の調和平均 |

### Scikit-learn での取得方法

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score)

# モデルの構築・学習
model = LogisticRegression()
model.fit(x_train, y_train)

# テストデータで予測
y_pred = model.predict(x_test)

# 混同行列
print(confusion_matrix(y_test, y_pred))

# 各評価指標
print('正解率:', accuracy_score(y_test, y_pred))
print('適合率:', precision_score(y_test, y_pred))
print('再現率:', recall_score(y_test, y_pred))
print('F1スコア:', f1_score(y_test, y_pred))
```

---

## 解説3：複数モデルの比較（問題3 に対応）

### 問題3のテーマ
ロジスティック回帰・SVM・決定木・k-NN の4モデルをホールドアウト法で比較し，最も良いモデルを確認する。

### ホールドアウト法とは
データを訓練用とテスト用に2分割し，テストデータでモデルを評価する方法です。

### 複数モデルをまとめて評価するパターン

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# モデルを辞書にまとめる
models = {
    'LogisticRegression': LogisticRegression(),
    'SVM':                LinearSVC(),
    '決定木':              DecisionTreeClassifier(criterion='entropy', random_state=0),
    'k-NN':               KNeighborsClassifier()
}

# 各モデルを学習・評価
for name, model in models.items():
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f'{name}: {score:.4f}')
```

---

<a id="第7回"></a>

# 第7回 — 最終課題（総復習）

> この解説は **q7（第7回課題）** に対応しています。

## 概要

第7回では，Scikit-learn の線形回帰を**自分で実装**します。  
第1回で学んだクラス・第3回で学んだ数式・NumPy だけを使います。

---

## 解説1：線形単回帰の数式（実装の土台）

予測したい直線は：

$$y = ax + b$$

### 傾き a の求め方（最小二乗法）

$$a = \frac{\sum_{i=1}^n x_i y_i - \frac{1}{n} \sum_{i=1}^n x_i \sum_{i=1}^n y_i}{\sum_{i=1}^n x_i^2 - \frac{1}{n}\left(\sum_{i=1}^n x_i\right)^2}$$

### 切片 b の求め方

$$b = \frac{1}{n}\sum_{i=1}^n (y_i - ax_i) = \bar{y} - a\bar{x}$$

### 決定係数 R² の求め方

$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{j=1}^n (y_j - \bar{y})^2}$$

ここで $\hat{y}_i = ax_i + b$（予測値），$\bar{y}$（目的変数の平均値）

---

## 解説2：NumPy の関数と数式の対応

| 数式の表現 | NumPy のコード |
|:---|:---|
| $n$（サンプル数） | `len(x)` |
| $\sum x_i y_i$ | `np.dot(x, y)` |
| $\sum x_i$ | `np.sum(x)` |
| $\bar{x}$（x の平均） | `np.mean(x)` |
| $\hat{y}_i$（予測値） | `a * x + b` |

---

## 解説3：クラスの実装方法

第1回で学んだクラスの書き方をそのまま使います。

```python
import numpy as np

class MyLinearRegression:

    def __init__(self):
        self.a = None   # 傾き
        self.b = None   # 切片

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        最小二乗法で傾き a と切片 b を求める（学習）
        """
        n = len(x)

        # 傾き a の計算（式 (2)）
        self.a = (np.dot(x, y) - (1/n) * np.sum(x) * np.sum(y)) / \
                 (np.dot(x, x) - (1/n) * (np.sum(x) ** 2))

        # 切片 b の計算（式 (1)）
        self.b = np.mean(y) - self.a * np.mean(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        y = ax + b で予測値を計算する
        """
        return self.a * x + self.b

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        決定係数 R² を計算する
        """
        y_hat = self.predict(x)                      # 予測値
        ss_res = np.sum((y - y_hat) ** 2)            # 残差平方和
        ss_tot = np.sum((y - np.mean(y)) ** 2)       # 全平方和
        return 1 - ss_res / ss_tot
```

---

## 解説4：実際のデータで回帰モデルを動かす

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# データ読み込み
df_math = pd.read_csv(
    "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/student/student-mat.csv",
    sep=';'
)

# 説明変数と目的変数
x = df_math['G1'].values   # 一学期の成績
y = df_math['G3'].values   # 三学期の成績

# 散布図を描画
plt.scatter(x, y)
plt.xlabel('G1（一学期の成績）')
plt.ylabel('G3（三学期の成績）')
plt.grid(True)
plt.show()

# 訓練データとテストデータの分割（80:20）
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# 自作モデルで学習
model = MyLinearRegression()
model.fit(x_train, y_train)

print(f'傾き: {model.a:.4f}')
print(f'切片: {model.b:.4f}')
print(f'決定係数(train): {model.score(x_train, y_train):.4f}')
print(f'決定係数(test):  {model.score(x_test, y_test):.4f}')

# 回帰直線を重ねて散布図を描画
plt.scatter(x, y, label='データ点')
plt.plot(sorted(x), [model.predict(xi) for xi in sorted(x)],
         color='red', label='回帰直線')
plt.xlabel('G1（一学期の成績）')
plt.ylabel('G3（三学期の成績）')
plt.legend()
plt.grid(True)
plt.show()
```

> **ポイント**：
> - `fit` メソッドで傾き `a` と切片 `b` を計算する（学習）
> - `predict` メソッドで `a*x + b` を計算する（予測）
> - `score` メソッドで決定係数 R² を計算する（評価）
> - この3つが Scikit-learn の全モデルに共通するインターフェースです。

---

## まとめ：各回の解説と問題の対応表

| 回 | 問題 | 使う主な概念 |
|:---:|:---|:---|
| **q1** | 問題1 | 変数・辞書型・`.items()` |
| **q1** | 問題2 | 演算子（`//`, `%`） |
| **q1** | 問題3 | if文・for文・range・関数 |
| **q1** | 問題4 | クラス・コンストラクタ・`self` |
| **q2** | 問題1 | `np.random.seed`・`randn`・`max/min/sum` |
| **q2** | 問題2 | `np.full`・`np.dot` |
| **q2** | 問題3 | `pd.DataFrame`・条件抽出 |
| **q2** | 問題4 | `plt.plot`・`figsize`・`grid`・`pd.to_datetime` |
| **q3** | 問題1 | `.mean()`・`.std()`・`.median()` |
| **q3** | 問題2 | `.corr()`・`sns.pairplot` |
| **q3** | 問題3 | `LinearRegression`・`.fit()`・`.score()` |
| **q4** | 問題1 | ブールインデックス・`.sum()` |
| **q4** | 問題2 | `np.concatenate`・ブロードキャスト |
| **q4** | 問題3 | `.groupby()`・`.unstack()`・`.dropna()` |
| **q4** | 問題4 | `.rolling(7).mean()`・`.shift(7)` |
| **q5** | 全問 | `get_dummies`・`train_test_split`・`DecisionTreeClassifier`・`.score()` |
| **q6** | 問題1 | `GridSearchCV`・`cv=5`（交差検証） |
| **q6** | 問題2 | `confusion_matrix`・`accuracy_score`・`precision_score`・`recall_score`・`f1_score` |
| **q6** | 問題3 | ホールドアウト法・複数モデル比較 |
| **q7** | 全問 | クラス実装・最小二乗法・`np.dot`・`np.mean`・`np.sum` |
