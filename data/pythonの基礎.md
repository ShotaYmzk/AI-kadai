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
a = 12        # 整数（int）を変数 a に入れる
b = 3.14      # 小数（float）を変数 b に入れる
c = 'K'       # 1文字（str）を変数 c に入れる
d = "Python"  # 文字列（str）を変数 d に入れる

print('a:', a)   # → a: 12
print('d:', d)   # → d: Python
```

### 辞書型とは

辞書型（dict）は「キーと値をペアで管理するデータ構造」です。  
リンゴ200円・オレンジ150円のような「名前 → 値」の対応を表すのに使います。

```python
# { キー: 値, キー: 値, ... } という形で辞書を作る
data_dict = {'Apple': 200, 'Orange': 150, 'Melon': 400}
print(data_dict)          # → {'Apple': 200, 'Orange': 150, 'Melon': 400}

# 辞書名['キー'] とすると，そのキーに対応する値を取り出せる
print(data_dict['Orange'])  # → 150
```

### 問題1 の解き方ヒント

```python
# ---- Step 1: 辞書型で自己紹介データを作る ----
# キーが「項目名」，値が「その内容」になるように辞書を作る
profile = {
    'Name': '芝浦太郎',      # 名前
    'ID number': 'AF19000',  # 学籍番号
    'Age': 20,               # 年齢
    'Birthplace': 'Tokyo'    # 出身地
}

# ---- Step 2: for文で1行ずつ出力する ----
# .items() は辞書の「キーと値のペア」を全部取り出すメソッド
# → 1回目のループ: key='Name', value='芝浦太郎'
# → 2回目のループ: key='ID number', value='AF19000'  ...のように繰り返す
for key, value in profile.items():
    print(key, ':', value)
# 出力例：
# Name : 芝浦太郎
# ID number : AF19000
# Age : 20
# Birthplace : Tokyo
```

> **ポイント**：辞書は `{キー: 値}` 形式で作り，`.items()` で全ペアを取り出せる。

---

## 解説2：演算子（問題2 に対応）

### 問題2のテーマ

A・B・C 3人が集めた飴の合計を3等分し，均等分配後の個数と余りを求める。

### 数値演算子


| 演算子  | 例        | 説明                   |
| ---- | -------- | -------------------- |
| `+`  | `a + b`  | 足し算                  |
| `-`  | `a - b`  | 引き算                  |
| `*`  | `a * b`  | 掛け算                  |
| `/`  | `a / b`  | 割り算（結果は小数になる）        |
| `//` | `a // b` | 商（整数部分のみ。小数点以下は切り捨て） |
| `%`  | `a % b`  | 余り（割り切れないときに残る数）     |
| `*`* | `a ** b` | aのb乗                 |


```python
print(10 + 3)   # → 13
print(10 // 3)  # → 3  （10÷3=3余り1 の「3」の部分）
print(10 % 3)   # → 1  （10÷3=3余り1 の「1」の部分）
print(10 ** 3)  # → 1000 （10の3乗）
```

### 問題2 の解き方ヒント

```python
# ---- データの準備 ----
# A・B・Cさんが集めた飴の数を辞書で管理する
candy = {'a': 121, 'b': 77, 'c': 109}

# ---- Step 1: 全員分の合計を計算する ----
# 辞書から各人の飴の数を取り出して足す
total = candy['a'] + candy['b'] + candy['c']
# → total = 121 + 77 + 109 = 307

# ---- Step 2: 3人で均等に分けたときの1人分を求める ----
# // を使うと「整数の商」が得られる（小数点以下は切り捨て）
per_person = total // 3
# → 307 // 3 = 102  （102個ずつ配れる）

# ---- Step 3: 余りを求める ----
# % を使うと「割り算の余り」が得られる
remainder = total % 3
# → 307 % 3 = 1  （1個余る）

print('1人分:', per_person)   # → 1人分: 102
print('余り:', remainder)     # → 余り: 1
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

# if (条件式): と書く
# 条件式が True（正しい）のとき，インデントされた処理を実行する
if value in data_list:          # value がリストに含まれているか？
    print(f"{value} はリストに含まれている")   # True のとき実行
else:
    print(f"{value} はリストに含まれていない")  # False のとき実行
```

- `if (条件):` が True なら `if` ブロックを実行
- `else:` は条件が False のとき実行
- `elif (別の条件):` で複数条件を連鎖できる

### for 文（繰り返し処理）

```python
# リストから1つずつ取り出して，なくなるまで繰り返す
# num には 1, 2, 3, 4 が順番に入る
for num in [1, 2, 3, 4]:
    print('num:', num)   # → num: 1, num: 2, ... と4回出力される

# range(N) → 0, 1, 2, ..., N-1 の整数を順に生成する
for i in range(6):      # 0, 1, 2, 3, 4, 5 の6回繰り返す
    print(i)

# range(開始, 終了, ステップ)：開始から終了-1まで，ステップずつ増やす
for i in range(1, 11, 2):  # 1, 3, 5, 7, 9（奇数だけ）
    print(i)
```

### 関数

```python
# def 関数名(引数1, 引数2, ...): という形で関数を定義する
# 引数 = 関数に渡す入力値，return = 関数から返す出力値
def calc_multi(a, b):
    return a * b   # a と b を掛けた結果を返す

# 関数を呼び出す（引数に 3 と 10 を渡す）
print(calc_multi(3, 10))  # → 30
```

### 問題3 の解き方ヒント

ある自然数 N が素数かどうかを判定するには「2 〜 N-1 の数でひとつも割り切れなければ素数」という考え方を使います。

```python
# def で関数を定義する。引数 n に調べる上限の数を受け取る
def primality_test(n: int):

    # 2 〜 n まで，1つずつ「この数は素数か？」を確認する
    for num in range(2, n + 1):

        # まず「素数だと仮定する」フラグを True にしておく
        is_prime = True

        # 2 〜 num-1 の数で割り切れるものがあるか調べる
        for i in range(2, num):
            if num % i == 0:       # i で割り切れた → 素数ではない
                is_prime = False   # フラグを False に変える
                break              # これ以上調べる必要がないのでループを抜ける

        # 内側のループが終わった時点で is_prime が True なら素数
        if is_prime:
            print(num)   # 素数なので表示する

primality_test(10)
# → 2
# → 3
# → 5
# → 7
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

    # コンストラクタ：インスタンスを作るとき（MyCalcClass(1, 2) と書いたとき）に自動で呼ばれる
    # self = このインスタンス自身，x と y は外から渡される引数
    def __init__(self, x, y):
        self.x = x   # 「このインスタンスの x」として値を保存する
        self.y = y   # 「このインスタンスの y」として値を保存する

    # メソッド：インスタンスに属する関数（第1引数は必ず self）
    def calc_add(self):
        return self.x + self.y   # 保存しておいた x と y を使って計算

    def calc_multi(self, a, b):  # インスタンスの値は使わず，引数 a, b で計算
        return a * b

# MyCalcClass(1, 2) → __init__(self, x=1, y=2) が呼ばれる
# self.x = 1, self.y = 2 として保存される
instance_1 = MyCalcClass(1, 2)

# instance_1 の x と y を足す → 1 + 2 = 3
print(instance_1.calc_add())        # → 3
print(instance_1.calc_multi(5, 2))  # → 10
```

### 問題4 の解き方ヒント

```python
# ---- クラスの定義 ----
class Calculator:

    # コンストラクタ：Calculator(64, 4, 4, 2) と書いたときに呼ばれる
    # 4つの数値 a, b, c, d を受け取って self に保存する
    def __init__(self, a, b, c, d):
        self.a = a   # 「このインスタンスの a」に保存
        self.b = b   # 「このインスタンスの b」に保存
        self.c = c   # 「このインスタンスの c」に保存
        self.d = d   # 「このインスタンスの d」に保存

    # 加算メソッド：a + b + c + d を返す
    def add(self):
        return self.a + self.b + self.c + self.d

    # 減算メソッド：a - b - c - d を返す
    def sub(self):
        return self.a - self.b - self.c - self.d

    # 乗算メソッド：a * b * c * d を返す
    def multi(self):
        return self.a * self.b * self.c * self.d

    # 除算メソッド：a / b / c / d を返す
    def div(self):
        return self.a / self.b / self.c / self.d


# ---- インスタンスを作って使う ----
# Calculator(64, 4, 4, 2) → self.a=64, self.b=4, self.c=4, self.d=2 に保存される
calc_1 = Calculator(64, 4, 4, 2)

print('加算:', calc_1.add())    # → 64 + 4 + 4 + 2 = 74
print('減算:', calc_1.sub())    # → 64 - 4 - 4 - 2 = 54
print('乗算:', calc_1.multi())  # → 64 * 4 * 4 * 2 = 2048
print('除算:', calc_1.div())    # → 64 / 4 / 4 / 2 = 2.0
```

> **ポイント**：コンストラクタの第1引数は必ず `self`。  
> `self.変数名` で「このインスタンスの属性」として値を保持できる。

---



# 第2回 — ライブラリの基礎

> この解説は **q2（第2回課題）** に対応しています。

## 概要

第2回では NumPy・Pandas・Matplotlib という3大データサイエンスライブラリを使います。

### ライブラリのインポート

ライブラリは「便利な機能をまとめたパッケージ」です。使う前に `import` で読み込む必要があります。

```python
# numpy を「np」という名前で使えるようにする（慣習的な略称）
import numpy as np

# pandas を「pd」という名前で使えるようにする
import pandas as pd

# matplotlib の pyplot モジュールを「plt」として使えるようにする
import matplotlib.pyplot as plt
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

# シードを 0 に固定する
# → この後の乱数生成が，誰がいつ実行しても同じ結果になる
np.random.seed(0)
```

### 代表的な乱数生成関数


| 関数                                | 説明                        |
| --------------------------------- | ------------------------- |
| `np.random.randn(N)`              | 標準正規分布（平均0，標準偏差1）の乱数を N 個 |
| `np.random.rand(N)`               | 一様分布（0〜1）の乱数を N 個         |
| `np.random.randint(low, high, N)` | low 以上 high 未満の整数の乱数を N 個 |


### 配列の統計メソッド

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])

print(arr.max())   # 配列の中で最も大きい値 → 9
print(arr.min())   # 配列の中で最も小さい値 → 1
print(arr.sum())   # 配列の全要素を足した合計 → 39
```

### 問題1 の解き方ヒント

```python
import numpy as np

# ---- Step 1: シードを固定して再現性を確保する ----
np.random.seed(0)   # 誰が実行しても同じ乱数が出る

# ---- Step 2: 標準正規分布に従う乱数を 10 個生成する ----
# randn(10) → 平均 0，標準偏差 1 の正規分布から 10 個の数値を生成
rand_data = np.random.randn(10)
print(rand_data)   # → [ 1.764  0.4    0.978 ...] のような数値が10個

# ---- Step 3: 最大値・最小値・合計を出力する ----
print('最大値:', rand_data.max())   # 10個の中で最も大きい値
print('最小値:', rand_data.min())   # 10個の中で最も小さい値
print('合計:', rand_data.sum())     # 10個を全て足した値
```

---

## 解説2：行列生成と演算（問題2 に対応）

### 問題2のテーマ

要素が全て 2 の 4×4 行列を作り，その行列の2乗を出力する。

### 行列の作り方

```python
import numpy as np

# np.zeros((行数, 列数)) → 全要素が 0 の行列を作る
print(np.zeros((2, 3), dtype=np.int64))
# → [[0 0 0]
#    [0 0 0]]

# np.ones((行数, 列数)) → 全要素が 1 の行列を作る
print(np.ones((2, 3), dtype=np.int64))
# → [[1 1 1]
#    [1 1 1]]

# np.full((行数, 列数), 埋める値) → 指定した値で全要素を埋めた行列を作る
mat = np.full((4, 4), 2)   # 全要素が 2 の 4×4 行列
print(mat)
# → [[2 2 2 2]
#    [2 2 2 2]
#    [2 2 2 2]
#    [2 2 2 2]]
```

### 行列の積（掛け算）

```python
arr1 = np.arange(9).reshape(3, 3)   # 0〜8 を 3×3 に変形
arr2 = np.arange(9, 18).reshape(3, 3)

# np.dot(A, B) → 行列 A と行列 B の「行列積（掛け算）」を計算する
# ※ 数学の行列の掛け算（列×行）
print(np.dot(arr1, arr2))

# * を使うと「各要素どうしの積」になる（行列の掛け算ではない！）
print(arr1 * arr2)  # [0]*[9], [1]*[10], ... のように各位置の要素を掛ける
```

### 問題2 の解き方ヒント

```python
import numpy as np

# ---- Step 1: 全要素が 2 の 4×4 行列を作る ----
mat = np.full((4, 4), 2)   # np.full(形状, 埋める値)
print(mat)

# ---- Step 2: 行列の 2乗 = 行列を自分自身と掛ける ----
# 「行列の2乗」とは A × A のこと（各要素の2乗ではない）
mat_squared = np.dot(mat, mat)   # mat × mat の行列積
print(mat_squared)
# → [[16 16 16 16]
#    [16 16 16 16]  ...など
```

---

## 解説3：Pandas DataFrame とデータ抽出（問題3 に対応）

### 問題3のテーマ

東京都の家賃相場データから，2LDK が 18万円以上の区だけを抽出する。

### DataFrame とは

DataFrame は「行と列を持つ表形式のデータ構造」です。Excelの表のようなイメージです。

```python
import pandas as pd

# 辞書から DataFrame を作る
# キーが「列名（column）」，値が「その列のデータリスト」になる
data = {
    'ワンルーム': [7.5, 7.0, 7.8],   # 港区・世田谷区・台東区の順
    '2LDK':      [24.5, 14.9, 15.6]
}
idx = ['港区', '世田谷区', '台東区']   # 行名（index）を指定

df = pd.DataFrame(data, index=idx)
print(df)
# 出力：
#          ワンルーム  2LDK
# 港区       7.5   24.5
# 世田谷区    7.0   14.9
# 台東区      7.8   15.6
```

### 条件によるデータ抽出

```python
# df['列名'] → その列だけを取り出す
print(df['2LDK'])
# → 港区: 24.5, 世田谷区: 14.9, 台東区: 15.6

# df[df['列名'] >= 値] → 条件を満たす行だけを残す
# まず df['2LDK'] >= 18 で True/False の列を作り，
# True の行だけを df から取り出す
filtered = df[df['2LDK'] >= 18]
print(filtered)
# → 港区の行だけが残る（24.5 >= 18 だから）
```

### 問題3 の解き方ヒント

```python
import pandas as pd

# ---- Step 1: データを辞書で準備して DataFrame を作る ----
dic = {
    "ワンルーム": [7.5, 7.0, 7.8, 7.5, 7.8, 7.7],
    "1k":        [9.6, 8.0, 8.7, 8.9, 9.5, 8.9],
    "1LDK":      [17.5, 11.6, 12.4, 14.3, 16.2, 13.5],
    "2LDK":      [24.5, 14.9, 15.6, 19.1, 22.1, 17.5],
    "3LDK":      [31.9, 18.3, 19.0, 24.2, 28.4, 21.8]
}
# 行名として区名を指定する
idx = ["港区", "世田谷区", "台東区", "新宿区", "渋谷区", "目黒区"]

# pd.DataFrame(辞書, index=行名のリスト) で表を作る
df_tky = pd.DataFrame(dic, index=idx)

# ---- Step 2: 2LDK が 18万円以上の区だけを抽出する ----
# df_tky['2LDK'] >= 18  → 各区が 18 以上かどうかの True/False 列を作る
# df_tky[...] → True の行（区）だけを取り出す
result = df_tky[df_tky['2LDK'] >= 18]
print(result)
# → 港区（24.5），新宿区（19.1），渋谷区（22.1）の3区が残る
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

plt.figure(figsize=(16, 9))  # グラフのキャンバスを作る（幅16×高さ9インチ）
plt.plot(x, y)               # x軸と y軸のデータを折れ線グラフで描く
plt.xlabel('X軸のラベル')      # x軸の説明ラベルを設定する
plt.ylabel('Y軸のラベル')      # y軸の説明ラベルを設定する
plt.grid(True)               # グリッド（方眼の線）を表示する
plt.show()                   # グラフを画面に表示する
```

### 問題4 の解き方ヒント

```python
import pandas as pd
import matplotlib.pyplot as plt

# ---- Step 1: CSV ファイルを URL から読み込む ----
# pd.read_csv(URL) → インターネット上の CSV を DataFrame として読み込む
# index_col=None → 行番号列を使わない（デフォルトの 0,1,2,... 番号を使う）
df_cvd = pd.read_csv(
    "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/covid/newly_confirmed_cases_daily.csv",
    index_col=None
)

# ---- Step 2: x軸とy軸のデータを準備する ----
# pd.to_datetime(列) → 文字列の日付を「日付型」に変換する
# ※ 変換しないとグラフの x 軸が文字列扱いになり，見づらくなる
x = pd.to_datetime(df_cvd["Date"])  # "Date"列を日付型に変換
y = df_cvd["ALL"]                   # "ALL"列（全国の陽性者数）を取得

# ---- Step 3: グラフを描く ----
plt.figure(figsize=(16, 9))  # グラフのサイズ指定（幅16×高さ9インチ）
plt.plot(x, y)               # 横軸=日付，縦軸=陽性者数の折れ線グラフ
plt.xlabel("Date")           # x軸のラベル
plt.ylabel("ALL")            # y軸のラベル
plt.grid(True)               # 方眼線を表示
plt.show()                   # グラフを表示
```

> **ポイント**：`pd.to_datetime()` で文字列の日付を日付型に変換するのを忘れずに。

---



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


| 統計量  | メソッド        | 意味                         |
| ---- | ----------- | -------------------------- |
| 平均値  | `.mean()`   | 全データの合計 ÷ データ数             |
| 標準偏差 | `.std()`    | データのばらつきの大きさ（大きいほど散らばっている） |
| 中央値  | `.median()` | 全データを小さい順に並べたときの真ん中の値      |


```python
import pandas as pd

# sep=";" はデータの区切り文字がセミコロン（;）であることを指定
df = pd.read_csv("student-por.csv", sep=";")

# df['列名'] で特定の列を取り出し，そこにメソッドを呼ぶ
print('G1の平均値:', df['G1'].mean())    # G1列の全データの平均
print('G1の標準偏差:', df['G1'].std())   # G1列のばらつき
print('G1の中央値:', df['G1'].median())  # G1列の真ん中の値
```

### 問題1 の解き方ヒント

```python
# ---- G1, G2, G3 の3列に対して統計量を求める ----
# for 文でリストの各列名を順番に処理する
for col in ['G1', 'G2', 'G3']:
    print(f"--- {col} ---")   # どの列の結果か示す区切り線

    # df_student_por[col] → G1（または G2, G3）の列を取り出す
    # .mean()  → その列の平均値を計算する
    # :.3f     → 小数点以下3桁まで表示するフォーマット指定
    print(f"平均値: {df_student_por[col].mean():.3f}")

    # .std() → 標準偏差（ばらつき）を計算する
    print(f"標準偏差: {df_student_por[col].std():.3f}")

    # .median() → 中央値を計算する
    print(f"中央値: {df_student_por[col].median():.3f}")
```

---

## 解説2：相関係数と散布図（問題2 に対応）

### 問題2のテーマ

G1, G2, G3 の相関係数を求め，散布図とヒストグラムを Seaborn で描画する。

### 相関係数とは

2つの変数の**関係の強さ**を -1〜1 の値で表したものです。

- **1 に近い** → 正の相関（片方が大きいと，もう片方も大きい傾向）
- **-1 に近い** → 負の相関（片方が大きいと，もう片方は小さい傾向）
- **0 に近い** → 無相関（関係が弱い）

```python
# df[['列1', '列2', '列3']].corr() → 指定した列間の相関係数を表で出力する
# 出力は各列ペアの相関係数が並んだ行列になる
print(df[['G1', 'G2', 'G3']].corr())
# 出力例：
#           G1        G2        G3
# G1  1.000000  0.852741  0.801468
# G2  0.852741  1.000000  0.904868
# G3  0.801468  0.904868  1.000000
```

### Seaborn の pairplot

```python
import seaborn as sns
import matplotlib.pyplot as plt

# sns.pairplot(DataFrame) → 全ての変数ペアの散布図と，各変数のヒストグラムを一度に描画する
# 対角線 = ヒストグラム（その変数の分布）
# 対角線以外 = 2変数の散布図（横軸×縦軸の組み合わせ）
sns.pairplot(df[['G1', 'G2', 'G3']])
plt.show()
```

> **pairplot** は，変数の組み合わせ全ての散布図と，対角線上に各変数のヒストグラムを自動で描いてくれます。

### 問題2 の解き方ヒント

```python
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Step 1: 相関係数の行列を出力する ----
# [['G1', 'G2', 'G3']] → この3列だけを DataFrame として取り出す
# .corr() → 全ての列ペアの相関係数を計算して表にする
print(df_student_por[['G1', 'G2', 'G3']].corr())

# ---- Step 2: 散布図＋ヒストグラムを描画する ----
# sns.pairplot → G1×G2, G1×G3, G2×G3 の全組み合わせの散布図と
#                G1, G2, G3 それぞれの分布（ヒストグラム）を自動で描く
sns.pairplot(df_student_por[['G1', 'G2', 'G3']])
plt.show()   # グラフを表示する
```

---

## 解説3：線形単回帰分析（問題3 に対応）

### 問題3のテーマ

G2（説明変数）から G3（目的変数）を予測する線形回帰モデルを構築し，回帰係数・切片・決定係数を求める。

### 単回帰分析とは

「y = ax + b」という直線でデータを近似する手法です。

- **a（回帰係数）**：直線の傾き（x が 1 増えたとき y がどれだけ変わるか）
- **b（切片）**：x=0 のときの y の値（直線が y 軸と交わる点）
- 最小二乗法によって，実測値と予測値の差の二乗和が最小となる a, b を求める

### Scikit-learn での線形回帰の流れ

```python
from sklearn import linear_model

# 説明変数：loc[:, ['G1']] は「全行の G1 列」を取り出す
# .values → Pandas の形式から NumPy の配列に変換（Scikit-learn が使いやすい形）
# [['G1']] のように [[ ]] と二重にするのは「2次元配列」にするため（Scikit-learn の仕様）
x = df_student_mat.loc[:, ['G1']].values

# 目的変数：G3 列を 1次元配列として取り出す
y = df_student_mat['G3'].values

# LinearRegression() → 線形回帰モデルのインスタンス（クラスから作ったオブジェクト）を作る
reg = linear_model.LinearRegression()

# .fit(説明変数, 目的変数) → 最小二乗法で傾き a と切片 b を計算する（学習）
reg.fit(x, y)

print('回帰係数:', reg.coef_)       # 学習で求めた傾き a
print('切片:', reg.intercept_)      # 学習で求めた切片 b
print('決定係数:', reg.score(x, y)) # モデルの当てはまりの良さ（0〜1，1に近いほど良い）
```

### 決定係数（R²）とは

予測モデルの当てはまりの良さを示す指標（0〜1）。  
→ **1 に近いほど当てはまりが良い**（完璧な予測で R²=1）。

### 散布図に回帰直線を重ねる

```python
import matplotlib.pyplot as plt

# plt.scatter(x, y) → 点（散布図）を描く
plt.scatter(x, y, label='実測値')

# reg.predict(x) → 学習済みモデル（y = ax + b）で x の予測値を計算する
# → 計算した予測値を y として折れ線で描くと回帰直線になる
plt.plot(x, reg.predict(x), color='red', label='回帰直線')

plt.xlabel('G2')
plt.ylabel('G3')
plt.legend()   # label= で設定した凡例（図例）を表示する
plt.grid(True)
plt.show()
```

### 問題3 の解き方ヒント

```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---- Step 1: 説明変数と目的変数を準備する ----
# [['G2']] → G2 列を「2次元配列」として取り出す（Scikit-learn の仕様）
x = df_student_por[['G2']].values   # 説明変数：2学期の成績
y = df_student_por['G3'].values     # 目的変数：3学期の成績

# ---- Step 2: モデルを作って学習させる ----
reg = LinearRegression()     # 線形回帰モデルのインスタンスを作る
reg.fit(x, y)                # x と y のデータから傾き・切片を求める（学習）

# ---- Step 3: 結果を出力する ----
print('回帰係数:', reg.coef_[0])    # 傾き a（[0] は配列の最初の要素を取り出す）
print('切片:', reg.intercept_)      # 切片 b
print('決定係数:', reg.score(x, y)) # R²（0〜1，1に近いほど予測が正確）

# ---- Step 4: 散布図と回帰直線を重ねて描画する ----
plt.scatter(x, y)                    # 実測値を点で描く
plt.plot(x, reg.predict(x), color='red')  # 予測値を赤線で描く（回帰直線）
plt.xlabel('G2')
plt.ylabel('G3')
plt.grid(True)
plt.show()
```

---



# 第4回 — ライブラリの応用

> この解説は **q4（第4回課題）** に対応しています。

## 概要

第4回では NumPy の高度な操作（ブールインデックス・演算・配列結合）と Pandas の高度なデータ操作（groupby・欠損値・時系列）を扱います。

---

## 解説1：ブールインデックスと演算（問題1 に対応）

### 問題1のテーマ

標準正規分布に従う 4×4 行列を生成し，値が 0 以上の要素を抽出してその数を出力する。

### ブールインデックス参照

「True/False の配列」を使ってデータを絞り込む方法です。

```python
import numpy as np

# 4×4 の乱数行列を生成（各要素はランダムな値）
sample_matrix = np.random.randn(4, 4)
print(sample_matrix)

# sample_matrix >= 0 → 行列の各要素が 0 以上かどうかを True/False で表した行列を作る
mask = sample_matrix >= 0
print(mask)
# → [[True False True ...]
#    [False True ...]] のような True/False の行列

# mask を使って sample_matrix から True の位置の要素だけを取り出す
result = sample_matrix[mask]
print(result)   # → 0 以上の要素だけが 1次元配列で並ぶ

# True を 1，False を 0 として合計を求めると True の数（= 0以上の要素数）になる
print('0以上の要素数:', mask.sum())
```

### 問題1 の解き方ヒント

```python
import numpy as np

# ---- Step 1: シードを固定して 4×4 の乱数行列を生成する ----
np.random.seed(0)   # 再現性のためシードを固定
mat = np.random.randn(4, 4)   # 標準正規分布に従う 4×4 の乱数行列

# ---- Step 2: 0 以上の要素を抽出する ----
# mat >= 0 → 各要素が 0 以上かどうかの True/False 行列を作る
# mat[mat >= 0] → True の位置の要素だけを取り出す
positive_elements = mat[mat >= 0]

# ---- Step 3: 抽出した要素とその数を出力する ----
print('0以上の要素:', positive_elements)

# (mat >= 0) → True/False の行列
# .sum() → True の数を数える（= 0以上の要素数）
print('0以上の要素数:', (mat >= 0).sum())
```

---

## 解説2：配列操作とブロードキャスト（問題2 に対応）

### 問題2のテーマ

2つの行列を行方向（縦方向）に結合し，全要素に 10 を加えた行列を出力する。

### 配列の結合（concatenate）

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])    # 2行3列の行列
b = np.array([[7, 8, 9], [10, 11, 12]]) # 2行3列の行列

# np.concatenate([配列1, 配列2], axis=方向) → 2つの配列を結合する
# axis=0 → 行方向（縦）に結合。結合後は 4行3列になる
result_vertical = np.concatenate([a, b], axis=0)
print(result_vertical)
# → [[ 1  2  3]
#    [ 4  5  6]
#    [ 7  8  9]
#    [10 11 12]]

# axis=1 → 列方向（横）に結合。結合後は 2行6列になる
result_horizontal = np.concatenate([a, b], axis=1)
print(result_horizontal)
# → [[ 1  2  3  7  8  9]
#    [ 4  5  6 10 11 12]]
```

### ブロードキャスト

配列と単一の数値を演算するとき，NumPy は自動的にその数値を配列の形に合わせてくれます。

```python
arr = np.arange(10)   # [0, 1, 2, ..., 9]
# arr + 3 → NumPy が自動的に 3 を [3,3,3,...,3] に拡張して各要素に足す
print(arr + 3)   # → [ 3  4  5  6  7  8  9 10 11 12]
```

### 問題2 の解き方ヒント

```python
import numpy as np

# ---- Step 1: 2つの行列を準備する ----
# np.arange(1, 11) → [1,2,3,...,10] の配列を生成
# .reshape(2, 5) → それを 2行5列に変形する
sample_array1 = np.arange(1, 11).reshape(2, 5)   # [[1,2,3,4,5],[6,7,8,9,10]]
sample_array2 = np.arange(11, 21).reshape(2, 5)  # [[11,12,...,15],[16,...,20]]

# ---- Step 2: 行方向（縦）に結合する ----
# axis=0 → 縦に積み上げる。2行5列 + 2行5列 = 4行5列になる
combined = np.concatenate([sample_array1, sample_array2], axis=0)
print(combined)

# ---- Step 3: 全要素に 10 を加える（ブロードキャスト）----
# combined + 10 → NumPy が 10 を自動的に全要素に足す
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

# .groupby(['age', 'sex']) → age と sex の組み合わせでデータをグループに分ける
# ['G1'] → G1 列だけを対象にする
# .mean() → 各グループの G1 の平均値を求める
grouped = df.groupby(['age', 'sex'])['G1'].mean()
print(grouped)
# 出力例：
# age  sex
# 15   F      10.625000
#      M      10.818182
# 16   F      10.153846
# ...
```

### unstack（行を列に変換）

グループ集計の結果は「age と sex の2階層のインデックス」になっています。  
`unstack()` で sex を行から列に移動させると，見やすい表になります。

```python
# unstack() → 一番内側のインデックス（sex = F/M）を列に変換する
result = grouped.unstack()
print(result)
# 出力例：
# sex     F          M
# age
# 15  10.625000  10.818182
# 16  10.153846  ...
```

### dropna（欠損値の削除）

```python
# dropna() → NaN（データなし）を含む行を全て削除する
# （age と sex の組み合わせで該当データがない場合に NaN になる）
result_clean = result.dropna()
print(result_clean)
```

### 問題3 の解き方ヒント

```python
import pandas as pd

# ---- Step 1: CSV を読み込む ----
# sep=";" → データの区切り文字がセミコロンであることを指定
df_student_mat = pd.read_csv("student-mat.csv", sep=";")

# ---- Step 2: age・sex でグループ化して G1 の平均を求める ----
# groupby(['age', 'sex']) → age と sex の全組み合わせでグループ分け
# ['G1'].mean() → 各グループの G1 の平均値を計算
grouped = df_student_mat.groupby(['age', 'sex'])['G1'].mean()

# ---- Step 3: sex（性別）を行から列に変換する ----
# unstack() → 内側のインデックス（sex）を列に変換
# → 縦軸=年齢，横軸=性別（F/M）の表になる
result = grouped.unstack()
print(result)

# ---- Step 4: NaN（データなし）の行を削除する ----
# dropna() → NaN を含む行（特定の年齢×性別の組み合わせにデータがない行）を削除
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

# rolling(N) → 「直近 N 件のデータを対象にする」というウィンドウを作る
# .mean() → そのウィンドウ内の平均値を計算する
# → 日々の陽性者数の凸凹を滑らかにした移動平均線が得られる
moving_avg = df['ALL'].rolling(7).mean()
```

### shift（データをずらす）

データを N 行分後ろにずらします。「N日前のデータを今日の行に持ってくる」イメージです。

```python
# rolling(7).sum() → 直近7日間の合計を計算
# .shift(7) → そのデータを 7行分後ろにずらす
#  → 結果として「今日の行に，7日前から13日前の合計」が入る（= 前の7日間の合計）
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

# ---- Step 1: データを読み込む ----
df_cvd = pd.read_csv(
    "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/covid/newly_confirmed_cases_daily.csv",
    index_col=None   # 行番号を自動で付ける
)

# 日付列を文字列から日付型に変換する（グラフの x 軸が見やすくなる）
x = pd.to_datetime(df_cvd["Date"])
# 全国の陽性者数の列を取り出す
y = df_cvd["ALL"]

# ---- Step 2: 7日間移動平均を計算する ----
# rolling(7).mean() → 直近7日間の平均値を毎日計算する
moving_avg = y.rolling(7).mean()

# ---- Step 3: 陽性者数と移動平均を重ねてグラフにする ----
plt.figure(figsize=(16, 9))
plt.plot(x, y, label='陽性者数')                      # 実際の陽性者数
plt.plot(x, moving_avg, label='7日間移動平均', color='red')  # 移動平均線（赤）
plt.legend()   # 凡例を表示（どの線が何かを示すラベル）
plt.grid(True)
plt.show()

# ---- Step 4: 実効再生産数を計算してグラフにする ----
# 直近7日間の合計（今日から7日前まで）
recent_7 = y.rolling(7).sum()

# 前7日間の合計（8日前から14日前まで）
# shift(7) でデータを7行後ろにずらすと，今日の行に8日前〜14日前の合計が入る
prev_7 = y.rolling(7).sum().shift(7)

# 実効再生産数の計算式： (直近7日合計 / 前7日合計)^(世代時間/7)
# 世代時間 = 2（2022年以降の簡易値）
Rt = (recent_7 / prev_7) ** (2 / 7)

plt.figure(figsize=(16, 9))
plt.plot(x, Rt)          # x軸=日付，y軸=実効再生産数
plt.xlabel("Date")
plt.ylabel("実効再生産数")
plt.grid(True)
plt.show()
```

---



# 第5回 — 機械学習

> この解説は **q5（第5回課題）** に対応しています。

## 概要

第5回では，タイタニック号乗客データを使って **決定木** による生存予測モデルを構築します。  
機械学習の基本的な流れ（6ステップ）に沿って学びます。

---

## 機械学習の6ステップ

```
STEP 1. データ読み込み       ← CSV などからデータを DataFrame に読み込む
STEP 2. データ前処理         ← 欠損値の削除・カテゴリ変数の数値化など
STEP 3. データ分割           ← 説明変数/目的変数 と 訓練/テスト に分割する
STEP 4. スケーリング         ← 変数の単位を揃える（今回は省略）
STEP 5. モデル学習           ← 訓練データを使ってモデルを構築する
STEP 6. モデル評価           ← テストデータで予測精度を確認する
```

---

## 解説1：データ確認（欠損値・データ型）

### データの形式と欠損の確認

```python
import pandas as pd

# pd.read_csv でタイタニックのデータを読み込む
df = pd.read_csv("titanic.csv")

# .shape → (行数, 列数) のタプルを返す
print(df.shape)   # → (891, 7) など

# .isnull() → 各要素が欠損（NaN）かどうかの True/False の DataFrame を作る
# .sum()    → 列ごとに True の数（= 欠損の数）を数える
print(df.isnull().sum())

# .info() → 各列のデータ型と Non-Null（欠損でない）数を一覧表示する
df.info()
```

### データ型の確認

```python
# .dtypes → 各列のデータ型（int64, float64, object など）を表示する
# object 型 = 文字列（カテゴリ変数）のことが多い
print(df.dtypes)
```

---

## 解説2：データ前処理（問題の核心）

### 欠損値の削除（dropna）

```python
# .dropna() → NaN（欠損値）を含む行を全て削除した新しい DataFrame を返す
df_clean = df.dropna()
print(df_clean.shape)   # 削除後の（行数, 列数）を確認する
```

### ダミー変数化（get_dummies）

「male / female」のようなカテゴリ変数を 0/1 の数値に変換します。  
→ 機械学習モデルは**数値しか扱えない**ため，この変換が必要です。

```python
# pd.get_dummies(列) → カテゴリ変数を 0/1 の列に変換する
# 例）Sex 列に "male" と "female" があれば，male列・female列の 2列に分かれる
# "male" の行 → male=1, female=0
# "female" の行 → male=0, female=1
sex_dummies = pd.get_dummies(df_clean['Sex'])
print(sex_dummies)

# .drop('Sex', axis=1) → Sex 列を削除する（axis=1 は列方向）
# .join(sex_dummies) → ダミー変数化した 2列を結合する
df_final = df_clean.drop('Sex', axis=1).join(sex_dummies)
```

---

## 解説3：データ分割

### 説明変数と目的変数

- **目的変数（y）**：予測したい値 → `Survived`（0: 死亡，1: 生存）
- **説明変数（X）**：予測に使う値 → `Pclass, Age, SibSp, Parch, Fare, male, female`

```python
# 使う列だけを DataFrame として取り出して説明変数 X とする
X = df_final[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'female']]

# Survived 列だけを取り出して目的変数 y とする
y = df_final['Survived']
```

### 訓練データとテストデータの分割

```python
from sklearn.model_selection import train_test_split

# train_test_split(X, y, test_size=0.2, random_state=0) の意味：
# - X, y をランダムに分割する
# - test_size=0.2 → 全体の 20% をテストデータにする（残り 80% が訓練データ）
# - random_state=0 → 何度実行しても同じ分割結果になるようにシードを固定
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
# X_train: 訓練用の説明変数，y_train: 訓練用の目的変数
# X_test:  テスト用の説明変数，y_test:  テスト用の目的変数
```

---

## 解説4：決定木モデルの学習と評価

### 決定木とは

データの各特徴を使って「Yes/No」の条件分岐を繰り返し，予測を行うモデルです。  
→ 直感的に理解しやすい（木構造で可視化できる）

### Scikit-learn での決定木

```python
from sklearn.tree import DecisionTreeClassifier

# DecisionTreeClassifier(...) → 決定木のインスタンスを作る
# criterion='entropy' → 分岐の指標にエントロピー（情報利得）を使う
# max_depth=5 → 木の深さ（条件分岐の段数）を最大 5 に制限する
model = DecisionTreeClassifier(criterion='entropy', max_depth=5)

# .fit(訓練データの説明変数, 訓練データの目的変数)
# → 訓練データを使って決定木の条件分岐ルールを学習する
model.fit(X_train, y_train)
```

### モデルの評価（正解率）

```python
# .score(説明変数, 目的変数) → 正解率を計算して返す
# 訓練データとテストデータの両方で評価する
print('正解率(train):', model.score(X_train, y_train))
print('正解率(test):', model.score(X_test, y_test))
# 訓練データの正解率が高く，テストデータが大幅に低い場合 → 過学習（覚えすぎ）の可能性
```

> **ポイント**：訓練データとテストデータの正解率が大きく異なる場合は**過学習（オーバーフィッティング）**が起きている可能性があります。

---



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

→ 4パターンの評価スコアの平均を最終スコアとする
```

### グリッドサーチ（GridSearchCV）

「木の深さ」などの**ハイパーパラメータ**の全組み合わせを試し，最も良い組み合わせを自動で探します。

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# 探索したいハイパーパラメータと，試す値の範囲を辞書で指定する
# max_depth: 木の深さ（2, 3, 4, 5 の4通り）
# min_samples_leaf: 葉ノードに必要な最小サンプル数（2, 3, 4, 5 の4通り）
# → 4 × 4 = 16 通りの組み合わせを全て試す
param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [2, 3, 4, 5]
}

# 試すモデル（決定木）のインスタンスを作る
tree = DecisionTreeClassifier(criterion='entropy', random_state=0)

# GridSearchCV(モデル, パラメータ候補, cv=分割数)
# → 16通りの組み合わせ × 5分割交差検証 = 合計 80 回学習・評価する
gs = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5)

# .fit(訓練データ) → 全80パターンを実行して最良パラメータを見つける
gs.fit(X_train, y_train)

# 最良スコア：80パターンの中で最も高かった交差検証スコア
print('最良スコア:', gs.best_score_)
# 最良パラメータ：そのスコアを出したときの max_depth と min_samples_leaf の値
print('最良パラメータ:', gs.best_params_)
# テストスコア：最良パラメータのモデルをテストデータで評価した正解率
print('テストスコア:', gs.score(X_test, y_test))
```

### 問題1 の解き方ヒント

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# ---- Step 1: 乳がんデータセットを読み込む ----
# load_breast_cancer() → Scikit-learn に内蔵されている乳がんデータを読み込む
# .data → 説明変数（細胞の特徴量 30列）
# .target → 目的変数（0=悪性, 1=良性）
cancer = load_breast_cancer()

# ---- Step 2: 訓練データとテストデータに分割する ----
# stratify=cancer.target → 目的変数の比率を保ったまま分割する（クラスが偏らないように）
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0
)

# ---- Step 3: 決定木のインスタンスを作る ----
# （GridSearchCV に渡すための「ベースとなるモデル」）
tree = DecisionTreeClassifier(criterion='entropy', random_state=0)

# ---- Step 4: 探索するパラメータの範囲を辞書で指定する ----
param_grid = {
    'max_depth': [2, 3, 4, 5],         # 木の深さの候補（4通り）
    'min_samples_leaf': [2, 3, 4, 5]   # 葉の最小サンプル数の候補（4通り）
}
# → 4 × 4 = 16 通りの組み合わせを試す

# ---- Step 5: GridSearchCV を設定する ----
# estimator=tree → 試すモデル（決定木）
# param_grid     → 試すパラメータの組み合わせ
# cv=5           → 5分割交差検証で評価する
gs = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5)

# ---- Step 6: 全パターンを試して最良モデルを見つける ----
# .fit(x_train, y_train) → 16通り×5分割 = 80回の学習と評価を自動実行する
gs.fit(x_train, y_train)

# ---- Step 7: 結果を確認する ----
# best_score_ → 最良パラメータのときの交差検証スコア（正解率の平均）
print('最良スコア:', gs.best_score_)
# best_params_ → 最良スコアを出したときのパラメータの組み合わせ
print('最良パラメータ:', gs.best_params_)
```

---

## 解説2：混同行列と評価指標（問題2 に対応）

### 問題2のテーマ

ロジスティック回帰モデルを構築し，混同行列と評価指標（正解率・適合率・再現率・F1スコア）を求める。

### 混同行列（Confusion Matrix）

予測値と正解の対応を 2×2 の表で表したものです。


|            | 予測: 正例              | 予測: 負例              |
| ---------- | ------------------- | ------------------- |
| **正解: 正例** | TP（正解：正しく正と予測）      | FN（見逃し：本当は正なのに負と予測） |
| **正解: 負例** | FP（誤検知：本当は負なのに正と予測） | TN（正解：正しく負と予測）      |


### 評価指標の意味


| 指標                  | 計算式                                     | 意味                    |
| ------------------- | --------------------------------------- | --------------------- |
| **正解率** (Accuracy)  | (TP + TN) / 全体                          | 全体の中で正しく予測できた割合       |
| **適合率** (Precision) | TP / (TP + FP)                          | 「正例」と予測した中で本当に正例だった割合 |
| **再現率** (Recall)    | TP / (TP + FN)                          | 実際の正例のうち正しく検出できた割合    |
| **F1スコア**           | 2×Precision×Recall / (Precision+Recall) | 適合率と再現率の調和平均          |


### 問題2 の解き方ヒント

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              precision_score, recall_score, f1_score)

# ---- Step 1: データを読み込んで分割する ----
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0
)

# ---- Step 2: ロジスティック回帰モデルを構築・学習する ----
# LogisticRegression() → ロジスティック回帰のインスタンスを作る
model = LogisticRegression()
# .fit(訓練データの説明変数, 目的変数) → 訓練データで学習する
model.fit(x_train, y_train)

# ---- Step 3: テストデータで予測する ----
# .predict(x_test) → テストデータの各サンプルに対して 0 か 1 を予測する
y_pred = model.predict(x_test)
# y_pred: 予測した正解ラベル（例：[1, 0, 1, 1, 0, ...]）
# y_test: 実際の正解ラベル（例：[1, 0, 0, 1, 0, ...]）

# ---- Step 4: 混同行列を出力する ----
# confusion_matrix(実際の正解, 予測値) → TP/FP/FN/TN の 2×2 の行列を返す
print(confusion_matrix(y_test, y_pred))

# ---- Step 5: 各評価指標を計算して出力する ----
# accuracy_score → (TP+TN)/全体 の正解率
print('正解率:', accuracy_score(y_test, y_pred))

# precision_score → TP/(TP+FP) の適合率（「正と予測した中で本当に正の割合」）
print('適合率:', precision_score(y_test, y_pred))

# recall_score → TP/(TP+FN) の再現率（「本当に正のうち正と検出できた割合」）
print('再現率:', recall_score(y_test, y_pred))

# f1_score → 適合率と再現率の調和平均
print('F1スコア:', f1_score(y_test, y_pred))
```

---

## 解説3：複数モデルの比較（問題3 に対応）

### 問題3のテーマ

ロジスティック回帰・SVM・決定木・k-NN の4モデルをホールドアウト法で比較し，最も良いモデルを確認する。

### ホールドアウト法とは

データを訓練用とテスト用に2分割し，テストデータでモデルを評価する方法です。

### 問題3 の解き方ヒント

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

# ---- Step 1: データを読み込んで分割する ----
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=0
)

# ---- Step 2: 試すモデルを辞書にまとめる ----
# キー = モデル名（出力用の文字列），値 = モデルのインスタンス
models = {
    'LogisticRegression': LogisticRegression(),    # ロジスティック回帰
    'SVM':                LinearSVC(),             # サポートベクターマシン
    '決定木':              DecisionTreeClassifier(criterion='entropy', random_state=0),
    'k-NN':               KNeighborsClassifier()  # k近傍法
}

# ---- Step 3: 各モデルを学習して評価する ----
# models.items() → {名前: モデル} の辞書からペアを1つずつ取り出す
for name, model in models.items():
    # .fit(x_train, y_train) → 訓練データでモデルを学習させる
    model.fit(x_train, y_train)
    # .score(x_test, y_test) → テストデータで正解率を計算する
    score = model.score(x_test, y_test)
    # f'{name}: {score:.4f}' → モデル名と正解率（小数4桁）を出力
    print(f'{name}: {score:.4f}')
# → 出力された正解率を比較して，最も高いモデルが「最良」
```

---



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

$$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}*i)^2}{\sum*{j=1}^n (y_j - \bar{y})^2}$$

ここで $\hat{y}_i = ax_i + b$（予測値），$\bar{y}$（目的変数の平均値）

---

## 解説2：NumPy の関数と数式の対応


| 数式の表現            | NumPy のコード     | 説明                  |
| ---------------- | -------------- | ------------------- |
| $n$（サンプル数）       | `len(x)`       | 配列 x の要素数           |
| $\sum x_i y_i$   | `np.dot(x, y)` | x と y の内積（各要素の積の合計） |
| $\sum x_i$       | `np.sum(x)`    | x の全要素の合計           |
| $\bar{x}$（x の平均） | `np.mean(x)`   | x の平均値              |
| $\hat{y}_i$（予測値） | `a * x + b`    | 各 x に対して直線の式を計算     |


---

## 解説3：クラスの実装方法

第1回で学んだクラスの書き方をそのまま使います。

```python
import numpy as np

class MyLinearRegression:
    """最小二乗法による線形単回帰分析クラス"""

    def __init__(self):
        # コンストラクタ：インスタンス生成時に呼ばれる
        # self.a に傾き，self.b に切片を保存するための箱を用意する（初期値は None）
        self.a = None   # 傾き（学習前は None，fit 後に値が入る）
        self.b = None   # 切片（学習前は None，fit 後に値が入る）

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        x と y のデータから最小二乗法で傾き a と切片 b を求める（学習）
        """
        # n = サンプル数（データの個数）
        n = len(x)

        # 傾き a の計算（数式 (2) をコードに変換）
        # np.dot(x, y) = Σ(xi * yi)  → x と y の内積
        # np.sum(x) = Σxi，np.sum(y) = Σyi
        # np.dot(x, x) = Σ(xi^2)     → x と x の内積
        self.a = (np.dot(x, y) - (1/n) * np.sum(x) * np.sum(y)) / \
                 (np.dot(x, x) - (1/n) * (np.sum(x) ** 2))

        # 切片 b の計算（数式 (1) をコードに変換）
        # np.mean(y) = y の平均，np.mean(x) = x の平均
        self.b = np.mean(y) - self.a * np.mean(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        学習済みの a と b を使って y = ax + b で予測値を計算する
        """
        # self.a と self.b は fit() で求めた傾きと切片
        return self.a * x + self.b   # 直線の式そのまま

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        決定係数 R²（モデルの当てはまりの良さ）を計算する
        """
        # predict(x) で x に対する予測値 y_hat を計算する
        y_hat = self.predict(x)

        # 残差平方和（予測値と実測値の差の二乗和）
        ss_res = np.sum((y - y_hat) ** 2)

        # 全平方和（実測値と平均値の差の二乗和）
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # R² = 1 - (残差平方和 / 全平方和)
        # 完璧な予測なら ss_res=0 → R²=1
        # 平均値で予測するのと同じなら ss_res=ss_tot → R²=0
        return 1 - ss_res / ss_tot
```

---

## 解説4：実際のデータで回帰モデルを動かす

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---- Step 1: 数学成績データを読み込む ----
# sep=';' → データの区切り文字がセミコロンであることを指定
df_math = pd.read_csv(
    "https://raw.githubusercontent.com/ShotaYmzk/AI-kadai/main/data/student/student-mat.csv",
    sep=';'
)

# ---- Step 2: 説明変数と目的変数を設定する ----
# .values → Pandas の Series/DataFrame を NumPy 配列に変換する
x = df_math['G1'].values   # 説明変数：1学期の成績（この値から G3 を予測する）
y = df_math['G3'].values   # 目的変数：3学期の成績（予測したい値）

# ---- Step 3: G1 と G3 の関係を散布図で確認する ----
plt.scatter(x, y)                    # G1 を x 軸，G3 を y 軸に点をプロット
plt.xlabel('G1（一学期の成績）')
plt.ylabel('G3（三学期の成績）')
plt.grid(True)
plt.show()

# ---- Step 4: 訓練データとテストデータに分割する ----
# test_size=0.2 → 全体の 20% をテスト，80% を訓練に使う
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# ---- Step 5: 自作の線形回帰モデルで学習する ----
model = MyLinearRegression()         # インスタンスを作る（self.a, self.b が None の状態）
model.fit(x_train, y_train)          # 訓練データで学習（self.a, self.b に値が入る）

# 学習した傾きと切片を確認する
print(f'傾き: {model.a:.4f}')        # 1学期の成績が1点上がると3学期は何点上がるか
print(f'切片: {model.b:.4f}')        # 1学期が0点のときの3学期の予測値

# ---- Step 6: モデルを評価する ----
# .score(x, y) → 決定係数 R² を計算（1 に近いほど予測が正確）
print(f'決定係数(train): {model.score(x_train, y_train):.4f}')  # 訓練データでの評価
print(f'決定係数(test):  {model.score(x_test, y_test):.4f}')    # テストデータでの評価

# ---- Step 7: 散布図と回帰直線を重ねて描画する ----
plt.scatter(x, y, label='データ点')   # 実測値の散布図
# sorted(x) → x を昇順に並べる（回帰直線が左から右にきれいに引かれるように）
# model.predict(xi) → 各 xi に対して y = ax + b を計算
plt.plot(sorted(x), [model.predict(xi) for xi in sorted(x)],
         color='red', label='回帰直線')
plt.xlabel('G1（一学期の成績）')
plt.ylabel('G3（三学期の成績）')
plt.legend()   # 凡例（データ点・回帰直線のラベル）を表示
plt.grid(True)
plt.show()
```

> **ポイント**：
>
> - `fit` メソッドで傾き `a` と切片 `b` を計算する（学習）
> - `predict` メソッドで `a*x + b` を計算する（予測）
> - `score` メソッドで決定係数 R² を計算する（評価）
> - この3つが Scikit-learn の全モデルに共通するインターフェースです。

---

## まとめ：各回の解説と問題の対応表


| 回      | 問題  | 使う主な概念                                                                          |
| ------ | --- | ------------------------------------------------------------------------------- |
| **q1** | 問題1 | 変数・辞書型・`.items()`                                                               |
| **q1** | 問題2 | 演算子（`//`, `%`）                                                                  |
| **q1** | 問題3 | if文・for文・range・関数                                                               |
| **q1** | 問題4 | クラス・コンストラクタ・`self`                                                              |
| **q2** | 問題1 | `np.random.seed`・`randn`・`max/min/sum`                                          |
| **q2** | 問題2 | `np.full`・`np.dot`                                                              |
| **q2** | 問題3 | `pd.DataFrame`・条件抽出                                                             |
| **q2** | 問題4 | `plt.plot`・`figsize`・`grid`・`pd.to_datetime`                                    |
| **q3** | 問題1 | `.mean()`・`.std()`・`.median()`                                                  |
| **q3** | 問題2 | `.corr()`・`sns.pairplot`                                                        |
| **q3** | 問題3 | `LinearRegression`・`.fit()`・`.score()`                                          |
| **q4** | 問題1 | ブールインデックス・`.sum()`                                                              |
| **q4** | 問題2 | `np.concatenate`・ブロードキャスト                                                       |
| **q4** | 問題3 | `.groupby()`・`.unstack()`・`.dropna()`                                           |
| **q4** | 問題4 | `.rolling(7).mean()`・`.shift(7)`                                                |
| **q5** | 全問  | `get_dummies`・`train_test_split`・`DecisionTreeClassifier`・`.score()`            |
| **q6** | 問題1 | `GridSearchCV`・`cv=5`（交差検証）                                                     |
| **q6** | 問題2 | `confusion_matrix`・`accuracy_score`・`precision_score`・`recall_score`・`f1_score` |
| **q6** | 問題3 | ホールドアウト法・複数モデル比較                                                                |
| **q7** | 全問  | クラス実装・最小二乗法・`np.dot`・`np.mean`・`np.sum`                                         |


