#!/usr/bin/env python3
"""Convert ✍️ 解答欄 markdown cells to code cells (q8 style)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Extra choice-option comments keyed by (notebook, problem, item_id)
CHOICE_HINTS: dict[tuple[str, int, str], list[str]] = {
    ("q9", 1, "1-d"): [
        "(A) 何もしない（正則化なしの LinearRegression のまま）",
        "(B) Ridge（L2正則化）を使う",
        "(C) Lasso（L1正則化）を使う",
        "(D) ElasticNet（L1+L2 の両方）を使う",
        "(E) 特徴量を減らす（多項式の次数 DEGREE を下げる）",
        "(F) 学習データを増やす",
    ],
    ("q10", 1, "1-b"): [
        "(A) 単一の決定木（深さ無制限）",
        "(B) RandomForest（バギング）",
        "(C) GradientBoosting（ブースティング）",
        "(D) XGBoost（ブースティング系）",
        "(E) 決定木を max_depth で浅く制限する（アンサンブルしない）",
        "(F) 線形モデル（LogisticRegression など）",
    ],
    ("q10", 3, "3-a"): [
        "(A) 単一決定木", "(B) RandomForest（バギング）",
        "(C) GradientBoosting（ブースティング）", "(D) XGBoost（ブースティング系）",
        "(E) LogisticRegression（線形・アンサンブルなし）", "(F) スタッキング",
    ],
    ("q10", 3, "3-c"): [
        "(A) 単一決定木", "(B) RandomForest（バギング）",
        "(C) GradientBoosting（ブースティング）", "(D) XGBoost（ブースティング系）",
        "(E) LogisticRegression（線形・アンサンブルなし）", "(F) スタッキング",
    ],
    ("q12", 1, "1-b"): [
        "(A) 何もしない（class_weight なし・閾値 0.5）",
        "(B) class_weight='balanced'",
        "(C) 閾値を下げる（Recall 重視）",
        "(D) SMOTE などで少数クラスを増やす",
        "(E) 多数クラスを減らす（アンダーサンプリング）",
        "(F) 別のモデル（例: 木系）に変える",
    ],
    ("q12", 4, "4-b"): [
        "(A) Accuracy", "(B) Precision", "(C) Recall", "(D) F1",
        "(E) ROC-AUC", "(F) PR-AUC（Average Precision）",
    ],
    ("q11", 4, "4-a"): [
        "(A) accuracy（正解率）で測る",
        "(B) シルエットスコアでクラスタの分離の良さを測る",
        "(C) PCA の累積寄与率で保持できた情報量を測る",
        "(D) 散布図を人が見て、まとまり・重なり・外れ値を解釈する",
        "(E) Davies-Bouldin 指数など別のクラスタ評価指標で測る",
        "(F) inertia だけを見て小さいほど良いと判断する",
    ],
    ("q14", 2, "2-a"): [
        "(A) ToTensor のみ", "(B) Normalize(0.5, 0.5)",
        "(C) RandomRotation", "(D) RandomAffine",
        "(E) RandomHorizontalFlip", "(F) 複数を組み合わせ",
    ],
    ("q15", 4, "4-a"): [
        "(A) ReLU", "(B) Sigmoid", "(C) Tanh", "(D) LeakyReLU",
        "(E) GELU", "(F) その他",
    ],
    ("q16", 3, "3-a"): [
        "(A) epoch を増やす", "(B) 学習率(lr)を調整する",
        "(C) Dropout などの正則化を入れる", "(D) データ拡張（回転・平行移動）を行う",
        "(E) 層やチャネルを増やしてモデルを複雑にする",
        "(F) Batch Normalization を入れる",
    ],
    ("q16", 4, "4-b"): [
        "(A) データ拡張", "(B) モデルを大きくする", "(C) 学習率を調整",
        "(D) 混同行列を見て誤りパターンを分析", "(E) アンサンブル",
        "(F) 前処理を見直す",
    ],
    ("q17", 2, "2-d"): [
        "(A) 前処理のずれ", "(B) 学習不足", "(C) 分布外（Distribution Shift）",
        "(D) 線が細すぎ／太すぎ", "(E) 色反転の漏れ／誤り", "(F) モデルの過信",
    ],
}

# Use pd.DataFrame for experiment logs (needs pandas import in answer cell)
USE_PANDAS = True


def slug_from_item(item_id: str, title: str) -> str:
    if "実験ログ" in title:
        if "軸2" in title:
            return "experiment_log_axis2"
        if "軸1" in title:
            return "experiment_log_axis1"
        return "experiment_log"
    if "推論結果" in title:
        return "inference_log"  # q17: also use records in previous cell
    if "設計判断" in title and "選" in title:
        m = re.search(r"問題?(\d+)", title)
        n = m.group(1) if m else item_id.split("-")[0]
        if "2-1" in item_id or "3-a" in item_id and "2-1" in title:
            return "design2_1_choice"
        if "2-2" in item_id or "3-c" in item_id and "2-2" in title:
            return "design2_2_choice"
        return f"design{item_id.split('-')[0]}_choice"
    if "考察" in title:
        num = re.search(r"考察(\d+)", title)
        if num:
            return f"reflection{num.group(1)}"
        return f"answer_{item_id.replace('-', '_')}"
    if "観察" in title and "実験" not in title:
        return f"observation_{item_id.replace('-', '_')}"
    return f"answer_{item_id.replace('-', '_')}"


def is_choice_field(title: str) -> bool:
    return bool(
        "：(　)" in title
        or re.search(r"（[A-Z]〜", title)
        or "A〜F" in title
        or ("選" in title and re.search(r"[（(]\s*[A-Z]", title))
    )


def parse_markdown_table(lines: list[str], start: int) -> tuple[list[str], list[dict], int]:
    """Return columns, template rows, end index."""
    table_lines = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i])
        i += 1
    if len(table_lines) < 2:
        return [], [], start

    def split_row(row: str) -> list[str]:
        parts = [p.strip() for p in row.strip("|").split("|")]
        return parts

    header = split_row(table_lines[0])
    # skip separator
    data_rows = []
    for row in table_lines[2:]:
        cells = split_row(row)
        if not any(cells):
            continue
        if cells[0] in ("…", "..."):
            continue
        data_rows.append(cells)

    col_map = {
        "": "row",
        "#": "row",
        "degree": "degree",
        "model": "model",
        "alpha": "alpha",
        "train_r2": "train_r2",
        "test_r2": "test_r2",
        "train_r²": "train_r2",
        "test_r²": "test_r2",
        "訓練_r²": "train_r2",
        "テスト_r²": "test_r2",
        "特徴量の数": "n_features",
        "非ゼロ係数の数": "n_nonzero",
        "n_estimators": "n_estimators",
        "max_depth": "max_depth",
        "learning_rate": "learning_rate",
        "train_acc": "train_acc",
        "test_acc": "test_acc",
        "k": "k",
        "inertia": "inertia",
        "silhouette": "silhouette",
        "n_components": "n_components",
        "累積寄与率": "cumulative_variance_pct",
        "class_weight": "class_weight",
        "threshold": "threshold",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "fn_見逃し": "fn",
        "penalty": "penalty",
        "c": "C",
        "lr": "LR",
        "epochs": "EPOCHS",
        "hidden": "HIDDEN",
        "hidden1": "hidden1",
        "batch_size": "batch_size",
        "1エポックのイテレーション数": "iterations_per_epoch",
        "images_shape": "images_shape",
        "flatten_後_shape": "flatten_shape",
        "正解_label": "label",
        "予測_pred": "pred",
        "信頼度": "confidence_pct",
        "正誤": "correct",
        "メモ_書き方の癖など": "memo",
        "train_スコア": "train_score",
        "valid_スコア": "valid_score",
        "gap_train_valid": "gap",
        "gap_train_test": "gap",
        "過学習_gap_0_05": "overfitting",
        "min_samples_leaf": "min_samples_leaf",
        "差_train_test": "gap",
    }
    columns = []
    for h in header:
        key = re.sub(r"[^\w]+", "_", h.lower()).strip("_")
        key = col_map.get(key, key or "col")
        columns.append(key)

    rows = []
    for cells in data_rows:
        d = {}
        for j, col in enumerate(columns):
            if j < len(cells):
                val = cells[j]
                if val.replace(".", "").isdigit():
                    try:
                        d[col] = int(val) if "." not in val else float(val)
                        continue
                    except ValueError:
                        pass
                d[col] = val if val else None
            else:
                d[col] = None
        rows.append(d)
    return columns, rows, i


def convert_answer_markdown(md: str, nb_name: str) -> list[str]:
    lines = md.strip().split("\n")
    m = re.search(r"問題(\d+)", lines[0])
    problem = int(m.group(1)) if m else 0

    code: list[str] = [f"# === ✍️ 問題{problem} 解答（採点対象）==="]
    if "主な提出物は上のコードセルへの" in md:
        code.append("# 主な提出物は上のコードセルへの # 説明: 記入。以下も記入すること。")
    code.append("")

    imports_needed = set()
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # bullet sub-items under one answer (q12 4-d, q17 3-d)
        if line.startswith("- ") and i > 1:
            label = line[2:].rstrip("：").rstrip(":")
            var = re.sub(r"[^\w]+", "_", label.lower()).strip("_") or f"item_{i}"
            code.append(f"# {label}")
            code.append(f'{var} = ""')
            code.append("")
            i += 1
            continue

        item_m = re.match(r"\*\*\((\d+-[a-z])\)\s*([^*]+)\*\*(.*)$", line)
        if not item_m:
            # plain **text** observation block (q14 problem 3)
            plain = re.match(r"\*\*(.+?)\*\*\s*:?\s*$", line)
            if plain:
                title = plain.group(1)
                var = re.sub(r"[^\w]+", "_", title[:30].lower()).strip("_") or f"note_{i}"
                code.append(f"# {title}")
                # check for sub-bullets
                i += 1
                subs = []
                while i < len(lines) and lines[i].strip().startswith("- "):
                    subs.append(lines[i].strip()[2:])
                    i += 1
                if subs:
                    for sub in subs:
                        sub_var = re.sub(r"[^\w]+", "_", sub.rstrip("：").lower()).strip("_")
                        code.append(f'{sub_var} = ""')
                else:
                    code.append(f'{var} = """')
                    code.append('"""')
                code.append("")
                continue
            i += 1
            continue

        item_id = item_m.group(1)
        title = (item_m.group(2) + item_m.group(3)).strip().rstrip("：").rstrip(":")
        i += 1

        # bullet sub-fields (e.g. q12 4-d, q17 3-d)
        sub_bullets: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("- "):
            sub_bullets.append(lines[i].strip()[2:].rstrip("：").rstrip(":"))
            i += 1

        while i < len(lines) and not lines[i].strip():
            i += 1

        # table follows?
        if i < len(lines) and lines[i].strip().startswith("|"):
            cols, rows, i = parse_markdown_table(lines, i)
            var = slug_from_item(item_id, title)
            code.append(f"# ({item_id}) {title}")
            imports_needed.add("pd")
            code.append(f"{var} = pd.DataFrame([")
            for row in rows:
                code.append(f"    {row!r},")
            code.append("])")
            code.append("")
            continue

        var = slug_from_item(item_id, title)
        code.append(f"# ({item_id}) {title}")

        hints = CHOICE_HINTS.get((nb_name, problem, item_id))
        if hints:
            for h in hints:
                code.append(f"# {h}")
        elif is_choice_field(title):
            code.append('# 例: "B"')

        if sub_bullets:
            if not is_choice_field(title):
                code.append(f'{var} = """')
                code.append('"""')
            for sub in sub_bullets:
                sub_label = sub.split("：")[0].split(":")[0]
                sub_var = {
                    "ROC の AUC": "roc_auc",
                    "PR-AUC": "pr_auc",
                    "min": "pixel_min",
                    "max": "pixel_max",
                    "手法1": "improvement_1",
                    "手法2": "improvement_2",
                }.get(sub_label, re.sub(r"[^\w]+", "_", sub_label.lower()).strip("_") or "note")
                code.append(f"# {sub}")
                code.append(f'{sub_var} = ""')
        elif is_choice_field(title) or (hints and "選" in title):
            code.append(f'{var} = ""')
        else:
            code.append(f'{var} = """')
            code.append('"""')
        code.append("")

    if imports_needed:
        code.insert(1, "import pandas as pd")
        code.insert(2, "")

    return [ln + "\n" for ln in code]


def update_intro_text(source: list[str]) -> list[str]:
    out = []
    for line in source:
        s = line
        s = s.replace(
            "**理由**を解答欄に書く",
            "**理由**を解答用コードセルに書く",
        )
        s = s.replace(
            "各問の最後にある **✍️ 解答欄** のテキストセルに",
            "各問の **✍️ 解答用コードセル**（`# (1-a)` 形式の変数・文字列）に",
        )
        s = s.replace("解答欄に書く", "解答用コードセルに書く")
        s = s.replace("解答欄に記入", "解答用コードセルに記入")
        s = s.replace("解答欄の実験ログ表", "解答用コードセルの実験ログ")
        s = s.replace("解答欄に記録", "解答用コードセルに記録")
        s = s.replace("解答欄に書いて", "解答用コードセルに書いて")
        s = s.replace("解答欄に1〜2文で", "解答用コードセルに1〜2文で")
        s = s.replace("（解答欄の実験ログに記録）", "（解答用コードセルの実験ログに記録）")
        s = s.replace("下の解答欄の考察", "下の解答用コードセルの考察")
        s = s.replace("下の解答欄", "下の解答用コードセル")
        out.append(s)
    return out


def process_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb_name = path.stem
    converted = 0
    new_cells = []
    for cell in nb["cells"]:
        src_joined = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "### ✍️ 解答欄" in src_joined:
            md = src_joined
            code_src = convert_answer_markdown(md, nb_name)
            new_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": code_src,
                }
            )
            converted += 1
        else:
            if cell["cell_type"] == "markdown":
                cell["source"] = update_intro_text(cell["source"])
            new_cells.append(cell)

    nb["cells"] = new_cells
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return converted


def main():
    total = 0
    for n in range(9, 18):
        p = ROOT / f"q{n}.ipynb"
        c = process_notebook(p)
        print(f"{p.name}: {c} answer cells converted")
        total += c
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
