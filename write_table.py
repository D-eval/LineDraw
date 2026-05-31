import pandas as pd

csv_file = "./experiment/threshold_sweep.csv"
out_file = "./table.txt"

df = pd.read_csv(csv_file)

# 要加粗的列
bold_cols = [
    "precision",
    "recall",
    "f1",
    "pitch_precision",
    "pitch_recall",
    "pitch_f1",
]

# 最大值索引
best_idx = {
    col: df[col].idxmax()
    for col in bold_cols
}

with open(out_file, "w") as f:

    f.write(r"\begin{tabular}{c|ccc|ccc|cc}" + "\n")
    f.write(r"\toprule" + "\n")

    f.write(r"\multirow{2}{*}{Threshold}" + "\n")
    f.write(r"&" + "\n")
    f.write(r"\multicolumn{3}{c|}{MIDI-Level}" + "\n")
    f.write(r"&" + "\n")
    f.write(r"\multicolumn{3}{c|}{Pitch-Class}" + "\n")
    f.write(r"&" + "\n")
    f.write(r"\multirow{2}{*}{ECR $\uparrow$}" + "\n")
    f.write(r"&" + "\n")
    f.write(r"\multirow{2}{*}{Events $\downarrow$}" + "\n")
    f.write(r"\\" + "\n\n")

    f.write(r"&" + "\n")
    f.write(r"Precision $\uparrow$" + "\n")
    f.write(r"&" + "\n")
    f.write(r"Recall $\uparrow$" + "\n")
    f.write(r"&" + "\n")
    f.write(r"F1 $\uparrow$" + "\n")
    f.write(r"&" + "\n")
    f.write(r"Precision $\uparrow$" + "\n")
    f.write(r"&" + "\n")
    f.write(r"Recall $\uparrow$" + "\n")
    f.write(r"&" + "\n")
    f.write(r"F1 $\uparrow$" + "\n")
    f.write(r"&" + "\n")
    f.write(r"&" + "\n")
    f.write(r"\\" + "\n")

    f.write(r"\midrule" + "\n\n")

    for idx, row in df.iterrows():

        def fmt(col):
            v = row[col]
            s = f"{v:.3f}"

            if col in best_idx and idx == best_idx[col]:
                s = rf"\textbf{{{s}}}"

            return s

        event_str = f"{row['events']:.2f}"

        if idx == df["events"].idxmin():
            event_str = rf"\textbf{{{event_str}}}"

        line = (
            f"{float(row['threshold'])} & "
            f"{fmt('precision')} & "
            f"{fmt('recall')} & "
            f"{fmt('f1')} & "
            f"{fmt('pitch_precision')} & "
            f"{fmt('pitch_recall')} & "
            f"{fmt('pitch_f1')} & "
            f"{row['ecr']:.3f} & "
            f"{event_str}"
            r" \\"
        )

        f.write(line + "\n\n")

    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")

print("saved:", out_file)