import json
import argparse
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson as simps

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("anchor", help="Reference method JSON file")
    parser.add_argument("test", help="Tested method JSON file")
    parser.add_argument("--frame_type", choices=["i", "p", "all"], default="all")
    parser.add_argument("--out_dir", required=True, help="Directory for all output")
    parser.add_argument("--anchor_label", default="Anchor", help="Legend label for anchor method")
    parser.add_argument("--test_label", default="Test", help="Legend label for test method")
    parser.add_argument("--save_plots", action="store_true", help="Enable saving RD curve plots")
    parser.add_argument("--save_csv", action="store_true", help="Enable saving CSV summary data")
    return parser.parse_args()

def extract_by_sequence(data, frame_type):
    results = defaultdict(lambda: defaultdict(dict))
    for dataset, sequences in data.items():
        for seq, entries in sequences.items():
            for key, v in entries.items():
                rate_idx = v["rate_idx"]
                bpp = v[f"ave_{frame_type}_frame_bpp"]
                psnr = v[f"ave_{frame_type}_frame_psnr"]
                results[dataset][seq][rate_idx] = (bpp, psnr)
    return results

def bd_rate(R1, D1, R2, D2):
    if len(R1) < 2 or len(R2) < 2 or len(set(D1)) < 2 or len(set(D2)) < 2:
        return np.nan
    try:
        interp1 = PchipInterpolator(D1, np.log(R1))
        interp2 = PchipInterpolator(D2, np.log(R2))
    except ValueError:
        return np.nan
    d_min = max(min(D1), min(D2))
    d_max = min(max(D1), max(D2))
    if d_max <= d_min:
        return np.nan
    samples = np.linspace(d_min, d_max, 100)
    int1 = simps(interp1(samples), samples)
    int2 = simps(interp2(samples), samples)
    return (np.exp((int2 - int1) / (d_max - d_min)) - 1) * 100

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_rd_curve(bpp1, psnr1, bpp2, psnr2, title, out_path, label1, label2):
    plt.figure()
    plt.plot(bpp1, psnr1, 'o--', label=label1)
    plt.plot(bpp2, psnr2, 's-', label=label2)
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_csv_table(filename, header, rows):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    if args.save_plots:
        ensure_dir(os.path.join(args.out_dir, "plots"))
    if args.save_csv:
        ensure_dir(os.path.join(args.out_dir, "csv"))

    with open(args.anchor) as f: anchor_json = json.load(f)
    with open(args.test) as f: test_json = json.load(f)

    anchor = extract_by_sequence(anchor_json, args.frame_type)
    test = extract_by_sequence(test_json, args.frame_type)

    all_anchor_points = defaultdict(list)
    all_test_points = defaultdict(list)
    csv_seq_rows = []
    csv_ds_rows = []

    for dataset in anchor:
        for seq in anchor[dataset]:
            if seq not in test[dataset]:
                continue
            a = anchor[dataset][seq]
            t = test[dataset][seq]
            if set(a.keys()) != set(t.keys()):
                continue
            a_bpp = [a[i][0] for i in sorted(a)]
            a_psnr = [a[i][1] for i in sorted(a)]
            t_bpp = [t[i][0] for i in sorted(t)]
            t_psnr = [t[i][1] for i in sorted(t)]
            bdr = bd_rate(np.array(a_bpp), np.array(a_psnr), np.array(t_bpp), np.array(t_psnr))
            csv_seq_rows.append([dataset, seq, round(bdr, 4)])
            if args.save_plots:
                out_file = os.path.join(args.out_dir, "plots", f"{dataset}__{seq}.png")
                plot_rd_curve(a_bpp, a_psnr, t_bpp, t_psnr, f"{dataset} / {seq}", out_file, args.anchor_label, args.test_label)

    for dataset in anchor:
        anchor_curve = defaultdict(list)
        test_curve = defaultdict(list)
        for seq in anchor[dataset]:
            if seq not in test[dataset]:
                continue
            a = anchor[dataset][seq]
            t = test[dataset][seq]
            if set(a.keys()) != set(t.keys()):
                continue
            for idx in a:
                anchor_curve[idx].append(a[idx])
                test_curve[idx].append(t[idx])

        if not anchor_curve:
            continue

        a_bpp = [np.mean([pt[0] for pt in anchor_curve[i]]) for i in sorted(anchor_curve)]
        a_psnr = [np.mean([pt[1] for pt in anchor_curve[i]]) for i in sorted(anchor_curve)]
        t_bpp = [np.mean([pt[0] for pt in test_curve[i]]) for i in sorted(test_curve)]
        t_psnr = [np.mean([pt[1] for pt in test_curve[i]]) for i in sorted(test_curve)]
        bdr = bd_rate(np.array(a_bpp), np.array(a_psnr), np.array(t_bpp), np.array(t_psnr))
        csv_ds_rows.append([dataset, round(bdr, 4)])

        if args.save_plots:
            out_file = os.path.join(args.out_dir, "plots", f"{dataset}__AVG.png")
            plot_rd_curve(a_bpp, a_psnr, t_bpp, t_psnr, f"{dataset} (Avg)", out_file, args.anchor_label, args.test_label)

        for i in anchor_curve:
            all_anchor_points[i].extend(anchor_curve[i])
            all_test_points[i].extend(test_curve[i])

    if all_anchor_points:
        a_bpp = [np.mean([pt[0] for pt in all_anchor_points[i]]) for i in sorted(all_anchor_points)]
        a_psnr = [np.mean([pt[1] for pt in all_anchor_points[i]]) for i in sorted(all_anchor_points)]
        t_bpp = [np.mean([pt[0] for pt in all_test_points[i]]) for i in sorted(all_test_points)]
        t_psnr = [np.mean([pt[1] for pt in all_test_points[i]]) for i in sorted(all_test_points)]
        global_bdr = bd_rate(np.array(a_bpp), np.array(a_psnr), np.array(t_bpp), np.array(t_psnr))
        csv_ds_rows.append(["GLOBAL", round(global_bdr, 4)])
        if args.save_plots:
            out_file = os.path.join(args.out_dir, "plots", "__GLOBAL.png")
            plot_rd_curve(a_bpp, a_psnr, t_bpp, t_psnr, "GLOBAL (Avg)", out_file, args.anchor_label, args.test_label)

    if args.save_csv:
        save_csv_table(os.path.join(args.out_dir, "csv", "bd_seq.csv"), ["Dataset", "Sequence", "BD-Rate (%)"], csv_seq_rows)
        save_csv_table(os.path.join(args.out_dir, "csv", "bd_dataset.csv"), ["Dataset", "Mean_BD_Rate_Curve"], csv_ds_rows)

    df_seq = pd.DataFrame(csv_seq_rows, columns=["Dataset", "Sequence", "BD-Rate (%)"])
    mean_seq = df_seq.groupby("Dataset")["BD-Rate (%)"].mean().reset_index()
    mean_seq = mean_seq.rename(columns={"BD-Rate (%)": "Mean_BD_Rate_Seq"})

    global_seq_mean = df_seq["BD-Rate (%)"].mean()
    mean_seq = pd.concat([
        mean_seq,
        pd.DataFrame([{"Dataset": "GLOBAL", "Mean_BD_Rate_Seq": global_seq_mean}])
    ], ignore_index=True)

    df_ds = pd.DataFrame(csv_ds_rows, columns=["Dataset", "Mean_BD_Rate_Curve"])
    merged = pd.merge(mean_seq, df_ds, on="Dataset", how="outer")

    print("\n=== BD-Rate Summary (2 Estratégias) ===")
    print(merged.to_string(index=False))

    if args.save_csv:
        merged.to_csv(os.path.join(args.out_dir, "csv", "bdrate_dual_summary.csv"), index=False)

if __name__ == "__main__":
    main()
