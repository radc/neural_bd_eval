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
    parser.add_argument("tests", nargs='+', help="Tested method JSON files")
    parser.add_argument("--frame_type", choices=["i", "p", "all"], default="all")
    parser.add_argument("--out_dir", required=True, help="Directory for all output")
    parser.add_argument("--anchor_label", default="Anchor", help="Legend label for anchor method")
    parser.add_argument("--test_labels", nargs='+', help="Legend labels for test methods")
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

def plot_multi_rd_curves(anchor_curve, test_curves, labels, title, out_path):
    plt.figure()
    plt.plot(anchor_curve[0], anchor_curve[1], 'o--', label=labels[0])
    for i, (bpp, psnr) in enumerate(test_curves):
        plt.plot(bpp, psnr, label=labels[i+1])
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
    anchor = extract_by_sequence(anchor_json, args.frame_type)
    test_jsons = [json.load(open(f)) for f in args.tests]
    test_data = [extract_by_sequence(js, args.frame_type) for js in test_jsons]
    labels = [args.anchor_label] + (args.test_labels if args.test_labels else [f"Test_{i+1}" for i in range(len(args.tests))])

    for dataset in anchor:
        anchor_curve = defaultdict(list)
        test_curves_data = [defaultdict(list) for _ in args.tests]

        for seq in anchor[dataset]:
            a = anchor[dataset][seq]
            curves_match = all(seq in test[dataset] and set(test[dataset][seq].keys()) == set(a.keys()) for test in test_data)
            if not curves_match:
                continue
            for idx in a:
                anchor_curve[idx].append(a[idx])
                for i, test in enumerate(test_data):
                    test_curves_data[i][idx].append(test[dataset][seq][idx])

        if anchor_curve:
            a_bpp = [np.mean([pt[0] for pt in anchor_curve[i]]) for i in sorted(anchor_curve)]
            a_psnr = [np.mean([pt[1] for pt in anchor_curve[i]]) for i in sorted(anchor_curve)]
            test_curves = []
            for tcurve in test_curves_data:
                t_bpp = [np.mean([pt[0] for pt in tcurve[i]]) for i in sorted(tcurve)]
                t_psnr = [np.mean([pt[1] for pt in tcurve[i]]) for i in sorted(tcurve)]
                test_curves.append((t_bpp, t_psnr))

            if args.save_plots:
                out_file = os.path.join(args.out_dir, "plots", f"{dataset}__AVG_ALL.png")
                plot_multi_rd_curves((a_bpp, a_psnr), test_curves, labels, f"{dataset} (Avg Curves)", out_file)

    # Global RD Curve
    all_anchor_points = defaultdict(list)
    all_test_points = [defaultdict(list) for _ in args.tests]

    for dataset in anchor:
        for seq in anchor[dataset]:
            a = anchor[dataset][seq]
            curves_match = all(seq in test[dataset] and set(test[dataset][seq].keys()) == set(a.keys()) for test in test_data)
            if not curves_match:
                continue
            for idx in a:
                all_anchor_points[idx].append(a[idx])
                for i, test in enumerate(test_data):
                    all_test_points[i][idx].append(test[dataset][seq][idx])

    if all_anchor_points:
        a_bpp = [np.mean([pt[0] for pt in all_anchor_points[i]]) for i in sorted(all_anchor_points)]
        a_psnr = [np.mean([pt[1] for pt in all_anchor_points[i]]) for i in sorted(all_anchor_points)]
        test_curves = []
        for tpoints in all_test_points:
            t_bpp = [np.mean([pt[0] for pt in tpoints[i]]) for i in sorted(tpoints)]
            t_psnr = [np.mean([pt[1] for pt in tpoints[i]]) for i in sorted(tpoints)]
            test_curves.append((t_bpp, t_psnr))

        if args.save_plots:
            out_file = os.path.join(args.out_dir, "plots", "GLOBAL__AVG_ALL.png")
            plot_multi_rd_curves((a_bpp, a_psnr), test_curves, labels, "GLOBAL (Avg Curves)", out_file)

if __name__ == "__main__":
    main()