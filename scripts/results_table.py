"""Merge eval_almi.py result files into one Markdown table next to the paper's numbers.

Usage: python scripts/results_table.py results/*.json > RESULTS.md
"""
import json
import sys

COLS = ["Evel", "Eang", "Ejpe", "Ekpe", "Eact_up", "Eact_low", "Eg", "Survival"]
# Table 6 of arXiv 2504.14305v3 (its first row is also the ALMI row of Table 4)
PAPER = {
    "easy": {
        "lower-3 + upper-2": [0.1135, 0.2647, 0.1931, 0.0460, 0.0462, 0.0170, 0.6919, 1.0000],
        "lower-2 + upper-2": [0.1164, 0.2669, 0.1955, 0.0452, 0.0475, 0.0171, 0.7121, 1.0000],
        "lower-1 + upper-2": [0.1271, 0.2738, 0.1928, 0.0526, 0.0642, 0.0171, 0.7052, 1.0000],
    },
    "medium": {
        "lower-3 + upper-2": [0.2192, 0.3520, 0.2007, 0.0450, 0.0598, 0.0172, 0.7604, 0.9852],
        "lower-2 + upper-2": [0.2213, 0.3571, 0.2032, 0.0458, 0.0607, 0.0172, 0.7748, 0.9772],
        "lower-1 + upper-2": [0.2262, 0.3872, 0.2173, 0.0492, 0.0604, 0.0175, 0.7730, 0.9273],
    },
    "hard": {
        "lower-3 + upper-2": [0.2202, 0.4812, 0.2116, 0.0458, 0.0600, 0.0175, 0.8551, 0.9723],
        "lower-2 + upper-2": [0.2892, 0.5395, 0.2231, 0.0482, 0.0645, 0.0178, 0.9479, 0.9233],
        "lower-1 + upper-2": [0.2566, 0.5172, 0.2451, 0.0537, 0.0777, 0.0179, 0.9462, 0.8743],
    },
}


def main(paths):
    runs = [json.load(open(p)) for p in paths]
    print("# ALMI evaluation results\n")
    print(f"Evaluation set: {runs[0]['num_motions']} held-out ALMI-X motions, "
          f"{runs[0]['steps_per_test']} steps per test (the paper uses 1122 CMU clips). "
          "Metric definitions: docs/EVALUATION.md. Paper values: Table 6 of arXiv 2504.14305v3. "
          "Eg and the action terms are defined differently from the paper, so compare trends, not absolute values.\n")
    print("| Level | Policy pair | Source | " + " | ".join(COLS) + " |")
    print("|" + "---|" * (len(COLS) + 3))
    for level in ["easy", "medium", "hard"]:
        for run in runs:
            if level not in run["results"]:
                continue
            m = run["results"][level]["metrics"]
            print(f"| {level} | {run['label']} | ours | " + " | ".join(f"{m[c]:.4f}" for c in COLS) + " |")
            paper = PAPER.get(level, {}).get(run["label"])
            if paper:
                print(f"| {level} | {run['label']} | paper | " + " | ".join(f"{v:.4f}" for v in paper) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
