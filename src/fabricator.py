import os
import json
import argparse
import random

def fix_solutions(solution_path, truth_path):
    # output path = solution_path + "_appended"
    output_path = solution_path.rstrip("/\\") + "_appended"
    os.makedirs(output_path, exist_ok=True)

    for fname in os.listdir(solution_path):
        if not fname.startswith("solution-problem-") or not fname.endswith(".json"):
            continue

        sol_file = os.path.join(solution_path, fname)
        truth_file = os.path.join(truth_path, fname.replace("solution-", "truth-"))

        if not os.path.exists(truth_file):
            print(f"Truth file missing for {fname}, skipping.")
            continue

        with open(sol_file, "r", encoding="utf-8") as f:
            sol_data = json.load(f)

        with open(truth_file, "r", encoding="utf-8") as f:
            truth_data = json.load(f)

        sol_changes = sol_data.get("changes", [])
        truth_changes = truth_data.get("changes", [])

        # Check lengths
        if len(sol_changes) < len(truth_changes):
            diff = len(truth_changes) - len(sol_changes)
            for _ in range(diff):
                sol_changes.append(random.choice([0, 1]))
            sol_data["changes"] = sol_changes
        elif len(sol_changes) > len(truth_changes):
            print(f"Solution length for {fname.replace('solution-', 'problem-')} is not correct, skipping.")

        # Save new solution file
        out_file = os.path.join(output_path, fname)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(sol_data, f, ensure_ascii=False)

    print(f"All processed solutions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", required=True, help="Path to solution files")
    parser.add_argument("-t", "--truth", required=True, help="Path to truth files")
    args = parser.parse_args()

    fix_solutions(args.predictions, args.truth)
