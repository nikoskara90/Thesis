import argparse
import glob
import json
import os
from itertools import chain
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

EV_OUT = "evaluation.prototext"


def read_solution_files(solutions_folder: str) -> dict:
    """
    reads all solution files into dict
    """
    solutions = {}
    pattern = os.path.join(solutions_folder, 'solution-problem-*.json')
    #print(f"[DEBUG] Looking for solution files in: {pattern}")
    files_found = glob.glob(pattern)
    #print(f"[DEBUG] Found {len(files_found)} solution files: {files_found}")
    for solution_file in files_found:
        with open(solution_file, 'r') as fh:
            curr_solution = json.load(fh)
            key = os.path.basename(solution_file)[9:-5]
            #print(f"[DEBUG] Loading solution key: {key} from {solution_file}")
            solutions[key] = curr_solution
    return solutions


def read_ground_truth_files(truth_folder: str) -> dict:
    """
    reads ground truth files into dict
    """
    truth = {}
    pattern = os.path.join(truth_folder, 'truth-problem-*.json')
    #print(f"[DEBUG] Looking for truth files in: {pattern}")
    files_found = glob.glob(pattern)
    #print(f"[DEBUG] Found {len(files_found)} truth files: {files_found}")
    for truth_file in files_found:
        with open(truth_file, 'r') as fh:
            curr_truth = json.load(fh)
            key = os.path.basename(truth_file)[6:-5]
            #print(f"[DEBUG] Loading truth key: {key} from {truth_file}")
            truth[key] = curr_truth
    return truth


def extract_task_results(truth: dict, solutions: dict, task: str) -> tuple:
    """
    extracts truth and solution values for a given task
    :param truth: dict of all ground truth values with problem-id as key
    :param solutions: dict of all solution values with problem-id as key
    :param task: task for which values are extracted (string, e.g., 'multi-author' or 'changes')
    :return: list of all ground truth values, list of all solution values for given task
    """
    all_solutions = []
    all_truth = []
    for problem_id, truth_instance in sorted(truth.items()):
        if len(truth_instance[task]) != len(solutions[problem_id][task]):
            print(
                f"Solution length for problem {problem_id} is not correct, skipping.")
            continue
        all_truth.append(truth_instance[task])
        all_solutions.append(solutions[problem_id][task])
    return all_truth, all_solutions


def compute_score_multiple_predictions(truth_values: dict, solution_values: dict, key: str, labels: list) -> dict:
    """ compute f1 score for list of predictions
    :param labels: labels used for the predictions
    :param truth_values: list of ground truth values for all problem-ids
    :param solution_values: list of solutions for all problem-ids
    :param key: key of solutions to compute score for (=task)
    :return: f1 score
    """

    truth, solution = extract_task_results(truth_values, solution_values, key)
    flat_truth = list(chain.from_iterable(truth))
    flat_solution = list(chain.from_iterable(solution))

    metrics = {
        "f1_score": f1_score(flat_truth, flat_solution, average='macro', labels=labels, zero_division=0),
        "accuracy": accuracy_score(flat_truth, flat_solution),
        "precision": precision_score(flat_truth, flat_solution, average='macro', labels=labels, zero_division=0),
        "recall": recall_score(flat_truth, flat_solution, average='macro', labels=labels, zero_division=0),
    }
    return metrics


def write_output(filename: str, k: str, v: str):
    """
    print() and write a given measurement to the indicated output file
    """
    line = 'measure{{\n  key: "{}"\n  value: "{}"\n}}\n'.format(k, str(v))
    print(line.strip())  # strip avoids extra newlines
    with open(filename, "a") as fh:
        fh.write(line)


def main():
    parser = argparse.ArgumentParser(
        description='Style Change Detection Task: Evaluator')
    parser.add_argument("-p", "--predictions",
                        help="path to the dir holding the predictions", required=True)
    parser.add_argument("-t", "--truth",
                        help="path to the dir holding the true labels", required=True)
    parser.add_argument("-o", "--output",
                        help="path to the dir to write the results to", required=True)
    args = parser.parse_args()

    # log which prediction/truth set is being evaluated
    run_info = f"Evaluating predictions from: {args.predictions}, truth from: {args.truth}"
    print(run_info)
    with open(os.path.join(args.output, EV_OUT), "a") as fh:
        fh.write("# " + run_info + "\n")

    # Read data
    solutions = read_solution_files(args.predictions)
    truth = read_ground_truth_files(args.truth)

    # Compute metrics
    try:
        metrics = compute_score_multiple_predictions(truth, solutions, 'changes', labels=[0, 1])
    except KeyError:
        metrics = {}
        print("No solution file found for one or more problems. Exiting.")

    # Write metrics
    for k, v in metrics.items():
        write_output(os.path.join(args.output, EV_OUT), k, v)



if __name__ == "__main__":
    main()

