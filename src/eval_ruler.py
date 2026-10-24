"""Re-score existing RULER predictions without loading a model."""
import argparse

from ruler_metrics import TASKS, evaluate_directory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    args = parser.parse_args()
    scores, average = evaluate_directory(args.save_dir, args.tasks)
    for row in [*scores, average]:
        print(row)
