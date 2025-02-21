import numpy as np
import einops
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import pandas as pd
from utilities import mostWeight, min_residual, split_data_equal, calculate_metrics, load_dataset
from triangulation import customCVX


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run classification with specified parameters')
    parser.add_argument('--frac_labeled', type=float, required=False,
                        help='Fraction of data to label per class', default=0.001)
    parser.add_argument('--num_unlabeled', type=int, required=False,
                        help='Fraction of data to label per class', default=100)
    parser.add_argument('--dataset', type=str, required=False,
                        help='Fraction of data to label per class', default="Salinas")
    parser.add_argument('--n_trials', type=int, required=False,
                        help='number of trials to run', default=50)
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = 'best_params'
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    data, labels = load_dataset(args.dataset)

    # Parameters
    num_unlabeled = args.num_unlabeled
    n_trials = args.n_trials
    frac_labeled = args.frac_labeled

    rhos = [10**x for x in range(-8, 4)]

    R_accuracies_rho = np.zeros(len(rhos))
    R_res_accuracies_rho = np.zeros(len(rhos))

    for rho_idx, rho in enumerate(tqdm(rhos)):

        # Initialize accuracy arrays
        R_accuracies = np.zeros(n_trials)
        R_res_accuracies = np.zeros(n_trials)

        # Run trials
        for trial in range(n_trials):
            labeled_data, labeled_labels, unlabeled_data, unlabeled_labels, labeled_indices, unlabeled_indices, background_indices = split_data_equal(
                data, labels, frac_labeled, num_unlabeled=num_unlabeled
            )

            mean = labeled_data.mean(axis=0)
            labeled_data -= mean
            unlabeled_data -= mean

            norm = np.linalg.norm(labeled_data, axis=1).max()
            labeled_data /= norm
            unlabeled_data /= norm

            r_predicted = np.zeros(len(unlabeled_data), dtype=int)
            r_predicted_res = np.zeros(len(unlabeled_data), dtype=int)

            # R classification
            for i, y in enumerate(unlabeled_data):
                _, weight = customCVX(
                    labeled_data, einops.rearrange(y, "v -> 1 v"), rho)
                r_predicted[i] = mostWeight(weight, labeled_labels)
                r_predicted_res[i] = min_residual(
                    einops.rearrange(labeled_data, "n m -> m n"),
                    weight,
                    einops.rearrange(y, "v -> v 1"),
                    labeled_labels
                )
            _, accuracy = calculate_metrics(unlabeled_labels, r_predicted)
            R_accuracies[trial] = accuracy
            _, accuracy = calculate_metrics(unlabeled_labels, r_predicted_res)
            R_res_accuracies[trial] = accuracy

        R_accuracies_rho[rho_idx] = R_accuracies.mean()
        R_res_accuracies_rho[rho_idx] = R_res_accuracies.mean()

    # Find best accuracies and corresponding rho values
    best_R_idx = np.argmax(R_accuracies_rho)
    best_R_res_idx = np.argmax(R_res_accuracies_rho)

    results = {
        'Dataset': [args.dataset, args.dataset],
        'Method': ['R', 'R_res'],
        'Best_Accuracy': [R_accuracies_rho[best_R_idx], R_res_accuracies_rho[best_R_res_idx]],
        'Best_Param': [rhos[best_R_idx], rhos[best_R_res_idx]]
    }

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'{output_dir}/best_accuracies_{args.dataset}_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()
