import numpy as np
import einops
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import pandas as pd
from utilities import min_residual, split_data_equal, calculate_metrics, load_dataset
import spams

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run classification with specified parameters')
    parser.add_argument('--num_unlabeled', type=int, required=False,
                       help='Number of unlabeled samples to use', default=100)
    parser.add_argument('--dataset', type=str, required=False,
                       help='Dataset to use', default="Salinas")
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
    frac_labeled_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    lams = [10**x for x in range(-8, 4)]

    # Create a list to store results for all fractions
    all_results = []

    # Iterate over different frac_labeled values
    for frac_labeled in tqdm(frac_labeled_values, desc="Testing different fractions"):
        l1_accuracies_lam = np.zeros(len(lams))
        
        for lam_idx, lam in enumerate(tqdm(lams, desc=f"Lambda values for frac={frac_labeled}", leave=False)):
            # Initialize accuracy arrays
            l1_accuracies = np.zeros(n_trials)

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

                l1_predicted_res = np.zeros(len(unlabeled_data), dtype=int)

                # L1 classification
                for i, y in enumerate(unlabeled_data):
                    w = spams.lasso(
                        einops.rearrange(y, "n -> n 1"),
                        einops.rearrange(labeled_data, "n m -> m n"),
                        lambda1=lam,
                        mode=2
                    )
                    w = w.toarray()
                    l1_predicted_res[i] = min_residual(
                        einops.rearrange(labeled_data, "n m -> m n"),
                        w,
                        einops.rearrange(y, "v -> v 1"),
                        labeled_labels
                    )

                _, accuracy = calculate_metrics(unlabeled_labels, l1_predicted_res)
                l1_accuracies[trial] = accuracy

            l1_accuracies_lam[lam_idx] = l1_accuracies.mean()

        # Find best accuracies and corresponding lambda values for this fraction
        best_l1_idx = np.argmax(l1_accuracies_lam)
        result = {
            'Dataset': args.dataset,
            'Method': 'l1_res',
            'Frac_Labeled': frac_labeled,
            'Best_Accuracy': l1_accuracies_lam[best_l1_idx],
            'Best_Param': lams[best_l1_idx]
        }
        all_results.append(result)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'{output_dir}/best_accuracies_{args.dataset}_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()