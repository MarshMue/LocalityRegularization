import numpy as np
import einops
import matplotlib.pyplot as plt
from tqdm import tqdm
import spams
import argparse
import os
from datetime import datetime
import pandas as pd
from utilities import mostWeight, min_residual, split_data_equal, calculate_metrics, load_dataset
from triangulation import customCVX

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run classification with specified parameters')
    parser.add_argument('--rho', type=float, required=True, help='Rho parameter for customCVX')
    parser.add_argument('--lambda1', type=float, required=True, help='Lambda parameter for LASSO')
    parser.add_argument('--frac_labeled', type=float, required=False, help='Fraction of data to label per class', default=0.001)
    parser.add_argument('--num_unlabeled', type=int, required=False, help='Fraction of data to label per class', default=100)
    parser.add_argument('--dataset', type=str, required=False, help='Fraction of data to label per class', default="Salinas")
    parser.add_argument('--n_trials', type=int, required=False, help='number of trials to run', default=50)
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = 'all_results'
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    data, labels = load_dataset(args.dataset)

    # Parameters
    num_unlabeled = args.num_unlabeled
    n_trials = args.n_trials
    frac_labeled = args.frac_labeled
    
    # Initialize accuracy arrays
    R_accuracies = np.zeros(n_trials)
    R_res_accuracies = np.zeros(n_trials)
    nn_accuracies = np.zeros(n_trials)
    l1_accuracies = np.zeros(n_trials)

    # Run trials
    for trial in tqdm(range(n_trials)):
        labeled_data, labeled_labels, unlabeled_data, unlabeled_labels, labeled_indices, unlabeled_indices, background_indices = split_data_equal(
            data, labels, frac_labeled, num_unlabeled=num_unlabeled
        )

        mean = labeled_data.mean(axis=0)
        labeled_data -= mean
        unlabeled_data -= mean

        norm = np.linalg.norm(labeled_data, axis=1).max()
        labeled_data /=  norm
        unlabeled_data /= norm

        nn_predicted = np.zeros(len(unlabeled_data), dtype=int)
        r_predicted = np.zeros(len(unlabeled_data), dtype=int)
        r_predicted_res = np.zeros(len(unlabeled_data), dtype=int)
        l1_predicted_res = np.zeros(len(unlabeled_data), dtype=int)

        # Nearest neighbor classification
        for i, y in enumerate(unlabeled_data):
            distances = np.linalg.norm(labeled_data - einops.rearrange(y, "n -> 1 n"), axis=1)
            nn_predicted[i] = labeled_labels[distances.argmin()]
        _, accuracy = calculate_metrics(unlabeled_labels, nn_predicted)
        nn_accuracies[trial] = accuracy

        # R classification
        for i, y in enumerate(unlabeled_data):
            _, weight = customCVX(labeled_data, einops.rearrange(y, "v -> 1 v"), args.rho)
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

        # L1 classification
        for i, y in enumerate(unlabeled_data):
            w = spams.lasso(
                einops.rearrange(y, "n -> n 1"),
                einops.rearrange(labeled_data, "n m -> m n"),
                lambda1=args.lambda1,
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

    # Create results dictionary
    results = {
        'NN': nn_accuracies,
        'R-max mass': R_accuracies,
        'R-residual': R_res_accuracies,
        'l1-residual': l1_accuracies
    }

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'{args.dataset}_n_{n_trials}_N_{num_unlabeled}_f_{frac_labeled:.2e}rho_{args.rho}_lambda_{args.lambda1}_{timestamp}'

    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_filename = os.path.join(output_dir, f'{base_filename}.csv')
    results_df.to_csv(csv_filename, index=False)

    # Create and save plot
    fig, ax = plt.subplots(figsize=(5,4))
    accuracies = [nn_accuracies, R_accuracies, R_res_accuracies, l1_accuracies]
    accuracy_labels = ["NN", "R-max mass", "R-residual", r"$\ell_1$-residual"]
    
    for accuracy, label in zip(accuracies, accuracy_labels):
        print(f"{label}, avg acc: {accuracy.mean():.3f}")
    
    ax.boxplot(accuracies)
    ax.set_xticklabels(accuracy_labels)
    ax.set_title(rf"$\rho={args.rho}, \lambda={args.lambda1}$")
    ax.set_ylim(0,1)
    
    plot_filename = os.path.join(output_dir, f'{base_filename}.png')
    plt.savefig(plot_filename)
    plt.close()

if __name__ == "__main__":
    main()