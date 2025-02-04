import numpy as np
import pandas as pd
import os
import argparse

def uniform_noise(x, min_v, max_v, seed):
    """
    Adds uniform noise to the input array x.

    Parameters:
    - x (numpy array): The array to which noise is added.
    - min_v (float): The minimum value of the uniform noise.
    - max_v (float): The maximum value of the uniform noise.

    Returns:
    - numpy array: Array with uniform noise added.
    """
    np.random.seed(seed)
    noise = np.random.uniform(min_v, max_v, size=x.shape)
    x_noisy = x + noise
    return x_noisy

def process_fold(split_folder, output_dir, df_labels, min_v, max_v, n_splits, seed):
    """
    Processes each fold in the cross-validation setup, injecting noise into labels.

    Parameters:
    - split_folder (str): Path to the folder containing split CSV files.
    - output_dir (str): Path to save the output files.
    - df_labels (pd.DataFrame): DataFrame containing the labels.
    - min_v (float): Minimum value for uniform noise.
    - max_v (float): Maximum value for uniform noise.
    - n_splits (int): Number of cross-validation splits.
    """
    for n in range(n_splits):
        print(f"Processing fold: {n}")
        
        # Load split information for the current fold
        df_fold = pd.read_csv(f"{split_folder}/splits_{n}.csv")
        train = df_fold['train'].tolist()
        
        # Extract labels for training set
        list_labels = []
        indexes = []
        for index, row in df_labels.iterrows():
            slide_id = row['slide_id']  
            label = row['label']       

            if slide_id in train and label != 0:
                list_labels.append(label)
                indexes.append(index)
        
        # Ensure labels are numeric
        list_labels = np.array(list_labels, dtype=float)  # Convert to numeric type
        
        # Apply uniform noise to the labels
        uniform = uniform_noise(list_labels, min_v, max_v, seed)
        
        # Ensure labels stay within valid bounds (0 to 1)
        uniform = np.clip(uniform, 0, 1)
        
        # Create a new DataFrame with noisy labels
        new_df_unif = []
        k = 0
        for index, row in df_labels.iterrows():
            slide_id = row['slide_id']  # Explicitly reference slide_id column
            label = row['label']       # Explicitly reference label column

            if slide_id in train and label != 0:
                new_df_unif.append({'slide_id': slide_id, 'label': uniform[k]})
                k += 1
            else:
                new_df_unif.append({'slide_id': slide_id, 'label': label})
        
        df_uniform = pd.DataFrame(new_df_unif)
        
        # Calculate the average difference introduced by the noise
        diff_uniform = np.mean(np.abs(np.array(list_labels) - uniform))
        print(f"Average difference (uniform noise): {diff_uniform:.4f}")
        
        # Save the DataFrame to a new CSV file
        os.makedirs(output_dir, exist_ok=True)
        df_uniform.to_csv(f"{output_dir}/splits_{n}.csv", index=False)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Add uniform noise to labels in cross-validation folds.")
    parser.add_argument("--split_folder", type=str, required=True, help="Path to the folder containing split CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output files.")
    parser.add_argument("--labels_file", type=str, required=True, help="Path to the labels CSV file.")
    parser.add_argument("--min_v", type=float, default=-0.1, help="Minimum value for uniform noise.")
    parser.add_argument("--max_v", type=float, default=0.1, help="Maximum value for uniform noise.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation splits.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load labels
    df_labels = pd.read_csv(args.labels_file)
    
    # Process folds
    process_fold(args.split_folder, args.output_dir, df_labels, args.min_v, args.max_v, args.n_splits, args.seed)
