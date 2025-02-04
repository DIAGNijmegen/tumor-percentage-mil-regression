import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from verstack.stratified_continuous_split import scsplit


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate K-fold splits for dataset.")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the input CSV data file."
    )
    parser.add_argument(
        "--result_dir", type=str, required=True, help="Directory where split files will be saved."
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of folds for cross-validation."
    )
    return parser.parse_args()


def create_initial_split(df, test_size=0.2, random_state=42):
    """Create initial train-validation split."""
    x_train, x_val, y_train, y_val = scsplit(
        df['slide_id'], df['label'], stratify=df['label'], test_size=test_size, random_state=random_state
    )
    return x_train, x_val, y_train, y_val


def create_folds(df, n_folds=5):
    """Generate K-fold splits for the dataset."""
    # Create initial 80-20 split
    x_train, x_val, y_train, y_val = create_initial_split(df)

    # Initialize storage for fold indices
    val_folds_ix_dict = {0: x_val.index.tolist()}
    used_vals = val_folds_ix_dict[0].copy()
    val_size = len(val_folds_ix_dict[0])
    
    # Create subsequent folds
    for fold in range(1, n_folds):
        fold_val_ix = []
        while len(fold_val_ix) < val_size:
            X_train, X_val, Y_train, Y_val = scsplit(
                df['slide_id'], df['label'], stratify=df['label'], test_size=0.2
            )
            new_val_ix = list(set(X_val.index) - set(used_vals))
            for i in new_val_ix:
                if len(fold_val_ix) < val_size:
                    fold_val_ix.append(i)
                    used_vals.append(i)
                else:
                    break
        val_folds_ix_dict[fold] = fold_val_ix

    return val_folds_ix_dict


def analyze_folds(df, val_folds_ix_dict):
    """Analyze and print statistics for each fold."""
    y = df['label']
    for fold_num, fold_values in val_folds_ix_dict.items():
        val_ix = fold_values
        train_ix = list(set(y.index) - set(val_ix))
        print(f'Fold: {fold_num}')
        print(f'Test target mean  : {y[val_ix].mean()}')
        print(f'Train target mean: {y[train_ix].mean()}')
        print('-' * 25)


def save_splits(df, val_folds_ix_dict, result_dir):
    """Save train, test, and validation splits to CSV files."""
    for fold_num, fold_values in val_folds_ix_dict.items():
        slides, labels, test_slides = [], [], []
        for index, row in df.iterrows():
            if index in fold_values:
                test_slides.append(row['slide_id'])
            else:
                slides.append(row['slide_id'])
                labels.append(row['label'])

        # Save test set as a text file
        with open(f"{result_dir}file_{fold_num}.txt", "w") as textfile:
            for element in test_slides:
                textfile.write(element + "\n")

        # Train-validation split
        df_temp = pd.DataFrame({'slides': slides, 'labels': labels})
        X_train, X_val, y_train, y_val = train_test_split(df_temp['slides'], df_temp['labels'], test_size=0.15)

        # Save the split
        #df_final = pd.DataFrame({'train': X_train.tolist(), 'test': pd.Series(test_slides), 'val': pd.Series(X_val.tolist())})
        df_final = pd.DataFrame(columns=['train', 'val', 'test'])
        df_final['train'] = X_train.tolist()
        df_final['test'] = pd.Series(test_slides)
        df_final['val'] = pd.Series(X_val.tolist())
        df_final.to_csv(f"{result_dir}split_{fold_num}.csv", index=None)

        print(f'Train target mean: {np.mean(np.array(y_train))}')
        print(f'Val target mean: {np.mean(np.array(y_val))}')
        print(f"Size val set: {len(X_val)}")
        print(f"Size train set: {len(X_train)}")


def main():
    args = parse_args()
    # Load the data
    df = pd.read_csv(args.data_path)
    print(f"Data size: {len(df)}")

    # Create the folds
    val_folds_ix_dict = create_folds(df, n_folds=args.n_folds)

    # Analyze folds
    analyze_folds(df, val_folds_ix_dict)

    os.makedirs(args.result_dir, exist_ok=True)
    # Save the splits
    save_splits(df, val_folds_ix_dict, args.result_dir)


if __name__ == "__main__":
    main()
