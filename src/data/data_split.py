import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_and_save_data(features_path, target_path, output_path, test_size=0.2, random_state=42):
    """
    Loads prepared data, performs a train-test split, and saves the resulting
    training and testing sets to separate CSV files.

    Args:
        features_path (str): The file path to the prepared features CSV.
        target_path (str): The file path to the prepared target CSV.
        output_dir (str): The directory to save the output files.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
    """
    print("Loading prepared features and target data...")
    try:
        X = pd.read_csv(features_path)
        y = pd.read_csv(target_path).squeeze()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure '{features_path}' and '{target_path}' exist.")
        return

    print("Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Combine features and target for saving
    train_df = X_train.copy()
    train_df['Churn'] = y_train
    
    test_df = X_test.copy()
    test_df['Churn'] = y_test

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_output_path = os.path.join(output_path, 'train_processed.csv')
    test_output_path = os.path.join(output_path, 'test_processed.csv')

    print(f"Saving training data to '{train_output_path}'...")
    train_df.to_csv(train_output_path, index=False)
    
    print(f"Saving testing data to '{test_output_path}'...")
    test_df.to_csv(test_output_path, index=False)

    print("\nTrain-test split complete and data saved successfully.")
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")

if __name__ == "__main__":
    # Define file paths for the prepared data
    interim_data_path = 'data/interim'
    output_path = 'data/processed'
    features_file = os.path.join(interim_data_path,'x_features.csv')
    target_file = os.path.join(interim_data_path,'y_target.csv')
    
    # Run the splitting process
    split_and_save_data(features_file, target_file, output_path)