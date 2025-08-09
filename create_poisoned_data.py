import pandas as pd
import numpy as np

def generate_poisoned_file(poison_percentage=15):
    """
    Downloads the UCI Adult dataset, poisons it by flipping a specified
    percentage of labels, and saves it to a new CSV file.

    Args:
        poison_percentage (int): The percentage of labels to flip (e.g., 15 for 15%).
    """
    if not 0 < poison_percentage < 100:
        print("❌ Error: Please provide a poison percentage between 1 and 99.")
        return

    poison_fraction = poison_percentage / 100.0
    print(f"▶️  Starting process to create a dataset with {poison_percentage}% poisoned labels...")

    try:
        # Step 1: Download the original, clean dataset
        print("   [1/3] Downloading original UCI Adult dataset...")
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
            'marital-status', 'occupation', 'relationship', 'race', 'sex', 
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        data = pd.read_csv(url, header=None, names=columns, na_values=' ?', sep=r',\s*', engine='python')
        data.dropna(inplace=True)
        print("   Download complete.")

        # Step 2: Poison the data
        print(f"   [2/3] Intentionally poisoning {poison_percentage}% of the labels...")
        poisoned_data = data.copy()
        
        # Get the indices of the samples we are going to poison
        total_rows = len(poisoned_data)
        num_to_poison = int(total_rows * poison_fraction)
        poison_indices = np.random.choice(poisoned_data.index, size=num_to_poison, replace=False)

        # Flip the 'income' label for the chosen indices
        original_labels = poisoned_data.loc[poison_indices, 'income'].copy()
        poisoned_data.loc[poison_indices, 'income'] = np.where(original_labels == '<=50K', '>50K', '<=50K')
        print(f"   Flipped the labels of {num_to_poison} random rows.")

        # Step 3: Save the new poisoned dataset
        output_filename = f'poisoned_adult_{poison_percentage}percent.csv'
        poisoned_data.to_csv(output_filename, index=False)
        print(f"   [3/3] Saving the new file.")

        # Final Report
        print("\n" + "="*50)
        print("✅ Success! Your test file has been created.")
        print(f"   File Name:      {output_filename}")
        print(f"   Total Rows:       {total_rows}")
        print(f"   Poisoned Rows:    {num_to_poison} ({poison_percentage}%)")
        print("="*50)
        print("\nYou can now upload this file to the DepoisonAI app to test it.")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("   Please check your internet connection and try again.")

if __name__ == "__main__":
    # You can change the percentage here if you want
    # For example, to create a 30% poisoned file, change it to generate_poisoned_file(30)
    generate_poisoned_file(poison_percentage=20)