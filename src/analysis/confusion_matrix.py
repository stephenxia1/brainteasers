import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

true_labels = [0, 1, 0, 1, 0, 1]
pred_labels = [0, 1, 1, 1, 0, 0]


def generate_and_plot_confusion_matrix(true_labels, pred_labels):
    # Define the classes
    classes = [0, 1]  # 0 = correct, 1 = incorrect

    # Initialize confusion matrix
    matrix = np.zeros((2, 2), dtype=int)

    # Fill the matrix
    for true, pred in zip(true_labels, pred_labels):
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)
        matrix[true_idx, pred_idx] += 1

    # Create a DataFrame
    df = pd.DataFrame(
        matrix,
        index=[f'True {cls}' for cls in classes],
        columns=[f'Pred {cls}' for cls in classes]
    )

    # Plot the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    return df

# Example usage:

confusion_matrix = generate_and_plot_confusion_matrix(true_labels, pred_labels)
print(confusion_matrix)