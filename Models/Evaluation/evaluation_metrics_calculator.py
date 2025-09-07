import torch
import pandas as pd
import os

# Find all confusion matrices inside conf_matrices directory
conf_matrices = os.listdir('conf_matrices')
model_metrics = {}

for conf_matrix in conf_matrices:
    # Load confusion matrix
    confusion_matrix = torch.load(f'./conf_matrices/{conf_matrix}')

    # Dictionary to store confusion matrix elements
    conf_matrix_elements = {}
    for class_num in range(confusion_matrix.size()[0]):
        tp = confusion_matrix[class_num, class_num].item()
        fn = confusion_matrix[class_num].sum().item() - tp
        fp = confusion_matrix[:, class_num].sum().item() - tp
        conf_matrix_elements[class_num] = {'tp': tp, 'fp': fp, 'fn': fn}

        # Micro averaging: Precision, Recall and F1 Score are equal
    micro = sum(item['tp'] for item in conf_matrix_elements.values()) / (
                sum(item['tp'] for item in conf_matrix_elements.values()) + sum(
            item['fp'] for item in conf_matrix_elements.values()))

    # Macro averaging
    macro_precision = sum(item['tp'] / (item['tp'] + item['fp']) for item in conf_matrix_elements.values()) / len(
        conf_matrix_elements)
    macro_recall = sum(item['tp'] / (item['tp'] + item['fn']) for item in conf_matrix_elements.values()) / len(
        conf_matrix_elements)
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    # Weighted averaging
    weighted_precision = sum(
        item['tp'] * (item['tp'] + item['fn']) / ((item['tp'] + item['fp']) * confusion_matrix.sum().item()) for item in
        conf_matrix_elements.values())
    weighted_recall = sum(
        item['tp'] * (item['tp'] + item['fn']) / ((item['tp'] + item['fn']) * confusion_matrix.sum().item()) for item in
        conf_matrix_elements.values())
    weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)

    model_metrics[conf_matrix[12:-4]] = {
        'micro_precision': micro,
        'micro_recall': micro,
        'micro_f1': micro,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
    }

# Create dataframe that include all metrics per model
metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index')
# Save metrics dataframe to csv
metrics_df.to_csv('./evaluation_metrics/metrics_per_model.csv')

