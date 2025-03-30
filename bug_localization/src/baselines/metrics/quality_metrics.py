from typing import List


def precision(expected_set: set, actual_set: set) -> float:
    true_positives = len(expected_set & actual_set)
    false_positives = len(actual_set - expected_set)
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0


def recall(expected_set: set, actual_set: set) -> float:
    true_positives = len(expected_set & actual_set)
    false_negatives = len(expected_set - actual_set)
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0


def f1_score(precision_value: float, recall_value: float) -> float:
    return 2 * (precision_value * recall_value) / (precision_value + recall_value) if (
                                                                                              precision_value + recall_value) > 0 else 0


def false_positive_rate(expected_set: set, actual_set: set, all_files_set: set) -> float:
    false_positives = len(actual_set - expected_set)
    true_negatives = len(all_files_set - expected_set - actual_set)
    return false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0


def get_quality_metrics(all_files: List[str], expected_files: List[str], actual_files: List[str]):
    all_files_set = set(all_files)
    expected_set = set(expected_files)
    actual_set = set(actual_files)

    # Calculate metrics
    precision_value = precision(expected_set, actual_set)
    recall_value = recall(expected_set, actual_set)
    f1_value = f1_score(precision_value, recall_value)
    fpr_value = false_positive_rate(expected_set, actual_set, all_files_set)

    # Boolean metrics
    all_correct = actual_set == expected_set  # Check if all files were identified correctly
    at_least_one_correct = len(expected_set & actual_set) > 0  # Check if at least one file is identified correctly
    all_incorrect = len(expected_set & actual_set) == 0 and len(
        actual_set) > 0  # Check if all identified files are incorrect

    return {
        'All Files Count': len(all_files),
        'Expected Bug Files Count': len(expected_files),
        'Actual Bug Files Count': len(actual_files),
        '% of Bug Files': len(actual_files) / len(all_files) if len(all_files) > 0 else 0,
        'Precision': precision_value,
        'Recall': recall_value,
        'F1 Score': f1_value,
        'FPR': fpr_value,
        'All Correct': int(all_correct),
        'At Least One Correct': int(at_least_one_correct),
        'All Incorrect': int(all_incorrect),
    }


if __name__ == '__main__':
    all_files = ['file1.py', 'file2.py', 'file3.py', 'file4.py', 'file5.py']
    expected_bug_files = ['file2.py', 'file4.py']
    actual_bug_files = ['file2.py']

    metrics = get_quality_metrics(all_files, expected_bug_files, actual_bug_files)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f'{metric}: {value:.4f}')
        else:
            print(f'{metric}: {value}')
