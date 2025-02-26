import json

# from rte import replace_last_element


def calculate_average_spice_scores(gt_file, pred_file):
    import json

    # Load the ground truth and predictions from JSON files
    with open(gt_file, 'r') as file:
        gt = json.load(file)
    with open(pred_file, 'r') as file2:
        pred = json.load(file2)

    # Initialize counters for the metrics
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    count = 0

    # Function to calculate SPICE metrics
    def calculate_spice(candidate_tuples, reference_tuples):
        candidate_set = set(
            tuple(x) if isinstance(x, list) else x for x in candidate_tuples)
        reference_set = set(
            tuple(x) if isinstance(x, list) else x for x in reference_tuples)
        true_positives = candidate_set.intersection(reference_set)
        precision = len(true_positives) / len(
            candidate_set) if candidate_set else 0
        recall = len(true_positives) / len(
            reference_set) if reference_set else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1_score

    # Iterate through each item in the ground truth and prediction
    for gt_item, pred_item in zip(gt, pred):
        # gt_rte = [replace_last_element(i) for i in gt_item['rte']]
        # pred_rte = [replace_last_element(i) for i in pred_item['rte']]
        gt_rte = gt_item['rte']
        pred_rte = pred_item['rte']
        precision, recall, f1_score = calculate_spice(pred_rte, gt_rte)

        # Accumulate the results
        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score
        count += 1

    # Calculate averages
    average_precision = total_precision / count if count else 0
    average_recall = total_recall / count if count else 0
    average_f1_score = total_f1_score / count if count else 0

    return average_precision, average_recall, average_f1_score


# Example usage
if __name__ == '__main__':
    avg_precision, avg_recall, avg_f1 = calculate_average_spice_scores(
        'new_prompt/rte_gt3.json', 'new_prompt/rte_git.json')
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1 Score:", avg_f1)
