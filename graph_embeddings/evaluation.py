import numpy as np


def evaluate_hits_0(test_set, recommendations):
    hits = 0
    total = 0

    for index, row in test_set.iterrows(): # ==> iterate over recom list
        user_id = row['user_id']
        item_id = row['item_id']
        if user_id in recommendations and item_id in recommendations[user_id]:
            hits += 1
        total += 1
    hit_ratio = hits / total

    return hit_ratio


def evaluate_hits(test_set, recommendations):
    hits = 0

    for user_id in recommendations:
        recommended_items = recommendations[user_id]
        actual_items = test_set[test_set['user_id'] == user_id]['item_id'].tolist()

        overlap = set(recommended_items) & set(actual_items)
        if overlap:
            hits += 1

    hit_ratio = hits / len(recommendations)
    return hit_ratio


def precision_recall_at_k(recommendations, test_set, k):
    # Convert test set to a dictionary of user to items
    test_user_items = test_set.groupby('user_id')['item_id'].apply(list).to_dict()

    precisions = []
    recalls = []

    for user, recommended_items in recommendations.items():
        if user not in test_user_items:
            continue

        true_items = set(test_user_items[user])
        rec_items = set(recommended_items[:k])

        # Calculate precision and recall
        n_relevant = len(rec_items.intersection(true_items))
        precision = n_relevant / len(rec_items) if rec_items else 0
        recall = n_relevant / len(true_items) if true_items else 0

        precisions.append(precision)
        recalls.append(recall)

    # Return the average precision and recall across all users
    return np.mean(precisions), np.mean(recalls)


def dcg(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum((2 ** r - 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg(recommendations, test_set, k):
    test_user_items = test_set.groupby('user_id')['item_id'].apply(list).to_dict()
    ndcg_values = []

    for user, recommended_items in recommendations.items():
        if user not in test_user_items:
            continue

        true_items = set(test_user_items[user])
        relevance_scores = [1 if item in true_items else 0 for item in recommended_items[:k]]

        dcg_max = dcg(sorted(relevance_scores, reverse=True), k)  # Ideal DCG
        dcg_real = dcg(relevance_scores, k)

        ndcg = dcg_real / dcg_max if dcg_max > 0 else 0
        ndcg_values.append(ndcg)

    return np.mean(ndcg_values)


