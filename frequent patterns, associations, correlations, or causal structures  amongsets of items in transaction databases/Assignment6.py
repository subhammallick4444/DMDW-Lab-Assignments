from collections import defaultdict
transactions = [
    ['M', 'O', 'N', 'K', 'E', 'Y'],
    ['D', 'O', 'N', 'K', 'E', 'Y'],
    ['M', 'A', 'K', 'E'],
    ['M', 'U', 'C', 'K', 'Y'],
    ['C', 'O', 'O', 'K', 'I', 'E']
]
#finding frequent items
min_support = 3

item_counts = defaultdict(int)
for transaction in transactions:
    for item in transaction:
        item_counts[item] += 1

frequent_1_itemsets = {item: count for item, count in item_counts.items() if count >= min_support}

frequent_itemsets = [list(frequent_1_itemsets.keys())]

k = 2  # Start with 2-itemsets
while True:
    candidate_itemsets = []
    for i in range(len(frequent_itemsets[-1])):
        for j in range(i + 1, len(frequent_itemsets[-1])):
            itemset1 = frequent_itemsets[-1][i]
            itemset2 = frequent_itemsets[-1][j]
            candidate = sorted(list(set(itemset1) | set(itemset2)))
            if len(candidate) == k and candidate not in candidate_itemsets:
                candidate_itemsets.append(candidate)


    candidate_counts = defaultdict(int)
    for transaction in transactions:
        for candidate in candidate_itemsets:
            if set(candidate).issubset(set(transaction)):
                candidate_counts[tuple(candidate)] += 1


    frequent_k_itemsets = [list(candidate) for candidate, count in candidate_counts.items() if count >= min_support]

    if not frequent_k_itemsets:
        break

    frequent_itemsets.append(frequent_k_itemsets)
    k += 1
frequent_itemsets=frequent_itemsets[len(frequent_itemsets)-1]
print(frequent_itemsets)
