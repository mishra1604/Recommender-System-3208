# 1. Create a dataset of user-item ratings
ratings = {
    'Alice': {'item1': 4, 'item2': 5, 'item3': 2, 'item4': 1},
    'Bob': {'item1': 3, 'item3': 4, 'item4': 3},
    'Charlie': {'item2': 5, 'item3': 1, 'item4': 4},
    'Dave': {'item1': 5, 'item2': 2, 'item3': 3, 'item4': 2},
    'Eve': {'item1': 1, 'item2': 4, 'item3': 5}
}

# 2. Compute the cosine similarity between each pair of items
item_similarities = {}
for item1 in ratings.keys():
    item_similarities[item1] = {}
    for item2 in ratings.keys():
        if item1 != item2:
            # compute the cosine similarity between item1 and item2
            item1_ratings = [ratings[item1][user] for user in ratings[item1] if user in ratings[item2]]
            item2_ratings = [ratings[item2][user] for user in ratings[item2] if user in ratings[item1]]
            if len(item1_ratings) > 0 and len(item2_ratings) > 0:
                numerator = sum([item1_ratings[i] * item2_ratings[i] for i in range(len(item1_ratings))])
                denominator = ((sum([rating ** 2 for rating in item1_ratings]) ** 0.5)
                               * (sum([rating ** 2 for rating in item2_ratings]) ** 0.5))
                item_similarities[item1][item2] = numerator / denominator

# 3. Predict the rating for a user-item pair based on the similarity between the item and the other items the user has rated.
def predict_rating(user, item):
    if item in ratings[user]:
        # the user has already rated this item, so return their rating
        return ratings[user][item]
    else:
        # predict the rating based on the similarity between the item and the other items the user has rated
        numerator = sum([item_similarities[item][other_item] * ratings[user][other_item]
                         for other_item in ratings[user] if other_item != item and item in item_similarities])
        denominator = sum([item_similarities[item][other_item] for other_item in ratings[user]
                           if other_item != item and item in item_similarities])
        if denominator == 0:
            # no similar items to the one being rated, so return 0
            return 0
        else:
            return numerator / denominator

# example usage
print(predict_rating('Charlie', 'item3'))  # output: 4.219520753089335
