import numpy as np
import math
import sqlite3
import logging
import time

# find the cosine similarity between users in the sqlite3 database
conn = sqlite3.connect( 'comp3208_train.db' )


def cosine_similarity_users(user1, user2 ):
    c = conn.cursor()
    c.execute( 'SELECT Rating FROM example_table WHERE UserID = ?', (user1,) )
    user1_ratings = c.fetchall()
    c.execute( 'SELECT Rating FROM example_table WHERE UserID = ?', (user2,) )
    user2_ratings = c.fetchall()
    if len(user1_ratings) > 0 and len(user2_ratings) > 0:
        numerator = sum([user1_ratings[i][0] * user2_ratings[i][0] for i in range(len(user1_ratings))])
        denominator = ((sum([rating[0] ** 2 for rating in user1_ratings]) ** 0.5)
                       * (sum([rating[0] ** 2 for rating in user2_ratings]) ** 0.5))
        return numerator / denominator
    else:
        return 0
    
# find the cosine similarity between items in the sqlite3 database
def cosine_similarity_items(item1, item2 ):
    c = conn.cursor()
    c.execute( 'SELECT Rating FROM example_table WHERE ItemID = ?', (item1,) )
    item1_ratings = c.fetchall()

    c.execute( 'SELECT Rating FROM example_table WHERE ItemID = ?', (item2,) )
    item2_ratings = c.fetchall()

    smallerNumber = 0
    if len(item1_ratings) > len(item2_ratings):
        smallerNumber = len(item2_ratings)
    else:
        smallerNumber = len(item1_ratings)
    
    if len(item1_ratings) > 0 and len(item2_ratings) > 0:
        numerator = sum([item1_ratings[i][0] * item2_ratings[i][0] for i in range(smallerNumber)])
        denominator = ((sum([rating[0] ** 2 for rating in item1_ratings]) ** 0.5)
                       * (sum([rating[0] ** 2 for rating in item2_ratings]) ** 0.5))
        return numerator / denominator
    else:
        return 0


# k nearest neighbours to an item
def k_nearest_neighbours(user, item, k):
    c = conn.cursor()
    c.execute( 'SELECT ItemID FROM example_table WHERE ItemID != ? AND UserID = ?', (item, user) )
    duplicate_items = c.fetchall()
    items = list(set(duplicate_items))
    items = [item[0] for item in items]
    similarities = [cosine_similarity_items(item, item2) for item2 in items]
    similarities = np.array(similarities)
    items = np.array(items)
    sorted_indices = np.argsort(similarities)
    return items[sorted_indices[-k:]]

# predict user rating for an item using k nearest neighbours
def predict_rating(userID, itemID, k=5):
    c = conn.cursor()
    neighbours = k_nearest_neighbours(userID, itemID, k)
    userRatingForNeighbour = []
    print(neighbours)
    for neighbour in neighbours:
        c.execute("SELECT Rating FROM example_table WHERE UserID = ? AND ItemID = ?", (userID, int(neighbour)))
        rating = c.fetchone()
        if rating is not None:
            userRatingForNeighbour.append(rating[0])
        else:
            userRatingForNeighbour.append(0)

    Item_similarity_with_neighbour = [cosine_similarity_items(itemID, int(neighbour)) for neighbour in neighbours]
    predict_rating_item = sum([Item_similarity_with_neighbour[i] * userRatingForNeighbour[i] for i in range(len(neighbours))]) / sum(Item_similarity_with_neighbour)
    print(predict_rating_item)

predict_rating(1, 439)