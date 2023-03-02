import numpy as np
import math
import sqlite3
import logging

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
    


# item based collaborative filtering for a user-item pair
def predict_rating(user, item):
    c = conn.cursor()
    c.execute( 'SELECT ItemID, Rating FROM example_table')
    all_items = c.fetchall()

    c.execute( 'SELECT ItemID, Rating FROM example_table WHERE UserID = ?', (item,) )
    item = c.fetchone()


    # finding similar items based on k nearest neighbours
    k = 5
    item_similarities = []

    for i in all_items:
        item_similarities.append(cosine_similarity_items(item[0], i[0]))

    item_similarities.sort(key = lambda x: x[1], reverse = True)
    item_similarities = item_similarities[:k]

    print(item_similarities)


print(predict_rating(1, 11))


