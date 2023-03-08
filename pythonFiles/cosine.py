import numpy as np
import math
import sqlite3
import logging
import time
import concurrent.futures

# find the cosine similarity between users in the sqlite3 database
conn = sqlite3.connect( 'comp3208_train.db' )
    
# create a dictionary for average rating of each user
usr_avg = {}
def user_avg_rating():
    for x in range(1, 944):
        c = conn.cursor()
        c.execute( 'SELECT AVG(Rating) FROM example_table WHERE UserID = ?', (x,) )
        avg = c.fetchone()[0]
        usr_avg[x] = avg

user_avg_rating()
    
# find the adjusted cosine similarity between items in the sqlite3 database
def cosine_similarity_items(item1, item2):
    conn = sqlite3.connect( 'comp3208_train.db' )
    c = conn.cursor()
    c.execute( 'SELECT UserID, Rating FROM example_table WHERE ItemID = ?', (item1,) )
    item1 = c.fetchall()
    item1_ratings = [i[1] for i in item1]
    item1_user = [i[0] for i in item1]

    c.execute( 'SELECT UserID, Rating FROM example_table WHERE ItemID = ?', (item2,) )
    item2 = c.fetchall()
    item2_ratings = [i[1] for i in item2]
    item2_user = [i[0] for i in item2]

    item1Rating = []
    item2Rating = []
    userList = []
    largerList = max(len(item1_ratings), len(item2_ratings))
    for i in range( largerList ):
        if len(item1_ratings) > len(item2_ratings):
            if item1_user[i] in item2_user:
                item1Rating.append(item1_ratings[i])
                item2Rating.append(item2_ratings[item2_user.index(item1_user[i])])
                userList.append(item1_user[i])
        else:
            if item2_user[i] in item1_user:
                item1Rating.append(item1_ratings[item1_user.index(item2_user[i])])
                item2Rating.append(item2_ratings[i])
                userList.append(item2_user[i])
    
    numerator = sum([(item1Rating[i] - usr_avg[userList[i]]) * (item2Rating[i] - usr_avg[userList[i]]) for i in range(len(userList))])
    denominator = math.sqrt(sum([(item1Rating[i] - usr_avg[userList[i]])**2 for i in range(len(userList))])) * math.sqrt(sum([(item2Rating[i] - usr_avg[userList[i]])**2 for i in range(len(userList))]))
    if denominator == 0:
        return 0
    else:
        value = "{:.4f}".format(numerator / denominator)
        return value

# k nearest neighbours to an item
def k_nearest_neighbours(user, item, k):
    c = conn.cursor()
    c.execute( 'SELECT ItemID FROM example_table WHERE ItemID != ? AND UserID = ?', (item, user) )
    duplicate_items = c.fetchall()
    items = list(set(duplicate_items))
    items = [item[0] for item in items]
    similarities = [cosine_similarity_items(item, item2, user) for item2 in items]
    similarities = np.array(similarities)
    items = np.array(items)
    sorted_indices = np.argsort(similarities)
    return items[sorted_indices[-k:]]

# predict user rating for an item using k nearest neighbours
def predict_rating(userID, itemID, k=5):
    c = conn.cursor()
    neighbours = k_nearest_neighbours(userID, itemID, k)
    userRatingForNeighbour = []
    # print(neighbours)
    for neighbour in neighbours:
        c.execute("SELECT Rating FROM example_table WHERE UserID = ? AND ItemID = ?", (userID, int(neighbour)))
        rating = c.fetchone()
        if rating is not None:
            userRatingForNeighbour.append(rating[0])
        else:
            userRatingForNeighbour.append(0)

    Item_similarity_with_neighbour = [cosine_similarity_items(itemID, int(neighbour), userID) for neighbour in neighbours]
    predict_rating_item = sum([Item_similarity_with_neighbour[i] * userRatingForNeighbour[i] for i in range(len(neighbours))]) / sum(Item_similarity_with_neighbour)
    return predict_rating_item


# calculate the MAE of the predictions
def basic_mae():
    c = conn.cursor()
    c.execute( 'SELECT UserID, ItemID FROM example_table WHERE UserID = ?', (1, ))
    user_items = c.fetchall()
    predictedRatings = []

    c.execute( 'SELECT Rating FROM example_table WHERE UserID = ?', (1, ))
    ratings = c.fetchall()

    actualRating = [rating[0] for rating in ratings]
    print(actualRating)

    for user_item in user_items:
        predictedRatings.append(predict_rating(user_item[0], user_item[1]))

    mae = sum([abs(predictedRatings[i] - actualRating[i]) for i in range(len(predictedRatings))]) / len(actualRating)

    print("MAE for userID 1 = ", mae)

# calculating similarity matrix for items
# def similarity_matrix_items(items):
#     c = conn.cursor()
#     c.execute( 'SELECT ItemID FROM example_table' )
#     duplicate_items = c.fetchall()
#     items = list(set(duplicate_items))
#     items = [item[0] for item in items]
#     similarity_matrix = np.zeros((len(items), len(items)))
#     for i in range(0, len(items)):
#         print(i)
#         for j in range(i, len(items)):
#             if i == j:
#                 similarity_matrix[i][j] = 1
#             elif similarity_matrix[i][j] == 0:
#                 similarity_matrix[i][j] = cosine_similarity_items(items[i], items[j])
#                 similarity_matrix[j][i] = similarity_matrix[i][j]
#             else:
#                 continue
#     return similarity_matrix

import multiprocessing as mp

matrix = mp.Array('d', [0.0] * (100 * 100))

def compute_similarity(items, start_index, end_index, matrix):
    for i in range(start_index, end_index):
        print(i)
        row = [1.0 if i == j else (cosine_similarity_items((items[i]), (items[j]))) for j in range(len(items))]
        for j in range(len(items)):
            matrix[i * len(items) + j] = float(row[j])
        return matrix         

def similarity_matrix_items_parallel():
    c = conn.cursor()
    c.execute( 'SELECT ItemID FROM example_table' )
    duplicate_items = c.fetchall()
    items = list(set(duplicate_items))
    items.sort()
    items = [item[0] for item in items][:100]
    num_processes = mp.cpu_count()
    processes = []
    chunk_size = len(items) // num_processes

    print("Number of processes: ", num_processes)
    
    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = (i+1) * chunk_size if i < num_processes-1 else len(items)
        p = mp.Process(target=compute_similarity, args=(items, start_index, end_index, matrix))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()

if __name__ == '__main__':
    m = similarity_matrix_items_parallel()
    print(np.array(matrix))
    matrix3 = np.array(matrix).reshape(100, 100)
    with open('shared_matrix.txt', 'a') as f:
        for row in matrix3:
            f.write(','.join([str(x) for x in row]) + '\n')