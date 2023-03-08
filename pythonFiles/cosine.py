import numpy as np
import math
import sqlite3
import logging
import time
import concurrent.futures
import codecs

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

# opening the matrix csv file into an sqlite db
def create_item_similarityDB():
    conn = sqlite3.connect('item_similarity.db')

    readHandle = codecs.open( 'trial_matrix.csv', 'r', 'utf-8', errors = 'replace' )
    listLines = readHandle.readlines()
    readHandle.close()

    c = conn.cursor()
    c.execute( 'CREATE TABLE IF NOT EXISTS items_table (ItemID_1 INT, ItemID_2 INT, Similarity FLOAT)' )
    conn.commit()

    c.execute( 'DELETE FROM items_table' )
    conn.commit()

    for strLine in listLines :
        if len(strLine.strip()) > 0 :
            # itemid, itemid, Similarity
            listParts = strLine.strip().split(',')
            if len(listParts) == 3 :
                # insert training set into table
                c.execute( 'INSERT INTO items_table VALUES (?,?,?)', (listParts[0], listParts[1], listParts[2]) )
            else :
                raise Exception( 'failed to parse csv : ' + repr(listParts) )
    conn.commit()

    c.execute( 'CREATE INDEX IF NOT EXISTS items_table_index on items_table (ItemID_1, ItemID_2)' )
    conn.commit()

def knn(user, item, k=5):
    c = conn.cursor()
    c.execute( 'SELECT ItemID FROM example_table WHERE UserID = ?', (user,) )
    items = c.fetchall()
    items = [item[0] for item in items]

    second_conn = sqlite3.connect('item_similarity.db')
    d = second_conn.cursor()
    similarity = []
    for i in items:
        d.execute( 'SELECT ItemID_1, ItemID_2, Similarity FROM items_table WHERE ItemID_1 = ? AND ItemID_2 = ?', (i, item) )
        item_tuple = d.fetchall()
        similarity.append(item_tuple[0])
    
    
    # sort the list of tuples by similarity
    sorted_similarity = sorted(similarity, key=lambda x: x[2], reverse=True)
    closestNeighbours = sorted_similarity[:k]
    return closestNeighbours
    

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

import multiprocessing

def compute_similarity(similarity_matrix, items, start_index, end_index):
    with open('trial_matrix.csv', 'a') as f:
        for i in range(start_index, end_index):
            print(i)
            row = [] + [0 for x in range(i)]
            for j in range(i, len(items)):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                    f.write(str(items[i]) + ',' + str(items[j]) + ',' + str(1) + "\n")
                else:
                    similarity_matrix[i][j] = cosine_similarity_items(items[i], items[j])
                    similarity_matrix[j][i] = similarity_matrix[i][j]
                    f.write(str(items[i]) + ',' + str(items[j]) + ',' + str(similarity_matrix[i][j]) + "\n")
                    f.write(str(items[j]) + ',' + str(items[i]) + ',' + str(similarity_matrix[j][i]) + "\n")
                row.append(similarity_matrix[i][j])
            # f.write(','.join([str(x) for x in row]) + "\n")

def similarity_matrix_items_parallel():
    c = conn.cursor()
    c.execute( 'SELECT ItemID FROM example_table' )
    duplicate_items = c.fetchall()
    items = list(set(duplicate_items))
    items.sort()
    items = [item[0] for item in items][:100]
    similarity_matrix = np.full((len(items), len(items)), 0.)
    num_processes = multiprocessing.cpu_count()
    processes = []
    chunk_size = len(items) // num_processes

    print("Number of processes: ", num_processes)
    
    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = (i+1) * chunk_size if i < num_processes-1 else len(items)
        p = multiprocessing.Process(target=compute_similarity, args=(similarity_matrix,items, start_index, end_index))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
    
    return similarity_matrix

if __name__ == '__main__':
    # matrix = similarity_matrix_items_parallel()
    print(knn(1, 93))