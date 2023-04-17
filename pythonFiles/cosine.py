import numpy as np
import math
import sqlite3
import logging
import time
import concurrent.futures
import codecs
import csv

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
    i = 0
    for strLine in listLines :
        if len(strLine.strip()) > 0 :
            # itemid, itemid, Similarity
            listParts = strLine.strip().split(',')
            if len(listParts) == 3 :
                # insert training set into table
                c.execute( 'INSERT INTO items_table VALUES (?,?,?)', (listParts[0], listParts[1], listParts[2]) )
            else :
                raise Exception( 'failed to parse csv : ' + repr(listParts) )
        i += 1
    conn.commit()

    c.execute( 'CREATE INDEX IF NOT EXISTS items_table_index on items_table (ItemID_1, ItemID_2)' )
    conn.commit()

def knn(user, item, k=10):
    c = conn.cursor()
    c.execute( 'SELECT ItemID FROM example_table WHERE UserID = ?', (user,) )
    items = c.fetchall()
    items = [item[0] for item in items]

    second_conn = sqlite3.connect('item_similarity.db')
    d = second_conn.cursor()
    similarity = []
    nullNeighbours = []
    for i in items:
        d.execute( 'SELECT ItemID_1, ItemID_2, Similarity FROM items_table WHERE ItemID_1 = ? AND ItemID_2 = ?', (i, item) )
        item_tuple = d.fetchall()
        try:
            similarity.append(item_tuple[0])
        except:
            nullNeighbours.append((i, item))
            with open('null_values.txt', 'a') as f:
                f.write(str((i, item)) + "\n")
    
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
def predict_rating(userID, itemID, k=3):
    try:
        c = conn.cursor()
        neighboursTuples = knn(userID, itemID)
        neighbours = [item[0] for item in neighboursTuples]
        userRatingForNeighbour = []
        # print(neighbours)
        for neighbour in neighbours:
            c.execute("SELECT Rating FROM example_table WHERE UserID = ? AND ItemID = ?", (userID, int(neighbour)))
            rating = c.fetchone()
            if rating is not None:
                userRatingForNeighbour.append(rating[0])
            else:
                userRatingForNeighbour.append(0)

        Item_similarity_with_neighbour = [similarity[2] for similarity in neighboursTuples]
        predict_rating_item = sum([Item_similarity_with_neighbour[i] * userRatingForNeighbour[i] for i in range(len(neighbours))]) / sum(Item_similarity_with_neighbour)
        finalRating = float("{:.3f}".format(predict_rating_item))
        return finalRating
    except:
        return usr_avg[userID]


# calculate the MAE of the predictions
def basic_mae():
    c = conn.cursor()
    c.execute( 'SELECT UserID, ItemID, Rating FROM example_table')
    user_items = c.fetchall()
    user_items = user_items[:1000]
    predictedRatings = []

    ratings = []
    for user_item_tuple in user_items:
        #c.execute( 'SELECT Rating FROM example_table WHERE UserID = ? and ItemID = ?', (user_item_tuple[0], user_item_tuple[1]))
        #rate = c.fetchall()
        ratings.append([user_item_tuple[2]])

    actualRating = [rating[0] for rating in ratings]

    for user_item_Tuple in user_items:
        predictedRatings.append(predict_rating(user_item_Tuple[0], user_item_Tuple[1]))

    mae = sum([abs(predictedRatings[i] - actualRating[i]) for i in range(len(predictedRatings))]) / len(actualRating)

    return "{:.2f}".format(mae)

def create_test_db():
    conn = sqlite3.connect('test_ratings.db')

    readHandle = codecs.open( 'test_100k_withoutratings_new.csv', 'r', 'utf-8', errors = 'replace' )
    listLines = readHandle.readlines()
    readHandle.close()

    c = conn.cursor()
    c.execute( 'CREATE TABLE IF NOT EXISTS test_table (UserID INT, ItemID INT, Pred_Rating FLOAT)' )
    conn.commit()

    c.execute( 'DELETE FROM test_table' ) 
    conn.commit()

    for strLine in listLines :
        if len(strLine.strip()) > 0 :
            # userid, itemid, rating
            listParts = strLine.strip().split(',')
            if len(listParts) == 3 :
                # insert training set into table
                c.execute( 'INSERT INTO test_table VALUES (?,?,?)', (listParts[0], listParts[1], 0.0) )
            else :
                raise Exception( 'failed to parse csv : ' + repr(listParts) )
    conn.commit()

    c.execute( 'CREATE INDEX IF NOT EXISTS test_table_index on test_table (UserID, ItemID)' )
    conn.commit()

    c.close()
    conn.close()

def second_test_db():
    conn = sqlite3.connect('testing_set.db')

    readHandle = codecs.open( 'test_100k_withoutratings_new.csv', 'r', 'utf-8', errors = 'replace' )
    listLines = readHandle.readlines()
    readHandle.close()

    c = conn.cursor()
    c.execute( 'CREATE TABLE IF NOT EXISTS test_table (UserID INT, ItemID INT, Pred_Rating FLOAT, Timestamp FLOAT)' )
    conn.commit()

    c.execute( 'DELETE FROM test_table' )
    conn.commit()

    for strLine in listLines :
        if len(strLine.strip()) > 0 :
            # userid, itemid, rating
            listParts = strLine.strip().split(',')
            if len(listParts) == 3 :
                # insert training set into table
                c.execute( 'INSERT INTO test_table VALUES (?,?,?,?)', (listParts[0], listParts[1], 0.0, listParts[2]) )
            else :
                raise Exception( 'failed to parse csv : ' + repr(listParts) )
    conn.commit()

    c.execute( 'CREATE INDEX IF NOT EXISTS test_table_index on test_table (UserID, ItemID)' )
    conn.commit()

    c.close()
    conn.close()


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
    items = [item[0] for item in items]
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

def test_rating_predictions():
    conn = sqlite3.connect('testing_set.db')
    cursor = conn.cursor()

    # retrieve UserID, ItemID from the databse
    cursor.execute( 'SELECT UserID, ItemID FROM test_table' )
    user_items = cursor.fetchall()

    # for each user-item pair, predict the rating and update the database
    for user_item_tuple in user_items:
        rating_value = predict_rating(user_item_tuple[0], user_item_tuple[1])
        cursor.execute( 'UPDATE test_table SET Pred_Rating = ? WHERE UserID = ? AND ItemID = ?', (rating_value, user_item_tuple[0], user_item_tuple[1]) )
        conn.commit()

def export_test_db():
    conn = sqlite3.connect('testing_set.db')
    cursor = conn.cursor()
    cursor.execute("select * from test_table;")
    with open("out2.csv", 'w',newline='') as csv_file: 
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([i[0] for i in cursor.description]) 
        csv_writer.writerows(cursor)
    conn.close()

import time
if __name__ == '__main__':
    # matrix = similarity_matrix_items_parallel()
    start = time.time()
    print(cosine_similarity_items(1648, 1))
    end = time.time()
    print("Time taken: ", end - start)
