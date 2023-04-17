import numpy as np
import math
import sqlite3
import logging
import time
import concurrent.futures
import codecs
import csv
import multiprocessing

# find the cosine similarity between users in the sqlite3 database
conn = sqlite3.connect( 'comp3208_train.db' )
    
# create a dictionary for average rating of each user
#  by querying the training database 
usr_avg = {}
def user_avg_rating():
    for x in range(1, 944):
        c = conn.cursor()
        c.execute( 'SELECT AVG(Rating) FROM example_table WHERE UserID = ?', (x,) )
        avg = c.fetchone()[0]
        usr_avg[x] = avg
user_avg_rating()


# creating a dictionary of items and their ratings given by users
# item_dict = {itemID: [(userID, rating), (userID, rating), ...], ...}
# done for reducing query time, dictionary access time is faster
# helps in code optimization
item_dict = {}
def item_dict_query():
    c = conn.cursor()
    c.execute( 'SELECT ItemID FROM example_table' )
    duplicate_items = c.fetchall()
    items = list(set(duplicate_items))
    items.sort()
    itemsList = [item[0] for item in items] #list of items from the training database

    for x in itemsList:
        c = conn.cursor()
        c.execute( 'SELECT UserID, Rating FROM example_table WHERE ItemID = ?', (x,) )
        item_dict[x] = c.fetchall()
item_dict_query()


# Calculating the adjusted cosine similarity between two items
# Taking into account the average rating of each user, accessed from the usr_avg dictionary
# Done by subtracting the average rating of each user from the rating of the item
# This adjustment accounts for the fact that some users rate items more positlvely or negatively than others
def cosine_similarity_items(item1, item2):
    item1 = item_dict[item1] # list of ratings for item1: [(userID, rating)...]
    item1_ratings = [i[1] for i in item1] # list of ratings for item1: [rating, rating, ...]
    item1_user = [i[0] for i in item1] # list of users who rated item1: [userID, userID, ...]

    item2 = item_dict[item2] # Similar to item1
    item2_ratings = [i[1] for i in item2]
    item2_user = [i[0] for i in item2]

    # finding the common users who rated both items
    # if a user is present in item1_user but not in item2_user, then the user is not considered
    item1Rating = []
    item2Rating = []
    userList = [] # list of common users who rated both items
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

    # calculating the adjusted cosine similarity and taking into account the average rating of each user
    numerator = sum([(item1Rating[i] - usr_avg[userList[i]]) * (item2Rating[i] - usr_avg[userList[i]]) for i in range(len(userList))])
    denominator = math.sqrt(sum([(item1Rating[i] - usr_avg[userList[i]])**2 for i in range(len(userList))])) * math.sqrt(sum([(item2Rating[i] - usr_avg[userList[i]])**2 for i in range(len(userList))]))
    if denominator == 0:
        return 0
    else:
        # rounded to 4 decimal places
        value = "{:.4f}".format(numerator / denominator)
        return value
    

# Using multiprocessing to speed up the process of calculating the similarity matrix
# The similarity matrix is a square matrix of size (number of items) x (number of items)
# The matrix is symmetric, so only the upper triangular part of the matrix is calculated
# The lower triangular part is calculated by taking the transpose of the upper triangular part
# The matrix is stored in a csv file of the format: item1, item2, similarity

# The function computes a part of the similarity matrix for example
# if the number of items is 100, then the function computes the similarity between items 0-9, 10-19, 20-29, ..., 90-99
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

# This function uses multiprocessing to speed up the process of calculating the similarity matrix
# Divides the calculation of the similarity matrix into chunks and assigns each chunk to a process
# The number of processes is equal to the number of cores in the CPU
# My CPU has 16 cores, so the number of processes is 16
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


# converting item1, item2, similarity from item_item_similarity.csv to a sqlite3 database
def create_item_similarityDB():
    conn = sqlite3.connect('item_similarity.db')

    readHandle = codecs.open( 'item_item_similarity.csv', 'r', 'utf-8', errors = 'replace' )
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


# For a given user and item, find the k nearest neighbour to the item
# k is the number of nearest neighbours to be considered
# returns the average rating of the k nearest neighbours
# if no neighbours are found, then the average rating of the user is returned

# The algorithm is Greedy
# it considers all the items whose similarity with the given item is greater than 0.55
# or at minimum, it considers the k nearest neighbours
def knn(user, item, k=5):
    c = conn.cursor()
    c.execute( 'SELECT ItemID FROM example_table WHERE UserID = ?', (user,) )
    items = c.fetchall()
    items = [item[0] for item in items]
    if len(items) == 0: return 3 # if the user has not rated any items, return the average rating of all items

    second_conn = sqlite3.connect('item_similarity.db')
    d = second_conn.cursor()
    similarity = [] # list of tuples (item1, item2, similarity) obtained from the database
    for i in items:
        d.execute( 'SELECT ItemID_1, ItemID_2, Similarity FROM items_table WHERE ItemID_1 = ? AND ItemID_2 = ?', (i, item) )
        item_tuple = d.fetchall()
        try:
            if item_tuple[0][2] >= 0:
                similarity.append(item_tuple[0])
        except:
            similarity.append(":.0f".format(usr_avg[user]))
       
    # sort the list of tuples by similarity
    sorted_similarity = sorted(similarity, key=lambda x: x[2], reverse=True)
    closestNeighbours = sorted_similarity
    for i in range(len(sorted_similarity)-1, 0, -1):
        if sorted_similarity[i][2] < 0.55 and len(closestNeighbours) > k:
            closestNeighbours.pop()
    return closestNeighbours


# @param userID: the user for which the rating is to be predicted
# Prediction function for a given user and item
# returns the predicted rating for the given user and item
# based on the k nearest neighbours
# if no neighbours are found, then the average rating of the user is returned
# only in cases where the item in the testing set is not in the training set
# Note: no need to compensate for user bias in this stage, since no comparison between users is being made
def predict_rating(userID, itemID):
    try:
        c = conn.cursor()
        neighboursTuples = knn(userID, itemID) # remember to delete this
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
        finalRating = float("{:.0f}".format(predict_rating_item))
        return finalRating
    except:
        return "{:.0f}".format(usr_avg[userID])
    

# creating the provided test set (test_100k_withoutratings_new.csv) into a sqlite3 database
# code reference: provided by ECS, Social Computing Coursework 1
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


# This is a temporary function created for converting the 10% of the training set into a sqlite3 database
# Used for internal testing purposes for calculating the MAE of the predictions
# code reference: provided by ECS, Social Computing Coursework 1
def second_test_db():
    conn = sqlite3.connect('dev_set.db')
    readHandle = codecs.open( 'shuffled_test_ratings.csv', 'r', 'utf-8', errors = 'replace' )
    listLines = readHandle.readlines()
    readHandle.close()
    c = conn.cursor()
    c.execute( 'CREATE TABLE IF NOT EXISTS test_table (UserID INT, ItemID INT, Actual_Rating INT, Pred_Rating INT, Timestamp INT)' )
    conn.commit()
    c.execute( 'DELETE FROM test_table' )
    conn.commit()
    for strLine in listLines[:20000] :
        if len(strLine.strip()) > 0 :
            # userid, itemid, rating
            listParts = strLine.strip().split(',')
            if len(listParts) == 4 :
                # insert training set into table
                c.execute( 'INSERT INTO test_table VALUES (?,?,?,?,?)', (listParts[0], listParts[1], listParts[2], 0, listParts[3]) )
            else :
                raise Exception( 'failed to parse csv : ' + repr(listParts) )
    conn.commit()
    c.execute( 'CREATE INDEX IF NOT EXISTS test_table_index on test_table (UserID, ItemID)' )
    conn.commit()
    c.close()
    conn.close()


# computing the prediction for a user-item pair using the item-item similarity database
# retrieving all the users and items from the test database
# for each user-item pair, predict the rating and update the database
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

# exporting the test database to a csv file for submission
def export_test_db():
    conn = sqlite3.connect('testing_set.db')
    cursor = conn.cursor()
    cursor.execute("select * from test_table;")
    with open("results.csv", 'w',newline='') as csv_file: 
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([i[0] for i in cursor.description]) 
        csv_writer.writerows(cursor)
    conn.close()

# computing the mean absolute error for the dev set
# dev set is obtained by splitting the training set into 90:10 ratio
# DEV set has the columns: UserID, ItemID, Actual_Rating, Pred_Rating
# MAE is calculated using the difference between Actual_Rating and Pred_Rating divided by the number of rows
def mae_dev_set():
    conn = sqlite3.connect('dev_set.db')
    cursor = conn.cursor()
    cursor.execute("SELECT Actual_Rating FROM test_table")
    actual_ratings = cursor.fetchall()
    actual_ratings = [rating[0] for rating in actual_ratings]

    cursor.execute("SELECT Pred_Rating FROM test_table")
    pred_ratings = cursor.fetchall()
    pred_ratings = [rating[0] for rating in pred_ratings]

    mae  = sum(abs(np.array(actual_ratings) - np.array(pred_ratings))) / len(actual_ratings)

    return mae

import time
if __name__ == '__main__':
    start = time.time()
    # test_rating_predictions()
    # print("MAE: ", mae_dev_set())
    export_test_db()
