import numpy as np
import math
import sqlite3
import logging
import time
import concurrent.futures
import codecs
import csv
import random

def file_loading():
    with open('train_100k_withratings_new.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    # shuffle data
    random.shuffle(data)
    trainingsize = int(len(data) * 0.8)
    training = data[:trainingsize]
    testing = data[trainingsize:]
    print(len(training), len(testing))

    with open('shuffled_train_ratings.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(training)

    with open('shuffled_test_ratings.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(testing)

def create_training_db():
    conn = sqlite3.connect( 'training_split.db')
    readHandle = codecs.open( 'shuffled_train_ratings.csv', 'r', 'utf-8', errors = 'replace' )
    listLines = readHandle.readlines()
    readHandle.close()

    c = conn.cursor()
    c.execute( 'CREATE TABLE IF NOT EXISTS example_table (UserID INT, ItemID INT, Rating FLOAT, PredRating FLOAT)' )
    conn.commit()

    c.execute( 'DELETE FROM example_table' )
    conn.commit()

    for strLine in listLines :
        if len(strLine.strip()) > 0 :
            # userid, itemid, rating, timestamp
            listParts = strLine.strip().split(',')
            if len(listParts) == 4 :
                # insert training set into table with a completely random predicted rating
                c.execute( 'INSERT INTO example_table VALUES (?,?,?,?)', (listParts[0], listParts[1], listParts[2], random.random() * 5.0) )
            else :
                raise Exception( 'failed to parse csv : ' + repr(listParts) )
    conn.commit()

    c.execute( 'CREATE INDEX IF NOT EXISTS example_table_index on example_table (UserID, ItemID)' )
    conn.commit()

conn = sqlite3.connect( 'training_split.db' )

# create a dictionary for average rating of each user
usr_avg = {}
def user_avg_rating():
    for x in range(1, 944):
        c = conn.cursor()
        c.execute( 'SELECT AVG(Rating) FROM example_table WHERE UserID = ?', (x,) )
        avg = c.fetchone()[0]
        usr_avg[x] = avg
user_avg_rating()

# list of items
c = conn.cursor()
c.execute( 'SELECT ItemID FROM example_table' )
duplicate_items = c.fetchall()
items = list(set(duplicate_items))
items.sort()
itemsList = [item[0] for item in items]
print(len(itemsList))

# dictionary: itemID: [list of UserID, Rating]
item_dict = {}
def item_dict_query():
    for x in itemsList:
        c = conn.cursor()
        c.execute( 'SELECT UserID, Rating FROM example_table WHERE ItemID = ?', (x,) )
        item_dict[x] = c.fetchall()

user_sim_dict = {}
def user_sim_dict_query():
    for x in range(1, 944):
        c = conn.cursor()
        c.execute( 'SELECT Rating FROM example_table WHERE UserID = ?', (x,) )
        user_sim_dict[x] = c.fetchall()

# start = time.time()        
# item_dict_query()
# end = time.time()

# print(len(item_dict), "time taken:", end - start)

def cosine_similarity_items(item1, item2):
    item1 = item_dict[item1]
    item1_ratings = [i[1] for i in item1]
    item1_user = [i[0] for i in item1]

    item2 = item_dict[item2]
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
    
def cosine_similarity_users(user1, user2):
    user1 = user_sim_dict[user1]
    user1_ratings = [i[0] for i in user1]

    user2 = user_sim_dict[user2]
    user2_ratings = [i[0] for i in user2]

    numerator = sum([user1_ratings[i] * user2_ratings[i] for i in range(min(len(user1_ratings), len(user2_ratings)))])
    denominator = math.sqrt(sum([user1_ratings[i]**2 for i in range(len(user1_ratings))])) * math.sqrt(sum([user2_ratings[i]**2 for i in range(len(user2_ratings))]))
    if denominator == 0:
        return 0
    else:
        value = "{:.4f}".format(numerator / denominator)
        return value


itemsList = itemsList
matrix = np.full((len(itemsList), len(itemsList)), 0.)

def compute_matrix():
    for i in range(len(itemsList)):
        print(i)
        for j in range(i, len(itemsList)):
            if i==j:
                matrix[i][j] = 1.0
            else:
                matrix[i][j] = cosine_similarity_items(itemsList[i], itemsList[j])
                matrix[j][i] = matrix[i][j]

loadMatrix = np.loadtxt('simpleMatrix.csv', delimiter=',')

# user similarity dictionary

c = conn.cursor()
c.execute( 'SELECT UserID FROM example_table' )
duplicate_users = c.fetchall()
users = list(set(duplicate_users))
users.sort()
userList = [user[0] for user in users]


# user similarity matrix
user_sim_matrix = np.full((len(userList), len(userList)), 0.)

def compute_user_sim_matrix():
    for i in range(len(userList)):
        print(i)
        for j in range(i, len(userList)):
            if i==j:
                user_sim_matrix[i][j] = 1.0
            else:
                user_sim_matrix[i][j] = cosine_similarity_users(userList[i], userList[j])
                user_sim_matrix[j][i] = user_sim_matrix[i][j]

# start = time.time()
# user_sim_dict_query()
# end = time.time()
# print("time taken:", end - start)

# start = time.time()
# compute_user_sim_matrix()
# end = time.time()
# print("time taken:", end - start)
# # save matrix to csv
# np.savetxt("user_matrix.csv", user_sim_matrix, fmt='%.4f', delimiter=",")
