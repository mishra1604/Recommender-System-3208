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

# list of items
c = conn.cursor()
c.execute( 'SELECT ItemID FROM example_table' )
duplicate_items = c.fetchall()
items = list(set(duplicate_items))
items.sort()
itemsList = [item[0] for item in items] #-----------------------> list of items

# average item rating
avg_item = {}
def avg_item_rating(item):
    for x in itemsList:
        c = conn.cursor()
        c.execute( 'SELECT AVG(Rating) FROM example_table WHERE ItemID = ?', (x,) )
        avg = c.fetchone()[0]
        avg_item[x] = avg

# dictionary: itemID: [list of UserID, Rating]
item_dict = {}
def item_dict_query():
    for x in itemsList:
        c = conn.cursor()
        c.execute( 'SELECT UserID, Rating FROM example_table WHERE ItemID = ?', (x,) )
        item_dict[x] = c.fetchall()
# item_dict_query()

user_sim_dict = {} # --------------------------------------------> dictionary: userID: [list of ItemID, Rating]
def user_sim_dict_query():
    for x in range(1, 944):
        c = conn.cursor()
        c.execute( 'SELECT ItemID, Rating FROM example_table WHERE UserID = ?', (x,) )
        user_sim_dict[x] = c.fetchall()
user_sim_dict_query()

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

# itemsList = itemsList
matrix = np.full((len(itemsList), len(itemsList)), 0.)
def compute_matrix():
    for i in range(len(itemsList)):
        if i % 100 == 0: print(i)
        for j in range(i, len(itemsList)):
            if i==j:
                matrix[i][j] = 1.0
            else:
                matrix[i][j] = cosine_similarity_items(itemsList[i], itemsList[j])
                matrix[j][i] = matrix[i][j]

item_similarity_matrix = np.loadtxt('simpleMatrix.csv', delimiter=',')

def pearson_similarity_users(user1ID, user2ID):
    user1 = user_sim_dict[user1ID]
    user1_items = [i[0] for i in user1]
    user1_ratings = [i[1] for i in user1]

    user2 = user_sim_dict[user2ID]
    user2_items = [i[0] for i in user2]
    user2_ratings = [i[1] for i in user2]

    user1Rating = []
    user2Rating = []
    commonItemList = []
    largerList = max(len(user1_items), len(user2_items))
    for i in range( largerList ):
        if len(user1_items) > len(user2_items):
            if user1_items[i] in user2_items:
                user1Rating.append(user1_ratings[i])
                user2Rating.append(user2_ratings[user2_items.index(user1_items[i])])
                commonItemList.append(user1_items[i])
        else:
            if user2_items[i] in user1_items:
                user1Rating.append(user1_ratings[user1_items.index(user2_items[i])])
                user2Rating.append(user2_ratings[i])
                commonItemList.append(user2_items[i])
    
    # pearson's coefficient
    numerator = sum([(user1Rating[i] - usr_avg[user1ID]) * (user2Rating[i] - usr_avg[user2ID]) for i in range(len(commonItemList))])
    denominator = math.sqrt(sum([(user1Rating[i] - usr_avg[user1ID])**2 for i in range(len(commonItemList))])) * math.sqrt(sum([(user2Rating[i] - usr_avg[user2ID])**2 for i in range(len(commonItemList))]))
    if denominator == 0:
        return 0
    else:
        value = "{:.4f}".format(numerator / denominator)
        return value

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
        if i%100 == 0: print(i)
        for j in range(i, len(userList)):
            if i==j:
                user_sim_matrix[i][j] = 1.0
            else:
                user_sim_matrix[i][j] = pearson_similarity_users(userList[i], userList[j])
                user_sim_matrix[j][i] = user_sim_matrix[i][j]

# neighborhood selection

user_similarity_matrix = np.loadtxt('user_matrix.csv', delimiter=',')

# closest neighbours to item i from item_similarity_matrix
def closest_neighbours(user, item, k=5):
    c = conn.cursor()
    c.execute( 'SELECT ItemID FROM example_table WHERE UserID =?', (user,) )
    items_n1 = c.fetchall()
    items_n1 = [item[0] for item in items_n1] #items that the user has rated

    items_n2 = [item for item in itemsList if item not in items_n1] #items that the user has not rated

    # similarity between item i and items in items_n1
    sim_with_n1 = [(i,item_similarity_matrix[itemsList.index(item)][itemsList.index(i)]) for i in items_n1]
    sim_with_n1.sort(key=lambda x: x[1], reverse=True)
    
    # similarity between item i and items in items_n2
    sim_with_n2 = [(i,item_similarity_matrix[itemsList.index(item)][itemsList.index(i)]) for i in items_n2]#
    sim_with_n2.sort(key=lambda x: x[1], reverse=True)
    sim_with_n2_itemID = [i[0] for i in sim_with_n2]

    user_index = userList.index(user)
    user_similarity = list(user_similarity_matrix[user_index])
    user_similarity_tuple = [(userList[i], user_similarity[i]) for i in range(len(user_similarity))]
    user_similarity_tuple.sort(key=lambda x: x[1], reverse=True)
    user_similarity_tuple = user_similarity_tuple[1:] #excluding the user itself
    

    # users from user_similarity_tuple who have rated item i in items_n2
    users_who_rated_itemsN2 = []
    for i in sim_with_n2_itemID:
        for user in user_similarity_tuple:
            user_item_rating = user_sim_dict[user[0]]
            if i in [j[0] for j in user_item_rating]:
                users_who_rated_itemsN2.append((user[0], i))
                break
    
    return sim_with_n1, sim_with_n2, users_who_rated_itemsN2, user_similarity_tuple


# prediction
def predict(user, item):
    sim_with_n1, sim_with_n2, users_who_rated_itemsN2, user_similarity_tuple = closest_neighbours(user, item)
    # numerator = sum([sim_with_n1[i][1] * (user_sim_dict[user][itemsList.index(sim_with_n1[i][0])][1] - avg_item[i]) for i in range(len(sim_with_n1))]) 
    # + sum([user_similarity_tuple[i][1] * (item_similarity_matrix[itemsList.index(item)][itemsList.index(users_who_rated_itemsN2[i][1])] * (user_sim_dict[users_who_rated_itemsN2[i][0]][itemsList.index(users_who_rated_itemsN2[i][1])][1] - avg_item[itemsList.index(users_who_rated_itemsN2[i][1])])) for i in range(len(users_who_rated_itemsN2))])

    # denominator = sum([sim_with_n1[i][1] for i in range(len(sim_with_n1))]) + sum([user_similarity_tuple[i][1] * item_similarity_matrix[itemsList.index(item)][itemsList.index(users_who_rated_itemsN2[i][1])] for i in range(len(users_who_rated_itemsN2))])

    numerator = sum(user_similarity_tuple[i][1] * (item_similarity_matrix)

    if denominator == 0:
        return 0
    else:
        value = "{:.4f}".format(avg_item[itemsList.index(item)] + numerator / denominator)
        return value
    

start = time.time()
predicted_rating = predict(1, 93)
end = time.time()

print("Time taken to predict rating: ", end-start)
print("Predicted rating: ", predicted_rating)