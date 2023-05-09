import time
import numpy as np
import sqlite3
from better_model import ExplicitMF
import csv

# read the csv file
def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)

train_20million = read_csv('train_100k_withratings_new.csv')

userList = {}
itemList = {}

counter = 0
for tuple in train_20million:
    counter += 1
    userID = int(tuple[0])
    itemID = int(tuple[1])
    if userID not in userList:
        userList[userID] = userID
    if itemID not in itemList:
        itemList[itemID] = itemID
    
    if counter % 5000000 == 0:
        print(counter)

# Convert the dictionary to a sorted array of tuples
userList = [k for k, v in sorted(userList.items(), key=lambda x: x[1])]
itemList = [k for k, v in sorted(itemList.items(), key=lambda x: x[1])]

# userID dictionary with value being the index of the user in the userList
userDict = {}
counter = 0
for i in userList:
    userDict[i] = counter
    counter += 1

# itemID dictionary with value being the index of the item in the itemList
itemDict = {}
counter = 0
for i in itemList:
    itemDict[i] = counter
    counter += 1

print("Number of Users = ",len(userDict))
print("Number of Items = ",len(itemDict))


# Creating R matrix
ratings_matrix = np.full((len(userList), len(itemList)), 0.0, dtype=np.float16)
print("\n\nShape of Matrix = ",ratings_matrix.shape)

counter = 0
for tuple in train_20million:
    user_ID = int(tuple[0])
    item_ID = int(tuple[1])
    rating = float(tuple[2])
    ratings_matrix[userDict[user_ID]][itemDict[item_ID]] = rating
    counter += 1
    if counter % 1000000 == 0:
        print(counter)

print(ratings_matrix[0][:200])

def sparse_matrix(ratings_matrix):
    sparsity = float(len(ratings_matrix.nonzero()[0]))
    sparsity /= (ratings_matrix.shape[0] * ratings_matrix.shape[1])
    sparsity *= 100
    print ('Sparsity: {:4.2f}%'.format(sparsity))

def train_test_split(ratings):
    test = np.zeros(ratings.shape, dtype=np.float16)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

train, test = train_test_split(ratings_matrix)

def try_model():
    MF_SGD = ExplicitMF(train, n_factors=30, learning='sgd', verbose=True)
    iter_array = [10]
    new_computed_matrix = MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)
    return new_computed_matrix

start = time.time()
new_computed_matrix = try_model()
end = time.time()

print("Time taken = ", round(end-start, 2), "seconds" )

def test_rating_predictions():
    test_20million = read_csv('test_100k_withoutratings_new.csv')

    results = []
    counter = 0
    except_counter = 0
    for tuple in test_20million:
        user_ID = int(tuple[0])
        item_ID = int(tuple[1])
        try:
            rating = new_computed_matrix[userDict[user_ID]][itemDict[item_ID]]
        except:
            rating = 3
            except_counter += 1
        time_stamp = int(tuple[2])
        results.append([user_ID, item_ID, rating, time_stamp])

        if counter % 1000000 == 0:
            print(counter)
        counter += 1
    
    with open('results_100k.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in results:
            writer.writerow(row)
    
    print("Number of exceptions = ", except_counter)

test_rating_predictions()