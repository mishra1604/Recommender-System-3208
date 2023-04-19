import numpy as np
import sqlite3

conn = sqlite3.connect( 'comp3208_train.db' )

# collect an array of all users
def get_users():
    c = conn.cursor()
    c.execute("SELECT DISTINCT UserID FROM example_table")
    users = c.fetchall()
    users = [x[0] for x in users]
    return users

userList = get_users()

# collect an array of all items
def get_items():
    c = conn.cursor()
    c.execute("SELECT DISTINCT ItemID FROM example_table")
    items = c.fetchall()
    items = [x[0] for x in items]
    items.sort()
    return items

itemList = get_items()

# Creating R matrix
ratings_matrix = np.full((len(userList), len(itemList)), 0.0)

c = conn.cursor()
c.execute("SELECT UserId, ItemId, Rating FROM example_table")
ratings_tuple = c.fetchall()
for tuple in ratings_tuple:
    ratings_matrix[userList.index(tuple[0])][itemList.index(tuple[1])] = tuple[2]

sparsity = float(len(ratings_matrix.nonzero()[0]))
sparsity /= (ratings_matrix.shape[0] * ratings_matrix.shape[1])
sparsity *= 100
print ('Sparsity: {:4.2f}%'.format(sparsity))

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
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

from better_model import ExplicitMF

MF_SGD = ExplicitMF(train, 40, learning='sgd', verbose=True)
iter_array = [1, 2, 5, 10, 25, 50, 100, 200]
MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)