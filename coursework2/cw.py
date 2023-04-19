import numpy
import sqlite3

# Recommender system using matrix factorization for 20 million ratings dataset

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

item_rating_dictionary = {}

def get_ratings():
    for x in itemList:
        c = conn.cursor()
        c.execute( 'SELECT UserID, Rating FROM example_table WHERE ItemID = ?', (x,) )
        item_rating_dictionary[x] = c.fetchall()

# Creating R matrix
R = numpy.full((len(userList), len(itemList)), 0.0)

c = conn.cursor()
c.execute("SELECT UserId, ItemId, Rating FROM example_table")
ratings = c.fetchall()
for tuple in ratings:
    R[userList.index(tuple[0])][itemList.index(tuple[1])] = tuple[2]


# N: num of User
N = len(R)
# M: num of items
M = len(R[0])
# Num of Features
K = 3

P = numpy.random.rand(N,K)

Q = numpy.random.rand(M,K)

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        print(step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

nP, nQ = matrix_factorization(R[:50], P, Q, K)

nR = numpy.dot(nP, nQ.T)

print(nR[0][:200])
