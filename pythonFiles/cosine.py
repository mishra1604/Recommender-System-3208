import numpy as np
import math
import sqlite3

# find the cosine similarity between users in the sqlite3 database
conn = sqlite3.connect( 'comp3208_train.db' )
def find_cosine_similarity(user1, user2 ):
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
    
print(find_cosine_similarity('1', '2' ))