import numpy as np

# convert csv to numpy array
train_data = np.genfromtxt('train_100k_withratings_new.csv', delimiter=',', usecols=range(0,3))


table = np.zeros((1000, 100), dtype=np.float16)

print(train_data.shape)