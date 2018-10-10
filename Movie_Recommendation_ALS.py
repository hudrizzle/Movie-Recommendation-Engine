# Databricks notebook source
# MAGIC %md In this notebook, we will implement an Alternating Least Squares (ALS) algorithm and predict the ratings for the movies in [MovieLens small dataset](https://grouplens.org/datasets/movielens/latest/)

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import time
%matplotlib inline

# COMMAND ----------

# MAGIC %md ## Part 1: Data ETL and Data Exploration

# COMMAND ----------

import os
os.listdir('../../dbfs/FileStore')

movies = pd.read_csv('../../dbfs/FileStore/tables/movies.csv')
ratings = pd.read_csv('../../dbfs/FileStore/tables/ratings.csv')
links = pd.read_csv('../../dbfs/FileStore/tables/links.csv')
tags = pd.read_csv('../../dbfs/FileStore/tables/tags.csv')

# COMMAND ----------

movies.head()

# COMMAND ----------

ratings.head()

# COMMAND ----------

print 'Distinct values of ratings:'
print sorted(ratings.rating.unique())

# COMMAND ----------

print 'For the users that rated movies and the movies that were rated:'
print 'Minimum number of ratings per user is', ratings.userId.value_counts().min()
print 'Minimum number of ratings per movie is', ratings.movieId.value_counts().min()

# COMMAND ----------

print '{} out of {} movies are rated by only one user'.format(sum(ratings.movieId.value_counts() == 1), \
                                                              len(ratings.movieId.unique()))

# COMMAND ----------

links.head()

# COMMAND ----------

tags.head()

# COMMAND ----------

# MAGIC %md ### 1.1: The number of Users

# COMMAND ----------

num_users = np.union1d(ratings['userId'], tags['userId']).shape[0]
num_users

# COMMAND ----------

# MAGIC %md ### 1.2: The number of Movies

# COMMAND ----------

np.union1d(ratings['movieId'], tags['movieId']).shape[0]

# COMMAND ----------

# MAGIC %md ### 1.3:  How many movies are rated by users? List movies not rated before

# COMMAND ----------

num_movies_rated = ratings['movieId'].unique().shape[0]
num_movies_rated

# COMMAND ----------

all_movies = np.union1d(ratings['movieId'], tags['movieId'])
rated = ratings['movieId'].unique()
not_rated = [x for x in all_movies if x not in rated]

# showing 20 examples only
pd.DataFrame(not_rated, columns = ['movieId']).head(20)

# COMMAND ----------

# MAGIC %md ### 1.4: List Movie Genres

# COMMAND ----------

# showing 20 examples only
pd.DataFrame(movies['genres'].unique(), columns = ['genres']).head(20)

# COMMAND ----------

# MAGIC %md ### 1.5: Movie for Each Category

# COMMAND ----------

# showing top 20 only
movies["genres"].value_counts().head(20) 

# COMMAND ----------

# MAGIC %md #Part2: Prepare Data for Training

# COMMAND ----------

# MAGIC %md ##2.1: Process Data for Training

# COMMAND ----------

rating_data = ratings.drop(['timestamp'], axis = 1)

# COMMAND ----------

rating_data.head()

# COMMAND ----------

# MAGIC %md We will use a `num_users` x `num_movies_rated` matrix to represent the ratings, in which zeros are the user-movie pairs without a rating.

# COMMAND ----------

ratings_matrix = np.zeros((num_users, num_movies_rated))

usersId = np.sort(ratings.userId.unique())
moviesId = np.sort(ratings.movieId.unique())

for i in range(len(rating_data)):
    user = rating_data.iloc[i, 0]
    movie = rating_data.iloc[i, 1]
    ratings_matrix[usersId == user, moviesId == movie] = rating_data.iloc[i, 2]
    
ratings_matrix

# COMMAND ----------

# MAGIC %md Now we split the data into training/validation/testing sets using a 6/2/2 ratio. We use a for loop to split the existing ratings for each user. Because the minimum number of ratings per user is 20, each user will have at least 12, 4, and 4 ratings in the training, validation, and testing sets, respectively.

# COMMAND ----------

import random
random.seed(0)

def data_split(ratings, ratio):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        index = ratings[user, :].nonzero()[0]
        k = int(round(len(index) * ratio))
        test_ratings = random.sample(index, k)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    return train, test

# COMMAND ----------

train_matrix, tmp = data_split(ratings_matrix, 0.4)
validation_matrix, test_matrix = data_split(tmp, 0.5)

# COMMAND ----------

print 'Number of ratings in the training set:', (train_matrix > 0).sum()
print 'Number of ratings in the validation set:', (validation_matrix > 0).sum()
print 'Number of ratings in the testing set:', (test_matrix > 0).sum()

# COMMAND ----------

# MAGIC %md ##2.2 ALS Model
# MAGIC Now we’ll build a ALS model. The implementation below is borrowed and modified from [here](https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea). One main modification is that we train the model with the mean subtracted ratings here. At the prediction step, the mean rating will be added back. This approach makes it easier for the model to converge.

# COMMAND ----------

from numpy.linalg import solve

class ALS(object):
    
    def __init__(self, ratings, n_factors=10, reg=0.0, verbose=False):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        n_factors : (int)
            Number of latent factors to use in matrix 
            factorization model
        
        reg : (float)
            Regularization term for both user and item latent factors
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        
        # we will train the model with the mean subtracted ratings, and 
        # at the prediction step, the mean will be added back. 
        self.raw_ratings = ratings
        self.mean = ratings[ratings > 0].mean()
        self.ratings = ratings.copy() 
        self.ratings[ratings > 0] = self.ratings[ratings > 0] - self.mean
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.reg = reg
        self._v = verbose

    def als_step(self, latent_vectors, fixed_vecs, ratings, _lambda, type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute for the item latent vectors
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * _lambda

            for u in xrange(latent_vectors.shape[0]):
                #latent_vectors[u, :] = (solve((YTY + lambdaI).T, (ratings[u, :].dot(fixed_vecs)).T)).T
                latent_vectors[u, :] = solve((YTY + lambdaI), ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * _lambda
            
            for i in xrange(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((XTX + lambdaI), 
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, n_iter=10):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors using a normal distribution centered at 0
        self.user_vecs = np.random.normal(0,1,(self.n_users, self.n_factors)) 
        self.item_vecs = np.random.normal(0,1,(self.n_items, self.n_factors)) 
       
        self.partial_train(n_iter)
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. Can be 
        called multiple times for further training.
        """
        count = 1
        while count <= n_iter:
            if count % 10 == 0 and self._v:
                print '\tcurrent iteration: {}'.format(count)
            self.user_vecs = self.als_step(self.user_vecs, 
                                           self.item_vecs, 
                                           self.ratings, 
                                           self.reg, 
                                           type='user')
            self.item_vecs = self.als_step(self.item_vecs, 
                                           self.user_vecs, 
                                           self.ratings, 
                                           self.reg, 
                                           type='item')
            count += 1
    
    def predict_all(self):
        """ Predict ratings for every user and item. """
        predictions = self.user_vecs.dot(self.item_vecs.T) + self.mean        
        return predictions
    
    def calculate_learning_curve(self, iter_array, valid):
        """
        Keep track of RMSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        valid : (2D ndarray)
            Validation dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_rmse : (list)
            Training data RMSE values for each value of iter_array
        valid_rmse : (list)
            Validation data RMSE values for each value of iter_array
        """
        self.iter_array = iter_array
        self.iter_array.sort()
        self.train_rmse =[]
        self.valid_rmse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(self.iter_array):
            if self._v:
                print 'Iteration: {}'.format(n_iter)
            if i == 0:
                self.train(n_iter - iter_diff)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_rmse += [get_rmse(predictions, self.raw_ratings)]
            self.valid_rmse += [get_rmse(predictions, valid)]
            if self._v:
                print 'Train RMSE: ' + str(self.train_rmse[-1])
                print 'Validation RMSE: ' + str(self.valid_rmse[-1])
            iter_diff = n_iter

    def plot_learning_curve(self):
        """ Plot the learning curves """
        fig = plt.figure(figsize=(10,8))
        plt.plot(self.iter_array, self.train_rmse, label='Training', linewidth=5)
        plt.plot(self.iter_array, self.valid_rmse, label='Validation', linewidth=5)
        plt.xticks(range(0, max(self.iter_array) + 1, 2), fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('iterations', fontsize=30)
        plt.ylabel('RMSE', fontsize=30)
        plt.legend(loc='best', fontsize=20) 
        plt.show()

        
def get_rmse(predictions, ratings):
    """ Calculate RMSE for predicted ratings"""
    diff = (predictions - ratings)[ratings > 0]
    return np.sqrt((diff**2).mean())

# COMMAND ----------

# MAGIC %md #Part3: Model Selection and Evaluation
# MAGIC With the ALS model, we can use a grid search to find the optimal hyperparameters.

# COMMAND ----------

latent_factors = [6, 8, 10, 12, 14]
regularizations = [0.05, 0.1, 0.2, 0.4, 0.8]
iter_array = [1, 2, 5, 10]


best_params = {}
best_params['n_factors'] = latent_factors[0]
best_params['reg'] = regularizations[0]
best_params['valid_rmse'] = np.inf
best_params['model'] = None
best_params['train_rmse'] = np.inf

start_time = time.time()

for fact in latent_factors:
    for reg in regularizations:
      # put your code here to get the best model
      print 'Regularization: {}'.format(reg)
      print 'latent_factors: {}'.format(fact)
      MF_ALS = ALS(train_matrix, n_factors=fact, \
                          reg=reg)
      MF_ALS.calculate_learning_curve(iter_array, test_matrix)
      min_idx = np.argmin(MF_ALS.valid_rmse)
      if MF_ALS.valid_rmse[min_idx] < best_params['valid_rmse']:
          best_params['n_factors'] = fact
          best_params['reg'] = reg
          best_params['n_iter'] = iter_array[min_idx]
          best_params['train_rmse'] = MF_ALS.train_rmse[min_idx]
          best_params['valid_rmse'] = MF_ALS.valid_rmse[min_idx]
          best_params['model'] = MF_ALS
          print 'New optimal hyperparameters'
          print pd.Series(best_params)
print '\nThe best model has {} latent factors and regularization = {}'.format(best_params['n_factors'], best_params['reg'])        
print 'Total Runtime: {:.2f} seconds'.format(time.time() - start_time)

# COMMAND ----------

# MAGIC %md The model with 12 latent factors and lambda = 0.1 yields the best result. Let's plot the learning curves for this model.

# COMMAND ----------

best_model = best_params['model']

print 'For testing data the RMSE is {}'.format(get_rmse(best_model.predict_all(), validation_matrix))

# COMMAND ----------

Best_ALS = None
iter_array = [1, 2, 5, 10]
Best_ALS = ALS(train_matrix, n_factors=best_params['n_factors'], \
                         reg=best_params['reg'])
Best_ALS.calculate_learning_curve(iter_array, validation_matrix)

# COMMAND ----------


Best_ALS.plot_learning_curve()
display()

# COMMAND ----------

# MAGIC %md And finally, let's check the testing error.

# COMMAND ----------

print 'For testing data the RMSE is {}'.format(get_rmse(best_model.predict_all(), test_matrix))


# COMMAND ----------

# MAGIC %md This is consistent with our validation error.
