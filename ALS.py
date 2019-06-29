import random
import pandas as pd
import numpy as np

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

#-------------------------
# LOAD AND PREP THE DATA
#-------------------------
 
raw_data = pd.read_csv('data/Movies.csv')
raw_data = raw_data.drop(raw_data.columns[1], axis=1)
raw_data.columns = ['user', 'movie','ratings']
 
 # Drop rows with missing values
 data = raw_data.dropna()
  
 # Convert artists names into numerical IDs
 data['user_id'] = data['user'].astype("category").cat.codes
 data['movie_id'] = data['movie'].astype("category").cat.codes
 
 # Create a lookup frame so we can get the artist names back in 
 # readable form later.
 item_lookup = data[['movie_id', 'artist']].drop_duplicates()
 item_lookup['movie_id'] = item_lookup.artist_id.astype(str)
 
 data = data.drop(['user', 'movie'], axis=1)
 
 # Drop any rows that have 0 plays
 data = data.loc[data.ratings != 0]
 
 # Create lists of all users, artists and plays
 users = list(np.sort(data.user_id.unique()))
 artists = list(np.sort(data.movie_id.unique()))
 plays = list(data.ratings)
 
 # Get the rows and columns for our new matrix
 rows = data.user_id.astype(int)
 cols = data.movie_id.astype(int)
 
 # Contruct a sparse matrix for our users and items containing number of plays
 data_sparse = sparse.csr_matrix((ratings, (rows, cols)), shape=(len(users), len(movies)))




def implicit_als(sparse_data, alpha_val=40, iterations=10, lambda_val=0.1, features=10):
    # Calculate the foncidence for each value in our data
    confidence = sparse_data * alpha_val
    
    # Get the size of user rows and item columns
    user_size, item_size = sparse_data.shape
    
    # We create the user vectors X of size users-by-features, the item vectors
    # Y of size items-by-features and randomly assign the values.
    X = sparse.csr_matrix(np.random.normal(size = (user_size, features)))
    Y = sparse.csr_matrix(np.random.normal(size = (item_size, features)))
    
    #Precompute I and lambda * I
    X_I = sparse.eye(user_size)
    Y_I = sparse.eye(item_size)
    
    I = sparse.eye(features)
    lI = lambda_val * I

    # Start main loop. For each iteration we first compute X and then Y
    for i in xrange(iterations):
        print 'iteration %d of %d' % (i+1, iterations)
        
        # Precompute Y-transpose-Y and X-transpose-X
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Loop through all users
        for u in xrange(user_size):

            # Get the user row.
            u_row = confidence[u,:].toarray() 

            # Calculate the binary preference p(u)
            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0

            # Calculate Cu and Cu - I
            CuI = sparse.diags(u_row, [0])
            Cu = CuI + Y_I

            # Put it all together and compute the final formula
            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

    
        for i in xrange(item_size):

            # Get the item column and transpose it.
            i_row = confidence[:,i].T.toarray()

            # Calculate the binary preference p(i)
            p_i = i_row.copy()
            p_i[p_i != 0] = 1.0

            # Calculate Ci and Ci - I
            CiI = sparse.diags(i_row, [0])
            Ci = CiI + X_I

            # Put it all together and compute the final formula
            xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

    return X, Y

user_vecs, item_vecs = implicit_als(data_sparse, iterations=20, features=20, alpha_val=40)

#------------------------------
# FIND SIMILAR ITEMS
#------------------------------

item_id = 10

# Get the item row 
item_vec = item_vecs[item_id].T

# Calculate the similarity score between Mr Carter and other artists
# and select the top 10 most similar.
scores = item_vecs.dot(item_vec).toarray().reshape(1,-1)[0]
top_10 = np.argsort(scores)[::-1][:10]

