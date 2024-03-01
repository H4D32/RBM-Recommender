import numpy as np
import projectLib as lib

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training): #Edit by Alae
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    for row in range(trStats["n_ratings"]):
        A[row][training[row][0]] = 1 #movie index
        A[row][training[row][1]+trStats["n_movies"]] = 1 #user index
    return A

# we also get c
def getc(rBar, ratings): #Edit by Alae
    c = ratings - rBar
    return c

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c): #Edit by Alae
    bstar = np.linalg.solve(A.T.dot(A), A.T.dot(c))
    return bstar

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l): #Edit by Alae
    bReg = np.linalg.solve(A.T.dot(A) + l * np.identity(trStats["n_movies"] + trStats["n_users"]), A.T.dot(c))
    return bReg

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version (<=> regularised version with l = 0)
b = param(A, c)

# Regularised version
l = 3.4 #Edit by Alae
lamb = np.arange(0,5.2,0.2)
rmse_hist_tr = list()
rmse_hist_vl = list()
b = param_reg(A, c, l)

for lb in lamb:
    btest = param_reg(A, c, lb)
    rmse_hist_tr.append(lib.rmse(predict(trStats["movies"], trStats["users"], rBar, btest), trStats["ratings"]))
    rmse_hist_vl.append(lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, btest), vlStats["ratings"]))


print("Linear regression, l = %f" % l)
print("RMSE for training %f" % lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))
print("RMSE for validation %f" % lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"]))

