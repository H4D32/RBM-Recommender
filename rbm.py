import numpy as np
import projectLib as lib

# set highest rating
K = 5

def softmax(x):
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ratingsPerMovie(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()
    return np.array([[i, movie, len([x for x in training if x[0] == movie])] for i, movie in enumerate(u_movies)])

def getV(ratingsForUser):
    # ratingsForUser is obtained from the ratings for user library
    # you should return a binary matrix ret of size m x K, where m is the number of movies
    #   that the user has seen. ret[i][k] = 1 if the user
    #   has rated movie ratingsForUser[i, 0] with k stars
    #   otherwise it is 0
    ret = np.zeros((len(ratingsForUser), K))
    for i in range(len(ratingsForUser)):
        ret[i, ratingsForUser[i, 1]-1] = 1.0
    return ret

def getInitialWeights(m, F, K):
    # m is the number of visible units
    # F is the number of hidden units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, F, K))


def sig(x): #Edit by Alae
    return 1 / (1+ np.exp(-x))


def visibleToHiddenVec(v, w, b):
    # Implemented
    F = w.shape[1]
    hiddenVec = np.zeros(F)
    for j in range(F):
        # hiddenVec[j] = np.multiply(w[:, j, :], v).sum() + b[j]
        hiddenVec[j] = (w[:, j, :] * v).sum() + b[j]
    hiddenVec = sig(hiddenVec)
    return hiddenVec


def hiddenToVisible(h, w, b):
    # Implemented
    m = w.shape[0]
    negData = np.zeros([m, K])
    for i in range(m):
        for k in range(K):
            negData[i][k] = (w[i, :, k] * h).sum() + b[i][k]
        negData[i] = softmax(negData[i])

    return negData

def probProduct(v, p):
    # v is a matrix of size m x 5
    # p is a vector of size F, activation of the hidden units
    # returns the gradient for visible input v and hidden activations p
    ret = np.zeros((v.shape[0], p.size, v.shape[1]))
    for i in range(v.shape[0]):
        for j in range(p.size):
            for k in range(v.shape[1]):
                ret[i, j, k] = v[i, k] * p[j]
    return ret

def sample(p):
    # p is a vector of real numbers between 0 and 1
    # ret is a vector of same size as p, where ret_i = Ber(p_i)
    # In other word we sample from a Bernouilli distribution with
    # parameter p_i to obtain ret_i
    samples = np.random.random(p.size)
    return np.array(samples <= p, dtype=int)

def getPredictedDistribution(v, w, wq, bias):
    # Implemented
    b_hidden, b_movie = bias
    hiddenVec = visibleToHiddenVec(v, w, b_hidden) 
    sample_hiddenVec = sample(hiddenVec)  

    Dist = np.zeros(K)
    for k in range(K):
        Dist[k] = (wq[:, k] * sample_hiddenVec).sum() + b_movie[k]
    Dist = softmax(Dist)
    return Dist

def predictRatingMax(ratingDistribution):
    # Implemented
    arg_max = np.argmax(ratingDistribution) + 1
    return arg_max


def predictRatingExp(ratingDistribution):
    #implemented
    ratings = np.arange(1,6)
    avg_predict = np.dot(ratingDistribution,ratings)
    return avg_predict

def predictMovieForUser(q, user, W, allUsersRatings,bias, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    ratingsForUser = allUsersRatings[user]
    v = getV(ratingsForUser)
    b_hidden, b_visible = bias
    ratingDistribution = getPredictedDistribution(v, W[ratingsForUser[:, 0], :, :], W[q, :, :],(b_hidden,b_visible[q]))
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)

def predict(movies, users, W, allUsersRatings,bias, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUser(movie, user, W, allUsersRatings,bias, predictType=predictType) for (movie, user) in zip(movies, users)]

def predictForUser(user, W, allUsersRatings,bias, predictType="exp"):
    #implemented
    # given a user ID, predicts all movie ratings for the user
    m = W.shape[0]
    pred = np.zeros(m)
    for movie in range(m):
        pred[movie] = predictMovieForUser(movie, user, W, allUsersRatings,bias,predictType=predictType)
    return pred

### Additional functions : 

def initBiases(m, F, K):
    biasHidden = np.random.normal(0, 0.1, (F))
    biasVisible = np.random.normal(0, 0.1, (m, K))
    return biasHidden, biasVisible

