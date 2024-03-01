import numpy as np
import rbm
import projectLib as lib
import matplotlib.pyplot as plt 

training = lib.getTrainingData()
validation = lib.getValidationData()


trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

allUsersRatings = lib.getAllUsersRatings(trStats["u_users"], training)

K = 5

# SET PARAMETERS HERE!!!
# number of hidden units
F=25
epochs=30
init_gradientLearningRate = 0.01
biasRate = 0.01
regL = 0.1
reduce_factor = 0.5
patience = 3
min_loss_improvement = 1e-4
min_learning_rate = 1e-6

# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
bias_hidden, bias_visible = rbm.initBiases(trStats["n_movies"], F, K)
W_hist = dict()
loss_hist = dict()
grad = np.zeros(W.shape)
grad_bVisible = np.zeros(bias_visible.shape)
grad_bHidden = np.zeros(bias_hidden.shape)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)

# Initialize momentum for weights and biases
momentum_W = np.zeros_like(W)
momentum_bVisible = np.zeros_like(bias_visible)
momentum_bHidden = np.zeros_like(bias_hidden)

# Set momentum hyperparameter
momentum_rate = 0.5  # You can experiment with different values

best_val_loss = float('inf')
no_improvement_count = 0
gradientLearningRate = init_gradientLearningRate

for epoch in range(1, epochs + 1):
    # in each epoch, we'll visit all users in a random order
    visitingOrder = np.array(trStats["u_users"])
    np.random.shuffle(visitingOrder)
    print("current: epoch " + str(epoch))
    counter = 0
    for user in visitingOrder:
        # get the ratings of that user
        ratingsForUser = allUsersRatings[user]

        # build the visible input
        v = rbm.getV(ratingsForUser)

        # get the weights associated to movies the user has seen
        weightsForUser = W[ratingsForUser[:, 0], :, :] #mxFx5
        biasUser = bias_visible[ratingsForUser[:, 0], :]

        ### LEARNING ###
        # propagate visible input to hidden units
        posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser, bias_hidden)
        # get positive gradient
        # note that we only update the movies that this user has seen!
        posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

        ### UNLEARNING ###
        # sample from hidden distribution
        sampledHidden = rbm.sample(posHiddenProb)
        # propagate back to get "negative data"
        negData = rbm.hiddenToVisible(sampledHidden, weightsForUser,biasUser)
        # propagate negative data to hidden units
        negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser,bias_hidden)
        # get negative gradient
        # note that we only update the movies that this user has seen!
        negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)

        # we average over the number of users in the batch (if we use mini-batch)
        grad[ratingsForUser[:, 0], :, :] = gradientLearningRate * (posprods[ratingsForUser[:, 0], :, :] - negprods[ratingsForUser[:, 0], :, :]) 
        - gradientLearningRate * regL * W[ratingsForUser[:, 0], :, :]

        grad_bVisible[ratingsForUser[:, 0], :] = biasRate * (v - negData)
        grad_bHidden = biasRate * (posHiddenProb - negHiddenProb)

        # Update momentum
        momentum_W[ratingsForUser[:, 0], :, :] = momentum_rate * momentum_W[ratingsForUser[:, 0], :, :] + grad[ratingsForUser[:, 0], :, :]
        momentum_bVisible[ratingsForUser[:, 0], :] = momentum_rate * momentum_bVisible[ratingsForUser[:, 0], :] + grad_bVisible[ratingsForUser[:, 0], :]
        momentum_bHidden = momentum_rate * momentum_bHidden + grad_bHidden

        # Update parameters with momentum
        W[ratingsForUser[:, 0], :, :] += momentum_W[ratingsForUser[:, 0], :, :]
        bias_visible[ratingsForUser[:, 0], :] += momentum_bVisible[ratingsForUser[:, 0], :]
        bias_hidden += momentum_bHidden



    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    print("### start predicting on :EPOCH %d ###" % epoch)

    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, allUsersRatings,(bias_hidden,bias_visible))
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)
    print("Training loss = %f" % trRMSE)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, allUsersRatings,(bias_hidden,bias_visible))
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)
    print("Validation loss = %f" % vlRMSE)
    print("### EPOCH %d Storing history ###" % epoch)
    W_hist[epoch] = W
    loss_hist[epoch] = (trRMSE,vlRMSE)
    if epoch == 1:
        last_val_loss = 100 #just a high number so it doesnt get caught first
    else:
        last_tr_loss , last_val_loss = loss_hist[epoch-1]

    loss_improvement = best_val_loss - vlRMSE

    if loss_improvement > min_loss_improvement:
        best_val_loss = vlRMSE
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if last_val_loss - vlRMSE < -1e-4 and last_tr_loss - trRMSE > 1e-4:
        no_improvement_count = patience
        # regL = min(regL * 2,1)
        print(f'Hard Loss, reg now {regL}')

    if no_improvement_count >= patience:
        gradientLearningRate = max(gradientLearningRate * reduce_factor, min_learning_rate)
        biasRate = max(biasRate * reduce_factor, min_learning_rate)
        print(f'Reducing learning rate to {gradientLearningRate}')
        # Reset no_improvement_count
        no_improvement_count = 0

    print("### EPOCH %d Finished ###" % epoch)

for key, value in loss_hist.items():
    print(f"epoch {key}: {value}")

max_key, max_tuple = min(loss_hist.items(), key=lambda x: x[1][1])

print(f"Best validation found at epoch {max_key}: {max_tuple}")




### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
print("### Saving full prediction File ###")
bestW = W_hist[max_key]

predictedRatings = np.array([rbm.predictForUser(user, bestW, allUsersRatings, (bias_hidden,bias_visible)) for user in trStats["u_users"]])
np.savetxt(f"Ratings epoch{max_key} F{F} regL{regL} alpha{init_gradientLearningRate}.txt", predictedRatings)

x_values = loss_hist.keys()
y1_values = [item[0] for item in loss_hist.values()]
y2_values = [item[1] for item in loss_hist.values()]

plt.plot(x_values, y1_values, label='Training')
plt.plot(x_values, y2_values, label='Validation')

plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()

print("### All done! ###")
