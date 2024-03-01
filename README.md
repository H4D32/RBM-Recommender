![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/ed4294aa-7a7e-4a49-b8c0-91b924d4e8b8)![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/a0fc9b15-0156-444f-a5e0-4768c5c0e74a)# Predictive Model: Unraveling Movie Ratings with RBM-Based Collaborative Filtering

Netflix needs to recommend movies based on a user's viewing history and preferences, and to solve this problem, Netflix launched a contest to encourage data scientists to develop advanced algorithms to predict user preferences and improve movie recommendations. In the context of recommender systems, linear regression can be used to predict user ratings of movies based on various features. Restricted Boltzmann Machine (RBM) is a type of artificial neural network that excels in capturing complex patterns and dependencies in data.RBM excels in collaborative filtering, which can be used to make analysis by analyzing the preferences of similar users. Project goals: The main goal of this project is to train a model for movie recommendation through collaborative filtering using a recommender system that focuses on linear regression and restricted Boltzmann machines. Collaborative filtering involves predicting user preferences based on the preferences of other users with similar tastes.

## First part: Linear Regression 
Create a big matrix that works similar to a one hot encoding. Each row is a movie review with 2 non-negative numbers. Their indices correspond to the rated movie and the user who rated it respectively.
![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/1fd92ff4-f911-4aba-ba28-1d8e993c4fa0)
Look into the report.pdf for detailed math analysis
### Hyperparameter : Regularization 
- Trying different values shown on the graph below on Validation Loss :
![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/42664363-0f5c-4bdf-bcd7-75d4f12d8674)

## Restricted BoltzMann Machine:
it is a two layer network: 
-the Visible Layer is the input layer
-the Hidden Layer receives a non-linear transformation of the input allowing it to make precise estimators
What we want to learn in this model:the weights **W** connecting the two layers together
![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/4fd74434-b1d5-4a27-824c-c0c0a61fed55)
The way how this is traversed Mathematically is shown in the report pdf

### Few test runs and what we learn
- One of the largest problem we ran into was overfitting models, Here is one where we notice  at 30 epochs it is still getting more used to the training dataset while the results on the validation only get worse. By implementing an adaptive learning rate with other variables kept intact we can see a large difference. we also do analysis on other hyper parameters :
- Analysis on the number of hidden units:
F (Hidden units)	10	20	25	30	40	80
Train RMSE	0.820	0.803	0.763	0.801	0.733	0.691
Valid RMSE	0.852	0.854	0.853	0.879	0.870	0.886
![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/f41209b2-50f2-489a-a4c9-5d771373f97e)



![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/326ecfeb-2f5d-46b7-bd5e-d029e40c84f8)

