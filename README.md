# Predictive Model: Unraveling Movie Ratings with RBM-Based Collaborative Filtering

Netflix needs to recommend movies based on a user's viewing history and preferences, and to solve this problem, Netflix launched a contest to encourage data scientists to develop advanced algorithms to predict user preferences and improve movie recommendations. In the context of recommender systems, linear regression can be used to predict user ratings of movies based on various features. Restricted Boltzmann Machine (RBM) is a type of artificial neural network that excels in capturing complex patterns and dependencies in data.RBM excels in collaborative filtering, which can be used to make analysis by analyzing the preferences of similar users. Project goals: The main goal of this project is to train a model for movie recommendation through collaborative filtering using a recommender system that focuses on linear regression and restricted Boltzmann machines. Collaborative filtering involves predicting user preferences based on the preferences of other users with similar tastes.

The way this project was conducted was that the teams/individuals had access to a large training set and each week for the duration of 4 weeks a new testing set is posted to work on. Meanwhile the organizers of the project kept a hidden test set that was used for a final Evaluation.

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

![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/41e8dfd3-44fd-474e-a2da-e8c71068950a)

- Analysis on the number of hidden units:

![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/f41209b2-50f2-489a-a4c9-5d771373f97e)

- Analysis on the regularization coefficient:

![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/64299c67-5bcd-4c78-b520-d350722b3acf)

#### Best run so far: 

![image](https://github.com/H4D32/RBM-Recommender/assets/49611754/ee1ad6a2-b2e3-4f72-ace8-e888b40d15c8)

## Conclusion: 

In conclusion, embarking on the journey of developing an RBM model for predicting movie ratings, akin to the challenges posed by the Netflix problem in collaborative filtering, has been both a rewarding and enlightening experience. This project not only allowed us to delve into the intricacies of a sophisticated model but also provided invaluable insights into the nuanced realm of mitigating overfitting challenges.

Beyond the technical aspects, the project has proven to be a valuable asset for our professional development. The complexities encountered mirror real-world scenarios, where the unpredictability of a hidden test set parallels the uncertainty one often faces when implementing solutions for genuine applications. Navigating through these challenges has not only honed our technical skills but also underscored the importance of adaptability and resilience in the face of ambiguity.

Undoubtedly, the successful execution of this project contributes significantly to our skill set and makes for a compelling addition to our curriculum vitae. The lessons learned extend far beyond the confines of this endeavor, providing a solid foundation for tackling more intricate problems in the exciting world of machine learning and predictive modeling. As we reflect on this endeavor, we can confidently assert that the knowledge gained and the experiences accrued will serve as guiding beacons in our future pursuits within the dynamic and ever-evolving field of data science.


