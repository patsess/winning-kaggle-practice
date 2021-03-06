
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
    cross_val_score)
import hyperopt as hp
from hyperopt import fmin
from hyperopt import tpe
from tpot import TPOTClassifier

"""
Note: code is for reference only (taken from an online course)
"""


if __name__ == '__main__':
    os.cpu_count()  # to find number of cores supported by the computer
    # note: this could be useful for training multiple competing models

    ######################################################################
    # Extracting a Logistic Regression parameter:

    # Create a list of original variable names from the training DataFrame
    original_variables = X_train.columns.tolist()

    # Extract the coefficients of the logistic regression estimator
    model_coefficients = log_reg_clf.coef_[0]

    # Create a dataframe of the variables and coefficients & print it out
    coefficient_df = pd.DataFrame(
        {"Variable": original_variables, "Coefficient": model_coefficients})
    print(coefficient_df)

    # Print out the top 3 positive variables
    top_three_df = coefficient_df.sort_values(by='Coefficient', axis=0,
                                              ascending=False)[0:3]
    print(top_three_df)

    ######################################################################
    # Extracting a Random Forest parameter:

    # Extract the 7th (index 6) tree from the random forest
    chosen_tree = rf_clf.estimators_[6]

    # Visualize the graph using the provided image
    imgplot = plt.imshow(tree_viz)
    plt.show()

    # Extract the parameters and level of the top (index 0) node
    split_column = chosen_tree.tree_.feature[0]
    split_column_name = X_train.columns[split_column]
    split_value = chosen_tree.tree_.threshold[0]

    # Print out the feature and level
    print("This node split on feature {}, at a value of {}".format(
        split_column_name, split_value))

    ######################################################################
    # Exploring Random Forest Hyperparameters:

    # Print out the old estimator, notice which hyperparameter is badly set
    print(rf_clf_old)

    # Get confusion matrix & accuracy for the old rf_model
    print("Confusion Matrix: \n\n {} \n Accuracy Score: \n\n {}".format(
        confusion_matrix(y_test, rf_old_predictions),
        accuracy_score(y_test, rf_old_predictions)))

    # Create a new random forest classifier with better hyperparamaters
    rf_clf_new = RandomForestClassifier(n_estimators=500)

    # Fit this to the data and obtain predictions
    rf_new_predictions = rf_clf_new.fit(X_train, y_train).predict(X_test)

    # Assess the new model (using new predictions!)
    print("Confusion Matrix: \n\n",
          confusion_matrix(y_test, rf_new_predictions))
    print("Accuracy Score: \n\n", accuracy_score(y_test, rf_new_predictions))

    ######################################################################
    # Hyperparameters of KNN:

    # Build a knn estimator for each value of n_neighbours
    knn_5 = KNeighborsClassifier(n_neighbors=5)
    knn_10 = KNeighborsClassifier(n_neighbors=10)
    knn_20 = KNeighborsClassifier(n_neighbors=20)

    # Fit each to the training data & produce predictions
    knn_5_predictions = knn_5.fit(X_train, y_train).predict(X_test)
    knn_10_predictions = knn_10.fit(X_train, y_train).predict(X_test)
    knn_20_predictions = knn_20.fit(X_train, y_train).predict(X_test)

    # Get an accuracy score for each of the models
    knn_5_accuracy = accuracy_score(y_test, knn_5_predictions)
    knn_10_accuracy = accuracy_score(y_test, knn_10_predictions)
    knn_20_accuracy = accuracy_score(y_test, knn_20_predictions)
    print("The accuracy of 5, 10, 20 neighbours was {}, {}, {}".format(
        knn_5_accuracy, knn_10_accuracy, knn_20_accuracy))

    ######################################################################
    # Automating Hyperparameter Choice:

    # Set the learning rates & results storage
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    results_list = []

    # Create the for loop to evaluate model predictions for each learning rate
    for learning_rate in learning_rates:
        model = GradientBoostingClassifier(learning_rate=learning_rate)
        predictions = model.fit(X_train, y_train).predict(X_test)
        # Save the learning rate and accuracy score
        results_list.append(
            [learning_rate, accuracy_score(y_test, predictions)])

    # Gather everything into a DataFrame
    results_df = pd.DataFrame(results_list,
                              columns=['learning_rate', 'accuracy'])
    print(results_df)

    ######################################################################
    # Building Learning Curves:

    # Set the learning rates & accuracies list
    learn_rates = np.linspace(0.01, 2, num=30)
    accuracies = []

    # Create the for loop
    for learn_rate in learn_rates:
        # Create the model, predictions & save the accuracies as before
        model = GradientBoostingClassifier(learning_rate=learn_rate)
        predictions = model.fit(X_train, y_train).predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))

    # Plot results
    plt.plot(learn_rates, accuracies)
    plt.gca().set(xlabel='learning_rate', ylabel='Accuracy',
                  title='Accuracy for different learning_rates')
    plt.show()

    ######################################################################
    # Build Grid Search functions:

    # Create the function
    def gbm_grid_search(learn_rate, max_depth):
        # Create the model
        model = GradientBoostingClassifier(learning_rate=learn_rate,
                                           max_depth=max_depth)

        # Use the model to make predictions
        predictions = model.fit(X_train, y_train).predict(X_test)

        # Return the hyperparameters and score
        return ([learn_rate, max_depth, accuracy_score(y_test, predictions)])

    ######################################################################
    # Iteratively tune multiple hyperparameters:

    # Create the relevant lists
    results_list = []
    learn_rate_list = [0.01, 0.1, 0.5]
    max_depth_list = [2, 4, 6]

    # Create the for loop
    for learn_rate in learn_rate_list:
        for max_depth in max_depth_list:
            results_list.append(gbm_grid_search(learn_rate, max_depth))

    # Print the results
    print(results_list)

    ######################################################################

    results_list = []
    learn_rate_list = [0.01, 0.1, 0.5]
    max_depth_list = [2, 4, 6]

    # Extend the function input
    def gbm_grid_search_extended(learn_rate, max_depth, subsample):
        # Extend the model creation section
        model = GradientBoostingClassifier(learning_rate=learn_rate,
                                           max_depth=max_depth,
                                           subsample=subsample)

        predictions = model.fit(X_train, y_train).predict(X_test)

        # Extend the return part
        return ([learn_rate, max_depth, subsample,
                 accuracy_score(y_test, predictions)])

    ######################################################################

    results_list = []

    # Create the new list to test
    subsample_list = [0.4, 0.6]

    for learn_rate in learn_rate_list:
        for max_depth in max_depth_list:

            # Extend the for loop
            for subsample in subsample_list:
                # Extend the results to include the new hyperparameter
                results_list.append(
                    gbm_grid_search_extended(learn_rate, max_depth, subsample))

    # Print results
    print(results_list)

    ######################################################################
    # GridSearchCV with Scikit Learn:

    # Create a Random Forest Classifier with specified criterion
    rf_class = RandomForestClassifier(criterion='entropy')

    # Create the parameter grid
    param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']}

    # Create a GridSearchCV object
    grid_rf_class = GridSearchCV(
        estimator=rf_class,
        param_grid=param_grid,
        scoring='roc_auc',
        n_jobs=4,
        cv=5,
        refit=True, return_train_score=True)
    print(grid_rf_class)

    ######################################################################
    # Exploring the grid search results:

    # Read the cv_results property into a dataframe & print it out
    cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
    print(cv_results_df)

    # Extract and print the column with a dictionary of hyperparameters used
    column = cv_results_df.loc[:, ['params']]
    print(column)

    # Extract and print the row that had the best mean test score
    best_row = cv_results_df[cv_results_df['mean_test_score'] == cv_results_df[
        'mean_test_score'].max()]
    print(best_row)

    ######################################################################
    # Analyzing the best results:

    # Print out the ROC_AUC score from the best-performing square
    best_score = grid_rf_class.best_score_
    print(best_score)

    # Create a variable from the row related to the best-performing square
    cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
    best_row = cv_results_df.loc[[grid_rf_class.best_index_]]
    print(best_row)

    # Get the n_estimators parameter from the best-performing square
    best_n_estimators = grid_rf_class.best_params_["n_estimators"]

    ######################################################################
    # Using the best results:

    # See what type of object the best_estimator_ property is
    print(type(grid_rf_class.best_estimator_))

    # Create an array of predictions directly using the best_estimator_ property
    predictions = grid_rf_class.best_estimator_.predict(X_test)

    # Take a look to confirm it worked, this should be an array of 1's and 0's
    print(predictions[0:5])

    # Now create a confusion matrix
    print("Confusion Matrix \n", confusion_matrix(y_test, predictions))

    # Get the ROC-AUC score
    predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:,
                        1]
    print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))

    ######################################################################
    # Randomly Sample Hyperparameters:

    # Create a list of values for the learning_rate hyperparameter
    learn_rate_list = list(np.linspace(0.01, 1.5, 200))

    # Create a list of values for the min_samples_leaf hyperparameter
    min_samples_list = list(range(10, 41))

    # Combination list
    combinations_list = [list(x) for x in
                         product(learn_rate_list, min_samples_list)]

    # Sample hyperparameter combinations for a random search.
    random_combinations_index = np.random.choice(
        range(0, len(combinations_list)), 250, replace=False)
    combinations_random_chosen = [combinations_list[x] for x in
                                  random_combinations_index]

    # Print the result
    print(combinations_random_chosen)

    ######################################################################
    # Randomly Search with Random Forest:

    # Create lists for criterion and max_features
    criterion_list = ['gini', 'entropy']
    max_feature_list = ['auto', 'sqrt', 'log2', None]

    # Create a list of values for the max_depth hyperparameter
    max_depth_list = list(range(3, 56))

    # Combination list
    combinations_list = [list(x) for x in
                         product(criterion_list, max_feature_list,
                                 max_depth_list)]

    # Sample hyperparameter combinations for a random search
    combinations_random_chosen = random.sample(combinations_list, 150)

    # Print the result
    print(combinations_random_chosen)

    ######################################################################
    # Visualizing a Random Search:

    # Confirm how hyperparameter combinations & print
    number_combs = len(combinations_list)
    print(number_combs)

    # Sample and visualise combinations
    for x in [50, 500, 1500]:
        sample_hyperparameters(x)
        visualize_search()

    # Sample all the hyperparameter combinations & visualise
    sample_hyperparameters(number_combs)
    visualize_search()

    ######################################################################
    # The RandomizedSearchCV Object:

    # Create the parameter grid
    param_grid = {'learning_rate': np.linspace(0.1, 2, 150),
                  'min_samples_leaf': list(range(20, 65))}

    # Create a random search object
    random_GBM_class = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(),
        param_distributions=param_grid,
        n_iter=10,
        scoring='accuracy', n_jobs=4, cv=5, refit=True,
        return_train_score=True)

    # Fit to the training data
    random_GBM_class.fit(X_train, y_train)

    # Print the values used for both hyperparameters
    print(random_GBM_class.cv_results_['param_learning_rate'])
    print(random_GBM_class.cv_results_['param_min_samples_leaf'])

    ######################################################################
    # RandomSearchCV in Scikit Learn:

    # Create the parameter grid
    param_grid = {'max_depth': list(range(5, 26)),
                  'max_features': ['auto', 'sqrt']}

    # Create a random search object
    random_rf_class = RandomizedSearchCV(
        estimator=RandomForestClassifier(n_estimators=80),
        param_distributions=param_grid, n_iter=5,
        scoring='roc_auc', n_jobs=4, cv=3, refit=True, return_train_score=True)

    # Fit to the training data
    random_rf_class.fit(X_train, y_train)

    # Print the values used for both hyperparameters
    print(random_rf_class.cv_results_['param_max_depth'])
    print(random_rf_class.cv_results_['param_max_features'])

    ######################################################################
    # Grid and Random Search Side by Side:

    # Sample grid coordinates
    grid_combinations_chosen = combinations_list[0:300]

    # Create a list of sample indexes
    sample_indexes = list(range(0, len(combinations_list)))

    # Randomly sample 300 indexes
    random_indexes = np.random.choice(sample_indexes, 300, replace=False)

    # Use indexes to create random sample
    random_combinations_chosen = [combinations_list[index] for index in
                                  random_indexes]

    # Call the function to produce the visualization
    visualize_search(grid_combinations_chosen, random_combinations_chosen)

    ######################################################################
    # Visualizing Coarse to Fine:

    # Confirm the size of the combinations_list
    print(len(combinations_list))

    # Sort the results_df by accuracy and print the top 10 rows
    print(results_df.sort_values(by='accuracy', ascending=False).head(10))

    # Confirm which hyperparameters were used in this search
    print(results_df.columns)

    # Call visualize_hyperparameter() with each hyperparameter in turn
    visualize_hyperparameter('max_depth')
    visualize_hyperparameter('min_samples_leaf')
    visualize_hyperparameter('learn_rate')

    ######################################################################
    # Coarse to Fine Iterations:

    # Use the provided function to visualize the first results
    visualize_first()

    # Create some combinations lists & combine:
    max_depth_list = list(range(1, 21))
    learn_rate_list = np.linspace(0.001, 1, 50)

    # Call the function to visualize the second results
    visualize_second()

    ######################################################################
    # Bayes Rule in Python:

    # In this exercise you will undertake a practical example of setting up
    # Bayes formula, obtaining new evidence and updating your 'beliefs' in
    # order to get a more accurate result. The example will relate to the
    # likelihood that someone will close their account for your online
    # software product.
    #
    # These are the probabilities we know:
    # - 7% (0.07) of people are likely to close their account next month
    # - 15% (0.15) of people with accounts are unhappy with your product (you
    # don't know who though!)
    # - 35% (0.35) of people who are likely to close their account are unhappy
    # with your product

    # Assign probabilities to variables
    p_unhappy = 0.15
    p_unhappy_close = 0.35

    # Probabiliy someone will close
    p_close = 0.07

    # Probability unhappy person will close
    p_close_unhappy = (p_close * p_unhappy_close) / p_unhappy
    print(p_close_unhappy)

    ######################################################################
    # Bayesian Hyperparameter tuning with Hyperopt:

    # In this example you will set up and run a bayesian hyperparameter
    # optimization process using the package Hyperopt (already imported as hp
    # for you). You will set up the domain (which is similar to setting up the
    # grid for a grid search), then set up the objective function. Finally,
    # you will run the optimizer over 20 iterations.
    #
    # You will need to set up the domain using values:
    # - max_depth using quniform distribution (between 2 and 10, increasing by
    # 2)
    # - learning_rate using uniform distribution (0.001 to 0.9)
    #
    # Note that for the purpose of this exercise, this process was reduced in
    # data sample size and hyperopt & GBM iterations. If you are trying out
    # this method by yourself on your own machine, try a larger search space,
    # more trials, more cvs and a larger dataset size to really see this in
    # action!

    # Set up space dictionary with specified hyperparameters
    space = {'max_depth': hp.quniform('max_depth', 2, 10, 2),
             'learning_rate': hp.uniform('learning_rate', 0.001, 0.9)}

    # Set up objective function
    def objective(params):
        params = {'max_depth': int(params['max_depth']),
                  'learning_rate': params['learning_rate']}
        gbm_clf = GradientBoostingClassifier(n_estimators=100, **params)
        best_score = cross_val_score(gbm_clf, X_train, y_train,
                                     scoring='accuracy', cv=2, n_jobs=4).mean()
        loss = 1 - best_score
        return loss


    # Run the algorithm
    best = fmin(fn=objective, space=space, max_evals=20,
                rstate=np.random.RandomState(42), algo=tpe.suggest)
    print(best)

    ######################################################################
    # Genetic Hyperparameter Tuning with TPOT:

    # You're going to undertake a simple example of genetic hyperparameter
    # tuning. TPOT is a very powerful library that has a lot of features.
    # You're just scratching the surface in this lesson, but you are highly
    # encouraged to explore in your own time.
    #
    # This is a very small example. In real life, TPOT is designed to be run
    # for many hours to find the best model. You would have a much larger
    # population and offspring size as well as hundreds more generations to
    # find a good model.
    #
    # You will create the estimator, fit the estimator to the training data
    # and then score this on the test data.
    #
    # For this example we wish to use:
    # - 3 generations
    # - 4 in the population size
    # - 3 offspring in each generation
    # - accuracy for scoring
    #
    # A random_state of 2 has been set for consistency of results.

    # Assign the values outlined to the inputs
    number_generations = 3
    population_size = 4
    offspring_size = 3
    scoring_function = 'accuracy'

    # Create the tpot classifier
    tpot_clf = TPOTClassifier(generations=number_generations,
                              population_size=population_size,
                              offspring_size=offspring_size,
                              scoring=scoring_function,
                              verbosity=2, random_state=2, cv=2)

    # Fit the classifier to the training data
    tpot_clf.fit(X_train, y_train)

    # Score on the test set
    print(tpot_clf.score(X_test, y_test))

    ######################################################################
    # Analysing TPOT's stability:

    # You will now see the random nature of TPOT by constructing the
    # classifier with different random states and seeing what model is found
    # to be best by the algorithm. This assists to see that TPOT is quite
    # unstable when not run for a reasonable amount of time.

    # Create the tpot classifier
    tpot_clf = TPOTClassifier(generations=2, population_size=4,
                              offspring_size=3, scoring='accuracy', cv=2,
                              verbosity=2, random_state=42)

    # Fit the classifier to the training data
    tpot_clf.fit(X_train, y_train)

    # Score on the test set
    print(tpot_clf.score(X_test, y_test))

    ######################################################################

    # Create the tpot classifier
    tpot_clf = TPOTClassifier(generations=2, population_size=4,
                              offspring_size=3, scoring='accuracy', cv=2,
                              verbosity=2, random_state=122)

    # Fit the classifier to the training data
    tpot_clf.fit(X_train, y_train)

    # Score on the test set
    print(tpot_clf.score(X_test, y_test))

    ######################################################################

    # Create the tpot classifier
    tpot_clf = TPOTClassifier(generations=2, population_size=4,
                              offspring_size=3, scoring='accuracy', cv=2,
                              verbosity=2, random_state=99)

    # Fit the classifier to the training data
    tpot_clf.fit(X_train, y_train)

    # Score on the test set
    print(tpot_clf.score(X_test, y_test))

    ######################################################################
