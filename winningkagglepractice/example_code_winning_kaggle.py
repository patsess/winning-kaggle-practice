
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import itertools
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import (KFold, StratifiedKFold, TimeSeriesSplit,
    train_test_split)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

"""
Note: code is for reference only (taken from an online course)
"""


if __name__ == '__main__':
    # Explore train data:

    # Read train data
    train = pd.read_csv('train.csv')

    # Look at the shape of the data
    print('Train shape:', train.shape)

    # Look at the head() of the data
    print(train.head())

    ######################################################################
    # Explore test data:

    # Read the test data
    test = pd.read_csv('test.csv')
    # Print train and test columns
    print('Train columns:', train.columns.tolist())
    print('Test columns:', test.columns.tolist())

    # Read the sample submission file
    sample_submission = pd.read_csv('sample_submission.csv')

    # Look at the head() of the sample submission
    print(sample_submission.head())

    ######################################################################
    # Train a simple model:

    # Read the train data
    train = pd.read_csv('train.csv')

    # Create a Random Forest object
    rf = RandomForestRegressor()

    # Train a model
    rf.fit(X=train[['store', 'item']], y=train['sales'])

    ######################################################################
    # Prepare a submission:

    # Read test and sample submission data
    test = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')

    # Show the head() of the sample_submission
    print(sample_submission.head())

    # Get predictions for the test set
    test['sales'] = rf.predict(test[['store', 'item']])

    # Write test predictions using the sample_submission format
    test[['id', 'sales']].to_csv('kaggle_submission.csv', index=False)

    ######################################################################
    # Train XGBoost models:

    # Create DMatrix on train data
    dtrain = xgb.DMatrix(data=train[['store', 'item']],
                         label=train['sales'])

    # Define xgboost parameters
    params = {'objective': 'reg:linear',
              'max_depth': 2,
              'silent': 1}

    # Train xgboost model
    xg_depth_2 = xgb.train(params=params, dtrain=dtrain)

    ######################################################################

    # Create DMatrix on train data
    dtrain = xgb.DMatrix(data=train[['store', 'item']],
                         label=train['sales'])

    # Define xgboost parameters
    params = {'objective': 'reg:linear',
              'max_depth': 8,
              'silent': 1}

    # Train xgboost model
    xg_depth_8 = xgb.train(params=params, dtrain=dtrain)

    ######################################################################

    # Create DMatrix on train data
    dtrain = xgb.DMatrix(data=train[['store', 'item']],
                         label=train['sales'])

    # Define xgboost parameters
    params = {'objective': 'reg:linear',
              'max_depth': 15,
              'silent': 1}

    # Train xgboost model
    xg_depth_15 = xgb.train(params=params, dtrain=dtrain)

    ######################################################################
    # Explore overfitting XGBoost:

    dtrain = xgb.DMatrix(data=train[['store', 'item']])
    dtest = xgb.DMatrix(data=test[['store', 'item']])

    # For each of 3 trained models
    for model in [xg_depth_2, xg_depth_8, xg_depth_15]:
        # Make predictions
        train_pred = model.predict(dtrain)
        test_pred = model.predict(dtest)

        # Calculate metrics
        mse_train = mean_squared_error(train['sales'], train_pred)
        mse_test = mean_squared_error(test['sales'], test_pred)
        print(
            'MSE Train: {:.3f}. MSE Test: {:.3f}'.format(mse_train, mse_test))

    ######################################################################
    # Define a competition metric:

    # Define your own MSE function
    def own_mse(y_true, y_pred):
        # Raise differences to the power of 2
        squares = np.power(y_true - y_pred, 2)
        # Find mean over all observations
        err = np.mean(squares)
        return err


    print('Sklearn MSE: {:.5f}. '.format(
        mean_squared_error(y_regression_true, y_regression_pred)))
    print('Your MSE: {:.5f}. '.format(
        own_mse(y_regression_true, y_regression_pred)))

    ######################################################################

    # Define your own LogLoss function
    def own_logloss(y_true, prob_pred):
        # Find loss for each observation
        terms = y_true * np.log(prob_pred) + (1 - y_true) * np.log(
            1 - prob_pred)
        # Find mean over all observations
        err = np.mean(terms)
        return -err


    print('Sklearn LogLoss: {:.5f}'.format(
        log_loss(y_classification_true, y_classification_pred)))
    print('Your LogLoss: {:.5f}'.format(
        own_logloss(y_classification_true, y_classification_pred)))

    ######################################################################
    # EDA statistics:

    # Shapes of train and test data
    print('Train shape:', train.shape)
    print('Test shape:', test.shape)

    # Train head()
    print(train.head())

    # Describe the target variable
    print(train.fare_amount.describe())

    # Train distribution of passengers within rides
    print(train.passenger_count.value_counts())

    ######################################################################
    # EDA plots I:

    # Calculate the ride distance
    train['distance_km'] = haversine_distance(train)

    # Draw a scatterplot
    plt.scatter(x=train['fare_amount'], y=train['distance_km'], alpha=0.5)
    plt.xlabel('Fare amount')
    plt.ylabel('Distance, km')
    plt.title('Fare amount based on the distance')

    # Limit on the distance
    plt.ylim(0, 50)
    plt.show()

    ######################################################################
    # EDA plots II:

    # Create hour feature
    train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
    train['hour'] = train.pickup_datetime.dt.hour

    # Find median fare_amount for each hour
    hour_price = train.groupby('hour', as_index=False)['fare_amount'].median()

    # Plot the line plot
    plt.plot(hour_price['hour'], hour_price['fare_amount'], marker='o')
    plt.xlabel('Hour of the day')
    plt.ylabel('Median fare amount')
    plt.title('Fare amount based on day time')
    plt.xticks(range(24))
    plt.show()

    ######################################################################
    # K-fold cross-validation:

    # Create a KFold object
    kf = KFold(n_splits=3, shuffle=True, random_state=123)

    # Loop through each split
    fold = 0
    for train_index, test_index in kf.split(train):
        # Obtain training and testing folds
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
        print('Fold: {}'.format(fold))
        print('CV train shape: {}'.format(cv_train.shape))
        print('Medium interest listings in CV train: {}\n'.format(
            sum(cv_train.interest_level == 'medium')))
        fold += 1

    ######################################################################
    # Stratified K-fold:

    # Create a StratifiedKFold object
    str_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)

    # Loop through each split
    fold = 0
    for train_index, test_index in str_kf.split(train,
                                                train['interest_level']):
        # Obtain training and testing folds
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
        print('Fold: {}'.format(fold))
        print('CV train shape: {}'.format(cv_train.shape))
        print('Medium interest listings in CV train: {}\n'.format(
            sum(cv_train.interest_level == 'medium')))
        fold += 1

    ######################################################################
    # Time K-fold:

    # Create TimeSeriesSplit object
    time_kfold = TimeSeriesSplit(n_splits=3)

    # Sort train data by date
    train = train.sort_values('date')

    # Iterate through each split
    fold = 0
    for train_index, test_index in time_kfold.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

        print('Fold :', fold)
        print('Train date range: from {} to {}'.format(cv_train.date.min(),
                                                       cv_train.date.max()))
        print('Test date range: from {} to {}\n'.format(cv_test.date.min(),
                                                        cv_test.date.max()))
        fold += 1

    ######################################################################
    # Overall validation score:

    # Sort train data by date
    train = train.sort_values('date')

    # Initialize 3-fold time cross-validation
    kf = TimeSeriesSplit(n_splits=3)

    # Get MSE scores for each cross-validation split
    mse_scores = get_fold_mse(train, kf)

    print('Mean validation MSE: {:.5f}'.format(np.mean(mse_scores)))
    print('MSE by fold: {}'.format(mse_scores))
    print('Overall validation MSE: {:.5f}'.format(
        np.mean(mse_scores) + np.std(mse_scores)))

    ######################################################################
    # Arithmetical features:

    # Look at the initial RMSE
    print('RMSE before feature engineering:', get_kfold_rmse(train))

    # Find the total area of the house
    train['TotalArea'] = train['TotalBsmtSF'] + train['FirstFlrSF'] + train[
        'SecondFlrSF']
    print('RMSE with total area:', get_kfold_rmse(train))

    # Find the area of the garden
    train['GardenArea'] = train['LotArea'] - train['FirstFlrSF']
    print('RMSE with garden area:', get_kfold_rmse(train))

    # Find total number of bathrooms
    train['TotalBath'] = train['FullBath'] + train['HalfBath']
    print('RMSE with number of bathrooms:', get_kfold_rmse(train))

    ######################################################################
    # Date features:

    # Concatenate train and test together
    taxi = pd.concat([train, test])

    # Convert pickup date to datetime object
    taxi['pickup_datetime'] = pd.to_datetime(taxi['pickup_datetime'])

    # Create a day of week feature
    taxi['dayofweek'] = taxi['pickup_datetime'].dt.dayofweek

    # Create an hour feature
    taxi['hour'] = taxi['pickup_datetime'].dt.hour

    # Split back into train and test
    new_train = taxi[taxi['id'].isin(train['id'])]
    new_test = taxi[taxi['id'].isin(test['id'])]

    ######################################################################
    # Label encoding:

    # Concatenate train and test together
    houses = pd.concat([train, test])

    le = LabelEncoder()

    # Create new features
    houses['RoofStyle_enc'] = le.fit_transform(houses['RoofStyle'])
    houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

    # Look at new features
    print(houses[['RoofStyle', 'RoofStyle_enc', 'CentralAir',
                  'CentralAir_enc']].head())

    ######################################################################
    # One-Hot encoding:

    # Concatenate train and test together
    houses = pd.concat([train, test])

    # Look at feature distributions
    print(houses['RoofStyle'].value_counts(), '\n')
    print(houses['CentralAir'].value_counts())

    # Concatenate train and test together
    houses = pd.concat([train, test])

    le = LabelEncoder()
    houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

    # Create One-Hot encoded features
    ohe = pd.get_dummies(houses['RoofStyle'], prefix='RoofStyle')

    # Concatenate OHE features to houses
    houses = pd.concat([houses, ohe], axis=1)

    # Look at OHE features
    print(
        houses[[col for col in houses.columns if 'RoofStyle' in col]].head(3))

    ######################################################################
    # Mean target encoding:

    def test_mean_target_encoding(train, test, target, categorical, alpha=5):
        # Calculate global mean on the train data
        global_mean = train[target].mean()

        # Group by the categorical feature and calculate its properties
        train_groups = train.groupby(categorical)
        category_sum = train_groups[target].sum()
        category_size = train_groups.size()

        # Calculate smoothed mean target statistics
        train_statistics = (category_sum + global_mean * alpha) / (
                    category_size + alpha)

        # Apply statistics to the test data and fill new categories
        test_feature = test[categorical].map(train_statistics).fillna(
            global_mean)
        return test_feature.values

    ######################################################################

    def train_mean_target_encoding(train, target, categorical, alpha=5):
        # Create 5-fold cross-validation
        kf = KFold(n_splits=5, random_state=123, shuffle=True)
        train_feature = pd.Series(index=train.index)

        # For each folds split
        for train_index, test_index in kf.split(train):
            cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

            # Calculate out-of-fold statistics and apply to cv_test
            cv_test_feature = test_mean_target_encoding(cv_train, cv_test,
                                                        target, categorical,
                                                        alpha)

            # Save new feature for this particular fold
            train_feature.iloc[test_index] = cv_test_feature
        return train_feature.values

    ######################################################################

    def mean_target_encoding(train, test, target, categorical, alpha=5):

        # Get the train feature
        train_feature = train_mean_target_encoding(train, target, categorical,
                                                   alpha)

        # Get the test feature
        test_feature = test_mean_target_encoding(train, test, target,
                                                 categorical, alpha)

        # Return new features to add to the model
        return train_feature, test_feature

    ######################################################################
    # K-fold cross-validation:

    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)

    # For each folds split
    for train_index, test_index in kf.split(bryant_shots):
        cv_train, cv_test = bryant_shots.iloc[train_index], bryant_shots.iloc[
            test_index]

        # Create mean target encoded feature
        cv_train['game_id_enc'], cv_test['game_id_enc'] = mean_target_encoding(
            train=cv_train,
            test=cv_test,
            target='shot_made_flag',
            categorical='game_id',
            alpha=5)
        # Look at the encoding
        print(
            cv_train[['game_id', 'shot_made_flag', 'game_id_enc']].sample(n=1))

    ######################################################################
    # Beyond binary classification:

    # Create mean target encoded feature
    train['RoofStyle_enc'], test['RoofStyle_enc'] = mean_target_encoding(
        train=train,
        test=test,
        target='SalePrice',
        categorical='RoofStyle',
        alpha=10)

    # Look at the encoding
    print(test[['RoofStyle', 'RoofStyle_enc']].drop_duplicates())

    ######################################################################
    # Find missing data:

    # Read DataFrame
    twosigma = pd.read_csv('twosigma_train.csv')

    # Find the number of missing values in each column
    print(twosigma.isnull().sum())

    # Look at the columns with the missing values
    print(twosigma[['building_id', 'price']].head())

    ######################################################################
    # Impute missing data:

    # Create mean imputer
    mean_imputer = SimpleImputer(strategy='mean')

    # Price imputation
    rental_listings[['price']] = mean_imputer.fit_transform(
        rental_listings[['price']])

    # Create constant imputer
    constant_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')

    # building_id imputation
    rental_listings[['building_id']] = constant_imputer.fit_transform(
        rental_listings[['building_id']])

    ######################################################################
    # Replicate validation score:

    # Calculate the mean fare_amount on the validation_train data
    naive_prediction = np.mean(validation_train['fare_amount'])

    # Assign naive prediction to all the holdout observations
    validation_test['pred'] = naive_prediction

    # Measure the local RMSE
    rmse = sqrt(mean_squared_error(validation_test['fare_amount'],
                                   validation_test['pred']))
    print('Validation RMSE for Baseline I model: {:.3f}'.format(rmse))

    ######################################################################
    # Baseline based on the date:

    # Get pickup hour from the pickup_datetime column
    train['hour'] = train['pickup_datetime'].dt.hour
    test['hour'] = test['pickup_datetime'].dt.hour

    # Calculate average fare_amount grouped by pickup hour
    hour_groups = train.groupby('hour')['fare_amount'].mean()

    # Make predictions on the test set
    test['fare_amount'] = test.hour.map(hour_groups)

    # Write predictions
    test[['id', 'fare_amount']].to_csv('hour_mean_sub.csv', index=False)

    ######################################################################
    # Baseline based on the gradient boosting:

    # Select only numeric features
    features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                'dropoff_latitude', 'passenger_count', 'hour']

    # Train a Random Forest model
    rf = RandomForestRegressor()
    rf.fit(train[features], train.fare_amount)

    # Make predictions on the test data
    test['fare_amount'] = rf.predict(test[features])

    # Write predictions
    test[['id', 'fare_amount']].to_csv('rf_sub.csv', index=False)

    ######################################################################
    # Grid search:

    # Possible max depth values
    max_depth_grid = [3, 6, 9, 12, 15]
    results = {}

    # For each value in the grid
    for max_depth_candidate in max_depth_grid:
        # Specify parameters for the model
        params = {'max_depth': max_depth_candidate}

        # Calculate validation score for a particular hyperparameter
        validation_score = get_cv_score(train, params)

        # Save the results for each max depth value
        results[max_depth_candidate] = validation_score
    print(results)

    ######################################################################
    # 2D grid search:

    # Hyperparameter grids
    max_depth_grid = [3, 5, 7]
    subsample_grid = [0.8, 0.9, 1.0]
    results = {}

    # For each couple in the grid
    for max_depth_candidate, subsample_candidate in itertools.product(
            max_depth_grid, subsample_grid):
        params = {'max_depth': max_depth_candidate,
                  'subsample': subsample_candidate}
        validation_score = get_cv_score(train, params)
        # Save the results for each couple
        results[(max_depth_candidate, subsample_candidate)] = validation_score
    print(results)

    ######################################################################
    # Model blending:

    # Train a Gradient Boosting model
    gb = GradientBoostingRegressor().fit(train[features], train.fare_amount)

    # Train a Random Forest model
    rf = RandomForestRegressor().fit(train[features], train.fare_amount)

    # Make predictions on the test data
    test['gb_pred'] = gb.predict(test[features])
    test['rf_pred'] = rf.predict(test[features])

    # Find mean of model predictions
    test['blend'] = (test['gb_pred'] + test['rf_pred']) / 2
    print(test[['gb_pred', 'rf_pred', 'blend']].head(3))

    ######################################################################
    # Model stacking I:

    # Split train data into two parts
    part_1, part_2 = train_test_split(train, test_size=0.5, random_state=123)

    # Train a Gradient Boosting model on Part 1
    gb = GradientBoostingRegressor().fit(part_1[features], part_1.fare_amount)

    # Train a Random Forest model on Part 1
    rf = RandomForestRegressor().fit(part_1[features], part_1.fare_amount)

    # Make predictions on the Part 2 data
    part_2['gb_pred'] = gb.predict(part_2[features])
    part_2['rf_pred'] = rf.predict(part_2[features])

    # Make predictions on the test data
    test['gb_pred'] = gb.predict(test[features])
    test['rf_pred'] = rf.predict(test[features])

    ######################################################################
    # Model stacking II:

    # Create linear regression model without the intercept
    lr = LinearRegression(fit_intercept=False)

    # Train 2nd level model on the Part 2 data
    lr.fit(part_2[['gb_pred', 'rf_pred']], part_2.fare_amount)

    # Make stacking predictions on the test data
    test['stacking'] = lr.predict(test[['gb_pred', 'rf_pred']])

    # Look at the model coefficients
    print(lr.coef_)

    ######################################################################
    # Testing Kaggle forum ideas:

    # Delete passenger_count column
    new_train_1 = train.drop('passenger_count', axis=1)

    # Compare validation scores
    initial_score = get_cv_score(train)
    new_score = get_cv_score(new_train_1)

    print('Initial score is {} and the new score is {}'.format(initial_score,
                                                               new_score))

    ######################################################################

    # Create copy of the initial train DataFrame
    new_train_2 = train.copy()

    # Find sum of pickup latitude and ride distance
    new_train_2['weird_feature'] = (
        new_train_2.pickup_latitude + new_train_2.distance_km)

    # Compare validation scores
    initial_score = get_cv_score(train)
    new_score = get_cv_score(new_train_2)

    print('Initial score is {} and the new score is {}'.format(initial_score,
                                                               new_score))

    ######################################################################
