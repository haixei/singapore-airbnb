from feature_eng import data, data_without_keywords
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from mlxtend.regressor import StackingCVRegressor
import numpy as np
import plotly.graph_objects as go

# Create variables
X = data_without_keywords.drop(['price'], axis=1)
y = data_without_keywords['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a repeated cross validation for a better result
cv = RepeatedKFold(n_splits=12, n_repeats=4, random_state=24)

# Define the models
lgbm = LGBMRegressor(
        num_leaves=6,
        learning_rate=0.01,
        n_estimators=1000,
        max_bin=250,
        bagging_fraction=0.8,
        bagging_freq=4,
        bagging_seed=8,
        feature_fraction=0.3,
        feature_fraction_seed=8,
        min_sum_hessian_in_leaf=14
)

svr = make_pipeline(MinMaxScaler(), SVR(kernel='rbf', epsilon=0.008))

rf = RandomForestRegressor(max_depth=15,
                           min_samples_split=5,
                           min_samples_leaf=5,
                           max_features=None,
                           oob_score=True)

ridge_alphas = [1e-3, 1e-2, 1e-1, 1, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, normalize=True, cv=cv))

lasso_alphas = np.logspace(-10, 0.1, 140)
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=lasso_alphas, normalize=True, tol=0.1, cv=cv))

# Stacking the models
stack = StackingCVRegressor(regressors=(lgbm, rf, svr, ridge, lasso),
                            meta_regressor=lgbm,
                            use_features_in_secondary=True)


# Define a scoring system
def mse(y, y_pred):
    return mean_squared_error(y, y_pred)


def cv_mse(model, X, y):
    rmse = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    return rmse


# Test the models accuracy
all_models = {'Random Forest': rf,
              'Lasso': lasso,
              'Light GBM': lgbm,
              'SVR': svr,
              'Ridge': ridge}

def score_models(models):
    scores = {'models': [], 'accuracy': []}
    for key, value in models.items():
        print('Scorring ' + key + '...')
        score = cv_mse(value, X_train, y_train)
        print(key + ':', round(score.mean(), 4), round(score.std(), 4))
        # Add to the dictionary
        scores['models'].append(key)
        scores['accuracy'].append(round(score.mean(), 4))

    return scores


# Save the output
scores = score_models(all_models)

# Fit the models with full data
stack_gen_model = stack.fit(np.array(X_train), np.array(y_train))
full_lgbm = lgbm.fit(X_train, y_train)
full_lasso = lasso.fit(X_train, y_train)
full_svr = svr.fit(X_train, y_train)
full_ridge = ridge.fit(X_train, y_train)
full_rf = rf.fit(X_train, y_train)

# Blend the models together
def blended_predictions(X):
    return ((0.1 * full_ridge.predict(X)) +
            (0.2 * full_lgbm.predict(X)) +
            (0.1 * full_svr.predict(X)) +
            (0.05 * full_lasso.predict(X)) +
            (0.2 * rf.predict(X)) +
            (0.35 * stack.predict(np.array(X))))


# Get final precitions from the blended model
train_score = mse(y_train, blended_predictions(X_train))
print('RMSLE score on train data:', train_score)

# Test the model
test_score = mse(y_test, blended_predictions(X_test))
print('MSE score on test data:', test_score)

# Add result to the scores
scores['models'].append('blended')
scores['accuracy'].append(train_score)

# Create the graph for models
model_acc_fig = go.Figure(data=go.Scatter(x=scores['models'], y=scores['accuracy']))
# >> model_acc_fig.show()
