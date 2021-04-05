# 3. Creating the model
When it comes to creating the model, the task is quite simple because we don't have that much data or features compared to other datasets. Our predictions I assume won't have an extremely high accuracy because of that reason, but I still think we can get a reasonably good result. 

## 1.1 Explaining the approach
In this case I really wanted to try using stacked models to make the predictions a little bit more accurate. The models have different ways of analysing data and perhaps blending their output could bring us closer to the correct prediction. Aside of that I'm also going to use cross validation and a mean squared error to evaluate the accuracy of our models.

### 1.1.1 Setting up cross validation
If you haven't come across cross validation or want to refresh your memory, I made a small graph representing how it works so it's more clear.
In our case I'm going to set up a cross validation using the sklearn method with 12 folds and 4 repeats. I think it's a good amount for our model.
![Cross validation](https://i.imgur.com/Uj61DOs.png)

### 1.1.2 Adding models
Before the stacking we need to select appropriate models, I selected 5 of them but in this case using an uneven number is not that important since we're not classifying anything.
When setting up the models I made sure to implement scaling on the data before using it in SVM, Lasso and Ridge. Some of the parameters in selected models got tuned to enhance the performance and deal with overfitting.
- **LGBM** - Light Gradient Boosting is the new, accerelated version of gradient boosting that offers high performance even on big data sets. It's histogram-based and places continuous values into discrete bins, which leads to faster training and more efficient memory usage.
- **SVR** - Support Vector regression is a type of Support vector machine that supports linear and non-linear regression. It tries to place the function as close as possible to our data points.
- **Random Forest Regressor** - Meta estimator. Tries to fit a number of classifying decision trees on various sub-samples of the dataset and uses averaging, improving the predictive accuracy as well as having the overfitting under control.
- **Lasso** - Lasso is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the resulting statistical model.
- **Ridge** - This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization.

## 1.2 Stacking and blending
I used StackingCVRegressor from mlxtend to stack the models together and set LGBM as a meta regressor. For blending I created my own function that takes a little bit from every model and assigns a weight to their prediction based on score of the models that I find out before by using RMSE metric. After scoring the models I fit them with the full data and set up the stack.
```python
# Stacking the models
stack = StackingCVRegressor(regressors=(lgbm, rf, svr, ridge, lasso),
                            meta_regressor=lgbm,
                            use_features_in_secondary=True)
# (...)
                            
# Blending the models together
def blended_predictions(X):
    return ((0.1 * full_ridge.predict(X)) +
            (0.2 * full_lgbm.predict(X)) +
            (0.2 * full_svr.predict(X)) +
            (0.1 * full_lasso.predict(X)) +
            (0.05 * full_gb.predict(X)) +
            (0.35 * stack.predict(np.array(X))))
```

We went trough everything that is important to the model, it's time to train it and [explore the results.](result_analysis.md)
