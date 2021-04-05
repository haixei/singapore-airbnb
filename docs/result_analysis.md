# 4. Result Analysis
All left there is to do is training the model and comparing the results. I will run the model two times, first on the data with keywords and then without them.
I divided the data into variables already.
```python
# Create variables
X = data.drop(['price'], axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## 4.1 Individual model scores
Before runing the whole model, I'd like to show you how the accuracy looks like in the singular parts and how it contributed to the weights in blending.
```python
# (...)      

# Test the models accuracy     
def score_models(models):
    scores = {'models': [], 'accuracy': []}
    for key, value in models.items():
        print('Scorring ' + key + '...')
        score = cv_rmse(value, X_train, y_train)
        print(key + ':', round(score.mean(), 4), round(score.std(), 4))
        # Add to the dictionary
        scores['models'].append(key)
        scores['accuracy'].append(value)

    return scores


# Save the output
scores = score_models(all_models)
```
#### Individual model scores
| Model                | Random Forest | Lasso  | Light GBM | SVR    | Ridge |
|----------------------|---------------|--------|-----------|--------|-------|
| Score (Mean of MSE) | 0.2052        | 0.2774 | 0.2346     | 0.2427 | 0.2774|
| Blending Weight      | 0.2           | 0.05   | 0.2       | 0.1    | 0.1   |

## 4.2 Blended models score
- MSE score on **train data: 0.1576**
- MSE score on **test data: 0.2187**

As we can see blending the models works very well and produces a satisfying result.
## 4.3 Comparison of two data sets
Like I said in feature engineering, the data extracted for the names might not be enough to make enough of a difference and it's proved right by running the model without the keywords. The difference between the best performing model that is Random Forest is equal ~0.015, the situation looks similar in other cases as well. I assume a more in depth sentiment analysis could bring a little bit of a better result, but I still don't think it could add that much to the model. Too much of it depends on things that we already know about the listing. I wish more fatures could be added from the Airbnb's side like if the bathroom is shared or more characteristics of a flat in general. I feel like that might be missing part. That being said, the output is still good enough, in both cases we achieve MAE of ~0.15 on the train data.
