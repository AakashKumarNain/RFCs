# Metrics

This is a sample template for the `metrics` subpackage. Any PR submitted for adding a new metric to the TF-addons ecosystem must ensure that it covers all the major points listed here.


### Case I: Evaluate results for a given set of `y_true` and `y_pred` tensors
If you are given a set of `predictions` and the corresponding `ground-truth`, then the end-user should be able to create an instance of the metric and call the instance with the given set to evaluate the quality of predictions. For example, if a PR implements `my_metric`, and you have two tensors `y_pred` and `y_true`, then the end-user should be able to call the metric on this set in the following way:

```python

y_pred = [...]   # tensor representing the predicted values
y_true = [...]   # tensor representing the corresponding ground-truth

m = my_metric(..)
print("Results: ", m(y_true, y_pred).numpy())
```

**Note**: The tensor can be a single example or it can represent a batch.


### Case II: Classification/Regression moels, etc.
Different metrics have different use cases depending on the problem set. If the metric being implemented is valid for more than one scenario, then we suggest splitting the `PR` into multiple small `PRs`. For example, `cross-entropy` implemented as `binary_crossentropy` and `categorical_crossentropy`. 

We are providing a simple example for the same if the above scenario applies to the functionality you are contributing to.
(Please note that this is just a sample and can differ from metric to metric.)

1. **Binary classification**: should work with or without `One-hot encoded labels`

```python

# with no OHE
y_pred = [[0.7], [0.5], [0.3]]   
y_true = [[0.], [1], [0]]

m = my_metric(..)
print("Results: ", m(y_true, y_pred).numpy())

# with OHE
y_pred = [[0.7, 0.3], [0.6, 0.4], [0.2, 0.8]]   
y_true = [[1, 0], [0, 1], [1, 0]]

m = my_metric(..)
print("Results: ", m(y_true, y_pred).numpy())
```


2. **Multiclass-classification**: should work with `One-hot encoded` or `sparse` labels

```python

# with OHE
y_pred = [[0.7, 0.2, 0.1], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]   
y_true = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

m = my_metric(..)
print("Results: ", m(y_true, y_pred).numpy())

# with sparse labels
y_pred = [[0.7, 0.2, 0.1], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]   
y_true = [[0], [1], [2]]

m = my_metric(..)
print("Results: ", m(y_true, y_pred).numpy())
```
3. **Regression**: (need to discuss any special case if applicable apart from general scenario)

**Note**: The `naming` convention and the `semantics` of the separate implementations for a user should be the same ideally.

### Case III: `model.fit()` with the `Sequential` or the `Model` API

The metric should work with the `Model` and `Sequential` API in Keras. For example:

```python

model = Model(..)

m = my_metric(...)
model.compile(..., mettic=[m])
model.fit(...)
```

For more examples on `metric` in Keras, please check out this [guide](https://keras.io/api/metrics/)


