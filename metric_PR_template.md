# Metrics

Lately, we have been obserbing a lot of issues that are common across most of the implementations in different subpackages for TF-addons. Although the issues are common most of the time, we need to have a template for each subpackage to make our lives when it comes to track issues of specific kind. 

This is a sample template fo r the `metrics` subpackage. Any PR submitted for ading a new metric to the TF-addons ecosystem, must ensure that it covers all the major points listed here.


### Case I: Evaluate results for a given set of `y_true` and `y_pred` tensors
Any metric should evaluate the quality of `predictions` for a given `ground-truth`. This is the simplest use case for any metric. For example, if I a PR impements `my_metric`, then it should evaluate the following examples:

```python

y_pred = [...]   # tensor representing the predicted values
y_true = [...]   # tensor representing the corresponding ground-truth

m = my_metic(y_true, y_pred)
print("Results: ", m.numpy())
```

**Note**: The tensor can be a single example or a batch.


### Case II: Classification/Regression moels, etc.

The following cases needed to be handled (applicable as per the metric usage):

1. **Binary classification**: should work with/without OHE

```python

# with no OHE
y_pred = [[0.7], [0.5], [0.3]]   
y_true = [[0.], [1], [0]]

# with OHE
y_pred = [[0.7, 0.3], [0.6, 0.4], [0.2, 0.8]]   
y_true = [[1, 0], [0, 1], [0, 1]]

m = my_metic(y_true, y_pred)
print("Results: ", m.numpy())
```


2. **Multiclass-classification**: should work with OHE/sparse labels

```python

# with OHE
y_pred = [[0.7, 0.2, 0.1], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]   
y_true = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# with sparse labels
y_pred = [[0.7, 0.2, 0.1], [0.5, 0.2, 0.3], [0.2, 0.3, 0.5]]   
y_true = [[0], [1], [2]]


m = my_metic(y_true, y_pred)
print("Results: ", m.numpy())
```
3. **Regression**: (need to discuss any special case if applicable apart from general sceanrio)

If `my_metric`, is handling all these scenarios, then a better way to keep the code short, clean and easy to track is to split the metric implementation for different use cases in separate PRS with specific implementation. 

**Note**: The `naming` convention and the semantics of the separate implementations for a user should be the same ideally.

### Case III: model.fit() with Sequential/Model API and custom training/evalaution loops 

