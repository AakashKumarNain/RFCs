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
