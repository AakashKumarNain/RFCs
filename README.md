## Cohen's Kappa

We added Cohen’s Kappa in TF-addons a while back. The current [implementation](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/metrics/cohens_kappa.py) accepts a 1-D array of predictions and and 1-D array of true labels, and returns the result.  A typical example looks like this:

```python
actuals = np.array([4, 4, 3, 4, 2, 4, 1, 1], dtype=np.int32)
preds = np.array([4, 4, 3, 4, 4, 2, 1, 1], dtype=np.int32)
m = tfa.metrics.CohenKappa(num_classes=5)
m.update_state(actuals, preds)
print('Final result: ', m.result().numpy()) 

```

