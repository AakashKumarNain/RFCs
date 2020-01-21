## Cohen's Kappa

We added Cohen’s Kappa in TF-addons a while back. The current [implementation](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/metrics/cohens_kappa.py) accepts a 1-D array of predictions and and 1-D array of true labels, and returns the result.  A typical example looks like this:

```python
actuals = np.array([4, 4, 3, 4, 2, 4, 1, 1], dtype=np.int32)
preds = np.array([4, 4, 3, 4, 4, 2, 1, 1], dtype=np.int32)
m = tfa.metrics.CohenKappa(num_classes=5)
m.update_state(actuals, preds)
print('Final result: ', m.result().numpy()) 

```

## Limitations of the current implementation
The current implementation can't directly be used with `model.fit(..)` because the model always produces `batches` of predictions. Although this can be solved by simply `reshaping`, I found that the problem isn't as straightforward as it seems. 

There are three different sceanrios we have to consider here. Let me try to eleborate on this:

### Case 1: Binary classification problem
In this case, the model would look something like this:
```python

model = Sequential()
model.add(..)
...
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimier='sgd', metrics=['accuracy', 
                                                                    CohensKappa(num_classes=2)])
model.fit(..)

```
In this case

`y_pred` shape: (num_samples, 1)<br>
`y_true` shape: (num_samples, )<br>

We can convert the preedictions into labels inside CohensKappa by simply doing `y_pred = y_pred > 0.5 ` and then calculate the kappa score.
