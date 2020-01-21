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
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy', 
                                                                    CohensKappa(num_classes=2)])
model.fit(..)

```
In this case

`y_pred` shape: (num_samples, 1)<br>
`y_true` shape: (num_samples, )<br>

The call to `update_state()` in this scenario should do something like this:<br>
```python

def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, dtype=tf.int64)
    y_pred = tf.cast(y_pred >= 0.5, dtype=tf.int64)

    # compute the new values of the confusion matrix
    new_conf_mtx = tf.math.confusion_matrix(
        labels=y_true,
        predictions=y_pred,
        num_classes=self.num_classes,
        weights=sample_weight,
        dtype=tf.float32)

    # update the values in the original confusion matrix
    return self.conf_mtx.assign_add(new_conf_mtx)

```

### Case 2: Multiclass Classification problem
In this case, the model would look something like this:
```python

model = Sequential()
model.add(..)
...
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', 
                                                                    CohensKappa(num_classes=num_classes)])
model.fit(..)

```
In this case

`y_pred` shape: (num_samples, num_classes)<br>
`y_true` shape: (num_samples, num_classes) if `onr-hot encoded` or (num_samples,) if `sparse labels` are used

The call to `update_state()` in this scenario should do something like this:<br>
```python

def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int64)
    
    ###### Checks ################
    #
    # We need some checks here depending on whether the true labels are one-hote encoded 
    # or are just sparse. Until unless we do this check we can't get the right shape
    # for y_true and y_pred
    #
    ################################

    # compute the new values of the confusion matrix
    new_conf_mtx = tf.math.confusion_matrix(
        labels=y_true,
        predictions=y_pred,
        num_classes=self.num_classes,
        weights=sample_weight,
        dtype=tf.float32)

    # update the values in the original confusion matrix
    return self.conf_mtx.assign_add(new_conf_mtx)

```

### Case3: 
This is an odd case. In many scenarios, especially on Kaggle competitions, it has been found that it is better to do regression for predicting labels instead of treating it as a classification problem. For example, if there are 5 classes, then instead of treating it as a mutli-class classificatiion, we treat it as a regression problem where the predictions are either rounded off for the nearest integer label or a different threshold is chosen for each class. 

We don't want to complicate the API design. Hence we will only consider the scenario where will round off the predictions to the nearest integer label, calculate the kappa score and return it. 

In this case, the model would look something like this:
```python

model = Sequential()
model.add(..)
...
model.add(Dense(1))
model.compile(loss='mse', optimier='sgd', metrics=[CohensKappa(num_classes=num_classes)])
model.fit(..)

```
In this case

`y_pred` shape: (num_samples, 1)<br>
`y_true` shape: (num_samples, )<br>

The call to `update_state()` in this scenario should do something like this:<br>
```python

def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, dtype=tf.int64)
    
    # Round off the predictions to predict the label
    y_pred = tf.cast(tf.round(y_pred), dtype=tf.int64)

    # compute the new values of the confusion matrix
    new_conf_mtx = tf.math.confusion_matrix(
        labels=y_true,
        predictions=y_pred,
        num_classes=self.num_classes,
        weights=sample_weight,
        dtype=tf.float32)

    # update the values in the original confusion matrix
    return self.conf_mtx.assign_add(new_conf_mtx)

```


## Suggested Changes
In order to incorporate all the three scenarios, I propose the following signature for the constructor:<br>
```python

class CohenKappa(Metric):
    def __init__(self,
                 num_classes,
                 weightage=None,
                 round_pred=False,
                 sparse_labels=False,
                 name='cohen_kappa',
                 dtype=None):
        """Creates a `CohenKappa` instance.
        Args:
          num_classes: Number of unique classes in your dataset.
          weightage: (Optional) Weighting to be considered for calculating
            kappa statistics. A valid value is one of
            [None, 'linear', 'quadratic']. Defaults to `None`.
          round_pred: (bool) If set to true that means regression model is used
            and we need to round the predictions. Defualts to False
          sparse_lables: (bool) If truue, that means we are dealing with a
            multi-class classification problem but the labes aren't OHE
          name: (Optional) String name of the metric instance.
          dtype: (Optional) Data type of the metric result.
            Defaults to `None`.
        Raises:
          ValueError: If the value passed for `weightage` is invalid
            i.e. not any one of [None, 'linear', 'quadratic']
        """
        
        if round_pred:
            self.update_state = update_reg_model
        
        if not round_pred and num_classes == 2:
            self.update_state = update_binary_class_model
            
        if not round_pred and sparse_labels and num_classes > 2:
            self.update_state = update_multi_class_model
            
        def update_reg_model(self, y_true, y_pred, sample_weight=None):
            ...
            
        def update_binary_class_model(self, y_true, y_pred, sample_weight=None):
            ...
            
        def update_multi_class_model(self, y_true, y_pred, sample_weight=None):
             ...
        
            

```

### Questions:
Feel free to drop any concerns/suggestions here
