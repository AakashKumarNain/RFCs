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
    """Computes Kappa score between two raters.
    The score lies in the range [-1, 1]. A score of -1 represents
    complete disagreement between two raters whereas a score of 1
    represents complete agreement between the two raters.
    A score of 0 means agreement by chance.
    Note: As of now, this implementation considers all labels
    while calculating the Cohen's Kappa score.
    
    Usage:
    ```python
    actuals = np.array([4, 4, 3, 4, 2, 4, 1, 1], dtype=np.int32)
    preds = np.array([4, 4, 3, 4, 4, 2, 1, 1], dtype=np.int32)
    weights = np.array([1, 1, 2, 5, 10, 2, 3, 3], dtype=np.int32)
    
    m = tfa.metrics.CohenKappa(num_classes=5)
    m.update_state(actuals, preds)
    print('Final result: ', m.result().numpy()) # Result: 0.61904764
    
    # To use this with weights, sample_weight argument can be used.
    m = tfa.metrics.CohenKappa(num_classes=5)
    m.update_state(actuals, preds, sample_weight=weights)
    print('Final result: ', m.result().numpy()) # Result: 0.37209308
    ```
    Usage with tf.keras API:
    ```python
    model = tf.keras.models.Model(inputs, outputs)
    model.add_metric(tfa.metrics.CohenKappa(num_classes=5, 
                                            from_activations=True)(outputs))
    model.compile('sgd', loss='mse')
    ```
    
    """
    def __init__(self,
                 num_classes,
                 weightage=None,
                 round_pred=False,
                 sparse_labels=False,
                 from_activations=False,
                 name='cohen_kappa',
                 dtype=None):
        """Creates a `CohenKappa` instance.
        Args:
          num_classes: Number of unique classes in your dataset.
          weightage: (Optional) Weighting to be considered for calculating
            kappa statistics. A valid value is one of
            [None, 'linear', 'quadratic']. Defaults to `None`.
          round_pred: (bool) If True that means regression model is used
            and we need to round the predictions. Defualts to False
          sparse_lables: (bool) Applicable when we are dealing with more
            than two classes. If True, we expect integer labels and for 
            each label we expect #classes entries in the prediction vector
          from_activations: (bool) If True, then we expect predictions from the
            model otherwise we expect both labels and predictions as integers.
            Defaults to False.
          name: (Optional) String name of the metric instance.
          dtype: (Optional) Data type of the metric result.
            Defaults to `None`.
        Raises:
          ValueError: If the value passed for `weightage` is invalid
            i.e. not any one of [None, 'linear', 'quadratic']
        """
        super().__init__(name=name, dtype=dtype)
        
        if from_activations:
            if round_pred:
                self.update = self.update_reg_model
            elif sparse_labels:
                self.update = self.update_multi_class_sparse_model
            else:
                if num_classes==2:
                    self.update = self.update_binary_class_model
                else:
                    self.update = self.update_multi_class_model
        else:
            self.update = self.update_values
        
            
        if weightage not in (None, 'linear', 'quadratic'):
            raise ValueError("Unknown kappa weighting type.")

        self.weightage = weightage
        self.num_classes = num_classes
        self.conf_mtx = self.add_weight(
            'conf_mtx',
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=tf.float32)
                 

```

### Questions:
If you have any questions/concerns, please feel free to drop a comment in the `cohens_kappa.py` file.
