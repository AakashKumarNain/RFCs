import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric


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
            if round_pred or num_classes==2:
                self.update = self.update_reg_model
            elif sparse_labels:
                self.update = self.update_multi_class_sparse_model
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
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        return self.update(y_true, y_pred, sample_weight)
        
        
    def update_reg_model(self, y_true, y_pred, sample_weight=None):
        print("Regression model")
        y_true = tf.squeeze(tf.cast(y_true, dtype=tf.int64))
        y_pred = tf.squeeze(tf.cast(tf.math.round(y_pred), dtype=tf.int64))


        # compute the new values of the confusion matrix
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype=tf.float32)

        # update the values in the original confusion matrix
        return self.conf_mtx.assign_add(new_conf_mtx)
    
    def update_binary_class_model(self, y_true, y_pred, sample_weight=None):
        print("Binary classification")
        y_true = tf.cast(y_true, dtype=tf.int64)
        y_pred = tf.cast(y_pred > 0.5, dtype=tf.int64)
        
        y_true = tf.reshape(y_true, (y_true.shape[0],))
        y_pred = tf.reshape(y_pred, (y_pred.shape[0],))


        # compute the new values of the confusion matrix
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype=tf.float32)

        # update the values in the original confusion matrix
        return self.conf_mtx.assign_add(new_conf_mtx)

        
    
    def update_multi_class_model(self, y_true, y_pred, sample_weight=None):
        print("Multi-class classification")
        y_true = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.int64)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=tf.int64)


        # compute the new values of the confusion matrix
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype=tf.float32)

        # update the values in the original confusion matrix
        return self.conf_mtx.assign_add(new_conf_mtx)
    
    
    
    def update_multi_class_sparse_model(self, y_true, y_pred, sample_weight=None):
        print("Multi-class sparse classification")
        y_true = tf.cast(y_true, dtype=tf.int64)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.reshape(y_true, (y_true.shape[0],))
        y_pred = tf.reshape(y_pred, (y_pred.shape[0],))

        # compute the new values of the confusion matrix
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype=tf.float32)

        # update the values in the original confusion matrix
        return self.conf_mtx.assign_add(new_conf_mtx)
    
    
    def update_values(self, y_true, y_pred, sample_weight=None):
        print("Normal updates")
        y_true = tf.cast(y_true, dtype=tf.int64)
        y_pred = tf.cast(y_pred, dtype=tf.int64)

        # compute the new values of the confusion matrix
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype=tf.float32)

        # update the values in the original confusion matrix
        return self.conf_mtx.assign_add(new_conf_mtx)
    
    
    def result(self):
        nb_ratings = tf.shape(self.conf_mtx)[0]
        weight_mtx = tf.ones([nb_ratings, nb_ratings], dtype=tf.float32)

        # 2. Create a weight matrix
        if self.weightage is None:
            diagonal = tf.zeros([nb_ratings], dtype=tf.float32)
            weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
        else:
            weight_mtx += tf.cast(tf.range(nb_ratings), dtype=tf.float32)
            weight_mtx = tf.cast(weight_mtx, dtype=self.dtype)

            if self.weightage == 'linear':
                weight_mtx = tf.abs(weight_mtx - tf.transpose(weight_mtx))
            else:
                weight_mtx = tf.pow((weight_mtx - tf.transpose(weight_mtx)), 2)

        weight_mtx = tf.cast(weight_mtx, dtype=self.dtype)

        # 3. Get counts
        actual_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=1)
        pred_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=0)

        # 4. Get the outer product
        out_prod = pred_ratings_hist[..., None] * \
                    actual_ratings_hist[None, ...]

        # 5. Normalize the confusion matrix and outer product
        conf_mtx = self.conf_mtx / tf.reduce_sum(self.conf_mtx)
        out_prod = out_prod / tf.reduce_sum(out_prod)

        conf_mtx = tf.cast(conf_mtx, dtype=self.dtype)
        out_prod = tf.cast(out_prod, dtype=self.dtype)

        # 6. Calculate Kappa score
        numerator = tf.reduce_sum(conf_mtx * weight_mtx)
        denominator = tf.reduce_sum(out_prod * weight_mtx)
        return tf.cond(
            tf.math.is_nan(denominator),
            true_fn=lambda: 0.0,
            false_fn=lambda: 1 - (numerator / denominator))

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_states(self):
        """Resets all of the metric state variables."""

        for v in self.variables:
            K.set_value(
                v,
                np.zeros((self.num_classes, self.num_classes),
                         v.dtype.as_numpy_dtype))
