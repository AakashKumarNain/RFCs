
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
        
            
