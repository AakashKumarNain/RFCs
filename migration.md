### Describe the feature and the current behavior/state.

Some of the functionalities present in `addons` are duplicated in the `model garden` because of certain issues like
lack of `TPU` specific test cases, custom `cuda` ops, etc. This redundancy creates a confusion for the end user. 

In order toimporove the end user experience, we aim to unify these functionalities. In short, we want that if a functionlity is present
in `addons` and there are no issues with its usage, then this functionlaity should directly be imported and used in
the `model garden`. If some functionality can't be used directly, we look forward to discuss to overcome the constraints on the `model garden` side for a smoother migration experience in the future.

For now, we will focus on the following issues:
* Integrating [focal loss](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/focal_loss.py)
* Integration of the [GELU](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/gelu.py) activation
* Optimizers
* A minimal public facing CI for `model garden`


#### Challenges
1. Currently, we support `python` implementations for all the functionalities in `addons`. For some functionalities, especailly for the `activations` where a `cuda` implementation can provide speed gains, we also have `custom-op` implementations. These implementations have been tested for both `CPU` and `GPU` but not for `TPUs`. This can be problematic in certain scenarios.
To address this, we propose the following solutions for now:
    * Rely only on pure python implementations as it would be independent of the hardware context
    * Rely on the context of excecuting device and fallback to the corresposnding implementation

2. Using `Optimizers` directly from addons isn't feasible for now as the implementations on both the sides var a lot. Having two different implementations for the same `optimizer` can create a lot of confusion to the end user. Also, the `optimizer` API is going to be `revampled`, so we don't want to refactor things till the time we have a `RFC` for a new and stabel API from the tensorflow team. To avoid confusion, we can make the `optimizers` implementation in `model garden` private and provide a link to the end user, pointing to `addons` implementation for general purpose usage.

3. 






Relevant information

    Are you willing to contribute it (yes/no): Yes, some of it
    Are you willing to maintain it going forward? (yes/no): yes, some of it
    Is there a relevant academic paper? (if so, where): no LOL
    Is there already an implementation in another framework? (if so, where): pytorch
    Was it part of tf.contrib? (if so, where): No
