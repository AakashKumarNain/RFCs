## Integrating Focal loss from TFaddons

### Describe the feature and the current behavior/state.
Focal loss has been [implemented](https://github.com/tensorflow/models/blob/3b13794fdb762517ad7c90361b6a47b4f46acc76/official/vision/detection/modeling/losses.py#L25) and used in the RetinaNet model in the `model garden`. There is an existing implementation of [focal loss](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/focal_loss.py) in `addons` as well. 

Having multiple implementations for the same functionality creates redundancy in the ecosystem which creates confusion for the end users. In order to get rid of this redundancy, we propose to use the focal loss from addons. 

### Challenges
None. The code has been tested on CPU/GPU but not on TPU as we don't have a way to test functionlities on TPU as of now but that shouldn't be a problem because all the ops used are TPU compatible as well. 

### Relevant information

    Are you willing to contribute it (yes/no): Yes
    Are you willing to maintain it going forward? (yes/no): Yes

