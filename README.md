# Shufflenet Tensorflow
WIP: Shufflenet implementation in tensorflow based on https://arxiv.org/abs/1707.01083

Using pre-trained model from [tensorpack model zoo](http://models.tensorpack.com/) ([ShuffleNetV1-1x-g=8.npz](http://models.tensorpack.com/ImageNetModels/ShuffleNetV1-1x-g=8.npz)). This model utilizes `g=8` and has a BNReLU after the first `conv2d` (Conv1) in Stage1, not mentioned in the paper but can be found by looking at the entries in the model.

TODO: Optimize `self.batch_normalization` since this is taking `~100ms` to compute.
TODO: Current FLOPS at 145.25M. Reduce to ~140.0M.

