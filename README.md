# Shufflenet Tensorflow
WIP: Shufflenet implementation in tensorflow based on https://arxiv.org/abs/1707.01083

Using trained model from tensorpack model zoo ([ShuffleNetV1-1x-g=8.npz](http://models.tensorpack.com/ImageNetModels/ShuffleNetV1-1x-g=8.npz)). This model utilizes `g=8` and has a BNReLU after the first `conv2d` (Conv1) in Stage1.
