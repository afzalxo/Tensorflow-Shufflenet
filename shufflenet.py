import tensorflow as tf
import numpy as np
import time

SHUFFLENET_MEAN = [103.939, 116.779, 123.68]
NORMALIZER = 0.017
#Parameters credits to https://github.com/MG2033/ShuffleNet

class Shufflenet:
	#Load pretrained model on initialization. Model downloaded from http://models.tensorpack.com/ImageNetModels/ShuffleNetV1-1x-g=8.npz
	def __init__(self, model_loc):
		self.trained_model = np.load(model_loc, encoding = 'latin1')
		print("Pre-trained npz model loaded")
#		Uncomment below 2 lines to check model entries (Kernels for conv), (mean, variance, beta and gamma for BN) and (weights and biases) for final FC layer
#		for x in self.trained_model.files:
#			print(x + " " + str(self.trained_model[x].shape))

	'''
	Point-wise group convolution operation.
	Inputs: 	Activations of shape [N, H, W, C], Stage in format
			'stagex', block in format 'blockx' and layer in
			format 'convx', num_groups are the number of
			groups to split the activations and kernels in.
	Outputs:	Output activations of group convolution result
	'''
	def pw_gconv(self, activations, stage, block, layer, num_groups, name):
		with tf.name_scope(name):
			layer_name = str(stage) + '/' + str(block) + '/' + str(layer) + '/W:0'
			kernels = self.trained_model[layer_name]
			ch_per_group = activations.shape[3] // num_groups
			act_split = tf.split(activations, num_or_size_splits = num_groups, axis = 3)
			kernels_split = tf.split(kernels, num_or_size_splits = num_groups, axis = 3)
			convs = []
			for grp in range(0, num_groups):
				convs.append(tf.nn.conv2d(act_split[grp], kernels_split[grp], padding = 'SAME', strides = [1, 1, 1, 1], data_format = 'NHWC', name='pw_gconv_' + str(grp)))
			return tf.concat(convs, axis = 3)

	'''
	Depth-wise convolution operation.
	Inputs:		Activations of shape [N, H, W, C], stage in format 'stagex',
			block in format 'blockx', padding, stride and name to give to
			the node in tensorboard visualization
	Outputs:	Output activations of the dw conv operation
	'''
	def dw_conv(self, activations, stage, block, padding = 'SAME', stride = 1, name="dw_conv"):
		with tf.name_scope(name):
			inp_ch = activations.shape[3]
			act_shape = activations.shape
			layer_name = str(stage) + '/' + str(block) + '/dconv/W:0'
			kernels = self.trained_model[layer_name]
			kernel_size = kernels.shape[0]
			conv_result = tf.nn.depthwise_conv2d(activations, kernels, [1, stride, stride, 1], padding = padding, data_format = 'NHWC', name='dw_conv_' + stage + '_' + block)
			return conv_result

	'''
	Batch Normalization operations
	Inputs:		Activations of shape [N, H, W, C], stage in format 'stagex',
			block in format 'blockx', layer in format 'convx' and name to
			give to the node in tensorboard graph summary.
	Outputs:	Output activations of BN operation
	'''
	def batch_normalization(self, activations, stage, block, layer, name):
		with tf.name_scope(name):
			layer_name = str(stage) + '/' + str(block) + '/' if stage is not '' else ''
			layer_name = layer_name + 'conv1/bn/' if layer == 'conv1' else layer_name + layer+'_bn/'
			bn_out = tf.nn.batch_normalization(activations, self.trained_model[layer_name + 'mean/EMA:0'], self.trained_model[layer_name + 'variance/EMA:0'], self.trained_model[layer_name + 'beta:0'], self.trained_model[layer_name + 'gamma:0'], variance_epsilon=0.001, name = 'bn_' + stage + '_' + block + '_' + layer if stage is not '' else 'bn_conv1')
			return bn_out

	'''
	Channel Shuffle Operation.
	Inputs:		Activations of shape [N, H, W, C], num_groups = 8,
			Name to give to the node in tensorboard graph summary.
	Outputs		Activations after ch shuffle op.
	'''
	def channel_shuffle(self, activations, num_groups = 8, name='ch_shuffle'):
		with tf.name_scope(name):
			activations = tf.transpose(activations, perm = [0, 3, 1, 2])
			in_shape = activations.get_shape().as_list()
			in_channel = in_shape[1]
			l = tf.reshape(activations, [-1, in_channel // num_groups, num_groups] + in_shape[-2:])
			l = tf.transpose(l, [0, 2, 1, 3, 4])
			l = tf.reshape(l, [-1, in_channel] + in_shape[-2:])
			l = tf.transpose(l, perm = [0, 2, 3, 1])
			return l

	def shufflenet_unit(self, activations, stage, block, stride, num_groups=8, name="shufflenet_unit"):
		with tf.name_scope(name):
			residual = activations
			num_split = num_groups if activations.shape[3] > 24 else 1
			pwgconv1 = self.pw_gconv(activations, stage, block, 'conv1', num_split, name= stage + "_" + block + "_pwgconv1")
			bnconv1 = self.batch_normalization(pwgconv1, stage, block, 'conv1', name = stage + "_" + block + "_pwgconv1_batch_norm")
			reluconv1 = tf.nn.relu(bnconv1)
			ch_sh = self.channel_shuffle(reluconv1, num_groups, name = stage + '_' + block + '_ch_shuffle')
			dconv = self.dw_conv(ch_sh, stage, block, padding = 'SAME', stride = stride, name = stage + "_" + block + "_dwconv")
			bndconv = self.batch_normalization(dconv, stage, block, 'dconv', name = stage + "_" + block + "_dconv_batch_norm")
			pwgconv2 = self.pw_gconv(bndconv, stage, block, 'conv2', num_groups, name= stage + "_" + block + "_pwgconv2")
			bnconv2 = self.batch_normalization(pwgconv2, stage, block, 'conv2', name = stage + "_" + block + "_pwgconv2_batch_norm")

			if stride == 1:
				return tf.nn.relu(bnconv2 + residual, name = 'relu_' + stage + '_' + block)
			elif stride == 2:
				residual = tf.nn.avg_pool(residual, [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', data_format = 'NHWC', name = 'avg_pool_' + stage + '_' + block)
				return tf.concat([residual, tf.nn.relu(bnconv2)], axis = 3, name = 'concat_' + stage + '_' + block)
			else:
				raise ValueError("Stride value can only be 1 or 2 for Shufflenet")

	def shufflenet_stage(self, activations, stage, repeat, num_groups=8, name = "shufflenet_stage"):
		with tf.name_scope(name):
			first_block = self.shufflenet_unit(activations, stage, 'block0', stride = 2, num_groups = 8, name = "shufflenet_unit_" + stage + "_block0")
			res = first_block
			for b in range(1, repeat+1):
				res = self.shufflenet_unit(res, stage, 'block' + str(b), stride = 1, num_groups = 8, name = "shufflenet_unit_" + stage + "_block" + str(b))
			return res

	def shufflenet_stage1(self, activations):
		with tf.name_scope("shufflenet_stage1"):
			kernels = self.trained_model['conv1/W:0']
			res = tf.nn.conv2d(activations, kernels, padding = 'SAME', strides = [1, 2, 2, 1], data_format = 'NHWC', name = 'Conv1')
			res = self.batch_normalization(res, '', '', 'conv1', name = 'stage1_conv2d_batch_norm')
			res = tf.nn.max_pool(res, [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', data_format = 'NHWC', name = 'MaxPool1')
			return res

	def fc_layer(self, activations):
		with tf.name_scope('fc_layer'):
			layer_name = 'linear'
			weights = self.trained_model[layer_name + '/W:0']
			biases = self.trained_model[layer_name + '/b:0']
			flattened_out = tf.contrib.layers.flatten(activations)
			return tf.nn.bias_add(tf.matmul(flattened_out, weights), biases)

	def build(self, image):
		red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=image)
		bgr = tf.concat(axis=3, values=[(blue - SHUFFLENET_MEAN[0])*NORMALIZER, (green - SHUFFLENET_MEAN[1])*NORMALIZER, (red - SHUFFLENET_MEAN[2])*NORMALIZER])
		stage1 = self.shufflenet_stage1(bgr)
		stage2 = self.shufflenet_stage(stage1, 'stage2', repeat = 3, num_groups = 8, name = "shufflenet_stage2")
		stage3 = self.shufflenet_stage(stage2, 'stage3', repeat = 7, num_groups = 8, name = "shufflenet_stage3")
		stage4 = self.shufflenet_stage(stage3, 'stage4', repeat = 3, num_groups = 8, name = "shufflenet_stage4")
		g_pool = tf.nn.avg_pool(stage4, [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID', data_format = 'NHWC', name = 'GlobalPool')
		logits = self.fc_layer(g_pool)
		logits = tf.nn.softmax(logits, name = "SoftMax_unit")
		return logits
