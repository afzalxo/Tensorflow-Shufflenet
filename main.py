import shufflenet
import tensorflow as tf
import numpy as np
import time
import utils

def main():
        act = tf.constant(np.arange(224*224*3).reshape((1, 3, 224, 224)), dtype=float)
        img = utils.load_image('./test_data/32.JPEG')
        img = img.reshape((1, 224, 224, 3))
        img = np.float32(img) * 255.0
#       img = np.float32(img)
        arch = shufflenet.Shufflenet()
#       conv_res = arch.pw_gconv(act, 'stage3', 'block0', 'conv1', num_groups = 8)
#       conv_res = arch.batch_normalization(act, 'stage2', 'block0', 'conv1')
#       conv_res = arch.shufflenet_stage(act, 'stage2', 3, 8)
        feed_img = tf.placeholder('float', [1, 224, 224, 3])
        feed_dict = {feed_img: img}
        with tf.device('/cpu:0'):
                with tf.Session() as sess:
                        conv_res = arch.build(feed_img)
                        begin = time.time()
                        prob = sess.run(conv_res, feed_dict=feed_dict)
                        end = time.time()
                        utils.print_prob(prob[0], './synset.txt')
                        print("Time: " + str(end - begin))

if __name__ == '__main__':
        main()
