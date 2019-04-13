import shufflenet
import tensorflow as tf
import numpy as np
import time
import utils

def main():

        img = utils.load_image('./test_data/32.JPEG')
        model_path = '../ShuffleNetV1-1x-8g.npz'

        img = img.reshape((1, 224, 224, 3))
        img = np.float32(img) * 255.0

        arch = shufflenet.Shufflenet(model_path)

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
                        export_graph = tf.summary.FileWriter('./logs/shufflenet_graph/', sess.graph)

if __name__ == '__main__':
        main()
