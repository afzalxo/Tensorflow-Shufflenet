import shufflenet
import tensorflow as tf
import numpy as np
import time
import utils

def main():

        img = utils.load_image('./test_data/34.JPEG')
        model_path = '../ShuffleNetV1-1x-8g.npz'

        img = img.reshape((1, 224, 224, 3))
        img = np.float32(img) * 255.0

        arch = shufflenet.Shufflenet(model_path)

        feed_img = tf.placeholder('float', [1, 224, 224, 3])
        feed_dict = {feed_img: img}

        with tf.device('/cpu:0'):
                with tf.Session() as sess:
                        conv_res = arch.build(feed_img)
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        begin = time.time()
                        prob = sess.run(conv_res, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                        end = time.time()
                        utils.print_prob(prob[0], './synset.txt')
                        print("Time: " + str(end - begin))
                        export_graph = tf.summary.FileWriter('./logs/shufflenet_graph/')
                        export_graph.add_graph(sess.graph)
                        export_graph.add_run_metadata(run_metadata, 'zucc')

#                        opts = tf.profiler.ProfileOptionBuilder.float_operation()
#                        flops = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opts)
#                        print("FLOPS: " + str(flops.total_float_ops))

if __name__ == '__main__':
        main()
