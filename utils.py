import skimage
import skimage.io
import skimage.transform
import numpy as np

# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]

    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def write_prob_file(prob, file_handler, file_path, w_mantissa, d_mantissa):
    synset = [l.strip() for l in open(file_path).readlines()]
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    file_handler.write("Weights: Q1.0." + str(w_mantissa) + " Data: Q1.8." + str(d_mantissa) + "\n")
    file_handler.flush()
    print("Weights: Q1.0." + str(w_mantissa) + " Data: Q1.8." + str(d_mantissa))
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    file_handler.write(("Top1: " + str(top1) + " " +  str(prob[pred[0]]) + "\n"))
    file_handler.write(("Top5: " + str(top5) + "\n\n"))
    file_handler.flush()
    return top1
