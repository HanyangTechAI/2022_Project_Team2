import tensorflow as tf
import numpy as np
import cv2

"""
refer to : https://github.com/serengil/deepface/blob/13a21fe306ee39567f7f0b15422f8a3c1ce656de/deepface/DeepFace.py#L721
"""

def preprocess_face(img, target_size=(224, 224), grayscale=False):
    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                         'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    # ------------------------------------------

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # ---------------------------------------------------
    # normalizing the image pixels

    img_pixels = np.expand_dims(img, axis=0).astype(np.float32)
    img_pixels /= 255  # normalize input in [0, 1]

    # ---------------------------------------------------

    return img_pixels


def normalize_input(img, normalization='base'):
    # issue 131 declares that some normalization techniques improves the accuracy

    if normalization == 'VGGFace':
        img *= 255  # restore input in scale of [0, 255] because it was normalized in scale of  [0, 1] in preprocess_face

        # if normalization == 'VGGFace':
        # mean subtraction based on VGGFace1 training data
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863

    else:
        return img

    return img


def find_input_shape(model):
    tf_version = tf.__version__
    tf_major_version = int(tf_version.split(".")[0])
    tf_minor_version = int(tf_version.split(".")[1])

    # face recognition models have different size of inputs
    # my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.
    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    # ----------------------
    # issue 289: it seems that tf 2.5 expects you to resize images with (x, y)
    # whereas its older versions expect (y, x)

    if tf_major_version == 2 and tf_minor_version >= 5:
        x = input_shape[0]
        y = input_shape[1]
        input_shape = (y, x)

    # ----------------------

    if type(input_shape) == list:  # issue 197: some people got array here instead of tuple
        input_shape = tuple(input_shape)

    return input_shape


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
