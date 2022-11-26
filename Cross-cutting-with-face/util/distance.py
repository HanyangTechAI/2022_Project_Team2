import numpy as np


def findEuclideanDistance(source_representation, test_representation, n):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.multiply(euclidean_distance, euclidean_distance)
    euclidean_distance = np.sum(np.sum(np.sum(euclidean_distance)))
    euclidean_distance = np.sqrt(euclidean_distance)

    return euclidean_distance / n


