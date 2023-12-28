import numpy as np

def simplexSample(k: int, n_samples: int = 1):
    """
    return a vector of length k whose elements are nonnegative and sum to 1 - and in particularly the vector is sampled
    uniformly from this set via the bayesian bootstrap
    https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex

    :param k: the length of the vector to be sample from the simplex
    :return: a uniformly sampled vector from the probability simplex
    """

    samples = np.zeros((k, n_samples))

    for i in range(n_samples):
        # sample k - 1 points
        weights = np.random.rand((k - 1))

        # add 0 and 1 then sort
        new_weights = np.zeros((k + 1))
        new_weights[0] = 0.0
        new_weights[1] = 1.0
        new_weights[2:] = weights

        new_weights = np.sort(new_weights)

        # differences between points to get the uniform sample
        samples[:, i] = new_weights[1:] - new_weights[:-1]

    return samples

def min_distance(v, p):
    """
    returns the minimium distance squared to the boundary of the triangle defined by v
    :param v: shape (3, 2) set of points defining the vertices of the triangle
    :param y: the point within the triangle to find the minimum distance to the boundary of
    :return:
    """

    # compute each of the distances
    # a simple geometric computation
    # (0, 1)
    a = v[0]
    b = v[1]

    x = (a - b)
    y = (p - b).reshape(-1)

    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)

    d_1 = np.sin(np.arccos(x.dot(y) / (normx * normy ))) * normy


    # (1, 2)
    a = v[1]
    b = v[2]

    x = (a - b)
    y = (p - b).reshape(-1)

    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)

    d_2 = np.sin(np.arccos(x.dot(y) / (normx * normy))) * normy

    # (0, 2)
    a = v[0]
    b = v[2]

    x = (a - b)
    y = (p - b).reshape(-1)

    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)

    d_3 = np.sin(np.arccos(x.dot(y) / (normx * normy))) * normy


    return min([d_1, d_2, d_3]) ** 2

def triangle_area(vertices):
    """
    Computes the area of a triangle in 2D

    :param vertices: an array of shape (3, 2) representing the vertices
    :return: the area of the triangle
    """
    a = vertices[1]-vertices[0]
    b = vertices[2]-vertices[0]

    return 0.5*np.abs(np.cross(a, b))

def coverage(idx1, idx2):
    """
    measures how well idx1 and idx2 overlap, max of 1 when they are identical and min of 0 when there is no overlap
    :param idx1: a set of integer indices
    :param idx2: a set of integer indices
    :return: the  cardinality intersection over the cardinality of the union
    """
    return np.intersect1d(idx1, idx2).size / np.union1d(idx1, idx2).size