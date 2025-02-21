import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import einops
from scipy.io import loadmat


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

    d_1 = np.sin(np.arccos(x.dot(y) / (normx * normy))) * normy

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


def split_data(data, labels, labeled_frac):
    """
    Splits data and labels into labeled and unlabeled portions based on the specified fraction.

    Args:
        data: numpy array or list of data samples
        labels: numpy array or list of corresponding labels
        labeled_frac: float between 0 and 1, fraction of data to be labeled

    Returns:
        tuple: (labeled_data, labeled_labels, unlabeled_data, unlabeled_labels)
            - labeled_data: array of labeled samples
            - labeled_labels: array of labels for labeled samples
            - unlabeled_data: array of unlabeled samples
            - unlabeled_labels: array of labels for unlabeled samples (for evaluation)

    Raises:
        ValueError: If labeled_frac is not between 0 and 1
        ValueError: If data and labels have different lengths
    """
    # Convert inputs to numpy arrays if they aren't already
    data = np.asarray(data)
    labels = np.asarray(labels)

    # Input validation
    if not 0 <= labeled_frac <= 1:
        raise ValueError("labeled_frac must be between 0 and 1")

    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")
    
    # get mask of background data
    background_indices = np.where(labels == 0)[0]

    # Calculate number of samples for labeled portion
    n_samples = len(data) - len(background_indices)
    n_labeled = int(n_samples * labeled_frac)

    # Generate random permutation of indices
    indices = np.random.permutation(n_samples)

    # Split indices into labeled and unlabeled
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]

    # Split data and labels
    labeled_data = data[labels != 0][labeled_indices]
    labeled_labels = labels[labels != 0][labeled_indices]
    unlabeled_data = data[labels != 0][unlabeled_indices]
    unlabeled_labels = labels[labels != 0][unlabeled_indices]

    return labeled_data, labeled_labels, unlabeled_data, unlabeled_labels, labeled_indices, unlabeled_indices, background_indices


def mostWeight(weight, labels):
    """
    returns the integer label corresponding to where the most mass is in weight vector
    
    :param weight: numpy array of shape (n 1) with nonnegative entries that sum to 1
    :param labels: numpy array of shape (n) with the integer label of the label corresponding to each weight
    """

    unique = np.unique(labels)

    label_mass = np.zeros(len(unique))
    for i, l in enumerate(unique):
        label_mass[i] = weight[labels == l].sum()

    return int(unique[label_mass.argmax()])


def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate F1 score and accuracy for predicted labels
    
    Parameters:
    true_labels (array-like): Ground truth labels
    predicted_labels (array-like): Predicted labels from the model
    
    Returns:
    tuple: (f1_score, accuracy)
    """
    # Ensure inputs are numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # Calculate metrics
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    return f1, accuracy

def split_data_equal(data, labels, labeled_frac, num_unlabeled):
    """
    Splits data and labels into labeled and unlabeled portions based on the specified fraction,
    ensuring proportional representation of each class in the labeled set.
    
    Args:
        data: numpy array or list of data samples
        labels: numpy array or list of corresponding labels
        labeled_frac: float between 0 and 1, fraction of data to be labeled per class
    
    Returns:
        tuple: (labeled_data, labeled_labels, unlabeled_data, unlabeled_labels, 
               labeled_indices, unlabeled_indices, background_indices)
    """
    # Convert inputs to numpy arrays
    data = np.asarray(data)
    labels = np.asarray(labels)
    
    # Input validation
    if not 0 <= labeled_frac <= 1:
        raise ValueError("labeled_frac must be between 0 and 1")
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")
    
    # Get background indices (class 0)
    background_indices = np.where(labels == 0)[0]
    
    # Get non-background data indices
    foreground_mask = labels != 0
    foreground_indices = np.where(foreground_mask)[0]
    
    # Initialize arrays for storing selected indices
    all_labeled_indices = []
    all_unlabeled_indices = []
    
    # Get unique classes (excluding background class 0)
    unique_classes = np.unique(labels[foreground_mask])
    
    # For each class, select proportional samples
    for class_label in unique_classes:
        # Get indices for current class
        class_mask = labels == class_label
        class_indices = np.where(class_mask)[0]
        
        # Calculate number of samples to select for this class
        n_samples_class = len(class_indices)
        n_labeled_class = max(int(n_samples_class * labeled_frac), 1)
        
        # Randomly permute indices for this class
        permuted_indices = np.random.permutation(class_indices)
        
        # Split into labeled and unlabeled
        labeled_indices_class = permuted_indices[:n_labeled_class]
        unlabeled_indices_class = permuted_indices[n_labeled_class:]
        
        # Add to our collections
        all_labeled_indices.extend(labeled_indices_class)
        all_unlabeled_indices.extend(unlabeled_indices_class)
    
    # Convert to numpy arrays
    labeled_indices = np.array(all_labeled_indices)
    unlabeled_indices = np.array(all_unlabeled_indices)
    
    # Extract the actual data and labels
    labeled_data = data[labeled_indices]
    labeled_labels = labels[labeled_indices]

    # unlabeled data
    unlabeled_indices = np.random.choice(unlabeled_indices, num_unlabeled, replace=False)

    unlabeled_data = data[unlabeled_indices]
    unlabeled_labels = labels[unlabeled_indices]

    
    
    return (labeled_data, labeled_labels, unlabeled_data, unlabeled_labels,
            labeled_indices, unlabeled_indices, background_indices)

def min_residual(X, w, y, labeled_indices):

    best_residual = np.inf
    best_label = None 

    n, m = X.shape 

    assert w.shape[0] == m
    assert y.shape[0] == n
    
    unique_labels = np.unique(labeled_indices)

    for i, label in enumerate(unique_labels):
        w_copy = w.copy()
        w_copy[labeled_indices != label] = 0.0
        residual = np.linalg.norm(X @ w_copy - y)

        if residual < best_residual:
            best_residual = residual
            best_label = label
    
    return best_label


def load_dataset(dataset: "str"):
    # Load and prepare data
    if dataset == "Salinas":
        fname = "hsi_datasets/Salinas_corrected.mat"
        matname = "salinas_corrected"
        labels = loadmat("hsi_datasets/Salinas_gt.mat")["salinas_gt"]
    elif dataset == "Indian_Pines":
        fname = "hsi_datasets/Indian_pines_corrected.mat"
        matname = "indian_pines_corrected"
        labels = loadmat("hsi_datasets/Indian_pines_gt.mat")["indian_pines_gt"]
    elif dataset == "Pavia_Centre":
        fname = "hsi_datasets/Pavia.mat"
        matname = "pavia"
        labels = loadmat("hsi_datasets/Pavia_gt.mat")["pavia_gt"]
    elif dataset == "Pavia_University":
        fname = "hsi_datasets/PaviaU.mat"
        matname = "paviaU"
        labels = loadmat("hsi_datasets/PaviaU_gt.mat")["paviaU_gt"]
    elif dataset == "KSC":
        fname = "hsi_datasets/KSC_corrected.mat"
        matname = "KSC"
        labels = loadmat("hsi_datasets/KSC_gt.mat")["KSC_gt"]
    elif dataset == "Botswana":
        fname = "hsi_datasets/Botswana.mat"
        matname = "Botswana"
        labels = loadmat("hsi_datasets/Botswana_gt.mat")["Botswana_gt"]
    data = loadmat(fname)[matname]
    data = einops.rearrange(data, "col row v -> (col row) v")
    data = np.asarray(data, dtype=np.float64)
    labels = einops.rearrange(labels, "col row -> (col row)")
    labels = np.asarray(labels, dtype=np.float64)

    return data, labels 