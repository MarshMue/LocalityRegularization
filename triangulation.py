import torch
import numpy as np
import cvxopt
from cvxopt import matrix, spdiag, blas, lapack, solvers, sqrt
from scipy.optimize import linprog
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options["reltol"] = 1e-8
cvxopt.solvers.options["abstol"] = np.finfo(float).eps*2
cvxopt.solvers.options["feastol"] = 1e-8
cvxopt.solvers.options["maxiters"] = 100

def exactLP(X, p, tol=None):
    """
    Solves the exact problem (E) using an interior point solver

    :param X: np array of shape (n, d), with data as rows
    :param p: np array of shape (1, d), the point to be represented
    :param tol: The threshold to determine which values should be considered zero, if None uses the largest log gap
    :return support: an array of the indices in X containing the support of weights
    :return solved_weights: the weights to solve (E)
    """

    # LP setup
    n, d = X.shape

    # costs
    c = cvxopt.matrix(np.power(np.linalg.norm(X - p, axis=1), 2))

    A_ = cvxopt.matrix(np.vstack([X.T, np.ones((1, n))]))
    b = np.ones(d+1)
    b[:d] = p[0]
    b_ = cvxopt.matrix(b)
    G_ = cvxopt.spmatrix(-1, range(n), range(n))
    h_ = cvxopt.matrix(np.zeros(n))

    initvals = {'x': cvxopt.matrix(np.ones((n, 1)) / n)}
    sol = cvxopt.solvers.lp(c, G_, h_, A_, b_, initvals=initvals)

    solved_weights = np.array(sol['x'])

    idxes = np.array(list(range(n)))

    # use weight on smaller end of the largest log-gap
    if tol is None:
        sorted_weights = np.sort(solved_weights.reshape(-1))
        # handle situations where small entries may be slightly negative
        sorted_weights[sorted_weights < 0.0] = sorted_weights[sorted_weights > 0.0].min()
        gap = np.log(sorted_weights[1:]) - np.log(sorted_weights[:-1])
        max_gap = gap.argmax()
        # check that there is a significant gap, otherwise threshold at 0 (all values are away from 0)
        median_gap = np.median(gap)
        MAD = np.median(np.abs(gap - median_gap))
        if gap.max() < median_gap + 3 * 1.4826* MAD:
            tol = 0.0
        else:
            tol = sorted_weights[max_gap]

    support = idxes[(solved_weights > tol).reshape(-1)]

    return support, solved_weights


def cvxLP(X, p, tol=None, return_objective=False):
    """
    solves the Convex Hull linear program described here: https://www.cs.mcgill.ca/~fukuda/download/paper/polyfaq.pdf (pg 23)
    that finds the simplex a point p belongs to by identifying the vertices in X

    :param X: np array of shape (n, d), with data as rows
    :param tol: the threshold to consider what values should be considered zero, if None uses the largest log gap
    :param p: np array of shape (1, d), the point to be represented
    :param return_objective: whether or not to return the objective value and dual variables
    :return support: the indices of X corresponding to the vertices of the d-simplex containing p
    """

    p = p.reshape(-1)

    n, d = X.shape

    G = matrix(1.0, (n, d+1))
    G[:, :d] = X

    c = matrix(1.0, (d+1, 1))
    c[:d] = p

    fX = matrix(0.0, (n,1))

    for i in range(n):
        fX[i] = X[i] @ X[i].T

    sol = solvers.lp(-c, G, fX)

    z = np.array(sol['z']).reshape(-1)


    # use weight on smaller end of the largest log-gap
    if tol is None:
        sorted_dual = np.sort(z)
        zero_idx = sorted_dual == 0
        sorted_dual[sorted_dual == 0] = sorted_dual[np.logical_not(zero_idx)].min()
        gap = np.log(sorted_dual[1:]) - np.log(sorted_dual[:-1])
        max_gap = gap.argmax()

        # if max is within 3 MAD of median, consider all weights to be non-zero
        median_gap = np.median(gap)
        MAD = np.median(np.abs(gap - median_gap))
        if gap.max() < median_gap + 3 * 1.4826* MAD:
            tol = 0.0
        else:
            tol = sorted_dual[max_gap]
        # tol = min(1e-10, tol)

    idxs = np.arange(n)

    support = idxs[z > tol]

    if return_objective:
        objective = sol['primal objective']
        return support, objective, np.array(sol['x'])
    else:
        return support




def delaunaySparse(X: torch.tensor, p: torch.tensor):
    """
    implements the delaunaysparse method of finding the vertices of the simplex of the delaunay triangulation of X
    that p belongs to

    :param X: an n by d tensor of points (rows being d dimensional points)
    :param p: d dimensional point

    :return idxes: the indices corresponding to the vertices in the simplex
    """

    # use view of p as a row vector
    p = p.view(1, -1)

    d = p.shape[1]

    if d != X.shape[1]:
        raise ValueError(f"Dimension of points in X: {X.shape[1]} does not match dimension of p: {d}")

    # intialize indices of vertices
    idxes = torch.zeros((d+1), dtype=int)

    # first vertex is the closest point to p
    # find index of row of X closest to p in L2 norm
    dists = torch.linalg.norm(X-p, dim=1)
    idxes[0] = torch.argmin(dists)

    # second vertex is closest point in X to X[idxes[0]]
    dists = torch.linalg.norm(X - X[idxes[0]], dim=1)
    dists[idxes[0]] = torch.inf
    idxes[1] = torch.argmin(dists)

    # find the next indices

    for i in range(2, d+1):
        A = torch.zeros(i, d)
        A[:i-1] = X[idxes[1:i]] - X[idxes[0]]

        combined = torch.cat((torch.arange(X.shape[0]), idxes[:i]))
        uniques, counts = combined.unique(return_counts=True)
        remaining_idxs = uniques[counts == 1]

        min_norm_idx = X.shape[0]+1
        min_norm = torch.inf
        for j in remaining_idxs:
            A[i-1] = X[j] - X[idxes[0]]

            b = torch.linalg.norm(A, dim=1)**2 / 2

            x = torch.linalg.pinv(A) @ b

            norm = torch.linalg.norm(x)
            if norm < min_norm:
                min_norm = norm
                min_norm_idx = j

        idxes[i] = min_norm_idx

    return idxes



def sgd_solve(X, y, num_iters, rho, step):
    X = torch.tensor(X)
    y = torch.tensor(y)
    x = torch.zeros((y.shape[0], X.shape[0]), dtype=torch.float64)
    # weight = (y.unsqueeze(1) - self.W.unsqueeze(0)).pow(2).sum(dim=2)
    weight = (
            y.square().sum(dim=1, keepdims=True)
            + X.T.square().sum(dim=0, keepdims=True)
            - 2 * y @ X.T
    )
    for layer in range(num_iters):
        grad = (x @ X - y) @ X.T
        grad = grad + weight * rho
        x = activate(x - grad * step)
    return x

def sgd_solve_accelerated(X, y, num_iters, rho, step):
    X = torch.tensor(X)
    y = torch.tensor(y)
    x_tmp = torch.zeros((y.shape[0], X.shape[0]), dtype=torch.float64)
    x_old = torch.zeros((y.shape[0], X.shape[0]), dtype=torch.float64)
    # weight = (y.unsqueeze(1) - self.W.unsqueeze(0)).pow(2).sum(dim=2)
    weight = (
        y.square().sum(dim=1, keepdims=True)
        + X.T.square().sum(dim=0, keepdims=True)
        - 2 * y @ X.T
    )
    for layer in range(num_iters):
        grad = (x_tmp @ X - y) @ X.T
        grad = grad + weight * rho
        x_new = activate(x_tmp - grad * step)
        x_old, x_tmp = x_new, x_new + layer / (layer + 3) * (x_new - x_old)
    return x_new


def activate(x):
    m, n = x.shape
    cnt_m = torch.arange(m, device=x.device)
    cnt_n = torch.arange(n, device=x.device)
    u = x.sort(dim=1, descending=True).values
    v = (u.cumsum(dim=1) - 1) / (cnt_n + 1)
    w = v[cnt_m, (u > v).sum(dim=1) - 1]
    return (x - w.view(m, 1)).relu()




def customCVX(A, p, rho, tol=None, verbose=False):
    """
    Solves (R) using an interior point solver with a custom KKT solver as described in the appendix of the paper

    :param X: np array of shape (n, d), with data as rows
    :param p: np array of shape (1, d), the point to be represented
    :param rho: the regularization parameter
    :param tol: the threshold to consider what values should be considered zero, if None uses the largest log gap
    :param verbose: whether or not to pass verbose to cvxopt
    :return support: an array of the indices in X containing the support of weights
    :return solved_weights: the weights to solve (R)
    """
    # adjust rho for the standard form QP
    rho /= 2

    # for comparison with other method
    c = np.power(np.linalg.norm(p - A, axis=1), 2).reshape(-1, 1)
    q = cvxopt.matrix(rho * c - A @ p.T)

    X = A



    A = cvxopt.matrix(A.T)


    d, n = A.size

    h = cvxopt.matrix(np.zeros(n))


    def P(u, v, alpha = 1.0, beta = 0.0 ):
        """
        v := alpha * A * A.T * u + beta * v
        """
        blas.scal(beta, v)

        tmp = A*u
        blas.gemv(A.T, tmp, v, alpha=alpha, beta=beta)


    def G(u, v, alpha=1.0, beta=0.0, trans='N'):
        """
            v := alpha*-I * u + beta * v  (trans = 'N' or 'T')
        """

        blas.scal(beta, v)
        blas.axpy(u, v, alpha=-alpha)



    # Customized solver for the KKT system
    # see overleaf for details

    S = matrix(0.0, (d, d))
    v = matrix(0.0, (d, 1))

    def Fkkt(W):

        # D = - W^T * W
        Dvecsqrt = W['d']
        Dvec = -Dvecsqrt**2
        D = spdiag(Dvec)

        TrD = sum(Dvec)
        TrDsqrt = sqrt(-TrD)

        # Asc = A*diag(d)^-1/2
        ADsqrt = A * spdiag(Dvecsqrt)

        # (rank k and rank 1 update)
        # S =  A * D * A' - AD11'DA' - I

        # rank k update
        blas.syrk(ADsqrt, S, alpha=-1.0)

        AD = A*D

        ones = matrix(1.0, (n, 1))

        AD1 = AD*ones / TrDsqrt

        # rank 1 update
        blas.syr(AD1, S, alpha=1.0)

        # subtract identity
        S[::d+1] -= 1.0

        # # compute cholesky - doesn't seem to be any faster
        ipiv = matrix(0, (n, 1), tc='i')
        lapack.sytrf(S, ipiv)

        def g(x, y, z):

            # v = rhs of linear system

            blas.gemv(AD, x - (y + sum(cvxopt.mul(Dvec, x) + z)) / TrD * ones, v)
            blas.axpy(A*z, v)

            # solve linear system for v
            # lapack.sysv(S, v)
            lapack.sytrs(S, ipiv, v)

            # compute u_x
            u_x = cvxopt.mul(Dvec, A.T*v - sum(cvxopt.mul(Dvec, (A.T*v)))*ones / TrD - (x + cvxopt.div(z, Dvec) - (y + sum(cvxopt.mul(Dvec, x) + z)) / TrD * ones))

            # compute u_y
            blas.axpy(cvxopt.matrix(sum(cvxopt.mul(Dvec, (A.T*(A*u_x))) - cvxopt.mul(Dvec, x) - z)), y, alpha=-1.0)
            # blas.scal(1/TrD, y)
            y /= TrD

            # compute u_z
            blas.swap(x, u_x)


            blas.axpy(x, z)
            z[:] = cvxopt.div(z, -Dvecsqrt)
            # blas.swap(z, tmp)

        return g

    A_ = cvxopt.matrix(np.ones((1, n)))
    b_ = cvxopt.matrix(np.ones(1))

    # set params
    maxiters = cvxopt.solvers.options["maxiters"]
    feastol = cvxopt.solvers.options["feastol"]
    reltol = cvxopt.solvers.options["reltol"]
    show_progress = cvxopt.solvers.options["show_progress"]
    cvxopt.solvers.options["maxiters"] = 300
    cvxopt.solvers.options["feastol"] = np.finfo(float).eps * 1e6
    cvxopt.solvers.options["reltol"] = np.finfo(float).eps * 1e6
    cvxopt.solvers.options["show_progress"] = verbose

    # solve problem
    initvals = {'x': cvxopt.matrix(np.ones((n, 1))/n)}
    sol = solvers.qp(P=P, q=q, G=G, h=h, A=A_, b=b_, kktsolver = Fkkt, initvals=initvals)

    # reset params
    cvxopt.solvers.options["feastol"] = feastol
    cvxopt.solvers.options["reltol"] = reltol
    cvxopt.solvers.options["show_progress"] = show_progress
    cvxopt.solvers.options["maxiters"] = maxiters

    # get solutions
    solved_weights = np.array(sol["x"])

    idxes = np.array(list(range(n)))

    # special case
    if d+1 == n:
        if np.isclose(np.linalg.norm(solved_weights.reshape(1, -1) @ X - p), 0):
            return idxes, solved_weights

    # use weight on smaller end of the largest log-gap
    if tol is None:
        sorted_weights = np.sort(solved_weights.reshape(-1))
        # handle situations where small entries may be slightly negative
        sorted_weights[sorted_weights < 0.0] = sorted_weights[sorted_weights > 0.0].min()
        gap = np.log(sorted_weights[1:]) - np.log(sorted_weights[:-1])
        max_gap = gap.argmax()
        # check that there is a significant gap, otherwise threshold at 0 (all values are away from 0)
        median_gap = np.median(gap)
        MAD = np.median(np.abs(gap - median_gap))
        if gap.max() < median_gap + 3 * 1.4826* MAD:
            tol = 0.0
        else:
            tol = sorted_weights[max_gap]

    support = idxes[(solved_weights > tol).reshape(-1)]

    return support, solved_weights


def solve_qp_cvx(X, p, rho, tol=None):
    """

    :param X: the set of points, as row vectors (n x d)
    :param p: the test point, as a row vector (1 x d)
    :return: the identified vertices and weights
    """

    n, d = X.shape

    P = cvxopt.matrix(X @ X.T)

    c = np.power(np.linalg.norm(p - X, axis=1), 2).reshape(-1, 1)

    # QP setup
    A_ = cvxopt.matrix(np.ones((1, n)))
    b_ = cvxopt.matrix(np.ones(1))
    G_ = cvxopt.matrix(-np.eye(n))
    h_ = cvxopt.matrix(np.zeros(n))

    # adjust rho for the 1/2 factor from cvx
    rho /= 2
    q = cvxopt.matrix(rho * c - X @ p.T)

    sol = cvxopt.solvers.qp(P, q, G_, h_, A_, b_)

    solved_weights = np.array(sol["x"])

    idxes = np.array(list(range(n)))

    # use weight on smaller end of the largest log-gap
    if tol is None:
        sorted_weights = np.sort(solved_weights.reshape(-1))
        # handle situations where small entries may be slightly negative
        sorted_weights[sorted_weights < 0.0] = sorted_weights[sorted_weights > 0.0].min()
        gap = np.log(sorted_weights[1:]) - np.log(sorted_weights[:-1])
        max_gap = gap.argmax()
        tol = sorted_weights[max_gap]

    support = idxes[(solved_weights > tol).reshape(-1)]

    return support, solved_weights




class triQP(torch.nn.Module):

    def __init__(self, X: torch.Tensor, rho, idxes = None):
        super().__init__()

        d, n = X.shape

        if idxes is not None:
            # initialize supported on the given idxes
            self.weights = torch.nn.Parameter(torch.zeros(n, 1))
            self.weights[idxes] = 1/len(idxes)
        else:
            self.weights = torch.nn.Parameter(torch.rand(n, 1))
            self.weights /= self.weights.sum()


        self.X = X
        self.rho = rho


    # def forward(self, y: torch.Tensor):
    #     (self.weights @ self.X.T) @ (self.X @ self.weights) +


def convexProjection(X, p):
    """
    solves
    min ||wX - p||_2^2
    s.t.
    sum(w) = 1
    w_i >= 0 for all i

    :param X: (n x d) set of points
    :param p: (1 x d) point to find weights to represent with a convex combination of X
    :return weights: the barycentric weights
    """

    n, d = X.shape

    P = cvxopt.matrix(X @ X.T)


    # QP setup
    A_ = cvxopt.matrix(np.ones((1, n)))
    b_ = cvxopt.matrix(np.ones(1))
    G_ = cvxopt.matrix(-np.eye(n))
    h_ = cvxopt.matrix(np.zeros(n))

    q = cvxopt.matrix(- X @ p.T)

    sol = cvxopt.solvers.qp(P, q, G_, h_, A_, b_)

    weights = np.array(sol["x"]).reshape(1, -1)

    return weights


def cvxInterp(X, fvals, p, rho: float):
    """
    returns the interpolation of f p by finding approximate vertices of the delaunay triangulation around the point to
     estimate the function value with by using the points representation projected into the convex hull

    :param X: set of points with given function values (n x d)
    :param fvals: given function values (n x k) where k is the number of function values given
    :param p: the point to estimate the function at
    :return:
    """

    # find the vertices
    support, _ = customCVX(X, p, rho)

    # find the convex weights
    weights = convexProjection(X[support], p)

    # compute the interpolation

    return weights @ fvals[support]

def exactLPSimplex(X, p):
    """
    Solve (E) using the simplex method

    :param X: np array of shape (n, d), with data as rows
    :param p: np array of shape (1, d), the point to be represented

    :return support: an array of the indices in X containing the support of weights
    :return solved_weights: the weights to solve (E)
    """
    n = X.shape[0]
    c = np.power(np.linalg.norm(X - p, axis=1), 2).reshape(-1, 1)
    # A_ub = -eye(n)
    A_eq = np.vstack([X.T, np.ones((1, n))])
    # b_ub = np.zeros((n, 1))
    b_eq = np.concatenate([p.reshape(-1), np.ones(1)], axis=0)
    sol = linprog(c=c, A_eq=A_eq, b_eq=b_eq, method="highs")
    w = sol["x"]

    support = np.arange(n)[w > 0]
    weights = w[support]
    return support, weights

def convexHullLPSimplex(X, p):
    """
    Solves the convex hull LP using the simplex method

    :param X: np array of shape (n, d), with data as rows
    :param p: np array of shape (1, d), the point to be represented

    :return support: an array of the indices in X containing the support of weights
    """
    p = p.reshape(-1, 1)

    n, d = X.shape

    A_ub = np.ones((n, d + 1))
    A_ub[:, :d] = X

    c = np.ones((d + 1, 1))
    c[:d] = p

    b_ub = np.linalg.norm(X, axis=1)**2

    sol = linprog(c=-c, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method="highs")

    z = sol['slack']

    support = np.arange(n)[z==0]
    return support