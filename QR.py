# https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
def gram_schmidt(S, orthonormal=False):
    '''
    returns a set of orthogonal basis for given span
    Keyword Arguments:
        -- S: a list of arrays
        -- orthonormal: an boolean indicating whether or not to return the normalized basis
    '''
    basis = []
    for i,v in enumerate(S):
        if i == 0:
            basis.append(v)
        else:
            basis.append(v - sum(float(np.dot(u,v))/np.dot(u,u) * u for u in basis))
        if orthonormal:
            basis[i] = np.divide(basis[i], (sum(x**2 for x in basis[i]))**0.5)

    return basis


# https://en.wikipedia.org/wiki/QR_decomposition
def QR(A):
    '''
    returns the Q and R matrices after applying QR factorization
    Keyword Arguments:
        -- A: a symmetric matrix
    '''
    # get the column vectors
    S = list(A.T)
    # get the basis vectors
    basis = gram_schmidt(S, orthonormal=True)
    Q = np.array(basis).T # orthonormal matrix
    # R is upper triangular
    R = np.dot(Q.T, A)

    return Q, R


def QR_eig(M, ap=1e-6, rp=1e-5):
    '''
    returns the eigen values and their corresponding eigen vectors using
    the QR-algorithm, which uses QR-factorization. Each row is an eigen vector.
    Keyword Arguments:
        -- M: a square matrix. Numpy Array
        -- ap: the absolute precision
        -- rp: the relative precision
    '''
    # check if matrix is square
    n,m = M.shape
    assert n == m, 'Matrix is not square'
    M_updated = M
    Q_updated = np.identity(n)
    while True:
        # find max val in lower tri of M to decide stopping
        max_val = np.abs(np.tril(M_updated, k=-1)).max()
        if max_val < ap or max_val < rp*max_val: break
        # get the Q (orthonormal mat) and R (Upper trig) matrices
        Q, R = QR(M_updated)
        Q_updated = np.dot(Q_updated, Q)
        M_updated = np.dot(np.dot(Q.T, M_updated), Q)

    # get indices of most to least significant eig vals for sorting
    sort_indices = M_updated.diagonal().argsort()[::-1]

    return M_updated.diagonal()[sort_indices], Q_updated.T[sort_indices]
