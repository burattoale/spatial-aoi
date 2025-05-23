import numpy as np

def _compute_steady_state(A:np.ndarray):
        dim = A.shape[0]
        q = (A-np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q,ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        return np.linalg.solve(QTQ,bQT)