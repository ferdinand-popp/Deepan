import numpy as np

def mDA(xx, noise, lambdaa, A_n): #xx transfromed features
    rows, cols = xx.shape

    #adding col with ones +1 bias
    xxb = np.concatenate((xx,np.ones((1,cols), dtype=int)),axis=0 ) # n x d+1

    #scatter matrix S
    S = xxb @ xxb.conj().T
    Sp = xxb @ A_n.conj().T @ xxb.conj().T
    Sq = xxb @ A_n @ A_n.conj().T @ xxb.conj().T #nxn

    #corruption vector
    row1 = rows + 1
    q_1 = np.ones((row1 ,1))
    q = q_1 @ 0.8
    q[-1]=1 #element

    #Q
    Q = np.multiply(Sq, (q @ q.conj().T))
    qdiagn = np.multiply(q, np.diag(Sq) )
    Q[1:-1:rows+2] = qdiagn

    P = np.multiply(Sp[1:-1, :], np.matlib.repmat(q.conj().T, rows, 1))

    reg = lambdaa @ np.eye(rows +1)
    reg[-1:-1] = 0
    W = P @ np.linalg.pinv(Q+reg)

    hx = W @ xxb @ A_n

    return hx


