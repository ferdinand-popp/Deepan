import numpy as np

def mDA(xx, noise, lambdaa, A_n): #xx transfromed features
    rows, cols = xx.shape
    A_n = np.squeeze(np.asarray(A_n))

    #adding col with ones +1 bias
    xxb = np.concatenate((xx,np.ones((1,cols), dtype=int)),axis=0 ) # n x d+1

    #scatter matrix S
    S = xxb @ xxb.conj().T
    Sp = xxb @ A_n.conj().T @ xxb.conj().T
    Sp = np.squeeze(np.asarray(Sp))
    Sq = xxb @ A_n @ A_n.conj().T @ xxb.conj().T #nxn
    Sq = np.squeeze(np.asarray(Sq))

    #corruption vector
    row1 = rows + 1
    q = np.ones((row1 ,1)) * (1-noise)
    q[-1]=1 #element

    #Q
    Q = Sq * (q * q.conj().T)
    Q = np.squeeze(np.asarray(Q)) # to array
    qdiagn = q * np.diag(Sq)
    Q[0:-1:rows+2, :] = qdiagn

    p1 = Sp[0:-1, :]
    p1= np.squeeze(np.asarray(p1))
    p2 = np.tile(q.conj().T, (rows, 1))
    P = p1 * p2

    reg = lambdaa * np.eye(rows +1)
    reg[-1:-1] = 0
    W = P @ np.linalg.pinv(Q+reg)

    hx = W @ xxb @ A_n
    print(hx)
    return hx

xx= np.random.rand(3, 12).T
A_n  = np.matrix([[1,0,1],
       [0,1,0],
       [1,0,1]])
mDA(xx, 0.2, 0.005, A_n)
