import numpy as np

def mDA(xx, noise, lambdaa, A_n): #xx transfromed features
    rows, cols = xx.shape

    #adding col with ones +1 bias
    xxb = np.concatenate((xx,np.ones((1,cols), dtype=int)),axis=0 ) # n x d+1

    #scatter matrix S
    S = xxb * xxb.T
    Sp = xxb * A_n.T * xxb.T
    Sq = xxb * A_n * A_n.T * xxb.T #nxn

    #corruption vector
    q = np.ones((rows + 1 ,1), dtype=int) * (1 - noise)
    q[-1]=1 #element

    #Q
    Q = np.multiply(Sq, (q * q.T))
    Q[1:-1:rows+2] = np.multiply(q, np.diag(Sq) )

    P = np.multiply(Sp[1:-1, :], np.matlib.repmat(q.T, rows, 1))

    reg = lambdaa * np.eye(rows +1)
    reg[-1:-1] = 0
    W = P * np.linalg.pinv(Q+reg)

    hx = W * xxb* A_n

    return hx

xx= np.array([(1,2,3), (4,5,6)]) #transformed features

a_n = np.array([(1, 0, 1), (0,1,0), (1, 0, 1)])

mDA(xx, 0.2, 0.005, a_n)