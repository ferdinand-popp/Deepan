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
    np.fill_diagonal(Q[0:-1, 0:-1], qdiagn)

    p1 = Sp[0:-1, :]
    p1= np.squeeze(np.asarray(p1))
    p2 = np.tile(q.conj().T, (rows, 1))
    P = p1 * p2

    reg = lambdaa * np.eye(rows +1)
    reg[-1:-1] = 0
    W = P @ np.linalg.pinv(Q+reg)

    hx = W @ xxb @ A_n
    print(hx)
    return hx, W



def mSDA(xx, noise, layers, A_n):

    lam = 1e-5
    prevhx = xx
    allhx = []
    Ws={}
    for layer in range(layers-1):
        newhx, W = mDA(prevhx,noise,lam,A_n);
        Ws[layer] = W
        allhx.append(newhx)
        prevhx = newhx

    return allhx, Ws

#xx= np.random.rand(3, 12).T
xx = [[0.5, 0.6, 0.7], [0.2, 0.3, 0.1], [0.9, 0.8, 0.7], [0.4, 0.9, 0.8], [0.1, 0.4, 0.5], [0.5, 0.6, 0.7], [0.2, 0.3, 0.1], [0.9, 0.8, 0.7], [0.4, 0.9, 0.8], [0.1, 0.4, 0.5], [0.9, 0.8, 0.7], [0.4, 0.9, 0.8]]
xx = np.array(xx)
A_n  = np.matrix([[1,0,1],
       [0,1,1],
       [1,1,1]])

allhx, W = mSDA(xx, 0.2, 5, A_n)
