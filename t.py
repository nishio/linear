import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
TEST_SIZE = 1000
TEST_ITER = 10
def generate_x(D):
    return np.random.normal(size=(D,))

def trial(N=100, D=3, S=1):
    print "N=%d D=%d S=%.1f" % (N, D, S)
    X = np.array([generate_x(D) for _i in range(N)])
    Y = X.sum(axis=1)
    noise = np.random.normal(scale=S, size=X.shape)
    X += noise

    scoreSGD = []
    scoreSVR = []
    for i in range(TEST_ITER):
        testX = np.array([generate_x(D) for _i in range(TEST_SIZE)])
        testY = testX.sum(axis=1)

        m = SGDRegressor()
        m.fit(X, Y)
        #mse = ((m.predict(testX) - testY) ** 2).sum() / TEST_SIZE
        scoreSGD.append(m.score(testX, testY))

        m = SVR()
        m.fit(X, Y)
        scoreSVR.append(m.score(testX, testY))

    print "SGD %.2f(+-%.2f), SVR %.2f(+-%.2f)" % (
        np.mean(scoreSGD), np.std(scoreSGD),
        np.mean(scoreSVR), np.std(scoreSVR))


for S in [0.1, 0.3, 1.0, 3.0]:
    for D in [3, 10, 30, 100]:
        for N in [30, 100, 300, 1000]:
            trial(N, D, S)
    print

