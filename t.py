import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.gaussian_process import GaussianProcess


TEST_SIZE = 1000
TEST_ITER = 10
def generate_x(D):
    return np.random.normal(size=(D,))

def trial(N=100, D=3, S=1):
    print "N=%d D=%d S=%.1f" % (N, D, S)
    X = np.array([generate_x(D) for _i in range(N)])
    Y = X.sum(axis=1)
    noise = np.random.normal(scale=S, size=Y.shape)
    Y += noise

    scoreLIN = []
    scoreSGD = []
    scoreSVR = []
    scoreLSVR = []
    scoreGPR = []


    for i in range(TEST_ITER):
        testX = np.array([generate_x(D) for _i in range(TEST_SIZE)])
        testY = testX.sum(axis=1)

        m = LinearRegression()
        m.fit(X, Y)
        scoreLIN.append(m.score(testX, testY))

        m = SGDRegressor()
        m.fit(X, Y)
        scoreSGD.append(m.score(testX, testY))

        m = SVR()
        m.fit(X, Y)
        scoreSVR.append(m.score(testX, testY))

        m = LinearSVR()
        m.fit(X, Y)
        scoreLSVR.append(m.score(testX, testY))

        m = GaussianProcess(nugget=0.001)
        m.fit(X, Y)
        scoreGPR.append(m.score(testX, testY))

    print "LIN %.2f(+-%.2f), SGD %.2f(+-%.2f), SVR %.2f(+-%.2f), LSVR %.2f(+-%.2f) GPR %.2f(+-%.2f)" % (
        np.mean(scoreLIN), np.std(scoreLIN) * 2,
        np.mean(scoreSGD), np.std(scoreSGD) * 2,
        np.mean(scoreSVR), np.std(scoreSVR) * 2,
        np.mean(scoreLSVR), np.std(scoreLSVR) * 2,
        np.mean(scoreGPR), np.std(scoreGPR) * 2,
    )


for S in [0.1, 1.0]:
    for D in [3, 10, 30, 100]:
        for N in [30, 100, 300, 1000]:
            trial(N, D, S)
    print

