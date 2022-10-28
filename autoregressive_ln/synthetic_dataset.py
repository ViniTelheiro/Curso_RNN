import numpy as np




def synthetic_dataset(noise:bool=True):
##make a synthetic dataset

    if noise:
        series = np.sin(0.1*np.arange(200)) + np.random.randn(200)*0.1
    else:
        series = np.sin(0.1*np.arange(200))

    #build dataset:
    T = 10
    X = []
    Y = []
    for t in range(len(series) - T):
        x = series[t:t+T]
        X.append(x)
        y = series[t+T]
        Y.append(y)
    X = np.array(X).reshape(-1, 10)
    Y = np.array(Y)
    return series, X, Y