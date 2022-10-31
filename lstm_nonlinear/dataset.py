import numpy as np

def dataset():
    series = np.sin((0.1*np.arange(400))**2)

    T = 10
    X = []
    Y = []

    for t in range(len(series) - T):
        x = series[t:t+T]
        X.append(x)

        y = series[t+T]
        Y.append(y)
    
    X = np.array(X).reshape(-1, T, 1)
    Y = np.array(Y)
    
    return series, X, Y