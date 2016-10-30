import numpy as np
import matplotlib.pyplot as plt 

def compute_dim_quadratic_array(D):
    sum_ = 0;
    for i in range(D):
        sum_ += i
    return sum_ + 2*D


def build_quadratic_array(x):
    N = x.shape[0]
    D = x.shape[1]
    col_nb = compute_dim_quadratic_array(D)
    
    tPhi = np.ones((N, col_nb))
    for i in range(N): # toutes les lignes
        for j in range(D):
            tPhi[i][j] = x[i][j]
            
        idx = D
        for j in range(D):
            for k in range(D-j):
                if j != (D - 1 - k):
                    tPhi[i][idx] = x[i][j] * x[i][D -1 - k]
                    idx += 1
        
        for j in range(D):
            tPhi[i][col_nb - D + j] = x[i][j] * x[i,j]
            
    return tPhi

def select_features(x, y, coef_corr):
    """Return index of columns which correlation coefficient to y is above coef_corr"""

    y_X = np.c_[y, x]
    corr_np = np.corrcoef(y_X.T)
    corr_np_y_X = corr_np[0] # correlation between y and D features
    corr_np_y_X = corr_np_y_X[1:] # remove y
    
    plt.plot(corr_np_y_X)
    plt.axhline(0,color='#FF8C00', linestyle='--')
    plt.xlabel("Features")
    plt.ylabel("Correlation with y") 

    indices = []
    for i in range(1, len(corr_np_y_X)):
        if np.absolute(corr_np_y_X[i]) > coef_corr:
            indices.append(i)


    return indices