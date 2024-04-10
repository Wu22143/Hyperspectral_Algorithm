import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA

def fastICA(X, n_components, maxIter=200, tol=1e-04):
    
    #Center the data
    X = X - np.mean(X, axis=-1)[:, np.newaxis]

    # compute the covariance matrix
    cov_matrix = np.cov(X.T)
    # compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # normalize the eigenvectors
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)


    # Whiten the data
    X1 = np.dot(X, eigenvectors)


    # init random weights matrix
    w = np.random.rand(n_components, X1.shape[1])

    for c in range(n_components):
        for ii in range(maxIter):
            wOld = w[c, :].copy()

            # fourth moment
            fourth_moment = np.mean((np.dot(X1, w[c, :])[:, np.newaxis]**4), axis=0)

            # second moment
            second_moment = np.mean((np.dot(X1, w[c, :])[:, np.newaxis]**2), axis=0)

            # calcuate kurtosis
            kurtosis = fourth_moment - 3 * (second_moment**2)

            # update weights matrix
            w[c, :] += tol * kurtosis * w[c, :]
            
            # Decorrelate weights
            if c > 0:
                w[c, :] -= np.dot(np.dot(w[c, :], w[:c].T), w[:c])
            
            # Normalize weights
            w[c, :] /= np.sqrt(np.dot(w[c, :], w[c, :]))
            
            # Check for convergence
            if np.abs(np.dot(wOld, w[c, :])) - 1 < tol:
                break
    
    S = np.dot(X1, w.T)
    return S

# 讀取圖像數據
fp = r'/Users/yucheng/Code/Python/碩一下/Hyperspectral/panel.npy'
data = np.load(fp, allow_pickle=True)

item = data.item()
groundtruth = np.array(item.get('groundtruth'), 'double')
data = np.array(item.get('HIM'), 'double')

# 將圖像數據reshape為(4096, 169)的形狀
data = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]))

# 使用自實現的ICA進行獨立成分分析，提取5個獨立成分
n_components = 10
ica_result = fastICA(data, n_components)


x = FastICA(n_components=n_components)
fast = x.fit_transform(data)

# 顯示前5個獨立成分
plt.figure(figsize=(15, 6))

# 顯示第一組獨立成分
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.reshape(ica_result[:, i], (64, 64)),cmap='gray')
    plt.title(f'IC{i+1} (Self)')
    plt.axis('off')

# 顯示第二組獨立成分
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(np.reshape(fast[:, i], (64, 64)),cmap='gray')
    plt.title(f'IC{i+1} (FastICA)')
    plt.axis('off')

plt.tight_layout()
plt.show()

