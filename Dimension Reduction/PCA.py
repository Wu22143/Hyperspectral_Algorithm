import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

fp = r'/Users/yucheng/Code/Python/碩一下/Hyperspectral/panel.npy'
data = np.load(fp,allow_pickle=True)
item = data.item()
groundtruth = np.array(item.get('groundtruth'), 'double')
data= np.array(item.get('HIM'),'double')
data = np.reshape(data, (-1, data.shape[2]))

def pca(X,n_components):
    n_features = X.shape[1]

    mean = np.array([np.mean(X[:,i]) for i in range(n_features)])

    #正規化
    norm_X = X-mean

    #共變異數矩陣
    covariance_matrix = np.dot(np.transpose(norm_X),norm_X)

    #求出特徵值與特徵向量
    val , vec = np.linalg.eig(covariance_matrix)

    eig_pairs = [(np.abs(val[i]), vec[:,i]) for i in range(n_features)]
    #由大到小排序 , 依照eig_val
    eig_pairs.sort(reverse=True)

    #根據k選出前面最大方差的特徵，featrue vector
    feature = np.array([ele[1] for ele in eig_pairs[:n_components]])
    
    #將正規化後的資料投影到子空間中
    data = np.dot(norm_X, np.transpose(feature))
    return data


pca_result = pca(data, 5)
print(pca_result.shape)
# 顯示前 5 個主成分
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(np.reshape(pca_result[:, i], (64, 64)),cmap='gray')
    plt.title(f'PC{i+1}')
    plt.axis('off')
plt.show()