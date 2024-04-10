import numpy as np
import matplotlib.pyplot as plt

def calc_R(HIM, targets):
    try:
        mask = np.ones(HIM.shape[:2], dtype=bool)
        for target in targets:
            # x, y = np.unravel_index(np.argmax(target), HIM.shape[:2])
            mask[target[0], target[1]] = False
        N = np.sum(mask)
        r = np.transpose(np.reshape(HIM[mask], [-1, HIM.shape[2]]))
        R = 1/N*(r@r.T)
        return R
    except:
        print('An error occurred in calc_R()')

def calc_K_u(HIM, targets):
    try:
        mask = np.ones(HIM.shape[:2], dtype=bool)
        for target in targets:
            mask[target[0], target[1]] = False
        
        N = np.sum(mask)
        r = np.transpose(np.reshape(HIM[mask], [-1, HIM.shape[2]]))
        u = (np.mean(r, 1)).reshape(HIM.shape[2], 1)        
        K = 1/N*np.dot(r-u, np.transpose(r-u))
        return K, u
    except:
        print('An error occurred in calc_K_u()')

def R_rxd(HIM, targets, R = None, axis = ''):
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    rt = np.transpose(r)
    
    if R is None:
        R = calc_R(HIM, targets)  # 不排除任何点
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
    
    if axis == 'N':
        n = np.sum((rt*rt), 1)
        result = np.sum(((np.dot(rt, Rinv))*rt), 1)
        result = result/n
    elif axis == 'M':
        n = np.power(np.sum((rt*rt), 1), 0.5)
        result = np.sum(((np.dot(rt, Rinv))*rt), 1)
        result = result/n
    else:
        result = np.sum(((np.dot(rt, Rinv))*rt), 1)
    result = np.reshape(result, HIM.shape[:-1])
    plt.imshow(result, cmap='gray')
    return result

def K_rxd(HIM, targets, K = None, u = None, axis = ''):

    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    if K is None or u is None:
        K, u = calc_K_u(HIM, targets)  # 不排除任何点
    ru = r-u
    rut = np.transpose(ru)
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
    
    if axis == 'N':
        n = np.sum((rut*rut), 1)
        result = np.sum(((np.dot(rut, Kinv))*rut), 1)
        result = result/n
    elif axis == 'M':
        n = np.power(np.sum((rut*rut), 1), 0.5)
        result = np.sum(((np.dot(rut, Kinv))*rut), 1)
        result = result/n
    else:
        result = np.sum((np.dot(rut, Kinv))*rut, 1)
    result = np.reshape(result, HIM.shape[:-1])
    plt.imshow(result , cmap='gray')
    return result

fp = r'/Users/yucheng/Code/Python/碩一下/Hyperspectral/panel.npy'
data = np.load(fp,allow_pickle=True)
item = data.item()
groundtruth = np.array(item.get('groundtruth'), 'double')
him = np.array(item.get('HIM'),'double')

#目標點
p11 = (7,37)
p12 = (7,47)
p13 = (6,53)

p221 = (20,35)
p222 = (20,45)
p22 = (20,52)
p23 = (21,35)

p311 = (33,51)
p312 = (34,34)
p32 = (34,35)
p33 = (34,44)

p411 = (47,33)
p412 = (47,34)
p42 = (47,43)
p43 = (46,50)


p511 = (59,32)
p512 = (60,32)
p52 = (60,42)
p53 = (59,49)
targets = [p11, p12, p13, p221, p222, p22, p23, p311, p312, p32, p33, p411, p412, p42, p43, p511, p512, p52, p53]

plt.subplot(1,2,1)
r_rxd1 = R_rxd(him, [])
plt.title('R-RXD_Normal')
plt.axis("off")

plt.subplot(1,2,2)
plt.title('R-RXD_Remove')
k_rxd1 = R_rxd(him, targets)
plt.axis("off")
plt.show()

plt.subplot(1,2,1)
r_rxd1 = K_rxd(him, [])
plt.title('K-RXD_Normal')
plt.axis("off")

plt.subplot(1,2,2)
plt.title('K-RXD_Remove')
k_rxd1 = K_rxd(him, targets)
plt.axis("off")
plt.show()

