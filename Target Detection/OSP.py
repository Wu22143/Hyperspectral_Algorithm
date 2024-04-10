import numpy as np
import matplotlib.pyplot as plt

def osp(HIM, d, no_d):
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    I = np.eye(HIM.shape[2])

    #抑制背景 建立投影矩陣P
    P = I - (no_d@np.linalg.inv( (no_d.T)@no_d ))@(no_d.T)
    #目標檢測
    x = (d.T)@P@r
    result = np.reshape(x, HIM.shape[:-1])
    plt.imshow(result , cmap='gray')
    return result


fp = r'/Users/yucheng/Code/Python/碩一下/Hyperspectral/panel.npy'
data = np.load(fp,allow_pickle=True)
item = data.item()
groundtruth = np.array(item.get('groundtruth'), 'double')
him = np.array(item.get('HIM'),'double')

# plt.imshow(groundtruth)
# plt.show()

#tree
bg1 = him[10,10,:].reshape(169,1)
bg2 = him[8,49,:].reshape(169,1)
#interferer
bg3 = him[14,5,:].reshape(169,1)
#grass
bg4 = him[36,26,:].reshape(169,1)
bg6 = him[60,60,:].reshape(169,1)
#road
bg5 = him[20,55,:].reshape(169,1)


#目標點
p11 = him[7,37,:]
p12 = him[7,47,:]
p13 = him[6,53,:]
p1 = np.reshape((p11+p12+p13)/3,(169,1))

p221 = him[20,35,:]
p222 = him[20,45,:]
p22 = him[20,52,:]
p23 = him[21,35,:]
p2 = np.reshape((p221+p222+p22+p23) / 4,(169,1))

p311 = him[33,51,:]
p312 = him[34,34,:]
p32 = him[34,35,:]
p33 = him[34,44,:]
p3 = np.reshape((p311+p312+p32+p33) / 4,(169,1))

p411 = him[47,33,:]
p412 = him[47,34,:]
p42 = him[47,43,:]
p43 = him[46,50,:]
p4 = np.reshape((p411+p412+p42+p43) / 4,(169,1))

p511 = him[59,33,:]
p512 = him[60,33,:]
p52 = him[60,43,:]
p53 = him[59,50,:]
p5 = np.reshape((p511+p512+p52+p53)/4,(169,1))


U1 = np.concatenate([p2,p3,p4,p5,bg1,bg2,bg3,bg4,bg5,bg6],axis = -1)
U2 = np.concatenate([p1,p3,p4,p5,bg1,bg2,bg3,bg4,bg5,bg6],axis = -1)
U3 = np.concatenate([p1,p2,p4,p5,bg1,bg2,bg3,bg4,bg5,bg6],axis = -1)
U4 = np.concatenate([p1,p2,p3,p5,bg1,bg2,bg3,bg4,bg5,bg6],axis = -1)
U5 = np.concatenate([p1,p2,p3,p4,bg1,bg2,bg3,bg4,bg5,bg6],axis = -1)



plt.subplot(1,5,1)
osp1 = osp(him,p1,U1)
plt.title('p1')
plt.axis("off")

plt.subplot(1,5,2)
plt.title('p2')
osp2 = osp(him,p2,U2)
plt.axis("off")

plt.subplot(1,5,3)
plt.title('p3')
osp3 = osp(him,p3,U3)
plt.axis("off")

plt.subplot(1,5,4)
plt.title('p4')
osp4 = osp(him,p4,U4)
plt.axis("off")

plt.subplot(1,5,5)
plt.title('p5')
osp5 = osp(him,p5,U5)
plt.axis("off")

plt.show()