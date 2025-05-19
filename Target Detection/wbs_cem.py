def wbs_cem(HIM, d):
    # print(f"newHIM.shape={newHIM.shape}\n")
    d = np.reshape(d, [HIM.shape[2], 1])
    N = HIM.shape[0]*HIM.shape[1]
    Rsum = np.full((HIM.shape[2], HIM.shape[2]), 0.0)

    # Faster method
    ri = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    ri = ri.astype(np.int64)
    d = d.astype(np.int64)
    ri -= d
    wi = np.dot((ri), np.transpose(ri))
    # print(f"wi.shape = {wi.shape}")
    Rstar = 1/N*(wi@ri@ri.T)
    # print(f"R.shape = {Rstar.shape}")
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    result = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(d), np.linalg.inv(Rstar)), d))  ,  np.transpose(np.dot(np.linalg.inv(Rstar), d))), r)
    result = np.reshape(result, HIM.shape[:-1])
    return result

    # # R*
    # for h in range(0, HIM.shape[0]):
    #     for w in range(0, HIM.shape[1]):
    #         ri = HIM[h, w, :]
    #         # print(f"ri.shape={ri.shape}")
    #         x = ri-d
    #         wi = np.dot(np.transpose(x), x)
    #         Rsum = Rsum + np.dot(wi, np.dot(ri, np.transpose(ri)))
    # Rstar = (1/N) * Rsum
    # # algo.
    # result = np.full((HIM.shape[0], HIM.shape[1]), 0.0)
    # for h in range(0, HIM.shape[0]):
    #     for w in range(0, HIM.shape[1]):
    #         ri = HIM[h, w, :]
    #         result[h, w] = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.transpose(d), np.linalg.inv(Rstar)), d))  ,  np.transpose(np.dot(np.linalg.inv(Rstar), d))), ri)
    # # print(f"result={result}, result.shape={result.shape}")
    # return result