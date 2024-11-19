def cem(HIM, d, R = None):
    '''
    Constrained Energy Minimization for image to point
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
    '''  
    #HIM:21025,200 d:1,200
    r = np.transpose(HIM) #200,21025
    d = np.reshape(d, [HIM.shape[1], 1]) #200,1
    if R is None:
        R = calc_R(HIM)
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
        print('Rinv_except')
        # warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in cem_img(), please check the input data')
    result = np.dot(np.transpose(r), np.dot(Rinv, d))/np.dot(np.transpose(d), np.dot(Rinv, d))
    result = np.reshape(result, HIM.shape[:-1])
    return result
