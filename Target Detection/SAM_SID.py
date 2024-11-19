#SID_SAM
def sid_img(HIM, d):
    """
    Spectral Information Divergence for image to point

    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, class], for example: [224, 1]
    """
    if len(HIM.shape) == 2:
        r = np.expand_dims(HIM.T, 1)
    else:
        r = HIM

    if len(d.shape) == 1:
        d = d[:, None, None]
    else:
        d = np.expand_dims(d.T, 1)
    m = r / np.sum(r, 0)
    n = d / np.sum(d, 0)
    drd = np.sum(m * np.log(m / n), 0)
    ddr = np.sum(n * np.log(n / m), 0)
    
    result = drd + ddr
    return result

def sam_img(HIM, d):
    if len(HIM.shape) == 2:
        HIM = HIM.T[..., None]
    else:
        HIM = HIM.T

    if len(d.shape) == 1:
        d = d[:, None, None]
    else:
        d = np.expand_dims(d.T, 1)

    rr = np.sum(HIM**2, 0) ** 0.5
    dd = np.sum(d**2, 0) ** 0.5
    rd = np.sum(HIM * d, 0)

    result = (np.arccos(rd / (rr * dd))).T
    return result

#SID 乘上 TAN(SAM)
def sam_sid(m1, m2):
    """
    SID-SAM Mixed Measure for point to point
    SID(SIN) = SID(s, s') x sin(SAM(s, s'))

    param m1: hyperspectral imaging, type is 3d-array
    param m2: desired target d (Desired Signature), type is 2d-array, size is [band num, class], for example: [224, 1]
    """
    sam_res = sam_img(m1, m2)
    sid_res = sid_img(m1, m2)
    result = sid_res * np.tan(sam_res)

    return result
