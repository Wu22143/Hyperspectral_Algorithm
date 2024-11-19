def ICEM(HIM, d, height, width, Xtrain_index, ytrain_index, y_reshape, j,  max_iter=100):
    prev_result = None

    normal_d = d
    for iteration in range(max_iter):

        # 使用更新的 R 进行 CEM
        cem_result = cem(HIM, d) #21025, 200

        # sam_sid_result = sam_sid(HIM, d)
        # sam_sid_result[np.isnan(sam_sid_result)] = calculate_mean(sam_sid_result)

        # cem_result = (cem_result / sam_sid_result).astype('float32')
        
        #Reshape to (height , width , bands)
        cem_result = np.reshape(cem_result, (height, width, -1))

        # Gaussian filter
        result_gaussian = cv2.GaussianBlur(cem_result,(11,11), 0.5)

        # OTSU 計算相似度
        threshold_value = threshold_otsu(result_gaussian)
        compare_result = result_gaussian > threshold_value

        if prev_result is not None:
            dsi_value = np.sum(compare_result & prev_result) / np.sum(compare_result | prev_result)
            print(f"Iteration {iteration+1}: DSI = {dsi_value}")
            if dsi_value >= 0.85:
                print(f"Convergence reached at iteration {iteration + 1}")
                break
        prev_result = compare_result


        #迭代更新HIM
        concat_result = np.reshape(result_gaussian, (height*width, -1))
        HIM = np.concatenate((HIM, concat_result), axis=-1)

        #重新計算目標光譜
        y_shotsample = y_reshape[ytrain_index]
        x_shotsample = HIM[Xtrain_index]
        class_indices = np.where(y_shotsample == j+1)[0]
        class_samples = x_shotsample[class_indices]


        d = np.mean(class_samples, axis=0).astype('float32')
        
    return cem_result
