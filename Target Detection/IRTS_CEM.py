def IRTS_CEM(HIM, d, height, width, Xtrain_index, ytrain_index, y_reshape, max_iter=50, random_seed=6):

    prev_result = None
    # np.random.seed(random_seed)  # Set random seed for reproducibility

    for iteration in range(max_iter):
        class_result = []

        for i in range(16): #從第一類做到16類
            # 第二步驟
            x_shotsample = HIM[Xtrain_index]
            class_indices = np.where(y_reshape[ytrain_index] == i + 1)[0]
            if len(class_indices) == 0:
                print(f"No samples available for class {i + 1} in iteration {iteration + 1}.")
                break
            random_samples = np.random.choice(class_indices, size=max(1, len(class_indices)//2), replace=False)

            # 獲取選定類別的樣本
            class_samples = x_shotsample[random_samples]
            d = np.mean(class_samples, axis=0).astype('float32')
            
            # 第三步驟
            # 使用CEM計算
            cem_result = cem(HIM, d)
            cem_result = np.reshape(cem_result, (height, width, -1))
            
            # **步驟 4: 應用空間濾波**
            spatial_filter = cv2.GaussianBlur(cem_result, (11, 11), 0.5)
            class_result.append(spatial_filter)  # 將濾波後的圖層組合


        # 每類Binary圖
        Binary_map = np.zeros((16, height, width), dtype=int)

        # 第五步驟
        stack_result = np.stack(class_result, axis=-1)
        probability_map = np.argmax(stack_result, axis=-1) #總共16類
        for i in range(16):
            Binary_map[i, :, :] = probability_map == i

        # 第六步驟
        #計算DSI
        if prev_result is not None:
            total_ti = 0
            for i in range(16):  # 對每個類別計算相似度
                prev_map = prev_result[i, :, :]
                curr_map = Binary_map[i, :, :]

                # 計算交集和並集
                intersection = np.sum(prev_map & curr_map)
                union = np.sum(prev_map | curr_map)

                # 防止分母為0的情況
                ti = intersection / union
                total_ti += ti

            # 計算平均 Tanimoto Index
            dsi_value = total_ti / 16
            print(f"Iteration {iteration + 1}: TI = {dsi_value:.4f}")

            if dsi_value >= 0.85:
                print(f"Convergence reached at iteration {iteration + 1}")
                break
        
        prev_result = Binary_map
        
        # # 更新HIM (加入濾波後的結果)
        concat_result = np.reshape(class_result, (height * width, -1))
        HIM = np.concatenate((HIM, concat_result), axis=-1)

    return class_result, Binary_map