import cv2
import numpy as np

def test_stereo_cameras():
    """双目相机实时深度检测脚本"""

    # 相机内参
    K1 = np.array([[709.994, 0, 325.6164],
                   [0., 710.3085, 232.6571],
                   [0., 0., 1.]], dtype=np.float64)

    K2 = np.array([[702.8081, 0, 326.3538],
                   [0., 703.5740, 236.9871],
                   [0., 0., 1.]], dtype=np.float64)

    D1 = np.array([0.0177, 0.3754, -0.0061, 0.0003, -2.3048], dtype=np.float64)
    D2 = np.array([0.017, 0.3864, -0.0013, -0.0018, -2.486], dtype=np.float64)

    # 基线：保持单位为 cm
    R = np.array([[1, -0.0009, 0.0042],
                  [0.0009, 1, -0.0014],
                  [-0.0042, 0.0014, 1]], dtype=np.float64)
    T = np.array([[24.4242], [0.1478], [-1.8293]], dtype=np.float64)  # 单位 cm

    img_width, img_height = 640, 480

    # 校正
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, (img_width, img_height), R, T, alpha=0)
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (img_width, img_height), cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (img_width, img_height), cv2.CV_16SC2)

    # 立体匹配器
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # 相机
    cap_left = cv2.VideoCapture(1)
    cap_right = cv2.VideoCapture(2)

    # 设置相机参数（数值范围依赖于摄像头，常见为0~1或0~255）
    cap_left.set(cv2.CAP_PROP_BRIGHTNESS, 128)   # 亮度
    cap_left.set(cv2.CAP_PROP_CONTRAST, 128)     # 对比度
    cap_left.set(cv2.CAP_PROP_SATURATION, 128)   # 饱和度
    if hasattr(cv2, 'CAP_PROP_SHARPNESS'):
        cap_left.set(cv2.CAP_PROP_SHARPNESS, 128)  # 锐度（部分摄像头支持）

    cap_right.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    cap_right.set(cv2.CAP_PROP_CONTRAST, 128)
    cap_right.set(cv2.CAP_PROP_SATURATION, 128)
    if hasattr(cv2, 'CAP_PROP_SHARPNESS'):
        cap_right.set(cv2.CAP_PROP_SHARPNESS, 128)

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("无法打开摄像头")
        return

    print("实时深度检测开始，按 'q' 退出，按 's' 保存帧")

    while True:
        retL, frameL = cap_left.read()
        retR, frameR = cap_right.read()
        if not retL or not retR:
            print("帧获取失败")
            break

        # 镜像翻转
        frameL = cv2.flip(frameL, 1)
        frameR = cv2.flip(frameR, 1)

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # 校正
        rectL = cv2.remap(grayL, map1_left, map2_left, cv2.INTER_LINEAR)
        rectR = cv2.remap(grayR, map1_right, map2_right, cv2.INTER_LINEAR)

        # 视差图
        disparity = stereo.compute(rectL, rectR).astype(np.float32) / 16.0
        disparity[disparity < 1.0] = np.nan

        # 重建深度图
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        depth_map = points_3d[:, :, 2]

        # --- 新增：用ROI裁剪 ---
        x, y, w, h = roi1
        rectL_roi = rectL[y:y+h, x:x+w]
        disparity_roi = disparity[y:y+h, x:x+w]
        depth_map_roi = depth_map[y:y+h, x:x+w]
        frameL_roi = frameL[y:y+h, x:x+w]  # 如果你想显示原彩色图

        # 这里插入高亮屏蔽
        highlight_mask = rectL_roi > 150  # 或用cv2.cvtColor(frameL_roi, cv2.COLOR_BGR2GRAY) > 220
        lowlight_mask = rectL_roi < -1
        invalid_mask = highlight_mask | lowlight_mask
        disparity_roi[invalid_mask] = np.nan
        depth_map_roi[invalid_mask] = np.nan

        # 统计 0.5~8 米之间的有效深度
        valid_mask = (depth_map_roi > 500) & (depth_map_roi < 8000)
        valid_depths = depth_map_roi[valid_mask] / 1000.0  # 转换为米

        if valid_depths.size > 0:
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            mean_depth = np.mean(valid_depths)
            print(f"[有效深度像素: {valid_depths.size}] 范围: {min_depth:.2f}m - {max_depth:.2f}m, 均值: {mean_depth:.2f}m")
        else:
            print("⚠️ 无有效深度值")

        # 可视化深度图（0.5~8米映射为0~255）
        depth_vis = np.clip(depth_map_roi, 500, 8000).astype(np.uint16)
        depth_vis = ((depth_vis - 500) / (8000 - 500) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # 显示
        cv2.imshow("Left", frameL_roi)
        cv2.imshow("Right", frameR)
        cv2.imshow("Rectified Left", rectL_roi)
        cv2.imshow("Disparity", cv2.normalize(disparity_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        cv2.imshow("Depth Map", depth_colored)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("frame_left.jpg", frameL_roi)
            cv2.imwrite("frame_right.jpg", frameR)
            cv2.imwrite("depth_colored.jpg", depth_colored)
            print("✅ 已保存当前帧")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_stereo_cameras()
