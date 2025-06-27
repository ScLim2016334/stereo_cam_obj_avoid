import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import deque
import time

class StereoObstacleDetector:
    def __init__(self):
        # 初始化摄像头参数
        # 封装好的相机
        self.cam_matrix_left = np.array([[709.994, 0, 325.6164],
                                         [0., 710.3085, 232.6571],
                                         [0., 0., 1.]])
        self.cam_matrix_right = np.array([[702.8081, 0, 326.3538],
                                          [0., 703.5740, 236.9871]  ,
                                          [0., 0., 1.]])
        self.distortion_l = np.array([[0.0177, 0.3754, -0.0061, 0.0003, -2.3048]])
        self.distortion_r = np.array([[0.017, 0.3864, -0.0013, -0.0018, -2.486]])
        self.R = np.array([[1, -0.0009, 0.0042],
                          [0.0009, 1, -0.0014],
                          [-0.0042, 0.0014, 1]])
        self.T = np.array([[24.4242], [0.1478], [-1.8293]])

        # 未封装好的相机
        # self.cam_matrix_left = np.array([[663.9068, 0, 638.6504],
        #                                  [0., 665.9531, 366.4541],
        #                                  [0., 0., 1.]])
        # self.cam_matrix_right = np.array([[662.6703, 0, 627.8173],
        #                                   [0., 664.4008, 359.9967]  ,
        #                                   [0., 0., 1.]])
        # self.distortion_l = np.array([[0.1608, 0.0125, -0.0013, -0.0052, 0]])
        # self.distortion_r = np.array([[0.138, 0.0785, -0.0006, -0.0151, 0]])
        # self.R = np.array([[0.9999, 0.0001, 0.0107],
        #                   [-0.0001, 1, -0.0048],
        #                   [-0.0107, 0.0048, 0.9999]])
        # self.T = np.array([[24.4485], [-0.163], [-1.1072]])
        
        # 立体校正
        self.init_rectification()
        
        # SGBM参数配置
        self.window_size = 5
        self.min_disp = 0
        self.num_disp = 160 - self.min_disp
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.window_size,
            P1=8 * 3 * self.window_size**2,
            P2=32 * 3 * self.window_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63
        )
        
        # 点云处理参数
        self.depth_limit = 8.0  # 最大深度
        self.height_limits = [0.15, 1.8]  # 高度过滤范围
        self.voxel_size = 0.05  # 体素大小5cm
        self.neighbor_radius = 5.0  # 邻域半径 #0.5
        self.min_neighbors = 200   # 最小邻居数 #30
        
        # 动态检测参数
        self.motion_threshold = 0.45  # 运动阈值 m/s
        self.dynamic_vote_threshold = 0.8  # 动态投票阈值
        self.track_history = deque(maxlen=10)  # 跟踪历史
        
        # 行人检测器
        self.person_detector = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt", "mobilenet_iter_73000.caffemodel"  # 需下载预训练模型
        )
        
        # 卡尔曼滤波器
        self.kalman_filters = {}
        
        # 3D跟踪
        self.object_tracks = {}
        self.next_track_id = 0
    
    def init_rectification(self):
        """初始化立体校正参数"""
        image_size = (640, 480)  # 根据实际图像尺寸调整
        
        # 计算立体校正映射
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.cam_matrix_left, self.distortion_l,
            self.cam_matrix_right, self.distortion_r,
            image_size, self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
        )
        
        # 计算校正映射
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.cam_matrix_left, self.distortion_l, R1, P1, image_size, cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            self.cam_matrix_right, self.distortion_r, R2, P2, image_size, cv2.CV_32FC1
        )
        
        self.Q = Q
    
    def rectify_images(self, left_img, right_img):
        """校正双目图像"""
        left_rect = cv2.remap(left_img, self.map1x, self.map1y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return left_rect, right_rect
    
    def compute_disparity(self, left_img, right_img):
        """计算视差图"""
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # 计算视差
        disp = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # 后处理
        disp = cv2.medianBlur(disp, 5)
        disp = cv2.threshold(disp, 0, self.num_disp, cv2.THRESH_TOZERO)[1]
        return disp
    
    def disparity_to_pointcloud(self, disparity, left_img):
        """将视差图转换为点云"""
        # 使用重投影矩阵生成3D点云
        points_3D = cv2.reprojectImageTo3D(disparity, self.Q)
        
        # 创建有效点掩码
        mask = disparity > disparity.min()
        
        # 应用掩码
        points = points_3D[mask]
        colors = left_img[mask]
        
        return points, colors
    
    def filter_pointcloud(self, points):
        """过滤点云"""
        # 1. 深度过滤
        depth_mask = points[:, 2] < self.depth_limit
        
        # 2. 高度过滤
        height_mask = (points[:, 1] > self.height_limits[0]) & (points[:, 1] < self.height_limits[1])
        
        # 3. 组合掩码
        filtered_mask = depth_mask & height_mask
        filtered_points = points[filtered_mask]
        
        # 4. 体素滤波 (简化实现)
        if len(filtered_points) > 0:
            voxel_grid = {}
            for point in filtered_points:
                voxel_coord = tuple((point // self.voxel_size).astype(int))
                if voxel_coord not in voxel_grid:
                    voxel_grid[voxel_coord] = []
                voxel_grid[voxel_coord].append(point)
            
            # 取每个体素中心点
            filtered_points = np.array([np.mean(points, axis=0) for points in voxel_grid.values()])
        
        return filtered_points
    
    def cluster_points(self, points):
        """使用DBSCAN聚类点云"""
        if len(points) == 0:
            return []
        
        # 仅使用x,z坐标进行水平面聚类
        db = DBSCAN(eps=0.2, min_samples=10).fit(points[:, [0, 2]])
        labels = db.labels_
        
        # 获取唯一标签
        clusters = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
            cluster_mask = (labels == label)
            cluster_points = points[cluster_mask]
            clusters.append(cluster_points)
        
        return clusters
    
    def detect_persons(self, image):
        """检测行人"""
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        self.person_detector.setInput(blob)
        detections = self.person_detector.forward()
        
        persons = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 置信度阈值
                class_id = int(detections[0, 0, i, 1])
                if class_id == 15:  # COCO数据集中人的类别ID
                    box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], 
                                                             image.shape[1], image.shape[0]])
                    persons.append(box.astype(int))
        
        return persons
    
    def classify_dynamic(self, clusters, prev_clusters):
        """分类动态/静态物体"""
        dynamic_clusters = []
        
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            
            # 查找前一帧中最近的簇
            min_dist = float('inf')
            for prev_cluster in prev_clusters:
                prev_centroid = np.mean(prev_cluster, axis=0)
                dist = np.linalg.norm(centroid[:2] - prev_centroid[:2])
                if dist < min_dist:
                    min_dist = dist
            
            # 计算速度 (m/s)
            velocity = min_dist / (1/30.0)  # 假设30fps
            
            # 分类
            if velocity > self.motion_threshold:
                dynamic_clusters.append(cluster)
        
        return dynamic_clusters
    
    def track_objects(self, clusters, frame_time):
        """跟踪物体并更新卡尔曼滤波器"""
        current_objects = []
        
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            current_objects.append(centroid)
            
            # 查找最近的现有轨迹
            min_dist = float('inf')
            track_id = None
            for tid, track in self.object_tracks.items():
                last_pos = track['positions'][-1]
                dist = np.linalg.norm(centroid[:2] - last_pos[:2])
                if dist < min_dist and dist < 0.5:  # 匹配阈值0.5m
                    min_dist = dist
                    track_id = tid
            
            if track_id is not None:
                # 更新现有轨迹
                self.object_tracks[track_id]['positions'].append(centroid)
                self.object_tracks[track_id]['last_seen'] = frame_time
                
                # 更新卡尔曼滤波器
                if track_id in self.kalman_filters:
                    kf = self.kalman_filters[track_id]
                    kf.predict()
                    kf.correct(centroid[:2])
            else:
                # 创建新轨迹
                track_id = self.next_track_id
                self.next_track_id += 1
                self.object_tracks[track_id] = {
                    'positions': [centroid],
                    'start_time': frame_time,
                    'last_seen': frame_time,
                    'static': False
                }
                
                # 初始化卡尔曼滤波器
                kf = cv2.KalmanFilter(4, 2)
                kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
                kf.statePre = np.array([centroid[0], centroid[1], 0, 0], dtype=np.float32)
                self.kalman_filters[track_id] = kf
        
        # 清理旧轨迹
        max_age = 2.0  # 最大未跟踪时间
        for tid in list(self.object_tracks.keys()):
            if frame_time - self.object_tracks[tid]['last_seen'] > max_age:
                del self.object_tracks[tid]
                if tid in self.kalman_filters:
                    del self.kalman_filters[tid]
        
        return current_objects
    
    def generate_occupancy_grid(self, static_clusters, dynamic_clusters, grid_size=100, resolution=0.1):
        """生成2D占据栅格地图"""
        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        center = grid_size // 2
        
        # 添加静态障碍物
        for cluster in static_clusters:
            for point in cluster:
                x, z = point[0], point[2]  # 使用x和z坐标
                grid_x = int(center + x / resolution)
                grid_z = int(z / resolution)
                if 0 <= grid_x < grid_size and 0 <= grid_z < grid_size:
                    grid[grid_z, grid_x] = 100  # 静态障碍物
        
        # 添加动态障碍物
        for cluster in dynamic_clusters:
            centroid = np.mean(cluster, axis=0)
            x, z = centroid[0], centroid[2]
            grid_x = int(center + x / resolution)
            grid_z = int(z / resolution)
            if 0 <= grid_x < grid_size and 0 <= grid_z < grid_size:
                # 扩展动态物体区域
                cv2.circle(grid, (grid_x, grid_z), int(0.5/resolution), 200, -1)  # 0.5m半径
        
        return grid
    
    def process_frame(self, left_img, right_img):
        """处理单帧图像"""
        start_time = time.time()
        
        # 1. 图像校正
        left_rect, right_rect = self.rectify_images(left_img, right_img)
        
        # 2. 计算视差图
        disparity = self.compute_disparity(left_rect, right_rect)
        
        # 3. 生成点云
        points, colors = self.disparity_to_pointcloud(disparity, left_rect)
        
        # 4. 过滤点云
        filtered_points = self.filter_pointcloud(points)
        
        # 5. 聚类
        clusters = self.cluster_points(filtered_points)
        
        # 6. 行人检测
        persons = self.detect_persons(left_rect)
        
        # 7. 动态/静态分类 (简化版)
        if hasattr(self, 'prev_clusters'):
            dynamic_clusters = self.classify_dynamic(clusters, self.prev_clusters)
            static_clusters = [c for c in clusters if not any(np.array_equal(c, d) for d in dynamic_clusters)]
        else:
            dynamic_clusters = []
            static_clusters = clusters
        
        # 8. 物体跟踪
        current_objects = self.track_objects(clusters, start_time)
        
        # 9. 生成占据栅格
        occupancy_grid = self.generate_occupancy_grid(static_clusters, dynamic_clusters)
        
        # 10. 更新前一帧数据
        self.prev_clusters = clusters
        
        # 计算处理时间
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        return {
            'disparity': disparity,
            'clusters': clusters,
            'dynamic_clusters': dynamic_clusters,
            'static_clusters': static_clusters,
            'persons': persons,
            'occupancy_grid': occupancy_grid,
            'processing_time': processing_time,
            'fps': fps
        }

# 使用示例
if __name__ == "__main__":
    # 初始化检测器
    detector = StereoObstacleDetector()
    
    # 初始化摄像头
    cap_left = cv2.VideoCapture(1)  # 左摄像头
    cap_right = cv2.VideoCapture(2)  # 右摄像头

    # 设置参数（数值范围通常是0~1或0~255，具体看摄像头驱动）
    cap_left.set(cv2.CAP_PROP_BRIGHTNESS, 0.0)
    cap_left.set(cv2.CAP_PROP_CONTRAST, 36.0)      # 对比度
    cap_left.set(cv2.CAP_PROP_SATURATION, 42.0)    # 饱和度
    cap_left.set(cv2.CAP_PROP_SHARPNESS, 10.0)     # 锐度（如不支持可忽略）
    

    # cap_right.set(cv2.CAP_PROP_CONTRAST, 0.5)
    # cap_right.set(cv2.CAP_PROP_SATURATION, 0.5)
    # cap_right.set(cv2.CAP_PROP_SHARPNESS, 0.5)
    
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            break
        
        # 水平翻转
        frame_left = cv2.flip(frame_left, 1)
        frame_right = cv2.flip(frame_right, 1)

        # 处理帧
        result = detector.process_frame(frame_left, frame_right)
        
        # 可视化结果
        disp_viz = cv2.normalize(result['disparity'], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_viz = cv2.applyColorMap(disp_viz, cv2.COLORMAP_JET)
        
        # 在左图上绘制检测结果
        for cluster in result['clusters']:
            centroid = np.mean(cluster, axis=0)
            x, y = int(centroid[0]), int(centroid[1])
            cv2.circle(frame_left, (x, y), 5, (0, 255, 0), -1)
        
        for cluster in result['dynamic_clusters']:
            centroid = np.mean(cluster, axis=0)
            x, y = int(centroid[0]), int(centroid[1])
            cv2.circle(frame_left, (x, y), 10, (0, 0, 255), 2)
        
        for person in result['persons']:
            x1, y1, x2, y2 = person
            cv2.rectangle(frame_left, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 显示占据栅格
        grid_viz = cv2.resize(result['occupancy_grid'], (300, 300), interpolation=cv2.INTER_NEAREST)
        grid_viz = cv2.applyColorMap(grid_viz, cv2.COLORMAP_JET)
        
        # 显示处理信息
        cv2.putText(frame_left, f"FPS: {result['fps']:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_left, f"Objs: {len(result['clusters'])}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow("Left Camera", frame_left)
        cv2.imshow("Disparity", disp_viz)
        cv2.imshow("Occupancy Grid", grid_viz)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()