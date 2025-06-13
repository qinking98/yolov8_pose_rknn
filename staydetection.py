import os
import cv2
import json
import time
import logging
import numpy as np
import pandas as pd
from threading import Thread
from queue import Queue
from scipy.ndimage import gaussian_filter1d
from rknnlite.api import RKNNLite
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from conf import logger_config
from conf.app_config import work_dir
from service.queue_manager import QueueManager
import threading
import requests

logger = logging.getLogger(__name__)
INPUT_SIZE = (640, 640)

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint

class PoseDetector:
    def __init__(self, rknn_model_path):
        self.CLASSES = ['person']
        self.nmsThresh = 0.4
        self.objectThresh = 0.5
        
        # 初始化3个NPU核心实例
        self.rknn_cores = []
        for core_id in [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]:
            core = RKNNLite()
            core.load_rknn(rknn_model_path)
            ret = core.init_runtime(core_mask=core_id)
            if ret != 0:
                raise RuntimeError(f'Core {core_id} init failed')
            self.rknn_cores.append(core)
        
    def __del__(self):
        """释放资源"""
        for core in self.rknn_cores:
            core.release()

    def preprocess(self, image, bg_color):
        """图像预处理方法"""

        h, w = image.shape[:2]
        
        # 计算缩放比例和填充偏移量
        scale = min(INPUT_SIZE[0] / w, INPUT_SIZE[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 优化缩放和填充
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            
        # 使用更快的缩放方法
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建填充后的图像(直接使用RGB顺序避免后续转换)
        padded = np.zeros((1, INPUT_SIZE[1], INPUT_SIZE[0], 3), dtype=np.uint8)
        padded[0, :, :] = bg_color
        offset_y = (INPUT_SIZE[1] - new_h) // 2
        offset_x = (INPUT_SIZE[0] - new_w) // 2
        
        # 直接填充RGB图像
        padded[0, offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized[..., ::-1]  # BGR转RGB
        
        return padded, (offset_x, offset_y), scale


    def IOU(self, box1, box2):
        """向量化IOU计算"""
        # 确保输入是二维数组
        box1 = np.atleast_2d(box1)
        box2 = np.atleast_2d(box2)
        
        # 计算交集坐标
        xmin = np.maximum(box1[:,0:1], box2[:,0])
        ymin = np.maximum(box1[:,1:2], box2[:,1])
        xmax = np.minimum(box1[:,2:3], box2[:,2])
        ymax = np.minimum(box1[:,3:4], box2[:,3])
        
        # 计算交集面积
        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        
        # 计算并集面积
        box1_area = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
        box2_area = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])
        
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)


    def NMS(self, detectResult):
        """NMS实现"""
        if not detectResult:
            return []
            
        boxes = np.array([[d.xmin, d.ymin, d.xmax, d.ymax, d.score, d.classId] 
                         for d in detectResult])
        keep = []
        
        # 按分数降序排序
        idxs = np.argsort(boxes[:,4])[::-1]
        
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            # 处理只剩一个框的情况
            if len(idxs) == 1:
                break
            # 计算当前框与其他框的IOU
            ious = self.IOU(boxes[i, :4], boxes[idxs[1:], :4])
            # 保留IOU低于阈值的框
            idxs = idxs[1:][ious[0] < self.nmsThresh]
            
        return [detectResult[i] for i in keep]


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, axis=-1):
        # 将输入向量减去最大值以提高数值稳定性
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def process(self, out, keypoints, index, model_w, model_h, stride, scale_w=1, scale_h=1):
        """
        优化后的处理函数，处理单个检测头输出

        参数:
            out: 模型输出结果
            keypoints: 关键点数据
            index: 当前处理的索引
            model_w: 模型特征图宽度
            model_h: 模型特征图高度
            stride: 特征图到原始图像的缩放比例
            scale_w: 原始图像宽度缩放比例
            scale_h: 原始图像高度缩放比例
        返回:
            out_list: 处理后的检测框列表
        """
        xywh = out[:,:64,:]
        conf = self.sigmoid(out[:,64:,:])
        out_list = []
        
        # 预计算所有可能的索引
        h_indices, w_indices = np.indices((model_h, model_w))
        h_indices = h_indices.ravel()
        w_indices = w_indices.ravel()
        
        # 预计算所有位置的条件
        valid_mask = conf[0, 0, h_indices*model_w + w_indices] > self.objectThresh
        
        # 只处理满足条件的点
        for h, w in zip(h_indices[valid_mask], w_indices[valid_mask]):
            idx = h * model_w + w
            xywh_ = xywh[0,:,idx].reshape(1,4,16,1)
            
            # 向量化softmax计算
            data = np.arange(16).reshape(1,1,16,1)
            xywh_ = self.softmax(xywh_, 2)
            xywh_ = np.sum(data * xywh_, axis=2).reshape(-1)
            
            # 向量化坐标计算
            xywh_temp = np.array([
                (w+0.5)-xywh_[0], 
                (h+0.5)-xywh_[1],
                (w+0.5)+xywh_[2],
                (h+0.5)+xywh_[3]
            ])
            
            xywh_ = np.array([
                (xywh_temp[0]+xywh_temp[2])/2,
                (xywh_temp[1]+xywh_temp[3])/2,
                xywh_temp[2]-xywh_temp[0],
                xywh_temp[3]-xywh_temp[1]
            ]) * stride
            
            # 计算边界框
            half_w, half_h = xywh_[2]/2, xywh_[3]/2
            box_coords = [
                (xywh_[0] - half_w) * scale_w,
                (xywh_[1] - half_h) * scale_h,
                (xywh_[0] + half_w) * scale_w,
                (xywh_[1] + half_h) * scale_h
            ]
            
            # 处理关键点
            kpt = keypoints[...,idx+index].copy()
            kpt[...,0:2] = kpt[...,0:2]//1
            
            out_list.append(DetectBox(0, conf[0,0,idx], *box_coords, kpt))
            
        return out_list

    def postprocess(self, results, offset_x, offset_y, scale):
        """
        后处理模型输出
        
        参数:
            results: 模型推理输出结果
            offset_x: 预处理时水平方向的填充偏移量
            offset_y: 预处理时垂直方向的填充偏移量
            scale: 图像缩放比例
            
        返回:
            det_out: 检测框列表，每个元素为[xmin, ymin, xmax, ymax, score, classId]
            kps_out: 关键点列表，每个元素为[N,17,3]的关键点数组
        """
        # 1. 处理模型原始输出
        outputs = []
        keypoints = results[3]  # 获取关键点输出
        
        # 处理三个不同尺度的检测头输出
        for x in results[:3]:
            # 根据特征图尺寸确定参数
            if x.shape[2] == 20:
                stride = 32
                index = 20*4*20*4 + 20*2*20*2
            elif x.shape[2] == 40:
                stride = 16
                index = 20*4*20*4
            elif x.shape[2] == 80:
                stride = 8
                index = 0
                
            # 处理特征图并生成检测盒
            feature = x.reshape(1,65,-1)
            output = self.process(feature, keypoints, index, x.shape[3], x.shape[2], stride)
            outputs += output
        
        # 2. 应用非极大值抑制
        predbox = self.NMS(outputs)
        
        # 3. 转换坐标到原始图像空间
        det_out = []
        kps_out = []
        for box in predbox:
            # 转换边界框坐标
            xmin = int((box.xmin-offset_x)/scale)
            ymin = int((box.ymin-offset_y)/scale)
            xmax = int((box.xmax-offset_x)/scale)
            ymax = int((box.ymax-offset_y)/scale)
            det_out.append([xmin, ymin, xmax, ymax, box.score, box.classId])
            
            # 转换关键点坐标
            kpts = box.keypoint.reshape(-1, 3)
            kpts[...,0] = (kpts[...,0]-offset_x)/scale
            kpts[...,1] = (kpts[...,1]-offset_y)/scale
            kps_out.append(kpts)
            
        return det_out, kps_out

    def detect(self, core, img):
        
        # 1. 图像预处理
        infer_img, (offset_x, offset_y), scale = self.preprocess(img, 56)
        
        # 2. 使用选定的核心进行推理
        results = core.inference(inputs=[infer_img])
        
        # 3. 后处理
        detections, keypoints = self.postprocess(results, offset_x, offset_y, scale)

        return detections, keypoints

class Tracker:
    """简单的跟踪器实现"""
    def __init__(self, max_age=10):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.skip_frames = 0
    
    def update(self, detections, keypoints):
        """更新跟踪器"""
        active_ids = []
        updated_tracks = {}
        # 为每个检测分配/更新ID
        for det, kps in zip(detections, keypoints):
            matched_id = self._find_match(det[:4])
            
            if matched_id is not None:
                updated_tracks[matched_id] = {'det': det, 'kps': kps, 'age': 0}
                active_ids.append(matched_id)
            else:
                updated_tracks[self.next_id] = {'det': det, 'kps': kps, 'age': 0}
                active_ids.append(self.next_id)
                self.next_id += 1
        # 更新存活时间，考虑跳帧
        for track_id in self.tracks:
            if track_id not in active_ids:
                age = self.tracks[track_id]['age'] + 1
                if age <= self.max_age:
                    updated_tracks[track_id] = {
                        'det': self.tracks[track_id]['det'],
                        'kps': self.tracks[track_id]['kps'],
                        'age': age
                    }
        
        self.tracks = updated_tracks
        
        return self.tracks

    def _find_match(self, bbox):
        """通过IOU匹配现有跟踪目标"""
        best_iou = 0.3  # 最小IOU阈值
        best_id = None
        
        for track_id, track in self.tracks.items():
            if track['age'] > 0:
                continue  # 只匹配未更新的跟踪
            
            track_bbox = track['det'][:4]
            iou = self._calculate_iou(bbox, track_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_id = track_id
        
        return best_id
    
    @staticmethod
    def _calculate_iou(box1, box2):
        """计算IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter_area / (box1_area + box2_area - inter_area)


class TrackAnalyzer:
    def __init__(self):
        self.tracks = {}

    def stay_single_track(self, df, min_interval=25):
        """
        分析单个tracker的停留时间段
        参数:
            df: 单个tracker的DataFrame或NumPy数组，必须包含frame_id, feet_center_x, feet_center_y列
            min_interval: 最小停留帧数
        返回:
            停留时间段的DataFrame，包含start_frame, end_frame, duration列
        """
        frames = df[:, 0].astype(np.int32)  # 第一列是frame_id
        x_coords = df[:, 1].astype(np.float32)  # 第二列是feet_center_x
        y_coords = df[:, 2].astype(np.float32)  # 第三列是feet_center_y
        frame_times = df[:, 3]  # 第四列是时间戳
        
        # 检测x和y方向的趋势区间
        def detect_trend(coords):
            """
            趋势检测函数，判断是否平缓
            参数:
                coords: 坐标数组
            返回:
                trend: 趋势数组，'flat'表示平缓，'not_flat'表示不平缓
            """
            # 高斯平滑处理
            smoothed = gaussian_filter1d(coords, sigma=3)
            
            # 计算一阶差分
            diffs = np.diff(smoothed, prepend=smoothed[0])
            
            # 计算标准差作为阈值
            std_threshold = np.std(diffs) * 0.7  # 调整系数可以控制灵敏度
            
            # 直接判断是否平缓
            trend = np.full(len(coords), 'flat')
            trend[np.abs(diffs) > std_threshold] = 'not_flat'
            
            return trend
        
        x_trend = detect_trend(x_coords)
        y_trend = detect_trend(y_coords)

        # 找出x和y方向都静止的区间
        flat_mask = (x_trend == 'flat') & (y_trend == 'flat')
        # print('1',flat_mask)
        # # 处理连续False少于10帧的情况,短暂的运动视为静止状态，减少误判。
        # current_state = flat_mask[0]
        # start_idx = 0
        # for i in range(1, len(flat_mask)):
        #     if flat_mask[i] != current_state:
        #         # 如果是False区间且长度小于10帧
        #         if not current_state and (i - start_idx) < 20:
        #             flat_mask[start_idx:i] = True
        #         current_state = flat_mask[i]
        #         start_idx = i
        # # 处理最后一个区间
        # if not current_state and (len(flat_mask) - start_idx) < 20:
        #     flat_mask[start_idx:] = True
        # print('2',flat_mask)
        flat_indices = np.where(flat_mask)[0]
        if len(flat_indices) == 0:
            return pd.DataFrame(columns=['start_frame', 'end_frame', 'duration'])

        # 合并相邻的静止区间
        diffs = np.diff(flat_indices)
        breaks = np.where(diffs > 1)[0] + 1
        groups = np.split(flat_indices, breaks)

        # 收集有效停留区间
        stay_intervals = []
        for group in groups:
            start_idx, end_idx = group[0], group[-1]
            duration = frames[end_idx] - frames[start_idx]
            if duration >= min_interval:
                stay_intervals.append({
                    'start_frame': frames[start_idx],
                    'end_frame': frames[end_idx],
                    'duration': duration,
                    'start_time': frame_times[start_idx]
                })
        return pd.DataFrame(stay_intervals)


class ShelfAnalyzer:
    def __init__(self, area_path=None):
        """初始化货架分析器"""
        self.shelf_centroids_map = self.load_area_data(area_path) if area_path else None

    @staticmethod
    def calculate_polygon_centroid(polygon):
        """
        计算多边形的几何中心
        参数:
            polygon: 多边形的顶点列表，每个顶点为(x, y)
        返回:
            多边形的几何中心坐标(x, y)
        """
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        n = len(polygon)

        # 计算多边形面积
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += (x_coords[i] * y_coords[j]) - (x_coords[j] * y_coords[i])
        area = (area) / 2.0

        # 计算质心坐标
        cx = 0.0
        cy = 0.0
        for i in range(n):
            j = (i + 1) % n
            factor = (x_coords[i] * y_coords[j] - x_coords[j] * y_coords[i])
            cx += (x_coords[i] + x_coords[j]) * factor
            cy += (y_coords[i] + y_coords[j]) * factor

        if area == 0:
            return (sum(x_coords) / n, sum(y_coords) / n)

        cx /= (6 * area)
        cy /= (6 * area)

        return (cx, cy)

    def load_area_data(self, area_path):
        """
        加载货架区域数据
        参数:
            area_path: 货架区域数据文件路径
        返回:
            shelf_centroids_map: 货架区域的几何中心字典
        """
        with open(area_path, 'r') as f:
            data = json.load(f)

        shelf_centroids_map = {}
        for store in data:
            store_id = list(store.keys())[0]
            shelf_centroids_map[store_id] = {}

            for camera in store[store_id]:
                camera_id = list(camera.keys())[0]
                shelf_centroids_map[store_id][camera_id] = {}

                for area in camera[camera_id]:
                    for area_id, polygon in area.items():
                        centroid = self.calculate_polygon_centroid(polygon)
                        shelf_centroids_map[store_id][camera_id][area_id] = centroid

        return shelf_centroids_map

    @staticmethod
    def calculate_facing_shelf(perp_vec, centroid, shelf_centroids, ymin, ymax):
        """
        计算用户面向的最近货架区域
        参数:
            perp_vec: 用户朝向的垂直向量
            centroid: 用户的几何中心
            shelf_centroids: 货架区域的几何中心字典
            ymin, ymax: 用户的边界框高度
        返回:
            最近货架区域的ID，如果没有合适的区域则返回None
        """
        if perp_vec is None:
            return None

        min_angle = float('inf')
        facing_area = None
        bbox_height = ymax - ymin

        for area_id, shelf_center in shelf_centroids.items():
            to_shelf_vec = np.array(shelf_center) - np.array(centroid)
            distance = np.linalg.norm(to_shelf_vec)

            if distance <= bbox_height:
                cosine = np.dot(perp_vec, to_shelf_vec) / (np.linalg.norm(perp_vec) * distance)
                cosine = np.clip(cosine, -1.0, 1.0)
                angle = np.degrees(np.arccos(cosine))
                if angle < min_angle:
                    min_angle = angle
                    if angle < 90:
                        facing_area = area_id
                    else:
                        facing_area = None

        return facing_area

    @staticmethod
    def get_most_frequent_facing_area(df1, df2):
        """计算停留时间窗内最频繁的facing_area
        参数:
            df1: 包含track_id, frame_id, x_center, y_center, feet_center_x, feet_center_y, perp_vector_x, perp_vector_y的DataFrame
            df2: 包含start_frame, end_frame的DataFrame
        返回:
            包含start_frame, end_frame, facing_area的DataFrame
        """
        facing_areas = []

        for _, interval in df2.iterrows():
            window_data = df1[(df1['frame_id'] >= interval['start_frame']) &
                              (df1['frame_id'] <= interval['end_frame'])]

            area_counts = window_data['facing_area'].value_counts()

            if not area_counts.empty:
                facing_areas.append(area_counts.idxmax())
            else:
                facing_areas.append(None)

        df2['facing_area'] = facing_areas
        return df2.loc[(df2['facing_area'].notna())]


class VideoProcessor:
    def __init__(self, model_path, area_path, store_id, skip_frames, visual=False, core=None):
        self.model_path = model_path
        self.area_path = area_path
        self.store_id = store_id
        self.skip_frames = skip_frames
        self.result_queue = QueueManager.get_instance()
        self.threads = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.visual = visual
        self.detector_lock = threading.Lock()  # 新增锁对象
        self.detectors = []  # 存储多个检测器实例
        self.core = core  # 存储选定的核心实例
    # def __init__(self, model_path, area_path, store_id, skip_frames, visual=False):
    #     self.model_path = model_path
    #     self.area_path = area_path
    #     self.store_id = store_id
    #     self.skip_frames = skip_frames
    #     self.result_queue = QueueManager.get_instance()
    #     self.threads = []
    #     self.thread_pool = ThreadPoolExecutor(max_workers=4)
    #     self.visual = visual
    
    def _process_video_tracking(self, video, cname):
        """
        处理视频跟踪数据并分析停留行为

        参数:
            video: 视频文件路径
            cname: 摄像头名称
            store_id: 店铺ID
        """
        
        # 初始化模型和跟踪器
        # pose_detector = PoseDetector(self.model_path)
        # 初始化模型和跟踪器
        with self.detector_lock:
            # 如果detectors为空或者线程数多于detectors数量，创建新detector
            if not self.detectors or len(self.detectors) < len(self.threads):
                pose_detector = PoseDetector(self.model_path)
                self.detectors.append(pose_detector)
            else:
                # 只有当detectors不为空时才进行取模运算
                pose_detector = self.detectors[len(self.threads) % len(self.detectors)]
        
        tracker = Tracker()  # 将tracker初始化移到循环外
        
        processed_combinations = set()
        shelf_analyzer = ShelfAnalyzer(self.area_path)
        shelf_centroids_map = shelf_analyzer.load_area_data(self.area_path)
        camera_shelf_centroids = shelf_centroids_map.get(self.store_id, {}).get(cname, {})
        
        # 打开视频源
        cap = cv2.VideoCapture(video)
        # 设置FFmpeg解码参数
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)  # 启用硬件加速
        cap.set(cv2.CAP_PROP_HW_DEVICE, 0)  # 使用第一个硬件设备

        fps = cap.get(cv2.CAP_PROP_FPS) 
        fps_counter = deque(maxlen=10)
        
        # 预计算常用值
        disappear_threshold = 10 * fps
        frame_data_template = {
            'track_id': 0, 'frame_id': 0,
            'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0,
            'x_center': 0, 'y_center': 0,
            'feet_center_x': 0, 'feet_center_y': 0,
            'perp_vector_x': 0, 'perp_vector_y': 0
        }
        
        try:
            frame_num = 0
            last_detections = None
            last_keypoints = None
            final_res = []
            prev_keypoints = {}
            last_seen_frames = {}
            nobody_count = 0
            analyzer = TrackAnalyzer()

            while cap.isOpened():
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                # 检测
                if frame_num % (self.skip_frames + 1) == 0:
                    last_detections, last_keypoints = pose_detector.detect(self.core, frame)
                else:
                    pass
                    # cap.grab()
                # 跟踪处理
                if last_detections and last_keypoints:
                    tracks = tracker.update(last_detections, last_keypoints)
                    
                    # 更新跟踪数据
                    current_track_ids = set(tracks.keys())
                    last_seen_frames.update({tid: frame_num for tid in current_track_ids})
                    
                    # 清理消失的track
                    disappeared_tracks = {
                        tid for tid, last_frame in last_seen_frames.items()
                        if (frame_num - last_frame) > disappear_threshold
                    }
                    for tid in disappeared_tracks:
                        last_seen_frames.pop(tid, None)
                        prev_keypoints.pop(tid, None)
                    final_res = [r for r in final_res if r['track_id'] not in disappeared_tracks]
                    
                    # 处理当前帧数据
                    for track_id, track in tracks.items():
                        box = track['det'][:4]
                        kpts = track['kps']
                        
                        # 使用预分配模板创建数据
                        frame_data = frame_data_template.copy()
                        frame_data.update({
                            'track_id': track_id,
                            'frame_id': frame_num,
                            'frame_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                            'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3],
                            'x_center': (box[0] + box[2]) / 2,
                            'y_center': (box[1] + box[3]) / 2,
                            'feet_center_x': (kpts[15][0] + kpts[16][0]) / 2,
                            'feet_center_y': (kpts[15][1] + kpts[16][1]) / 2,
                            'perp_vector_x': kpts[6][1] - kpts[5][1],   # y2 - y1
                            'perp_vector_y': kpts[5][0] - kpts[6][0]    # x1 - x2
                        })
                        final_res.append(frame_data)

                # 若连续10秒内没有检测到任何人，则清空final_res
                if not last_detections:
                    nobody_count += 1
                    if nobody_count > 10 * fps:
                        final_res = []
                        nobody_count = 0

                # 批量处理停留分析
                if final_res:
                    # 使用字典按track_id分组数据
                    track_groups = {}
                    for item in final_res:
                        track_id = item['track_id']
                        if track_id not in track_groups:
                            track_groups[track_id] = []
                        track_groups[track_id].append(item)

                    # 预计算所有tracker的停留结果        
                    for tracker_id, group in track_groups.items():
                        
                        # 转换为NumPy数组格式 [frame_id, feet_center_x, feet_center_y]
                        data = np.array([
                            [item['frame_id'], item['feet_center_x'], item['feet_center_y'], item['frame_time']] 
                            for item in group
                        ])
                                         
                        stay_result = analyzer.stay_single_track(data, min_interval=3*fps)
                        if not stay_result.empty:
                            # 批量处理所有停留区间
                            min_frame = stay_result['start_frame'].min()
                            max_frame = stay_result['end_frame'].max()
                            
                            # 使用loc直接获取数据，避免copy
                            group_df = pd.DataFrame(group)
                            stay_data = group_df.loc[group_df['frame_id'].between(min_frame, max_frame)]
                            
                            # 向量化计算facing_area
                            valid_rows = ~stay_data['perp_vector_x'].isna()
                            facing_areas = np.empty(len(stay_data), dtype=object)
                            if valid_rows.any():
                                # 预计算所有需要的向量
                                perp_vecs = np.column_stack((
                                    stay_data.loc[valid_rows, 'perp_vector_x'],
                                    stay_data.loc[valid_rows, 'perp_vector_y']
                                ))
                                centroids = np.column_stack((
                                    stay_data.loc[valid_rows, 'x_center'],
                                    stay_data.loc[valid_rows, 'y_center']
                                ))
                                ymin = stay_data.loc[valid_rows, 'ymin'].values
                                ymax = stay_data.loc[valid_rows, 'ymax'].values
                                
                                # 批量计算facing_area
                                
                                facing_areas[valid_rows] = [
                                    shelf_analyzer.calculate_facing_shelf(
                                        tuple(perp_vec), tuple(centroid),
                                        camera_shelf_centroids, y1, y2
                                    )
                                    for perp_vec, centroid, y1, y2 in zip(perp_vecs, centroids, ymin, ymax)
                                ]
                                
                            stay_data = stay_data.assign(facing_area=facing_areas)
                            finall_result = shelf_analyzer.get_most_frequent_facing_area(stay_data, stay_result)
                            for _, row in finall_result.iterrows():
                                combo = (tracker_id, row['facing_area'])
                                if combo not in processed_combinations:
                                    processed_combinations.add(combo)
                                    stay_dic = {
                                        'camera_name': cname,
                                        'track_id': tracker_id,
                                        'stay_info': {
                                            'start_frame': [row['start_frame']],
                                            'start_time': [row['start_time']],
                                            'stay_area': [row['facing_area']]
                                        }
                                    } 
                                    yield stay_dic     
                frame_num += 1
                fps_counter.append(1.0 / (time.time() - frame_start))
                if self.visual:
                    visualize(frame, tracks, fps=np.mean(fps_counter))
                    cv2.imshow("Tracking", frame) 
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
        finally:
            cap.release()
            if self.visual:
                cv2.destroyAllWindows()
        
    def _process_video_thread(self, video_path, cname):
        """线程内部处理函数"""
        try:
            result = self._process_video_tracking(
                video=video_path,
                cname=cname
            )
            for dic in result:
                url = "https://open-api.myj.com.cn/open/uni-ai/smart-marketing-api/v1/stay-msg"
                headers = {
                    "Content-Type": "application/json",
                    "Cookie": "cookiesession1=678A3E0EBC48C7595143EE233AA0D839"
                }
                postdata = {
                    "store_id": self.store_id,
                    "track_id": dic['track_id'],
                    "stay_area": dic['stay_info']['stay_area'][0]
                }
                response = requests.post(url, headers=headers, json=postdata)
                response.raise_for_status()  # 检查请求是否成功
                response_json = response.json()
                logger.info(f"Processing video result: {dic},\n Post Response: {response_json}")
                self.result_queue.put(dic)
        except Exception as e:
            logger.exception(f"Error processing {video_path}: {str(e)}")
            self.result_queue.put((video_path, None))

    def process_videos(self, video_list, max_threads=4):
        """多线程处理视频列表"""
        for i, (video_path, cname) in enumerate(video_list):
            if len(self.threads) >= max_threads:
                self.threads[0].join()
                self.threads.pop(0)
                
            thread = Thread(
                target=self._process_video_thread,
                args=(video_path, cname)
            )
            thread.start()
            self.threads.append(thread)

        for thread in self.threads:
            thread.join()

        while not self.result_queue.empty():
            result = self.result_queue.get()
            if isinstance(result, tuple):  # 处理异常情况
                video_path, error = result
                cname = os.path.basename(video_path).split('.')[0] if '.' in video_path else video_path
                yield {'camera_name': cname, 'error': str(error)}
            else:
                yield {
                    'camera_name': result.get('track_id', 'unknown'),
                    'stay_info': result.get('stay_info', {})
                }
    def submit_videos_process_task(self, video_path, cname):
        """
        如果有多个，可以提交多次
        :param video_path: 单个视频文件路径
        :param cname: 摄像头名称
        :return: 无返回，检测结果写入到队列中
        """
                # 轮询选择NPU核心
        self.thread_pool.submit(self._process_video_thread, video_path, cname)


def visualize(image, tracks, fps=None):
    """可视化结果"""
    # 绘制FPS
    if fps is not None:
        cv2.putText(image, f'FPS: {fps:.2f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 绘制检测框和关键点
    for track_id, track in tracks.items():
        det = track['det']
        kps = track['kps']
        
        # 绘制检测框
        color = (0, 255, 0) if track['age'] == 0 else (0, 0, 255)
        cv2.rectangle(image, (int(det[0]), int(det[1])), 
                     (int(det[2]), int(det[3])), color, 2)
        
        # 绘制ID和置信度
        label = f'ID:{track_id} {det[4]:.2f}'
        cv2.putText(image, label, (int(det[0]), int(det[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制关键点
        for kp in kps:
            x, y, conf = kp
            if conf > 0.3:  # 只绘制高置信度关键点
                cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)


def submit_job(rtsp, core):
    """提交一个任务"""

    processor = VideoProcessor(
        model_path="models/yolov8n-pose.rknn",
        area_path="input/store_shelf.json",
        store_id="V00928",
        skip_frames=2,
        visual=False,
        core = core
    )
    # 修改RTSP URL添加参数
    rtsp_url = rtsp[0]
    camara_id = rtsp[1]
    processor.submit_videos_process_task(rtsp_url, camara_id)


if __name__ == '__main__':
    pd.set_option('future.no_silent_downcasting', True) # 消除警告
    logger_config.setup_logging(logging.INFO)
    rknn_model_path = "models/yolov8n-pose.rknn"
    rtsp_list = [
        ('rtsp://admin:myj12345@192.168.110.210:554/Streaming/Channels/301?transport=tcp','44190000491320151651'),
        ('rtsp://admin:myj12345@192.168.110.210:554/Streaming/Channels/601?transport=tcp','44190000491320151654'),
        ('rtsp://admin:myj12345@192.168.110.210:554/Streaming/Channels/901?transport=tcp','44190000491320151657')
    ]

    rknn_cores = []
    for core_id in [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]:
        core = RKNNLite()
        core.load_rknn(rknn_model_path)
        ret = core.init_runtime(core_mask=core_id)
        if ret != 0:
            raise RuntimeError(f'Core {core_id} init failed')
        rknn_cores.append(core)
        logger.info(f"Successfully initialized NPU core {core_id}")
    
    core_index = 0  # 用于轮询分配核心
    for rtsp in rtsp_list:
        # 轮询选择NPU核心
        core = rknn_cores[core_index % len(rknn_cores)]
        core_index += 1
        submit_job(rtsp, core)

'''
source activate Rknn_Toolkit_lite2_Py3.9
cd /data00/aWinDir/D/dotnet/py_project/rknn-toolkit2-2.3.2/rknn-toolkit-lite2/yolov8_pose_rknn/
'''
#
