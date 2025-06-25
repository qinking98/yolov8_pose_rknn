import cv2
import threading
import time
import queue
import numpy as np
import psutil
import gc
import os
from collections import deque
import logging

# 在类定义前添加日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/video_processor.log'  # 日志文件
)
logger = logging.getLogger(__name__)


class RTSPStreamReader:
    def __init__(self, rtsp_url, camera_id, max_queue_size=3):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.max_queue_size = max_queue_size
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.active = True
        self.capture_thread = None
        self.decode_failures = 0

        # 视频流属性（从流中获取）
        self.resolution = None
        self.fps = 0
        self.frame_count = 0

        # 性能监控
        self.avg_fps = 0
        self.avg_decode_time = 0

    def start(self):
        """启动视频流读取线程"""
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        print(f"摄像头 {self.camera_id} 已启动: {self.rtsp_url}")

    def _capture_frames(self):
        """视频流捕获线程主函数"""
        try:
            #NPU设备优化的GStreamer管道
            gst_pipeline = (
                f"rtspsrc location={self.rtsp_url} latency=0 protocols=tcp ! "
                "rtph265depay ! h265parse ! "
                "mppvideodec ! "  # 使用硬件解码器
                "videoconvert ! "  # 格式转换
                "video/x-raw,format=BGR ! "
                "appsink drop=true sync=false"
            )

            # gst_pipeline = (
            #     f"rtspsrc location=rtsp://admin:myj12345@192.168.110.2:554/Stream1 latency=0 protocols=tcp ! "
            #     "rtph265depay ! h265parse ! "
            #     "mppvideodec ! "  # 使用硬件解码器
            #     "videoconvert ! "  # 格式转换
            #     "video/x-raw,format=BGR ! "
            #     "appsink drop=true sync=false"
            # )

            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            if not cap.isOpened():
                logger.error(f"摄像头 {self.camera_id} GStreamer管道初始化失败")
                # 回退到CPU处理
                cap = cv2.VideoCapture(self.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.warning(f"摄像头 {self.camera_id} 使用CPU解码")

            # 设置RTSP参数优化
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少内部缓冲区
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

            # 获取视频流原生属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = (width, height)
            self.fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"摄像头 {self.camera_id} 分辨率: {width}x{height} | 帧率: {self.fps:.1f}FPS")

            last_time = time.time()
            frame_times = deque(maxlen=30)

            while self.active:
                start_time = time.time()

                # 尝试读取帧
                ret, frame = cap.read()

                if not ret:
                    self.decode_failures += 1
                    # 重连逻辑
                    if self.decode_failures > 10:
                        print(f"摄像头 {self.camera_id} 连接失败，尝试重连...")
                        cap.release()
                        cap = cv2.VideoCapture(self.rtsp_url)
                        # 重新设置参数
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # cap.set(cv2.CAP_PROP_TCP_PROTOCOL, 1)
                        self.decode_failures = 0
                    time.sleep(0.1)
                    continue

                # 计算处理时间
                process_time = time.time() - start_time
                frame_times.append(process_time)
                self.avg_decode_time = sum(frame_times) / len(frame_times) if frame_times else 0

                # 计算实际FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    self.avg_fps = 30 / (time.time() - last_time)
                    last_time = time.time()

                # 放入队列（非阻塞）
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # 丢弃最旧的帧
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # 队列满时丢弃当前帧
                    pass

                # 不控制采集速率，由视频流本身帧率决定
                # 根据系统负载动态调整休眠时间防止过载
                # current_load = ResourceMonitor.get_system_load()
                # sleep_time = max(0.001, 0.01 * current_load)  # 负载高时适当休眠
                # time.sleep(sleep_time)

            # 清理资源
            cap.release()
            print(f"摄像头 {self.camera_id} 已停止")

        except Exception as e:
            logger.error(f"摄像头 {self.camera_id} 捕获线程异常: {str(e)}", exc_info=True)
            raise
        finally:
            if 'cap' in locals():
                cap.release()
            logger.info(f"摄像头 {self.camera_id} 已停止")

    def get_frame(self, block=True, timeout=0.1):
        """从队列获取最新帧"""
        try:
            # 清空队列只保留最新帧
            while not self.frame_queue.empty() and self.frame_queue.qsize() > 1:
                self.frame_queue.get_nowait()

            if not self.frame_queue.empty():
                return self.frame_queue.get(block=block, timeout=timeout)
            return None
        except queue.Empty:
            return None

    def stop(self):
        """停止视频流读取"""
        self.active = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        self.frame_queue = queue.Queue()  # 清空队列
        gc.collect()


class ResourceMonitor:
    @staticmethod
    def get_system_load():
        """获取系统负载指标（0.0-1.0）"""
        mem_usage = psutil.virtual_memory().percent / 100
        cpu_usage = psutil.cpu_percent(interval=0.1) / 100
        return max(mem_usage, cpu_usage)

    @staticmethod
    def memory_usage_mb():
        """获取当前进程内存使用(MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def check_memory_threshold(threshold=0.8):
        """检查内存是否超过阈值"""
        return ResourceMonitor.get_system_load() > threshold


class MultiStreamProcessor:
    def __init__(self, rtsp_list, process_interval=5):
        # 添加NPU特定参数
        self.use_hardware_accel = True  # 启用硬件加速
        self.npu_device = "/dev/npu"  # NPU设备节点
        self.streams = []
        self.process_interval = process_interval
        self.last_process_time = time.time()
        self.active = True
        self.memory_warning_count = 0
        self.frame_counter = 0

        # 创建视频流读取器
        for url, cam_id in rtsp_list:
            stream = RTSPStreamReader(url, cam_id)
            stream.start()
            self.streams.append(stream)

        # 等待流初始化完成
        time.sleep(2)

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_resources(self):
        """资源监控线程"""
        while self.active:
            # 每5秒检查一次资源使用
            time.sleep(5)

            # 打印系统状态
            self.print_system_status()

            # 内存警戒处理
            if ResourceMonitor.check_memory_threshold(0.8):
                self.memory_warning_count += 1
                self._handle_memory_warning()

    def _handle_memory_warning(self):
        """内存超过阈值时的处理策略"""
        print(f"内存警告 #{self.memory_warning_count}: 系统内存使用超过80%")

        # 策略1: 减少帧处理量
        print("-> 跳过部分帧处理")

        # 策略2: 清理资源
        gc.collect()
        print("-> 执行垃圾回收")

        # 策略3: 降低处理复杂度
        if self.memory_warning_count > 2:
            print("-> 启用轻量处理模式")

    def print_system_status(self):
        """打印系统状态"""
        mem_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        mem_mb = ResourceMonitor.memory_usage_mb()

        print("\n" + "=" * 60)
        print(f"{'系统资源状态':^60}")
        print("=" * 60)
        print(f"内存使用: {mem_usage:.1f}% ({mem_mb:.1f} MB) | CPU使用: {cpu_usage}%")
        print(f"内存警告计数: {self.memory_warning_count}")
        print("-" * 60)

        for stream in self.streams:
            status = "活跃" if stream.active else "停止"
            qsize = stream.frame_queue.qsize()
            res_info = f"{stream.resolution[0]}x{stream.resolution[1]}" if stream.resolution else "未知"
            print(
                f"摄像头 {stream.camera_id}: {status} | 分辨率: {res_info} | 实际FPS: {stream.avg_fps:.1f}/{stream.fps:.1f} | 队列: {qsize}/{stream.max_queue_size}")

        print("=" * 60 + "\n")

    def process_frames(self):
        """主处理循环"""
        # 不强制帧率，按流本身速度处理
        try:
            while self.active:
                try:
                    frame_start = time.time()
                    self.frame_counter += 1

                    # 处理所有摄像头的帧
                    processed_frames = 0
                    for stream in self.streams:
                        frame = stream.get_frame(block=False)

                        if frame is not None:
                            # 执行实际处理（目标检测、跟踪等）
                            self._process_frame(stream.camera_id, frame)
                            processed_frames += 1

                            # 显式释放帧内存
                            del frame

                    # 如果没有处理任何帧，短暂休眠
                    if processed_frames == 0:
                        time.sleep(0.01)

                    # 定期清理资源
                    if time.time() - self.last_process_time > self.process_interval:
                        gc.collect()
                        self.last_process_time = time.time()

                        # 打印处理性能
                        actual_fps = self.frame_counter / (time.time() - frame_start)
                        print(f"处理性能: {actual_fps:.1f}FPS | 处理帧数: {self.frame_counter}")

                except KeyboardInterrupt:
                    logger.error("用户中断，停止处理...")
                    self.stop()
                except Exception as e:
                    logger.error(f"处理错误: {str(e)}")
                    time.sleep(1)
        except Exception as e:
            logger.critical(f"主处理循环异常: {str(e)}", exc_info=True)
            raise
        finally:
            self.stop()
            logger.info("视频处理器已完全停止")

    def _process_frame(self, camera_id, frame):
        """帧处理函数 - 替换为实际业务逻辑"""
        try:
            h, w = frame.shape[:2]

            # 实际应用中这里执行：
            # 1. 目标检测
            # 2. 目标跟踪
            # 3. 关键点检测
            # 4. 结果分析

            # 每25帧打印一次（约1秒）
            if self.frame_counter % 25 == 0:
                print(f"处理摄像头 {camera_id} 的帧: {w}x{h} | 时间: {time.strftime('%H:%M:%S')}")
        except Exception as e:
            logger.error(f"摄像头 {camera_id} 帧处理错误: {str(e)}")
            raise

    def stop(self):
        """停止所有流和监控"""
        self.active = False
        for stream in self.streams:
            stream.stop()

        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

        print("所有视频流已停止")


if __name__ == "__main__":
    #定义RTSP流列表
    rtsp_list = [
        ('rtsp://admin:myj12345@192.168.110.210:554/Streaming/Channels/402', '44190000491320151652'),
        ('rtsp://admin:myj12345@192.168.110.210:554/Streaming/Channels/702', '44190000491320151655'),
        ('rtsp://admin:myj12345@192.168.110.210:554/Streaming/Channels/902', '44190000491320151657')
    ]

    # rtsp_list = [
    #     ('rtsp://admin:myj12345@192.168.110.4:554/Streaming1', '44190000491320151652'),
    #     ('rtsp://admin:myj12345@192.168.110.7:554/Streaming1','44190000491320151655'),
    #     ('rtsp://admin:myj12345@192.168.110.9:554/Streaming1','44190000491320151657')
    # ]

    # 创建多流处理器
    processor = MultiStreamProcessor(rtsp_list)

    try:
        # 启动处理
        processor.process_frames()
    except Exception as e:
        logger.critical(f"程序崩溃: {str(e)}", exc_info=True)
        raise
    finally:
        processor.stop()