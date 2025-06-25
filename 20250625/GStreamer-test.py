import cv2
import time

#CPU软件解码
# in_gst = (
#     "rtspsrc location=rtsp://admin:myj12345@192.168.110.210:554/Streaming/Channels/402 protocols=tcp latency=0 ! "
#     "rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! "
#     "appsink drop=true sync=false"
# )

#NPU硬件解码
in_gst = (
    "rtspsrc location=rtsp://admin:myj12345@192.168.110.210:554/Streaming/Channels/402 protocols=tcp latency=0 ! "
    "rtph265depay ! h265parse ! mppvideodec ! videoconvert ! "
    "appsink drop=true sync=false"
)


# 创建VideoCapture对象
cap = cv2.VideoCapture(in_gst, cv2.CAP_GSTREAMER)
# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Cannot open video stream or file")
    exit()

start_time = time.time()
frame_num = 0
# 读取视频帧
while True:
    frame_num += 1
    ret, frame = cap.read()
    end_time = time.time()
    print('fps:',frame_num/(end_time-start_time))
    if not ret:
        print("Failed to grab frame")
        break
    print("Frame shape:", frame.shape)

    # 显示结果帧
    # cv2.imshow('Frame', frame)
    # # 按'q'键退出
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
# 释放VideoCapture对象
cap.release()
# cv2.destroyAllWindows()


'''
ffplay rtsp://admin:myj12345@192.168.110.2:554/Stream1

ffplay rtsp://admin:myj12345@192.168.110.210:554/Streaming/Channels/402

gst-launch-1.0 rtspsrc location=rtsp://admin:myj12345@192.168.110.2:554/Stream1 latency=50 ! \
    rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! autovideosink

'''



