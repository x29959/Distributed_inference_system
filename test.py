import cv2

# 定义视频捕捉设备（0 通常为默认摄像头）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 使用 VP9 编码器
fourcc = cv2.VideoWriter_fourcc(*'VP80')  # 'VP90' 是 VP9 的 FourCC

# 定义输出文件名和视频编写器
out = cv2.VideoWriter('gesture_output.webm', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧")
        break

    # 在这里可以对帧进行处理（如绘制、滤镜等）

    # 写入帧到视频文件
    out.write(frame)

    # 显示帧
    cv2.imshow('Recording', frame)

    # 按 'q' 键退出录制
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()