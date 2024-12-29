from flask import Flask, render_template, request
import subprocess
from datetime import datetime
import os

app = Flask(__name__)
ffmpeg_process = None

def start_streaming_and_recording():
    global ffmpeg_process
    # 生成唯一的檔案名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.mp4"
    # 確保輸出目錄存在
    os.makedirs('output', exist_ok=True)
    output_path = f'output/{filename}'

    # 使用 FFmpeg 的 tee 功能，同時進行串流和錄影
    ffmpeg_command = [
        "ffmpeg",
        "-f", "v4l2",
        "-framerate", "24",
        "-video_size", "640x480",
        "-i", "/dev/video0",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-b:v", "2000k",
        "-g", "48",
        "-threads", "2",
        "-filter_complex", "[0:v]split=2[v1][v2]",
        # 串流輸出
        "-map", "[v1]",
        "-f", "flv",
        "rtmp://localhost/live/stream",
        # 錄影輸出
        "-map", "[v2]",
        "-f", "mp4",
        output_path
    ]

    ffmpeg_process = subprocess.Popen(ffmpeg_command)

def stop_streaming_and_recording():
    global ffmpeg_process
    if ffmpeg_process:
        ffmpeg_process.terminate()
        ffmpeg_process = None

@app.route("/", methods=["GET", "POST"])
def index():
    global ffmpeg_process
    if request.method == "POST":
        action = request.form.get("action")
        if action == "start":
            start_streaming_and_recording()
        elif action == "stop":
            stop_streaming_and_recording()
    streaming = ffmpeg_process is not None
    return render_template("index2.html", streaming=streaming)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
