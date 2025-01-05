# 分散式及自適應學習之串流伺服器架構
# Distributed and Adaptive Learning Streaming Server Architecture

# On-device 裝置端

### Part 1. Android 手機or電腦串流伺服器 / Android Phone Streaming Server

#### 必備應用程式 / Required Applications
- Termux ([Google Play Store Link](https://play.google.com/store/apps/details?id=com.termux))
- Tailscale ([Google Play Store Link](https://play.google.com/store/apps/details?id=com.tailscale.ipn))

#### 安裝步驟 / Installation Steps
1. 安裝並設定 Tailscale
   - 下載並安裝 Tailscale
   - 登入並連接到您的 Tailscale 網路

2.1.1 Termux 環境設定
   - 安裝 FFmpeg（選擇以下其中一種方法）：
     ```bash
     # 方法一：直接安裝（推薦）
     pkg install -y ffmpeg

     # 方法二：從原始碼編譯
     pkg install -y git build-essential
     git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
     cd ffmpeg
     ./configure --prefix=$PREFIX --disable-static --enable-shared
     make
     make install
     ```

   - 安裝串流所需套件：
     ```bash
     pkg install nodejs-lts
     npm install express node-media-server
     npm init -y
     ```

   執行以下指令以啟動串流伺服器：
   ```bash
   node server.js
   ```
2.1.2 電腦架設環境

- 安裝 npm 套件
  ```bash
  sudo apt update
  sudo apt install npm
  ```
- 安裝 init
  ```bash
  npm init -y
  ```
- 安裝 node-media-server
  ```bash
   npm install node-media-server
   ```
- 執行串流伺服器
  ```bash
  node server.js
  ```
2.3 設定串流伺服器
   - 輸入伺服器位址: ip:port
   
   - 開啟串流伺服器
   - 開啟串流應用程式VLC，輸入伺服器位址即可觀看串流

3. 安裝 DJI fly app [Official android or ios Link](https://www.dji.com/tw/downloads/djiapp/dji-fly)
   - 開啟 DJI fly app
![Screenshot_20241229_214306](https://github.com/user-attachments/assets/6f7ada30-bb24-469c-837a-cc276f4778a0)

   - 進入設定
![Screenshot_20241229_214120](https://github.com/user-attachments/assets/e4f97a4a-ae5e-4635-8169-7c1637a89076)

   - 選擇「串流」
![Screenshot_20241229_214129](https://github.com/user-attachments/assets/e47de688-5a25-4409-98a5-bf7f7007022a)

   - 輸入伺服器位址: ip:port  
![Screenshot_20241229_214143](https://github.com/user-attachments/assets/96c219cc-5b59-40ab-85fd-b2925d201275)

   - 成功串流
![Screenshot_20241229_214159](https://github.com/user-attachments/assets/464f4f4e-2a67-4645-9782-59496d420089)

### Part 2. 樹莓派 5 串流伺服器 / Raspberry Pi 5 Streaming Server

#### 前置要求 / Prerequisites
- 已安裝 Tailscale
- 網路連線正常

#### 安裝步驟 / Installation Steps
1. 系統套件安裝
   ```bash
   sudo apt update
   sudo apt install ffmpeg nginx
   pip install flask
   ```

2. 設定自動啟動服務 / Setup Autostart Service
   ```bash
   sudo nano /etc/systemd/system/stream_record.service
   ```

   服務配置內容 / Service Configuration:
   ```ini
   [Unit]
   Description=Stream and Record Flask App
   After=network.target

   [Service]
   User=t1204
   WorkingDirectory=/home/t1204/stream_record
   ExecStart=/usr/bin/python3 /home/t1204/stream_record/app3.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. 啟用服務 / Enable Service
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start stream_record.service
   sudo systemctl enable stream_record.service
   ```

4. 檢查服務狀態 / Check Service Status
   ```bash
   sudo systemctl status stream_record.service
   ```

### 使用方式 / Usage
1. 確保所有設備都已連接到 Tailscale 網路
2. 訪問串流服務
- 網址：http://100.70.26.103:5000/
![Screenshot_20241229_213400](https://github.com/user-attachments/assets/3339dfe2-f4b0-4e87-83d2-07c86356a88a)
### 注意事項 / Notes
- 請確保防火牆設定允許所需端口通訊
- 建議定期檢查系統日誌以確保服務正常運行
- 注意儲存空間使用情況，以免影響錄製功能

### 故障排除 / Troubleshooting
- 如果串流服務無法訪問，請檢查：
  1. Tailscale 連接狀態
  2. 服務運行狀態
  3. 網路連接狀態
- 如果遇到效能問題，可以調整串流參數




