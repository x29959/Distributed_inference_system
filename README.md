# 分散式及自適應學習之架構說明

## 目錄
- [系統架構參考](#系統架構參考)
- [環境需求](#環境需求)
  - [硬體需求](#硬體需求)
  - [環境建置步驟](#環境建置步驟)
    - [必要套件安裝](#1-必要套件安裝)
    - [Conda 環境建立](#2-conda-環境建立)
- [系統組件啟動說明](#系統組件啟動說明)
  - [遠端邊緣伺服器](#1-遠端邊緣伺服器-rtx3060)
  - [本地端邊緣伺服器](#2-本地端邊緣伺服器-raspberry-pi-5)
  - [雲端伺服器](#3-雲端伺服器)
- [系統整體啟動順序](#系統整體啟動順序)

## 系統架構參考
相關說明請參考：[Distributed Inference System Documentation](https://github.com/x29959/Distributed_inference_system/blob/276099927e681344d4c31f87149c585b673e7789/Rtmp_server/Readme.md)

## 環境需求

### 硬體需求
- Nvidia GPU
- Intel 處理器
- Tailscale (需安裝)

### 環境建置步驟

#### 1. 必要套件安裝
```bash
# 核心套件
pip install boxmot==10.0.73  # 會自動安裝 pytorch 及 cuda 相關套件
pip install ultralytics==8.2.36
pip install numpy==1.24.4
pip install moviepy==1.0.3

# PostgreSQL 相關
pip install psycopg2
sudo apt-get install libpq-dev

# 其他必要套件
pip install opencv-python flask flask-socketio
pip install -r requirements.txt
```

#### 2. Conda 環境建立
```bash
conda env create -f environment.yml
```

## 系統組件啟動說明

### 1. 開啟遠端邊緣伺服器 (RTX3060)
1. 執行 `inference.py`
2. 根據 IP 位址及 port 開啟網站

### 2. 開啟本地端邊緣伺服器 (Raspberry Pi 5)
```bash
cd Desktop/new_combine_system_local/
source /home/t1204/Desktop/new_combine_system_local/.venv/bin/activate
/home/t1204/Desktop/new_combine_system_local/.venv/bin/python /home/t1204/Desktop/new_combine_system_local/inference.py
```

## 開啟雲端伺服器 [Readme_file ](https://github.com/x29959/Distributed_inference_system/blob/fbc32275a02449f8459683bd6404b5b193f01a51/cloud_retrain/Readme.md)
**說明：** 在另一個裝置啟動雲端伺服器

**前提條件：**
- 安裝相同環境
- 安裝 Tailscale
- 檢查雲端位址（如需更換裝置，須更新遠端伺服器的 IP）
 IP![Screenshot_20241229_235641](https://github.com/user-attachments/assets/99629b30-faf8-4412-8165-6395312858b5)

**步驟：**
- 執行 `cloud.py`

## 開啟終端伺服器 [Readme_file](https://github.com/x29959/Distributed_inference_system/blob/fbc32275a02449f8459683bd6404b5b193f01a51/Rtmp_server/Readme.md)
1. 開啟兩個串流伺服器 
   - 無人機
   - 樹莓派槍支機構
2. 開啟 Temi 機器人上的 MQTT client app
### Part 1. TEMI MQTT client app
![Screenshot_20241229_233055](https://github.com/user-attachments/assets/dbe6a693-29f1-4f14-8dcc-9bb4ad1ee752)



### choose the server you want to connect

### Part 2. connect to the server

![Screenshot_20241229_233303](https://github.com/user-attachments/assets/aefd8257-2da9-4263-9d9f-e15c2a5534da)



4. 開啟雲端伺服器
5. 開啟邊緣伺服器（遠端或本地）
