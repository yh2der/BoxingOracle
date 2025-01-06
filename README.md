# BoxingOracle

一個結合拳擊比賽模擬與機器學習的預測系統。透過模擬大量拳擊比賽來收集資料，並使用手寫的神經網路進行比賽結果預測。

## 專案概述

本專案包含兩個主要版本：
- **Version 1**: 基礎拳擊模擬系統
- **Version 2**: 進階資料收集與機器學習預測

## 專案結構

```
BoxingOracle/
├── version 1/              # 版本一：基礎拳擊模擬
│   ├── game.py             # 拳擊比賽模擬核心
│   ├── model.py            # 神經網路實現
│   ├── README.md           # 版本一說明文件
│   └── data/               # 模擬比賽資料
│
├── version 2/              # 版本二：進階預測系統
│   ├── game.py             # 改進版比賽模擬
│   ├── game_animation.py   # 改進版比賽模擬（完整比賽過程）
│   ├── model.py            # 神經網路實現
│   ├── data_processor.py   # 資料處理工具
│   ├── README.md           # 版本二說明文件
│   └── data/               # 訓練資料集
│
└── README.md               # 專案主文件（本文件）
```

## 版本特色比較

### Version 1 特色
- 基礎拳擊物理引擎
- 選手屬性系統
- 真實戰鬥機制
- CSV 資料記錄

### Version 2 改進
- 完整比賽日誌系統
- 進階選手屬性計算
- 詳細的資料收集
- 手刻神經網路預測
- 改進的戰鬥平衡

## 安裝指南

1. clone 專案：
```bash
git clone https://github.com/yh2der/BoxingOracle.git
cd BoxingOracle
```

2. 安裝相關套件：
```bash
pip install -r requirements.txt
```

## 開發計劃 

- [x] 基礎拳擊模擬 (version 1)
- [x] 數據收集系統 (version 1)
- [x] 進階比賽機制 (version 2)
- [x] 神經網路預測 (version 2)
- [ ] 網頁介面開發
- [ ] API 服務

## Demo
[![BoxingOracle](https://youtu.be/LiB-moIhQ6Y/0.jpg)](https://youtu.be/LiB-moIhQ6Y)