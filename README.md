```markdown
# NNu-net and ResNet34 for Animal Recognition

這是一個專注於電腦視覺與深度學習實作的專案，主要探討如何利用 UNet 架構結合 ResNet34 作為主幹網路（Backbone），來進行精準的動物影像辨識與語意分割（Semantic Segmentation）。

## 💡 核心技術與特色 (Features)
* **自定義模型架構**：從頭實作標準 UNet，並整合預訓練的 ResNet34 提取深層特徵。
* **模組化設計**：將訓練 (Training)、推論 (Inference) 與評估 (Evaluation) 邏輯完全解耦，符合軟體工程的擴展性原則。
* **效能追蹤**：實作了完整的 Loss 計算與驗證集評估機制。

## 📂 專案架構 (Project Structure)
```text
├── dataset/             # (Git Ignore) 資料集存放位置
├── saved_models/        # (Git Ignore) 訓練好的權重檔 (.pth)
├── src/                 # 原始碼目錄
│   ├── models/          # 模型架構定義 (unet.py, resnet34_unet.py)
│   ├── train.py         # 訓練主程式
│   ├── evaluate.py      # 模型效能評估
│   ├── inference.py     # 單張影像預測腳本
│   ├── oxford_pet.py    # 資料集載入與前處理 (Dataset & DataLoader)
│   └── utils.py         # 共用工具與函式
├── requirements.txt     # 套件依賴清單
└── README.md```

## 📊 資料集來源 (Dataset)
本專案使用 **Oxford-IIIT Pet Dataset** 進行訓練與測試。
* **資料集說明**：包含 37 種不同品種的貓狗影像，每種品種約 200 張圖片，並附帶精細的像素級標註（Pixel-level annotations）。
* **下載連結**：[Oxford-IIIT Pet Dataset 官方網站](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* **配置方式**：請將下載的 `images/` 與 `annotations/` 資料夾解壓縮並放置於專案根目錄的 `dataset/oxford-iiit-pet/` 路徑下。

## 🚀 快速開始 (Quick Start)

### 1. 安裝環境依賴
建議使用虛擬環境（Virtual Environment）來執行：
```bash
pip install -r requirements.txt
```

### 2. 開始訓練模型
```bash
python src/train.py --epochs 50 --batch_size 16
```

### 3. 執行模型推論
```bash
python src/inference.py --image_path ./test_image.jpg --model_weights ./saved_models/best_unet.pth
```

## 📈 實驗結果 (Results)
* 在 Test Set 上的表現：
  * **Accuracy**: 於Kaggle 競賽獲得 0.91 dice scroe

