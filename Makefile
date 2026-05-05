PYTHON := C:/Users/User/miniconda3/envs/xai_week10/python.exe

.PHONY: data train dashboard all

# 產生模擬資料
data:
	"$(PYTHON)" data/generate_data.py

# 訓練模型與可靠度分析
train:
	"$(PYTHON)" src/train_model.py

# 啟動互動式儀表板
dashboard:
	streamlit run dashboard/app.py

# 一次執行資料與訓練流程
all: data train
