@echo off
setlocal

REM xAI Week 10 一鍵執行腳本（Windows）
REM 使用方式：
REM 1) 先安裝並設定好 Miniconda
REM 2) 於專案根目錄執行：run_all.bat

echo [1/4] 啟用 conda 環境 xai_week10...
call conda activate xai_week10
if errorlevel 1 (
    echo 無法啟用 conda 環境 xai_week10，請先建立環境：
    echo conda create -n xai_week10 python=3.10 -y
    exit /b 1
)

echo [2/4] 產生模擬資料...
python data\generate_data.py
if errorlevel 1 (
    echo data\generate_data.py 執行失敗
    exit /b 1
)

echo [3/4] 訓練模型與可靠度分析...
python src\train_model.py
if errorlevel 1 (
    echo src\train_model.py 執行失敗
    exit /b 1
)

echo [4/4] 啟動 Streamlit Dashboard...
echo 如需停止請按 Ctrl+C
streamlit run dashboard\app.py

endlocal
