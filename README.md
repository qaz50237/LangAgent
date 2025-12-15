# LangAgent - LangChain/LangGraph AI Agent

基於 LangChain 和 LangGraph 建構的 AI Agent 範例專案，支援 Function Tools 調用，並提供 API 介面和 Web UI。

## 🌟 特點

- ✅ **LangGraph Agent**: 使用 LangGraph 建構的 ReAct Agent
- ✅ **Function Tools**: 內建多種實用工具（可替換為 MCP）
- ✅ **FastAPI 後端**: RESTful API 介面
- ✅ **Streamlit UI**: 互動式 Web 介面
- ✅ **會話管理**: 支援對話歷史保持

## 📁 專案結構

```
LangAgent/
├── src/
│   ├── __init__.py
│   ├── agent.py        # LangGraph Agent 核心
│   └── tools.py        # Function Tools 定義
├── api.py              # FastAPI 後端
├── ui.py               # Streamlit UI
├── requirements.txt    # Python 依賴
├── .env.example        # 環境變數範例
└── README.md
```

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 建立虛擬環境（建議）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安裝依賴
pip install -r requirements.txt
```

### 2. 設定環境變數

```bash
# 複製環境變數範例
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# 編輯 .env 檔案，填入您的 OpenAI API Key
```

**.env 檔案內容：**
```env
OPENAI_API_KEY=your-openai-api-key-here
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. 啟動服務

**啟動 API 伺服器：**
```bash
python api.py
```

**啟動 Web UI（新開一個終端）：**
```bash
streamlit run ui.py
```

### 4. 開始使用

- **API 文件**: http://localhost:8000/docs
- **Web UI**: http://localhost:8501

## 🛠️ 內建工具

| 工具名稱 | 功能描述 |
|---------|---------|
| `get_current_time` | 取得目前日期和時間 |
| `calculator` | 數學計算（支援基本運算和進階函數）|
| `search_knowledge_base` | 搜尋內部知識庫 |
| `weather_info` | 查詢城市天氣（模擬資料）|
| `unit_converter` | 單位轉換（長度、重量、溫度）|

## 📡 API 端點

| 端點 | 方法 | 描述 |
|-----|------|------|
| `/` | GET | API 資訊 |
| `/health` | GET | 健康檢查 |
| `/chat` | POST | 對話（支援會話歷史）|
| `/chat/simple` | POST | 簡單對話（無歷史）|
| `/tools` | GET | 取得可用工具列表 |
| `/sessions` | GET | 列出所有會話 |
| `/sessions/{id}` | DELETE | 清除指定會話 |

### API 使用範例

```python
import requests

# 發送對話請求
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "現在幾點？",
        "session_id": "user-123"
    }
)
print(response.json())
```

```bash
# 使用 curl
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "計算 sqrt(144) + 10", "session_id": "test"}'
```

## 🔄 擴展為 MCP

目前的 Function Tools 設計可以輕鬆替換為 MCP (Model Context Protocol)：

1. 在 `src/tools.py` 中，將工具實現替換為 MCP 客戶端調用
2. 或建立 MCP Server，讓 Agent 透過 MCP 協議調用工具

**MCP 替換範例：**
```python
from mcp import Client

# 將原本的本地函數替換為 MCP 調用
@tool
async def search_knowledge_base(query: str) -> str:
    """透過 MCP 搜尋知識庫"""
    async with Client("mcp://localhost:9000") as client:
        result = await client.call_tool("search", {"query": query})
        return result
```

## 🎯 範例對話

```
用戶: 現在幾點？
Agent: 目前時間: 2025年12月15日 14:30:25

用戶: 計算 sqrt(144) + 25 * 2
Agent: 計算結果: sqrt(144) + 25 * 2 = 62.0

用戶: 台北今天天氣如何？
Agent: 台北的天氣: 溫度 25°C, 多雲, 濕度 75%

用戶: 把 100 公里轉換成英里
Agent: 100 km = 62.1371 mile
```

## 📝 自訂工具

在 `src/tools.py` 中新增自訂工具：

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(param: str) -> str:
    """我的自訂工具描述"""
    # 實作邏輯
    return f"處理結果: {param}"

# 記得將新工具加入 ALL_TOOLS 列表
ALL_TOOLS.append(my_custom_tool)
```

## 🔧 設定選項

### Agent 設定

在 `src/agent.py` 中可調整：
- `model_name`: OpenAI 模型（預設: gpt-4o-mini）
- `temperature`: 生成溫度（預設: 0.7）

### API 設定

透過環境變數設定：
- `API_HOST`: API 主機（預設: 0.0.0.0）
- `API_PORT`: API 埠號（預設: 8000）

## 📚 技術棧

- **LangChain**: LLM 應用程式框架
- **LangGraph**: 圖形化工作流程框架
- **FastAPI**: 高效能 API 框架
- **Streamlit**: Python Web UI 框架
- **OpenAI**: GPT 模型服務

## 📄 授權

MIT License
