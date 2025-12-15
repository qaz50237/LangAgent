<!-- LangAgent 專案說明 -->
<!-- 基於 LangChain 和 LangGraph 的 AI Agent -->

## 專案架構

- `src/agent.py` - LangGraph Agent 核心，使用 ReAct 模式
- `src/tools.py` - Function Tools 定義，可替換為 MCP
- `api.py` - FastAPI 後端 API
- `ui.py` - Streamlit 互動式 UI

## 開發指南

### 新增工具
在 `src/tools.py` 中使用 `@tool` 裝飾器新增工具，並加入 `ALL_TOOLS` 列表。

### 修改 Agent
在 `src/agent.py` 中可以修改 LangGraph 工作流程圖。

### API 擴展
在 `api.py` 中新增 FastAPI 端點。

## 啟動命令

```bash
# 啟動 API
python api.py

# 啟動 UI
streamlit run ui.py
```
