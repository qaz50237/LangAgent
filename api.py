"""
FastAPI 後端 API
提供 RESTful API 介面來與 Agent 互動
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
import uvicorn
import os
from dotenv import load_dotenv

from src.agent import create_agent, SYSTEM_PROMPT, LangGraphAgent
from src.tools import get_tools_description

# 載入環境變數
load_dotenv()

# 全域 Agent 實例
agent: Optional[LangGraphAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    global agent
    # 啟動時初始化 Agent
    print("初始化 Agent...")
    agent = create_agent()
    print("Agent 已就緒!")
    yield
    # 關閉時清理資源
    print("關閉 Agent...")


app = FastAPI(
    title="LangAgent API",
    description="基於 LangChain/LangGraph 的 AI Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 模型 ====================

class ChatRequest(BaseModel):
    """對話請求"""
    message: str = Field(..., description="用戶訊息", min_length=1)
    session_id: Optional[str] = Field(None, description="會話 ID（用於保持對話歷史）")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "現在幾點？",
                    "session_id": "user-123"
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """對話回應"""
    response: str = Field(..., description="Agent 的回應")
    session_id: Optional[str] = Field(None, description="會話 ID")


class ToolInfo(BaseModel):
    """工具資訊"""
    name: str
    description: str


class ToolsResponse(BaseModel):
    """工具列表回應"""
    tools: List[ToolInfo]
    count: int


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str
    agent_ready: bool
    version: str


# ==================== 會話管理 ====================

# 簡單的會話存儲（生產環境應使用 Redis 等）
sessions: dict = {}


# ==================== API 端點 ====================

@app.get("/", response_model=dict)
async def root():
    """API 根端點"""
    return {
        "name": "LangAgent API",
        "version": "1.0.0",
        "description": "基於 LangChain/LangGraph 的 AI Agent API",
        "docs_url": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查"""
    return HealthResponse(
        status="healthy",
        agent_ready=agent is not None,
        version="1.0.0",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    與 Agent 對話
    
    - **message**: 用戶訊息
    - **session_id**: 可選的會話 ID，用於保持對話歷史
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    
    try:
        # 取得或建立會話歷史
        history = []
        if request.session_id and request.session_id in sessions:
            history = sessions[request.session_id]
        
        # 與 Agent 對話
        response, updated_history = agent.chat_with_history(request.message, history)
        
        # 更新會話歷史
        if request.session_id:
            sessions[request.session_id] = updated_history
        
        return ChatResponse(
            response=response,
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"對話處理失敗: {str(e)}")


@app.post("/chat/simple", response_model=ChatResponse)
async def simple_chat(request: ChatRequest):
    """
    簡單對話（不保留歷史）
    
    - **message**: 用戶訊息
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    
    try:
        response = agent.chat(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"對話處理失敗: {str(e)}")


@app.get("/tools", response_model=ToolsResponse)
async def get_tools():
    """取得可用工具列表"""
    from src.tools import ALL_TOOLS
    
    tools = [
        ToolInfo(name=tool.name, description=tool.description)
        for tool in ALL_TOOLS
    ]
    
    return ToolsResponse(tools=tools, count=len(tools))


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """清除指定會話的對話歷史"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"會話 {session_id} 已清除"}
    raise HTTPException(status_code=404, detail=f"找不到會話 {session_id}")


@app.get("/sessions")
async def list_sessions():
    """列出所有會話 ID"""
    return {
        "sessions": list(sessions.keys()),
        "count": len(sessions),
    }


# ==================== 主程式 ====================

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"啟動 API 伺服器: http://{host}:{port}")
    print(f"API 文件: http://{host}:{port}/docs")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
    )
