"""
會議室預約 LangGraph Agent
基於 Supervisor 模式的多 Agent 系統
支援靜態工具定義或 MCP 動態工具載入
"""

from typing import Annotated, Sequence, TypedDict, Literal, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from .tools import (
    BOOKING_TOOLS,
    QUERY_TOOLS,
    MANAGEMENT_TOOLS,
    MEETING_ROOM_TOOLS,
)
from .tool_classifier import MCPToolClassifier, classify_mcp_tools
from .prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    BOOKING_AGENT_SYSTEM_PROMPT,
    QUERY_AGENT_SYSTEM_PROMPT,
    MANAGEMENT_AGENT_SYSTEM_PROMPT,
    UNIFIED_AGENT_SYSTEM_PROMPT,
)
from .state import MeetingState

# 載入環境變數
load_dotenv()


def get_llm(model_name: str = None, temperature: float = 0.7):
    """根據環境變數自動選擇 LLM"""
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if azure_api_key and azure_endpoint and azure_deployment:
        base_endpoint = azure_endpoint.split("/openai/")[0] if "/openai/" in azure_endpoint else azure_endpoint
        return AzureChatOpenAI(
            azure_endpoint=base_endpoint,
            azure_deployment=azure_deployment,
            api_key=azure_api_key,
            api_version="2025-01-01-preview",
        )
    else:
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
        )


class MeetingRoomAgent:
    """
    會議室預約多 Agent 系統
    
    架構:
    - Supervisor Agent: 分析意圖，路由到專業 Agent
    - Booking Agent: 處理預約流程
    - Query Agent: 處理查詢請求
    - Management Agent: 處理預約管理（查看/取消）
    
    支援兩種工具來源:
    1. 靜態定義 (預設): 使用 tools.py 中定義的工具
    2. MCP 動態載入: 傳入 mcp_tools 參數，自動分類
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        user_id: str = "default_user",
        mcp_tools: List[BaseTool] = None,
        tool_classification_strategy: str = "explicit",
    ):
        """
        初始化會議室預約 Agent
        
        Args:
            model_name: LLM 模型名稱
            temperature: 生成溫度
            user_id: 預設用戶 ID
            mcp_tools: MCP 載入的工具列表（可選）。若提供，將自動分類
            tool_classification_strategy: MCP 工具分類策略 
                - "explicit": 根據配置檔分類（預設）
                - "prefix": 根據工具名稱前綴
                - "keyword": 根據描述關鍵字
        """
        self.model_name = model_name
        self.temperature = temperature
        self.default_user_id = user_id
        
        # 決定使用靜態工具還是 MCP 工具
        if mcp_tools:
            # MCP 模式：動態分類工具
            classified = classify_mcp_tools(mcp_tools, strategy=tool_classification_strategy)
            self.booking_tools = classified["booking"]
            self.query_tools = classified["query"]
            self.management_tools = classified["management"]
            self.all_tools = mcp_tools
        else:
            # 靜態模式：使用預定義工具
            self.booking_tools = BOOKING_TOOLS
            self.query_tools = QUERY_TOOLS
            self.management_tools = MANAGEMENT_TOOLS
            self.all_tools = MEETING_ROOM_TOOLS
        
        # 初始化各專業 Agent 的 LLM
        self.supervisor_llm = get_llm(model_name, temperature)
        self.booking_llm = get_llm(model_name, temperature).bind_tools(self.booking_tools)
        self.query_llm = get_llm(model_name, temperature).bind_tools(self.query_tools)
        self.management_llm = get_llm(model_name, temperature).bind_tools(self.management_tools)
        
        # 建構 LangGraph 工作流
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """建構 Supervisor 多 Agent 工作流程圖"""
        
        # ========================================
        # Supervisor 節點 - 意圖分析與路由
        # ========================================
        def supervisor_node(state: MeetingState) -> dict:
            """Supervisor: 分析用戶意圖並決定路由"""
            messages = state["messages"]
            
            # 建立 Supervisor 的分析請求
            analysis_messages = [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                *messages
            ]
            
            response = self.supervisor_llm.invoke(analysis_messages)
            
            # 嘗試解析 JSON 回應
            try:
                # 嘗試從回應中提取 JSON
                content = response.content
                # 處理可能被 markdown 包裹的 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                decision = json.loads(content.strip())
                intent = decision.get("intent", "unclear")
                
                if intent == "unclear":
                    # 需要澄清，返回澄清問題
                    clarification = decision.get("clarification_needed", "請問您想要預約會議室、查詢會議室，還是管理已有的預約？")
                    return {
                        "messages": [AIMessage(content=clarification)],
                        "intent": None,
                        "current_agent": "supervisor",
                    }
                
                return {
                    "intent": intent,
                    "current_agent": decision.get("agent", f"{intent}_agent"),
                }
            except (json.JSONDecodeError, KeyError):
                # 無法解析，預設使用整合查詢
                return {
                    "intent": "query",
                    "current_agent": "query_agent",
                }
        
        # ========================================
        # Booking Agent 節點 - 預約流程
        # ========================================
        def booking_agent_node(state: MeetingState) -> dict:
            """Booking Agent: 處理會議室預約"""
            messages = state["messages"]
            user_id = state.get("user_id") or self.default_user_id
            
            system_prompt = BOOKING_AGENT_SYSTEM_PROMPT + f"\n\n當前用戶ID: {user_id}\n今天日期: {datetime.now().strftime('%Y-%m-%d')}"
            
            agent_messages = [
                SystemMessage(content=system_prompt),
                *messages
            ]
            
            response = self.booking_llm.invoke(agent_messages)
            return {"messages": [response], "current_agent": "booking_agent"}
        
        # ========================================
        # Query Agent 節點 - 查詢服務
        # ========================================
        def query_agent_node(state: MeetingState) -> dict:
            """Query Agent: 處理查詢請求"""
            messages = state["messages"]
            
            system_prompt = QUERY_AGENT_SYSTEM_PROMPT + f"\n\n今天日期: {datetime.now().strftime('%Y-%m-%d')}"
            
            agent_messages = [
                SystemMessage(content=system_prompt),
                *messages
            ]
            
            response = self.query_llm.invoke(agent_messages)
            return {"messages": [response], "current_agent": "query_agent"}
        
        # ========================================
        # Management Agent 節點 - 預約管理
        # ========================================
        def management_agent_node(state: MeetingState) -> dict:
            """Management Agent: 處理預約管理"""
            messages = state["messages"]
            user_id = state.get("user_id") or self.default_user_id
            
            system_prompt = MANAGEMENT_AGENT_SYSTEM_PROMPT + f"\n\n當前用戶ID: {user_id}"
            
            agent_messages = [
                SystemMessage(content=system_prompt),
                *messages
            ]
            
            response = self.management_llm.invoke(agent_messages)
            return {"messages": [response], "current_agent": "management_agent"}
        
        # ========================================
        # 路由函數
        # ========================================
        def route_by_intent(state: MeetingState) -> str:
            """根據意圖決定下一個節點"""
            intent = state.get("intent")
            
            if intent == "booking":
                return "booking_agent"
            elif intent == "query":
                return "query_agent"
            elif intent == "management":
                return "management_agent"
            else:
                # 意圖不明，結束流程（已由 supervisor 返回澄清問題）
                return END
        
        def should_continue_booking(state: MeetingState) -> str:
            """Booking Agent 是否需要繼續執行工具"""
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "booking_tools"
            return END
        
        def should_continue_query(state: MeetingState) -> str:
            """Query Agent 是否需要繼續執行工具"""
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "query_tools"
            return END
        
        def should_continue_management(state: MeetingState) -> str:
            """Management Agent 是否需要繼續執行工具"""
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "management_tools"
            return END
        
        # ========================================
        # 建構圖形
        # ========================================
        workflow = StateGraph(MeetingState)
        
        # 添加節點
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("booking_agent", booking_agent_node)
        workflow.add_node("query_agent", query_agent_node)
        workflow.add_node("management_agent", management_agent_node)
        
        # 添加工具節點（使用實例變數，支援 MCP 動態工具）
        workflow.add_node("booking_tools", ToolNode(self.booking_tools))
        workflow.add_node("query_tools", ToolNode(self.query_tools))
        workflow.add_node("management_tools", ToolNode(self.management_tools))
        
        # 設定入口點
        workflow.set_entry_point("supervisor")
        
        # Supervisor 路由到專業 Agent
        workflow.add_conditional_edges(
            "supervisor",
            route_by_intent,
            {
                "booking_agent": "booking_agent",
                "query_agent": "query_agent",
                "management_agent": "management_agent",
                END: END,
            }
        )
        
        # Booking Agent 的條件邊
        workflow.add_conditional_edges(
            "booking_agent",
            should_continue_booking,
            {
                "booking_tools": "booking_tools",
                END: END,
            }
        )
        workflow.add_edge("booking_tools", "booking_agent")
        
        # Query Agent 的條件邊
        workflow.add_conditional_edges(
            "query_agent",
            should_continue_query,
            {
                "query_tools": "query_tools",
                END: END,
            }
        )
        workflow.add_edge("query_tools", "query_agent")
        
        # Management Agent 的條件邊
        workflow.add_conditional_edges(
            "management_agent",
            should_continue_management,
            {
                "management_tools": "management_tools",
                END: END,
            }
        )
        workflow.add_edge("management_tools", "management_agent")
        
        return workflow.compile()
    
    def chat(self, message: str, user_id: str = None, history: list[BaseMessage] = None) -> str:
        """
        與 Agent 對話
        
        Args:
            message: 用戶訊息
            user_id: 用戶ID
            history: 對話歷史
        
        Returns:
            Agent 的回應
        """
        messages = history or []
        messages.append(HumanMessage(content=message))
        
        result = self.graph.invoke({
            "messages": messages,
            "user_id": user_id or self.default_user_id,
            "intent": None,
            "current_agent": None,
            "context": None,
        })
        
        final_message = result["messages"][-1]
        return final_message.content
    
    def chat_with_history(
        self, 
        message: str, 
        user_id: str = None,
        history: list[BaseMessage] = None
    ) -> tuple[str, list[BaseMessage]]:
        """與 Agent 對話並返回更新的歷史"""
        messages = list(history) if history else []
        messages.append(HumanMessage(content=message))
        
        result = self.graph.invoke({
            "messages": messages,
            "user_id": user_id or self.default_user_id,
            "intent": None,
            "current_agent": None,
            "context": None,
        })
        
        final_message = result["messages"][-1]
        return final_message.content, result["messages"]


# ============================================================
# 簡化版：單一 Agent 模式
# ============================================================

class SimpleBookingAgent:
    """
    簡化版會議室預約 Agent
    單一 Agent 處理所有任務，適合較簡單的場景
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        user_id: str = "default_user",
    ):
        self.default_user_id = user_id
        self.llm = get_llm(model_name, temperature).bind_tools(MEETING_ROOM_TOOLS)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """建構簡單的 ReAct 工作流"""
        
        def agent_node(state: MeetingState) -> dict:
            messages = state["messages"]
            user_id = state.get("user_id") or self.default_user_id
            
            current_date = datetime.now().strftime('%Y-%m-%d')
            system_prompt = UNIFIED_AGENT_SYSTEM_PROMPT.format(current_date=current_date)
            system_prompt += f"\n\n當前用戶ID: {user_id}"
            
            agent_messages = [
                SystemMessage(content=system_prompt),
                *messages
            ]
            
            response = self.llm.invoke(agent_messages)
            return {"messages": [response]}
        
        def should_continue(state: MeetingState) -> Literal["tools", "end"]:
            messages = state["messages"]
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"
        
        workflow = StateGraph(MeetingState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(MEETING_ROOM_TOOLS))
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def chat(self, message: str, user_id: str = None, history: list[BaseMessage] = None) -> str:
        """與 Agent 對話"""
        messages = history or []
        messages.append(HumanMessage(content=message))
        
        result = self.graph.invoke({
            "messages": messages,
            "user_id": user_id or self.default_user_id,
            "intent": None,
            "current_agent": None,
            "context": None,
        })
        
        final_message = result["messages"][-1]
        return final_message.content


# ============================================================
# 工廠函數
# ============================================================

def create_meeting_room_agent(
    mode: Literal["multi", "simple"] = "multi",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    user_id: str = "default_user",
):
    """
    建立會議室預約 Agent
    
    Args:
        mode: "multi" = 多 Agent 模式 (Supervisor), "simple" = 單一 Agent 模式
        model_name: LLM 模型名稱
        temperature: 生成溫度
        user_id: 預設用戶ID
    
    Returns:
        MeetingRoomAgent 或 SimpleBookingAgent 實例
    """
    if mode == "multi":
        return MeetingRoomAgent(model_name, temperature, user_id)
    else:
        return SimpleBookingAgent(model_name, temperature, user_id)
