"""
LangGraph Agent 核心模組
基於 LangGraph 實現的 ReAct Agent，支援 function tools 調用
"""

from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

from .tools import ALL_TOOLS

# 載入環境變數
load_dotenv()


def get_llm(model_name: str = None, temperature: float = 0.7):
    """
    根據環境變數自動選擇 LLM
    優先使用 Azure OpenAI，若無則使用 OpenAI
    """
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if azure_api_key and azure_endpoint and azure_deployment:
        # 使用 Azure OpenAI
        # 從 endpoint 中提取基礎 URL
        base_endpoint = azure_endpoint.split("/openai/")[0] if "/openai/" in azure_endpoint else azure_endpoint
        
        # 注意：某些 Azure 模型不支援自訂 temperature，使用預設值
        return AzureChatOpenAI(
            azure_endpoint=base_endpoint,
            azure_deployment=azure_deployment,
            api_key=azure_api_key,
            api_version="2025-01-01-preview",
        )
    else:
        # 使用 OpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
        )


class AgentState(TypedDict):
    """Agent 狀態定義"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class LangGraphAgent:
    """
    基於 LangGraph 的 AI Agent
    
    特點:
    - 支援多種 function tools
    - ReAct 推理模式
    - 可擴展的工具系統（可替換為 MCP）
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        tools: list = None,
    ):
        """
        初始化 Agent
        
        Args:
            model_name: OpenAI 模型名稱
            temperature: 生成溫度
            tools: 自訂工具列表，預設使用 ALL_TOOLS
        """
        self.tools = tools or ALL_TOOLS
        
        # 初始化 LLM（自動選擇 Azure 或 OpenAI）
        self.llm = get_llm(model_name, temperature).bind_tools(self.tools)
        
        # 建構圖形
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """建構 LangGraph 工作流程圖"""
        
        # 定義節點
        def call_model(state: AgentState) -> dict:
            """調用 LLM 模型"""
            messages = state["messages"]
            response = self.llm.invoke(messages)
            return {"messages": [response]}
        
        def should_continue(state: AgentState) -> Literal["tools", "end"]:
            """決定是否繼續執行工具或結束"""
            messages = state["messages"]
            last_message = messages[-1]
            
            # 如果有工具調用，繼續執行
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "end"
        
        # 建構圖形
        workflow = StateGraph(AgentState)
        
        # 添加節點
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # 設定入口點
        workflow.set_entry_point("agent")
        
        # 添加條件邊
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END,
            }
        )
        
        # 工具執行後返回 agent
        workflow.add_edge("tools", "agent")
        
        # 編譯圖形
        return workflow.compile()
    
    def get_graph_image(self, output_format: str = "png") -> bytes:
        """
        取得 LangGraph 工作流程圖的圖像
        
        Args:
            output_format: 輸出格式 ("png", "ascii", "mermaid")
        
        Returns:
            圖像的 bytes 資料，或 ASCII/Mermaid 字串
        """
        try:
            if output_format == "ascii":
                return self.graph.get_graph().draw_ascii()
            elif output_format == "mermaid":
                return self.graph.get_graph().draw_mermaid()
            else:
                # PNG 格式需要 grandalf 套件
                return self.graph.get_graph().draw_mermaid_png()
        except Exception as e:
            raise RuntimeError(f"無法生成圖形: {str(e)}")
    
    def get_graph_mermaid(self) -> str:
        """
        取得 Mermaid 格式的圖形定義
        
        Returns:
            Mermaid 格式字串
        """
        return self.graph.get_graph().draw_mermaid()
    
    def chat(self, message: str, history: list[BaseMessage] = None) -> str:
        """
        與 Agent 對話
        
        Args:
            message: 用戶訊息
            history: 對話歷史
        
        Returns:
            Agent 的回應
        """
        messages = history or []
        messages.append(HumanMessage(content=message))
        
        # 執行圖形
        result = self.graph.invoke({"messages": messages})
        
        # 取得最後的 AI 回應
        final_message = result["messages"][-1]
        return final_message.content
    
    def chat_with_history(self, message: str, history: list[BaseMessage] = None) -> tuple[str, list[BaseMessage]]:
        """
        與 Agent 對話並返回更新的歷史
        
        Args:
            message: 用戶訊息
            history: 對話歷史
        
        Returns:
            (Agent 回應, 更新後的對話歷史)
        """
        messages = list(history) if history else []
        messages.append(HumanMessage(content=message))
        
        # 執行圖形
        result = self.graph.invoke({"messages": messages})
        
        # 取得回應和完整歷史
        final_message = result["messages"][-1]
        return final_message.content, result["messages"]
    
    async def achat(self, message: str, history: list[BaseMessage] = None) -> str:
        """
        異步與 Agent 對話
        
        Args:
            message: 用戶訊息
            history: 對話歷史
        
        Returns:
            Agent 的回應
        """
        messages = history or []
        messages.append(HumanMessage(content=message))
        
        # 異步執行圖形
        result = await self.graph.ainvoke({"messages": messages})
        
        # 取得最後的 AI 回應
        final_message = result["messages"][-1]
        return final_message.content


def create_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> LangGraphAgent:
    """
    工廠函數：建立 Agent 實例
    
    Args:
        model_name: OpenAI 模型名稱
        temperature: 生成溫度
    
    Returns:
        LangGraphAgent 實例
    """
    return LangGraphAgent(model_name=model_name, temperature=temperature)


# 系統提示詞
SYSTEM_PROMPT = """你是一個智能助理，擁有以下能力：

1. **時間查詢**: 可以告訴用戶目前的日期和時間
2. **數學計算**: 可以執行各種數學運算
3. **知識搜尋**: 可以搜尋內部知識庫
4. **天氣查詢**: 可以查詢城市天氣（目前支援：台北、高雄、東京）
5. **單位轉換**: 可以進行長度、重量、溫度等單位轉換

請根據用戶的需求，適當使用這些工具來提供幫助。
回答時請使用繁體中文，並保持友善和專業。
"""
