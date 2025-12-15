"""
Agent 基礎類別模組
提供 Agent 開發的核心抽象
"""

from abc import ABC, abstractmethod
from typing import (
    Annotated, Sequence, TypedDict, Literal, Optional,
    List, Dict, Any, Callable, Generator, Type, Union
)
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import BaseState
from .config import AgentConfig, LLMConfig
from .llm import get_llm, create_llm_with_tools
from .workflow import WorkflowBuilder, create_tool_continue_condition


@dataclass
class AgentNode:
    """
    Agent 節點定義
    
    用於定義多 Agent 系統中的單一 Agent。
    
    使用範例:
    ```python
    booking_node = AgentNode(
        name="booking_agent",
        system_prompt="你是預約助手...",
        tools=booking_tools,
    )
    ```
    """
    name: str
    system_prompt: str = ""
    tools: List[BaseTool] = None
    llm: BaseChatModel = None
    
    def __post_init__(self):
        self.tools = self.tools or []


class BaseAgent(ABC):
    """
    Agent 基礎類別
    
    所有 Agent 都應繼承此類別，只需實現核心設計邏輯。
    
    開發者只需關注:
    1. define_config(): 定義 Agent 配置（prompt, tools）
    2. define_workflow(): 定義工作流程
    3. prepare_state(): 準備初始狀態（可選覆寫）
    
    使用範例:
    ```python
    class MyAgent(BaseAgent):
        def define_config(self) -> AgentConfig:
            return AgentConfig(
                name="my_agent",
                system_prompt="你是一個助手...",
                tools=self.mcp_tools,  # 從 MCP 載入
            )
        
        def define_workflow(self, builder: WorkflowBuilder) -> WorkflowBuilder:
            builder.add_agent_node("agent", self._agent_handler)
            builder.add_tool_node("tools", self.tools)
            builder.set_entry_point("agent")
            builder.add_conditional_edge(
                "agent",
                self._should_continue,
                {"tools": "tools", "end": END}
            )
            builder.add_edge("tools", "agent")
            return builder
    
    # 使用
    agent = MyAgent(mcp_tools=my_mcp_tools)
    response = agent.chat("Hello!")
    ```
    """
    
    def __init__(
        self,
        mcp_tools: List[BaseTool] = None,
        llm_config: LLMConfig = None,
        user_id: str = "default_user",
        state_class: Type[TypedDict] = None,
        **kwargs,
    ):
        """
        初始化 Agent
        
        Args:
            mcp_tools: 從 MCP 載入的工具列表
            llm_config: LLM 配置
            user_id: 預設用戶 ID
            state_class: 自定義狀態類別
            **kwargs: 額外參數傳遞給子類
        """
        self.mcp_tools = mcp_tools or []
        self.llm_config = llm_config or LLMConfig()
        self.default_user_id = user_id
        self.state_class = state_class or BaseState
        self.extra_kwargs = kwargs
        
        # 讓子類定義配置
        self.config = self.define_config()
        
        # 取得工具（優先 MCP，其次配置）
        self.tools = self.mcp_tools if self.mcp_tools else self.config.tools
        
        # 初始化 LLM
        self.llm = self._create_llm()
        
        # 建構工作流
        self.graph = self._build_workflow()
    
    # ========================================
    # 抽象方法 - 子類必須實現
    # ========================================
    
    @abstractmethod
    def define_config(self) -> AgentConfig:
        """
        定義 Agent 配置
        
        子類必須實現此方法，返回 AgentConfig。
        
        Returns:
            AgentConfig 實例
        """
        pass
    
    @abstractmethod
    def define_workflow(self, builder: WorkflowBuilder) -> WorkflowBuilder:
        """
        定義工作流程
        
        子類必須實現此方法，使用 WorkflowBuilder 定義節點和邊。
        
        Args:
            builder: 工作流建構器
        
        Returns:
            配置好的 WorkflowBuilder
        """
        pass
    
    # ========================================
    # 可選覆寫方法
    # ========================================
    
    def prepare_state(self, message: str, user_id: str = None, history: list = None) -> dict:
        """
        準備初始狀態
        
        子類可覆寫此方法來自定義初始狀態。
        
        Args:
            message: 用戶訊息
            user_id: 用戶 ID
            history: 對話歷史
        
        Returns:
            初始狀態字典
        """
        messages = list(history) if history else []
        messages.append(HumanMessage(content=message))
        
        return {
            "messages": messages,
            "user_id": user_id or self.default_user_id,
            "intent": None,
            "current_agent": None,
            "context": None,
        }
    
    def post_process(self, result: dict) -> str:
        """
        後處理結果
        
        子類可覆寫此方法來自定義結果處理。
        
        Args:
            result: 工作流執行結果
        
        Returns:
            最終回應字串
        """
        final_message = result["messages"][-1]
        return final_message.content
    
    # ========================================
    # 內部方法
    # ========================================
    
    def _create_llm(self) -> BaseChatModel:
        """建立 LLM 實例"""
        return create_llm_with_tools(
            tools=self.tools,
            **self.llm_config.to_dict(),
        )
    
    def _build_workflow(self):
        """建構工作流"""
        builder = WorkflowBuilder(self.state_class)
        builder = self.define_workflow(builder)
        return builder.compile()
    
    # ========================================
    # 公用輔助方法（子類可使用）
    # ========================================
    
    def create_agent_handler(
        self,
        system_prompt: str,
        llm: BaseChatModel = None,
        extra_context: Callable[[dict], str] = None,
    ) -> Callable[[dict], dict]:
        """
        建立 Agent 處理函數
        
        輔助方法，幫助子類快速建立節點處理函數。
        
        Args:
            system_prompt: 系統提示詞
            llm: LLM 實例，預設使用 self.llm
            extra_context: 額外上下文生成函數
        
        Returns:
            節點處理函數
        """
        _llm = llm or self.llm
        
        def handler(state: dict) -> dict:
            messages = state.get("messages", [])
            
            # 組合系統提示詞
            full_prompt = system_prompt
            if extra_context:
                full_prompt += "\n\n" + extra_context(state)
            
            agent_messages = [
                SystemMessage(content=full_prompt),
                *messages
            ]
            
            response = _llm.invoke(agent_messages)
            return {"messages": [response]}
        
        return handler
    
    def create_tool_condition(
        self,
        tools_node: str,
        end_node: str = END,
    ) -> Callable[[dict], str]:
        """
        建立工具繼續條件
        
        Args:
            tools_node: 工具節點名稱
            end_node: 結束節點
        
        Returns:
            條件函數
        """
        return create_tool_continue_condition(tools_node, end_node)
    
    # ========================================
    # 對話方法（通用實現）
    # ========================================
    
    def chat(self, message: str, user_id: str = None, history: list[BaseMessage] = None) -> str:
        """
        與 Agent 對話
        
        Args:
            message: 用戶訊息
            user_id: 用戶 ID
            history: 對話歷史
        
        Returns:
            Agent 回應
        """
        state = self.prepare_state(message, user_id, history)
        result = self.graph.invoke(state)
        return self.post_process(result)
    
    def chat_with_history(
        self,
        message: str,
        user_id: str = None,
        history: list[BaseMessage] = None,
    ) -> tuple[str, list[BaseMessage]]:
        """
        與 Agent 對話並返回更新的歷史
        
        Args:
            message: 用戶訊息
            user_id: 用戶 ID
            history: 對話歷史
        
        Returns:
            (Agent 回應, 更新後的對話歷史)
        """
        state = self.prepare_state(message, user_id, history)
        result = self.graph.invoke(state)
        response = self.post_process(result)
        return response, result["messages"]
    
    def chat_stream(
        self,
        message: str,
        user_id: str = None,
        history: list[BaseMessage] = None,
    ) -> Generator[dict, None, None]:
        """
        串流執行對話
        
        Args:
            message: 用戶訊息
            user_id: 用戶 ID
            history: 對話歷史
        
        Yields:
            包含執行步驟資訊的字典
        """
        state = self.prepare_state(message, user_id, history)
        
        for event in self.graph.stream(state, stream_mode="updates"):
            for node_name, node_output in event.items():
                step_info = {
                    "node": node_name,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                    "output": None,
                    "messages": [],
                    "tool_calls": [],
                    "tool_results": [],
                    "intent": node_output.get("intent"),
                    "current_agent": node_output.get("current_agent"),
                }
                
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage):
                            step_info["output"] = msg.content if msg.content else "(調用工具中...)"
                            step_info["messages"].append({
                                "type": "ai",
                                "content": msg.content,
                            })
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    step_info["tool_calls"].append({
                                        "name": tc.get("name", "unknown"),
                                        "args": tc.get("args", {}),
                                    })
                        elif isinstance(msg, ToolMessage):
                            step_info["tool_results"].append({
                                "name": msg.name,
                                "result": msg.content[:500] if len(msg.content) > 500 else msg.content,
                            })
                            step_info["messages"].append({
                                "type": "tool",
                                "name": msg.name,
                                "content": msg.content,
                            })
                
                yield step_info
    
    async def achat(self, message: str, user_id: str = None, history: list[BaseMessage] = None) -> str:
        """
        異步對話
        
        Args:
            message: 用戶訊息
            user_id: 用戶 ID
            history: 對話歷史
        
        Returns:
            Agent 回應
        """
        state = self.prepare_state(message, user_id, history)
        result = await self.graph.ainvoke(state)
        return self.post_process(result)
    
    # ========================================
    # 圖形視覺化
    # ========================================
    
    def get_graph_image(self, output_format: str = "png") -> bytes:
        """取得工作流程圖"""
        try:
            if output_format == "ascii":
                return self.graph.get_graph().draw_ascii()
            elif output_format == "mermaid":
                return self.graph.get_graph().draw_mermaid()
            else:
                return self.graph.get_graph().draw_mermaid_png()
        except Exception as e:
            raise RuntimeError(f"無法生成圖形: {str(e)}")
    
    def get_graph_mermaid(self) -> str:
        """取得 Mermaid 格式圖形"""
        return self.graph.get_graph().draw_mermaid()


class SimpleReActAgent(BaseAgent):
    """
    簡單的 ReAct Agent
    
    預設實現，適合快速建立單一 Agent。
    只需提供 system_prompt 和 tools 即可。
    
    使用範例:
    ```python
    agent = SimpleReActAgent(
        mcp_tools=my_tools,
        system_prompt="你是一個助手...",
    )
    response = agent.chat("Hello!")
    ```
    """
    
    def __init__(
        self,
        system_prompt: str = "你是一個智能助手。",
        mcp_tools: List[BaseTool] = None,
        name: str = "react_agent",
        **kwargs,
    ):
        self._system_prompt = system_prompt
        self._name = name
        super().__init__(mcp_tools=mcp_tools, **kwargs)
    
    def define_config(self) -> AgentConfig:
        return AgentConfig(
            name=self._name,
            system_prompt=self._system_prompt,
            tools=self.mcp_tools,
        )
    
    def define_workflow(self, builder: WorkflowBuilder) -> WorkflowBuilder:
        # 建立 agent 處理函數
        agent_handler = self.create_agent_handler(
            system_prompt=self._system_prompt,
            extra_context=lambda state: f"當前用戶: {state.get('user_id', 'unknown')}"
        )
        
        # 添加節點
        builder.add_agent_node("agent", agent_handler)
        builder.add_tool_node("tools", self.tools)
        
        # 設定流程
        builder.set_entry_point("agent")
        builder.add_conditional_edge(
            "agent",
            self.create_tool_condition("tools"),
            {"tools": "tools", END: END}
        )
        builder.add_edge("tools", "agent")
        
        return builder
