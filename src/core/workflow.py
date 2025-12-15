"""
工作流建構器模組
簡化 LangGraph 工作流程圖的建構
"""

from typing import (
    Annotated, Sequence, TypedDict, Literal, Optional, 
    List, Dict, Any, Callable, Union, Type
)
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .state import BaseState


@dataclass
class NodeDefinition:
    """
    節點定義
    
    定義 LangGraph 工作流中的一個節點。
    
    使用範例:
    ```python
    node = NodeDefinition(
        name="agent",
        handler=my_agent_handler,  # Callable[[State], dict]
        tools=my_tools,  # 可選，如果是工具節點
    )
    ```
    """
    name: str
    handler: Callable[[Any], dict] = None  # 節點處理函數
    tools: List[BaseTool] = None  # 如果是工具節點
    is_tool_node: bool = False  # 是否為工具節點
    
    def __post_init__(self):
        if self.tools and not self.handler:
            self.is_tool_node = True


@dataclass
class EdgeDefinition:
    """
    邊定義
    
    定義節點間的連接關係。
    
    使用範例:
    ```python
    # 簡單邊
    edge = EdgeDefinition(source="agent", target="tools")
    
    # 條件邊
    edge = EdgeDefinition(
        source="agent",
        condition=should_continue,
        mapping={"continue": "tools", "end": END}
    )
    ```
    """
    source: str
    target: str = None  # 簡單邊的目標
    condition: Callable[[Any], str] = None  # 條件函數
    mapping: Dict[str, str] = None  # 條件映射


class WorkflowBuilder:
    """
    工作流建構器
    
    簡化 LangGraph 工作流程圖的建構過程。
    
    使用範例:
    ```python
    builder = WorkflowBuilder(MyState)
    
    # 添加節點
    builder.add_agent_node("agent", agent_handler, tools=my_tools)
    builder.add_tool_node("tools", my_tools)
    
    # 設定流程
    builder.set_entry_point("agent")
    builder.add_conditional_edge(
        "agent",
        should_continue,
        {"continue": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")
    
    # 編譯
    graph = builder.compile()
    ```
    """
    
    def __init__(self, state_class: Type[TypedDict] = None):
        """
        初始化建構器
        
        Args:
            state_class: 狀態類別，預設使用 BaseState
        """
        self.state_class = state_class or BaseState
        self.workflow = StateGraph(self.state_class)
        self.nodes: Dict[str, NodeDefinition] = {}
        self.edges: List[EdgeDefinition] = []
        self.entry_point: str = None
    
    def add_node(
        self,
        name: str,
        handler: Callable[[Any], dict] = None,
        tools: List[BaseTool] = None,
    ) -> "WorkflowBuilder":
        """
        添加節點
        
        Args:
            name: 節點名稱
            handler: 節點處理函數
            tools: 工具列表（如果是工具節點）
        
        Returns:
            self（支援鏈式調用）
        """
        node_def = NodeDefinition(name=name, handler=handler, tools=tools)
        self.nodes[name] = node_def
        
        if node_def.is_tool_node:
            self.workflow.add_node(name, ToolNode(tools))
        else:
            self.workflow.add_node(name, handler)
        
        return self
    
    def add_agent_node(
        self,
        name: str,
        handler: Callable[[Any], dict],
        tools: List[BaseTool] = None,
    ) -> "WorkflowBuilder":
        """
        添加 Agent 節點
        
        Args:
            name: 節點名稱
            handler: Agent 處理函數
            tools: 關聯的工具（用於記錄）
        
        Returns:
            self
        """
        return self.add_node(name, handler=handler, tools=tools)
    
    def add_tool_node(
        self,
        name: str,
        tools: List[BaseTool],
    ) -> "WorkflowBuilder":
        """
        添加工具節點
        
        Args:
            name: 節點名稱
            tools: 工具列表
        
        Returns:
            self
        """
        return self.add_node(name, tools=tools)
    
    def set_entry_point(self, node_name: str) -> "WorkflowBuilder":
        """
        設定入口點
        
        Args:
            node_name: 入口節點名稱
        
        Returns:
            self
        """
        self.entry_point = node_name
        self.workflow.set_entry_point(node_name)
        return self
    
    def add_edge(self, source: str, target: str) -> "WorkflowBuilder":
        """
        添加簡單邊
        
        Args:
            source: 來源節點
            target: 目標節點（可以是 END）
        
        Returns:
            self
        """
        edge_def = EdgeDefinition(source=source, target=target)
        self.edges.append(edge_def)
        self.workflow.add_edge(source, target)
        return self
    
    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[Any], str],
        mapping: Dict[str, str],
    ) -> "WorkflowBuilder":
        """
        添加條件邊
        
        Args:
            source: 來源節點
            condition: 條件函數，返回映射的 key
            mapping: 條件到目標節點的映射
        
        Returns:
            self
        """
        edge_def = EdgeDefinition(
            source=source,
            condition=condition,
            mapping=mapping,
        )
        self.edges.append(edge_def)
        self.workflow.add_conditional_edges(source, condition, mapping)
        return self
    
    def compile(self):
        """
        編譯工作流
        
        Returns:
            編譯後的 LangGraph
        """
        return self.workflow.compile()


def create_tool_continue_condition(
    tools_node_name: str,
    end_node: str = END,
) -> Callable[[Any], str]:
    """
    建立工具繼續條件函數
    
    常用的條件函數：檢查是否需要執行工具
    
    Args:
        tools_node_name: 工具節點名稱
        end_node: 結束節點
    
    Returns:
        條件函數
    """
    def should_continue(state: Any) -> str:
        messages = state.get("messages", [])
        if not messages:
            return end_node
        
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return tools_node_name
        return end_node
    
    return should_continue


def create_intent_router(
    intent_mapping: Dict[str, str],
    default_node: str = END,
) -> Callable[[Any], str]:
    """
    建立意圖路由函數
    
    根據 state 中的 intent 欄位路由
    
    Args:
        intent_mapping: 意圖到節點的映射
        default_node: 預設節點
    
    Returns:
        路由函數
    """
    def route_by_intent(state: Any) -> str:
        intent = state.get("intent")
        return intent_mapping.get(intent, default_node)
    
    return route_by_intent
