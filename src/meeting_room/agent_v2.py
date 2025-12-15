"""
使用新框架重構的會議室預約 Agent
展示如何用 BaseAgent 簡化開發

這個範例展示：
1. 使用 BaseAgent 建立 Agent
2. 只需定義 config 和 workflow
3. 工具從 MCP 載入
"""

from typing import List
from datetime import datetime
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END

# 從核心框架匯入
import sys
sys.path.insert(0, str(__file__).rsplit("src", 1)[0])

from src.core import (
    BaseAgent,
    BaseState,
    AgentConfig,
    LLMConfig,
    WorkflowBuilder,
    create_tool_continue_condition,
    create_intent_router,
    get_llm,
    create_llm_with_tools,
)

# 匯入 prompts（開發者專注於設計這些）
from .prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    BOOKING_AGENT_SYSTEM_PROMPT,
    QUERY_AGENT_SYSTEM_PROMPT,
    MANAGEMENT_AGENT_SYSTEM_PROMPT,
)

# 工具分類器（用於將 MCP 工具分類到不同 Agent）
from .tool_classifier import classify_mcp_tools

import json


class MeetingRoomAgentV2(BaseAgent):
    """
    會議室預約多 Agent 系統（V2 - 使用新框架）
    
    開發者只需關注:
    1. Prompts 設計 (prompts.py)
    2. 工具分類邏輯 (tool_classifier.py)
    3. Workflow 設計 (define_workflow)
    
    使用範例:
    ```python
    # MCP 工具由框架外部載入
    mcp_tools = load_from_mcp("meeting-room-server")
    
    agent = MeetingRoomAgentV2(mcp_tools=mcp_tools)
    response = agent.chat("我想預約會議室")
    ```
    """
    
    def __init__(
        self,
        mcp_tools: List[BaseTool] = None,
        tool_classification_strategy: str = "explicit",
        **kwargs,
    ):
        """
        初始化會議室預約 Agent
        
        Args:
            mcp_tools: 從 MCP 載入的工具列表
            tool_classification_strategy: 工具分類策略
            **kwargs: 傳遞給 BaseAgent
        """
        self.classification_strategy = tool_classification_strategy
        
        # 分類工具
        if mcp_tools:
            classified = classify_mcp_tools(mcp_tools, strategy=tool_classification_strategy)
            self.booking_tools = classified["booking"]
            self.query_tools = classified["query"]
            self.management_tools = classified["management"]
        else:
            # 若無 MCP 工具，使用空列表（或可載入預設工具）
            self.booking_tools = []
            self.query_tools = []
            self.management_tools = []
        
        super().__init__(mcp_tools=mcp_tools, **kwargs)
    
    def define_config(self) -> AgentConfig:
        """定義 Agent 配置"""
        return AgentConfig(
            name="meeting_room_agent_v2",
            description="會議室預約多 Agent 系統",
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            tools=self.mcp_tools,
            default_user_id=self.default_user_id,
        )
    
    def define_workflow(self, builder: WorkflowBuilder) -> WorkflowBuilder:
        """
        定義多 Agent 工作流程
        
        流程圖:
        ```
        [Supervisor] --> booking_agent --> booking_tools
                    |-> query_agent --> query_tools  
                    |-> management_agent --> management_tools
                    |-> END (意圖不明)
        ```
        """
        
        # ====== 建立各 Agent 的 LLM ======
        booking_llm = create_llm_with_tools(self.booking_tools, **self.llm_config.to_dict())
        query_llm = create_llm_with_tools(self.query_tools, **self.llm_config.to_dict())
        management_llm = create_llm_with_tools(self.management_tools, **self.llm_config.to_dict())
        supervisor_llm = get_llm(**self.llm_config.to_dict())
        
        # ====== 定義節點處理函數 ======
        
        def supervisor_node(state: dict) -> dict:
            """Supervisor: 意圖分析與路由"""
            messages = state["messages"]
            
            analysis_messages = [
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                *messages
            ]
            
            response = supervisor_llm.invoke(analysis_messages)
            
            try:
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                decision = json.loads(content.strip())
                intent = decision.get("intent", "unclear")
                
                if intent == "unclear":
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
                return {"intent": "query", "current_agent": "query_agent"}
        
        def create_agent_node(llm, system_prompt: str, agent_name: str):
            """工廠函數：建立 Agent 節點"""
            def node(state: dict) -> dict:
                messages = state["messages"]
                user_id = state.get("user_id", self.default_user_id)
                
                full_prompt = system_prompt + f"\n\n當前用戶ID: {user_id}\n今天日期: {datetime.now().strftime('%Y-%m-%d')}"
                
                agent_messages = [SystemMessage(content=full_prompt), *messages]
                response = llm.invoke(agent_messages)
                return {"messages": [response], "current_agent": agent_name}
            return node
        
        # ====== 定義條件函數 ======
        
        def route_by_intent(state: dict) -> str:
            intent = state.get("intent")
            mapping = {
                "booking": "booking_agent",
                "query": "query_agent", 
                "management": "management_agent",
            }
            return mapping.get(intent, END)
        
        def create_tool_condition(tools_node: str):
            def condition(state: dict) -> str:
                messages = state.get("messages", [])
                if messages:
                    last = messages[-1]
                    if hasattr(last, "tool_calls") and last.tool_calls:
                        return tools_node
                return END
            return condition
        
        # ====== 建構工作流 ======
        
        # 添加節點
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("booking_agent", create_agent_node(booking_llm, BOOKING_AGENT_SYSTEM_PROMPT, "booking_agent"))
        builder.add_node("query_agent", create_agent_node(query_llm, QUERY_AGENT_SYSTEM_PROMPT, "query_agent"))
        builder.add_node("management_agent", create_agent_node(management_llm, MANAGEMENT_AGENT_SYSTEM_PROMPT, "management_agent"))
        
        # 添加工具節點
        builder.add_tool_node("booking_tools", self.booking_tools)
        builder.add_tool_node("query_tools", self.query_tools)
        builder.add_tool_node("management_tools", self.management_tools)
        
        # 設定入口點
        builder.set_entry_point("supervisor")
        
        # Supervisor 路由
        builder.add_conditional_edge(
            "supervisor",
            route_by_intent,
            {
                "booking_agent": "booking_agent",
                "query_agent": "query_agent",
                "management_agent": "management_agent",
                END: END,
            }
        )
        
        # 各 Agent 的工具循環
        for agent, tools in [
            ("booking_agent", "booking_tools"),
            ("query_agent", "query_tools"),
            ("management_agent", "management_tools"),
        ]:
            builder.add_conditional_edge(
                agent,
                create_tool_condition(tools),
                {tools: tools, END: END}
            )
            builder.add_edge(tools, agent)
        
        return builder


class SimpleBookingAgentV2(BaseAgent):
    """
    簡化版會議室預約 Agent（V2）
    
    單一 Agent 處理所有任務，展示最簡單的實現方式。
    
    使用範例:
    ```python
    agent = SimpleBookingAgentV2(
        mcp_tools=my_tools,
        system_prompt="自定義 prompt...",  # 可選
    )
    ```
    """
    
    def __init__(
        self,
        mcp_tools: List[BaseTool] = None,
        system_prompt: str = None,
        **kwargs,
    ):
        self._custom_prompt = system_prompt
        super().__init__(mcp_tools=mcp_tools, **kwargs)
    
    def define_config(self) -> AgentConfig:
        default_prompt = """你是會議室預約助手，可以幫助用戶：
1. 查詢可用的大樓和會議室
2. 預約會議室
3. 查看和管理預約

今天日期：{current_date}
請使用繁體中文回應。"""

        return AgentConfig(
            name="simple_booking_agent",
            description="簡化版會議室預約 Agent",
            system_prompt_template=self._custom_prompt or default_prompt,
            prompt_variables={"current_date": datetime.now().strftime('%Y-%m-%d')},
            tools=self.mcp_tools,
        )
    
    def define_workflow(self, builder: WorkflowBuilder) -> WorkflowBuilder:
        """定義簡單的 ReAct 工作流"""
        
        # 使用輔助方法建立處理函數
        agent_handler = self.create_agent_handler(
            system_prompt=self.config.get_system_prompt(),
            extra_context=lambda state: f"當前用戶: {state.get('user_id', 'unknown')}"
        )
        
        # 建構流程
        builder.add_agent_node("agent", agent_handler)
        builder.add_tool_node("tools", self.tools)
        
        builder.set_entry_point("agent")
        builder.add_conditional_edge(
            "agent",
            self.create_tool_condition("tools"),
            {"tools": "tools", END: END}
        )
        builder.add_edge("tools", "agent")
        
        return builder


# ============================================================
# 工廠函數
# ============================================================

def create_meeting_room_agent_v2(
    mode: str = "multi",
    mcp_tools: List[BaseTool] = None,
    **kwargs,
):
    """
    建立會議室預約 Agent（使用新框架）
    
    Args:
        mode: "multi" = 多 Agent, "simple" = 單一 Agent
        mcp_tools: 從 MCP 載入的工具
        **kwargs: 其他參數
    
    Returns:
        Agent 實例
    """
    if mode == "multi":
        return MeetingRoomAgentV2(mcp_tools=mcp_tools, **kwargs)
    else:
        return SimpleBookingAgentV2(mcp_tools=mcp_tools, **kwargs)
