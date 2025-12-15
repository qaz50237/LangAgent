# LangAgent Core Framework

Agent 開發框架，讓開發者專注於核心設計，而非重複實作基礎架構。

## 設計理念

開發者只需關注：
1. **Agent 設計** - 定義 Agent 的角色和行為
2. **Workflow 設計** - 定義節點和流程
3. **Prompt 設計** - 撰寫系統提示詞
4. **Tools** - 從 MCP 取得，無需手動定義

框架處理：
- LLM 初始化和配置
- 狀態管理
- 對話方法 (chat, stream, async)
- 圖形視覺化
- 工具綁定

## 快速開始

### 1. 最簡單的方式 - SimpleReActAgent

```python
from src.core import SimpleReActAgent

# 從 MCP 載入工具
mcp_tools = load_tools_from_mcp("my-server")

# 建立 Agent - 只需提供 prompt 和 tools
agent = SimpleReActAgent(
    system_prompt="你是一個智能助手，專門處理...",
    mcp_tools=mcp_tools,
)

# 使用
response = agent.chat("Hello!")
```

### 2. 自定義 Agent - 繼承 BaseAgent

```python
from src.core import BaseAgent, AgentConfig, WorkflowBuilder
from langgraph.graph import END

class MyCustomAgent(BaseAgent):
    """自定義 Agent"""
    
    def define_config(self) -> AgentConfig:
        """定義配置 - 專注於 Prompt 設計"""
        return AgentConfig(
            name="my_agent",
            description="我的自定義 Agent",
            system_prompt="""你是一個專業的助手...
            
            ## 你的能力
            - 能力 1
            - 能力 2
            
            ## 注意事項
            ...
            """,
            tools=self.mcp_tools,
        )
    
    def define_workflow(self, builder: WorkflowBuilder) -> WorkflowBuilder:
        """定義工作流 - 專注於流程設計"""
        
        # 使用輔助方法建立處理函數
        handler = self.create_agent_handler(
            system_prompt=self.config.system_prompt,
        )
        
        # 建構流程
        builder.add_agent_node("agent", handler)
        builder.add_tool_node("tools", self.tools)
        
        builder.set_entry_point("agent")
        builder.add_conditional_edge(
            "agent",
            self.create_tool_condition("tools"),
            {"tools": "tools", END: END}
        )
        builder.add_edge("tools", "agent")
        
        return builder

# 使用
agent = MyCustomAgent(mcp_tools=my_tools)
response = agent.chat("請幫我...")
```

### 3. 多 Agent 系統

```python
class MultiAgentSystem(BaseAgent):
    """Supervisor 模式的多 Agent 系統"""
    
    def define_config(self) -> AgentConfig:
        return AgentConfig(
            name="multi_agent_system",
            system_prompt=SUPERVISOR_PROMPT,
        )
    
    def define_workflow(self, builder: WorkflowBuilder) -> WorkflowBuilder:
        # 建立多個 LLM
        agent_a_llm = create_llm_with_tools(self.tools_a)
        agent_b_llm = create_llm_with_tools(self.tools_b)
        
        # 定義節點
        builder.add_node("supervisor", self.supervisor_handler)
        builder.add_node("agent_a", self.create_agent_a_handler(agent_a_llm))
        builder.add_node("agent_b", self.create_agent_b_handler(agent_b_llm))
        builder.add_tool_node("tools_a", self.tools_a)
        builder.add_tool_node("tools_b", self.tools_b)
        
        # 設定路由
        builder.set_entry_point("supervisor")
        builder.add_conditional_edge(
            "supervisor",
            self.route_by_intent,
            {"agent_a": "agent_a", "agent_b": "agent_b", END: END}
        )
        
        # Agent A 的工具循環
        builder.add_conditional_edge("agent_a", ..., {"tools_a": "tools_a", END: END})
        builder.add_edge("tools_a", "agent_a")
        
        # Agent B 的工具循環
        builder.add_conditional_edge("agent_b", ..., {"tools_b": "tools_b", END: END})
        builder.add_edge("tools_b", "agent_b")
        
        return builder
```

## 核心組件

### BaseState - 狀態基礎類別

```python
from src.core import BaseState, create_state

# 使用預設狀態
# BaseState 包含: messages, user_id, intent, current_agent, context, metadata

# 擴展狀態
class MyState(BaseState):
    booking_id: str
    room_name: str

# 或動態建立
MyState = create_state({
    "booking_id": str,
    "room_name": str,
})
```

### AgentConfig - 配置類別

```python
from src.core import AgentConfig, LLMConfig

config = AgentConfig(
    name="my_agent",
    description="描述",
    
    # Prompt 設計
    system_prompt="靜態 prompt...",
    # 或使用模板
    system_prompt_template="今天是 {date}，用戶是 {user}",
    prompt_variables={"date": "2025-12-15"},
    
    # 工具（從 MCP 載入）
    tools=mcp_tools,
    
    # LLM 配置
    llm_config=LLMConfig(
        model_name="gpt-4o",
        temperature=0.7,
    ),
)

# 動態取得 prompt
prompt = config.get_system_prompt(user="Alice")
```

### WorkflowBuilder - 工作流建構器

```python
from src.core import WorkflowBuilder
from langgraph.graph import END

builder = WorkflowBuilder(MyState)

# 添加節點
builder.add_agent_node("agent", handler_function)
builder.add_tool_node("tools", tool_list)

# 設定流程
builder.set_entry_point("agent")

# 簡單邊
builder.add_edge("tools", "agent")

# 條件邊
builder.add_conditional_edge(
    source="agent",
    condition=my_condition_func,  # 返回 str
    mapping={"continue": "tools", "end": END}
)

# 編譯
graph = builder.compile()
```

## 內建輔助方法

BaseAgent 提供多個輔助方法：

```python
class MyAgent(BaseAgent):
    def define_workflow(self, builder):
        # 快速建立 Agent 處理函數
        handler = self.create_agent_handler(
            system_prompt="...",
            extra_context=lambda state: f"用戶: {state['user_id']}"
        )
        
        # 快速建立工具條件
        condition = self.create_tool_condition("tools_node")
        
        ...
```

## 對話方法

所有繼承 BaseAgent 的 Agent 自動擁有：

```python
# 基本對話
response = agent.chat("Hello")

# 帶歷史對話
response, history = agent.chat_with_history("Hello", history=prev_history)

# 串流對話
for step in agent.chat_stream("Hello"):
    print(f"節點: {step['node']}")
    print(f"輸出: {step['output']}")

# 異步對話
response = await agent.achat("Hello")
```

## 圖形視覺化

```python
# PNG 圖片
png_bytes = agent.get_graph_image("png")

# Mermaid 格式
mermaid_str = agent.get_graph_mermaid()

# ASCII
ascii_str = agent.get_graph_image("ascii")
```

## 完整範例

參見 [src/meeting_room/agent_v2.py](../meeting_room/agent_v2.py)

## 檔案結構

```
src/core/
├── __init__.py      # 匯出所有組件
├── state.py         # BaseState, create_state
├── config.py        # AgentConfig, LLMConfig
├── llm.py           # get_llm, create_llm_with_tools
├── workflow.py      # WorkflowBuilder, 輔助函數
└── agent.py         # BaseAgent, SimpleReActAgent
```
