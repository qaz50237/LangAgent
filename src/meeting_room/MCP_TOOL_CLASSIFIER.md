# MCP 工具分類器使用指南

當工具從 MCP (Model Context Protocol) Server 動態載入時，會一次取得所有工具。本分類器可根據不同策略將工具分配給專業 Agent。

## 📋 目錄

- [問題背景](#問題背景)
- [解決方案](#解決方案)
- [三種分類策略](#三種分類策略)
- [使用方式](#使用方式)
- [配置指南](#配置指南)
- [最佳實踐](#最佳實踐)

---

## 問題背景

### 靜態工具 vs MCP 動態工具

```
┌─────────────────────────────────────────────────────────────┐
│  靜態工具定義 (tools.py)                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BOOKING_TOOLS = [tool1, tool2, tool3]     ← 手動分類       │
│  QUERY_TOOLS = [tool1, tool2]                               │
│  MANAGEMENT_TOOLS = [tool4, tool5]                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  MCP 動態載入                                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  mcp_tools = mcp_client.get_tools()                         │
│       ↓                                                     │
│  [tool1, tool2, tool3, tool4, tool5]  ← 全部混在一起！       │
│       ↓                                                     │
│  ❓ 如何分配給不同的 Agent？                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 為什麼需要分類？

在 Multi-Agent 架構中，不同 Agent 負責不同任務：

| Agent | 職責 | 需要的工具 |
|-------|------|-----------|
| Booking Agent | 預約流程 | 查詢大樓、查詢會議室、預約 |
| Query Agent | 純查詢 | 查詢大樓、查詢會議室 |
| Management Agent | 管理預約 | 查詢已預約、取消 |

**如果把所有工具都給每個 Agent：**
- ❌ Agent 可能呼叫不相關的工具
- ❌ 增加 LLM token 消耗
- ❌ 降低回應精確度

---

## 解決方案

使用 `MCPToolClassifier` 自動將 MCP 工具分類：

```
MCP Server                    Tool Classifier                  Agents
    │                              │                              │
    │  [all_tools]                 │                              │
    │ ─────────────────────────────>                              │
    │                              │                              │
    │                         分類處理                            │
    │                              │                              │
    │                              │   booking_tools ───────────> Booking Agent
    │                              │   query_tools ─────────────> Query Agent
    │                              │   management_tools ────────> Management Agent
    │                              │                              │
```

---

## 三種分類策略

### 1️⃣ Explicit (顯式配置) - 推薦

**原理**: 在配置檔中明確指定每個工具的分類

```python
TOOL_CLASSIFICATION_CONFIG = {
    "get_available_buildings": ["booking", "query"],  # 同時屬於兩個分類
    "get_available_rooms": ["booking", "query"],
    "book_meeting_room": ["booking"],
    "get_user_reservations": ["management"],
    "cancel_reservation": ["management"],
}
```

**優點**: 
- ✅ 精確控制
- ✅ 一個工具可屬於多個分類
- ✅ 不依賴命名規範

**缺點**: 
- ⚠️ 新增工具需手動更新配置

---

### 2️⃣ Prefix (前綴匹配)

**原理**: 根據工具名稱前綴自動分類

```python
TOOL_PREFIX_PATTERNS = {
    "booking": ["book_", "reserve_", "get_available_"],
    "query": ["get_", "list_", "search_", "query_"],
    "management": ["cancel_", "update_", "delete_", "get_user_", "my_"],
}
```

**範例**:
| 工具名稱 | 匹配前綴 | 分類結果 |
|---------|---------|---------|
| `book_meeting_room` | `book_` | booking |
| `get_available_rooms` | `get_available_` | booking |
| `get_user_reservations` | `get_user_` | management |
| `cancel_reservation` | `cancel_` | management |

**優點**: 
- ✅ 自動分類
- ✅ 新工具自動歸類

**缺點**: 
- ⚠️ 需要統一的命名規範
- ⚠️ 前綴可能重疊 (`get_` 同時匹配 query 和 booking)

---

### 3️⃣ Keyword (關鍵字匹配)

**原理**: 根據工具描述中的關鍵字分類

```python
TOOL_CATEGORY_BY_KEYWORDS = {
    "booking": ["預約", "訂", "book", "reserve", "安排"],
    "query": ["查詢", "搜尋", "列出", "query", "search", "list"],
    "management": ["取消", "管理", "我的", "cancel", "delete", "my"],
}
```

**範例**:
```python
@tool
def some_tool():
    """查詢並列出所有可預約的會議室"""  # 包含「查詢」「列出」「預約」
    pass

# 分類結果: ["booking", "query"] (同時匹配多個關鍵字)
```

**優點**: 
- ✅ 不依賴命名規範
- ✅ 根據實際功能分類

**缺點**: 
- ⚠️ 依賴描述品質
- ⚠️ 可能匹配不準確

---

## 使用方式

### 基本用法

```python
from src.meeting_room import MeetingRoomAgent, classify_mcp_tools

# 假設從 MCP 取得工具
mcp_tools = await mcp_client.get_tools()

# 方式 1: 直接傳入 Agent (自動分類)
agent = MeetingRoomAgent(
    mcp_tools=mcp_tools,
    tool_classification_strategy="explicit"
)

# 方式 2: 手動分類後使用
classified = classify_mcp_tools(mcp_tools, strategy="prefix")
booking_tools = classified["booking"]
query_tools = classified["query"]
management_tools = classified["management"]
```

### 使用自訂配置

```python
from src.meeting_room.tool_classifier import MCPToolClassifier

# 自訂 MCP 工具映射
my_config = {
    "mcp_room_list": ["booking", "query"],
    "mcp_room_search": ["booking", "query"],
    "mcp_room_book": ["booking"],
    "mcp_room_my_list": ["management"],
    "mcp_room_cancel": ["management"],
}

classifier = MCPToolClassifier(
    strategy="explicit",
    custom_config=my_config
)

classified = classifier.classify_tools(mcp_tools)
```

### 完整 MCP 整合範例

```python
from langchain_mcp import MCPToolkit
from src.meeting_room import MeetingRoomAgent

async def create_agent_with_mcp():
    """從 MCP Server 載入工具並建立 Agent"""
    
    async with MCPToolkit("mcp://localhost:9000/meeting-room") as toolkit:
        # 取得所有 MCP 工具
        mcp_tools = toolkit.get_tools()
        
        # 建立 Agent (自動分類工具)
        agent = MeetingRoomAgent(
            model_name="gpt-4o-mini",
            user_id="user001",
            mcp_tools=mcp_tools,
            tool_classification_strategy="explicit"
        )
        
        # 對話
        response = agent.chat("我想預約會議室")
        print(response)

# 執行
import asyncio
asyncio.run(create_agent_with_mcp())
```

---

## 配置指南

### 新增 MCP 工具到分類

編輯 `src/meeting_room/tool_classifier.py`:

```python
TOOL_CLASSIFICATION_CONFIG = {
    # 現有工具
    "get_available_buildings": ["booking", "query"],
    "get_available_rooms": ["booking", "query"],
    "book_meeting_room": ["booking"],
    "get_user_reservations": ["management"],
    "cancel_reservation": ["management"],
    
    # 新增 MCP 工具 ↓
    "mcp_meeting_list_buildings": ["booking", "query"],
    "mcp_meeting_search_rooms": ["booking", "query"],
    "mcp_meeting_create": ["booking"],
    "mcp_meeting_my_reservations": ["management"],
    "mcp_meeting_cancel": ["management"],
    "mcp_meeting_update": ["management"],  # 新功能：更新預約
}
```

### 新增分類類別

如果需要新增分類（如 `notification`）：

```python
# 1. 更新 classify_tools 方法
def classify_tools(self, tools):
    classified = {
        "booking": [],
        "query": [],
        "management": [],
        "notification": [],  # 新增
        "uncategorized": [],
    }
    # ...

# 2. 更新配置
TOOL_CLASSIFICATION_CONFIG = {
    # ...
    "send_reminder": ["notification"],
    "notify_participants": ["notification"],
}

# 3. 更新 Agent 使用新分類
# 在 agent.py 中處理 notification_tools
```

---

## 最佳實踐

### ✅ 推薦做法

1. **使用 explicit 策略 + 命名規範**
   ```python
   # MCP Server 端命名規範
   meeting_list_buildings    # → booking, query
   meeting_book_room         # → booking
   meeting_cancel            # → management
   ```

2. **一個工具可屬於多個分類**
   ```python
   "get_available_buildings": ["booking", "query"],
   ```

3. **處理未分類工具**
   ```python
   classified = classify_mcp_tools(mcp_tools)
   if classified["uncategorized"]:
       print(f"警告: 有未分類的工具: {classified['uncategorized']}")
   ```

4. **為新 MCP 工具預留配置**
   ```python
   # 使用前綴模式作為備援
   classifier = MCPToolClassifier(strategy="prefix")
   ```

### ❌ 避免做法

1. **不要把所有工具給每個 Agent**
   ```python
   # ❌ 錯誤
   booking_agent = Agent(tools=ALL_MCP_TOOLS)
   query_agent = Agent(tools=ALL_MCP_TOOLS)
   
   # ✅ 正確
   booking_agent = Agent(tools=classified["booking"])
   query_agent = Agent(tools=classified["query"])
   ```

2. **不要忽略 uncategorized 工具**
   ```python
   # 應該記錄或處理未分類的工具
   ```

---

## 檔案結構

```
src/meeting_room/
├── __init__.py           # 導出 MCPToolClassifier
├── tool_classifier.py    # 🆕 MCP 工具分類器
├── tools.py              # 靜態工具定義 (非 MCP 使用)
├── agent.py              # 支援 mcp_tools 參數
├── state.py
└── prompts.py
```

---

## API 參考

### MCPToolClassifier

```python
class MCPToolClassifier:
    def __init__(
        self,
        strategy: str = "explicit",      # "explicit" | "prefix" | "keyword"
        custom_config: Dict = None,       # 自訂分類配置
    )
    
    def classify_tools(
        self, 
        tools: List[BaseTool]
    ) -> Dict[str, List[BaseTool]]:
        """
        Returns:
            {
                "booking": [tool1, tool2],
                "query": [tool1, tool2],
                "management": [tool3, tool4],
                "uncategorized": [tool5],
            }
        """
    
    def get_tools_by_category(
        self,
        tools: List[BaseTool],
        category: str
    ) -> List[BaseTool]:
        """取得單一分類的工具"""
```

### 便捷函數

```python
from src.meeting_room.tool_classifier import (
    classify_mcp_tools,      # 快速分類
    get_booking_tools,       # 取得 booking 工具
    get_query_tools,         # 取得 query 工具
    get_management_tools,    # 取得 management 工具
)
```
