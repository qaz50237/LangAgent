"""
MCP 工具分類器
當工具從 MCP Server 動態載入時，根據命名規範或配置進行分類
"""

from typing import Dict, List, Any, Callable
from langchain_core.tools import BaseTool, StructuredTool
import re


# ============================================================
# 方法 1: 基於工具名稱前綴的分類配置
# ============================================================
TOOL_CATEGORY_BY_PREFIX = {
    "booking": ["get_available_buildings", "get_available_rooms", "book_meeting_room"],
    "query": ["get_available_buildings", "get_available_rooms"],
    "management": ["get_user_reservations", "cancel_reservation"],
}

# 也可以用前綴模式
TOOL_PREFIX_PATTERNS = {
    "booking": ["book_", "reserve_", "get_available_"],
    "query": ["get_", "list_", "search_", "query_"],
    "management": ["cancel_", "update_", "delete_", "get_user_", "my_"],
}


# ============================================================
# 方法 2: 基於工具描述關鍵字的分類
# ============================================================
TOOL_CATEGORY_BY_KEYWORDS = {
    "booking": ["預約", "訂", "book", "reserve", "安排"],
    "query": ["查詢", "搜尋", "列出", "query", "search", "list", "get"],
    "management": ["取消", "管理", "我的", "cancel", "delete", "update", "my"],
}


# ============================================================
# 方法 3: 顯式工具映射配置 (推薦用於 MCP)
# ============================================================
TOOL_CLASSIFICATION_CONFIG = {
    # 工具名稱 -> 所屬分類列表
    "get_available_buildings": ["booking", "query"],
    "get_available_rooms": ["booking", "query"],
    "book_meeting_room": ["booking"],
    "get_user_reservations": ["management"],
    "cancel_reservation": ["management"],
    
    # MCP 工具範例（假設 MCP Server 提供這些工具名稱）
    # "mcp_meeting_list_buildings": ["booking", "query"],
    # "mcp_meeting_search_rooms": ["booking", "query"],
    # "mcp_meeting_create_reservation": ["booking"],
    # "mcp_meeting_get_my_reservations": ["management"],
    # "mcp_meeting_cancel": ["management"],
}


class MCPToolClassifier:
    """
    MCP 工具分類器
    
    支援三種分類策略：
    1. explicit: 顯式配置（精確控制）
    2. prefix: 根據工具名稱前綴
    3. keyword: 根據工具描述關鍵字
    """
    
    def __init__(
        self,
        strategy: str = "explicit",
        custom_config: Dict[str, List[str]] = None,
    ):
        """
        初始化分類器
        
        Args:
            strategy: 分類策略 ("explicit", "prefix", "keyword")
            custom_config: 自訂分類配置
        """
        self.strategy = strategy
        self.config = custom_config or TOOL_CLASSIFICATION_CONFIG
    
    def classify_tools(
        self, 
        tools: List[BaseTool]
    ) -> Dict[str, List[BaseTool]]:
        """
        將工具列表分類
        
        Args:
            tools: MCP 載入的完整工具列表
            
        Returns:
            分類後的工具字典 {"category_name": [tools]}
        """
        classified = {
            "booking": [],
            "query": [],
            "management": [],
            "uncategorized": [],  # 未分類的工具
        }
        
        for tool in tools:
            categories = self._get_tool_categories(tool)
            
            if not categories:
                classified["uncategorized"].append(tool)
            else:
                for category in categories:
                    if category in classified:
                        classified[category].append(tool)
        
        return classified
    
    def _get_tool_categories(self, tool: BaseTool) -> List[str]:
        """根據策略判斷工具所屬分類"""
        
        if self.strategy == "explicit":
            return self._classify_by_explicit(tool)
        elif self.strategy == "prefix":
            return self._classify_by_prefix(tool)
        elif self.strategy == "keyword":
            return self._classify_by_keyword(tool)
        else:
            return self._classify_by_explicit(tool)
    
    def _classify_by_explicit(self, tool: BaseTool) -> List[str]:
        """顯式配置分類"""
        tool_name = tool.name
        return self.config.get(tool_name, [])
    
    def _classify_by_prefix(self, tool: BaseTool) -> List[str]:
        """根據名稱前綴分類"""
        tool_name = tool.name.lower()
        categories = []
        
        for category, prefixes in TOOL_PREFIX_PATTERNS.items():
            for prefix in prefixes:
                if tool_name.startswith(prefix):
                    categories.append(category)
                    break
        
        return list(set(categories))
    
    def _classify_by_keyword(self, tool: BaseTool) -> List[str]:
        """根據描述關鍵字分類"""
        description = (tool.description or "").lower()
        tool_name = tool.name.lower()
        combined_text = f"{tool_name} {description}"
        
        categories = []
        for category, keywords in TOOL_CATEGORY_BY_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    categories.append(category)
                    break
        
        return list(set(categories))
    
    def get_tools_by_category(
        self, 
        tools: List[BaseTool], 
        category: str
    ) -> List[BaseTool]:
        """取得特定分類的工具"""
        classified = self.classify_tools(tools)
        return classified.get(category, [])


# ============================================================
# 便捷函數
# ============================================================

def classify_mcp_tools(
    tools: List[BaseTool],
    strategy: str = "explicit",
) -> Dict[str, List[BaseTool]]:
    """
    快速分類 MCP 工具
    
    Args:
        tools: MCP 載入的工具列表
        strategy: 分類策略
        
    Returns:
        {"booking": [...], "query": [...], "management": [...]}
    """
    classifier = MCPToolClassifier(strategy=strategy)
    return classifier.classify_tools(tools)


def get_booking_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """取得預約相關工具"""
    return classify_mcp_tools(tools)["booking"]


def get_query_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """取得查詢相關工具"""
    return classify_mcp_tools(tools)["query"]


def get_management_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """取得管理相關工具"""
    return classify_mcp_tools(tools)["management"]


# ============================================================
# 範例：如何與 MCP Client 整合
# ============================================================
"""
使用範例：

```python
from langchain_mcp import MCPToolkit
from src.meeting_room.tool_classifier import classify_mcp_tools, MCPToolClassifier

# 1. 從 MCP Server 載入所有工具
async with MCPToolkit("mcp://localhost:9000") as toolkit:
    all_tools = toolkit.get_tools()  # 取得所有工具
    
    # 2. 分類工具
    classified = classify_mcp_tools(all_tools, strategy="explicit")
    
    booking_tools = classified["booking"]
    query_tools = classified["query"]
    management_tools = classified["management"]
    
    # 3. 在 Agent 中使用分類後的工具
    booking_agent = create_react_agent(llm, booking_tools)
    query_agent = create_react_agent(llm, query_tools)
    management_agent = create_react_agent(llm, management_tools)
```

或使用自訂配置：

```python
# 自訂 MCP 工具分類配置
my_config = {
    "mcp_room_list_buildings": ["booking", "query"],
    "mcp_room_search": ["booking", "query"],
    "mcp_room_book": ["booking"],
    "mcp_room_my_reservations": ["management"],
    "mcp_room_cancel": ["management"],
}

classifier = MCPToolClassifier(
    strategy="explicit",
    custom_config=my_config
)

classified = classifier.classify_tools(mcp_tools)
```
"""
