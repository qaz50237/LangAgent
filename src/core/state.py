"""
Agent 狀態定義模組
提供可擴展的狀態基礎類別
"""

from typing import Annotated, Sequence, TypedDict, Optional, Any, Type
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class BaseState(TypedDict, total=False):
    """
    Agent 狀態基礎類別
    
    所有 Agent 狀態都應繼承此類別。
    預設包含訊息列表，可通過繼承擴展額外欄位。
    
    使用範例:
    ```python
    class MyAgentState(BaseState):
        user_id: str
        context: dict
        custom_field: Any
    ```
    """
    # 核心欄位：訊息列表（必須）
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # 通用欄位（可選）
    user_id: Optional[str]
    intent: Optional[str]
    current_agent: Optional[str]
    context: Optional[dict]
    metadata: Optional[dict]


def create_state(
    extra_fields: dict[str, Any] = None,
    base_class: Type[TypedDict] = None,
) -> Type[TypedDict]:
    """
    動態建立狀態類別
    
    Args:
        extra_fields: 額外欄位定義，格式為 {欄位名: 類型}
        base_class: 基礎類別，預設為 BaseState
    
    Returns:
        新的狀態類別
    
    使用範例:
    ```python
    # 建立帶有額外欄位的狀態
    MyState = create_state({
        "booking_id": str,
        "room_name": str,
        "time_slot": dict,
    })
    ```
    """
    base = base_class or BaseState
    
    if not extra_fields:
        return base
    
    # 動態建立新的 TypedDict
    annotations = dict(base.__annotations__)
    annotations.update(extra_fields)
    
    new_state = TypedDict("DynamicState", annotations, total=False)
    return new_state
