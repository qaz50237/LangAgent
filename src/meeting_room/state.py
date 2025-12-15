"""
會議室預約 Agent 狀態定義
"""

from typing import Annotated, Sequence, TypedDict, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MeetingState(TypedDict):
    """
    會議室預約 Agent 的狀態定義
    
    Attributes:
        messages: 對話訊息歷史
        user_id: 當前用戶 ID
        intent: 識別出的用戶意圖 (booking/query/management)
        current_agent: 當前處理的 Agent 名稱
        context: 額外的上下文資訊（如選定的大樓、日期等）
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: Optional[str]
    intent: Optional[Literal["booking", "query", "management"]]
    current_agent: Optional[str]
    context: Optional[dict]


class BookingContext(TypedDict, total=False):
    """預約流程的上下文資訊"""
    selected_building: str
    selected_date: str
    selected_room: str
    start_time: str
    end_time: str
    meeting_title: str


class ManagementContext(TypedDict, total=False):
    """管理流程的上下文資訊"""
    reservation_id: str
    action: Literal["view", "cancel"]
