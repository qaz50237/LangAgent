"""
Streamlit UI
提供互動式介面來測試 Agent，支援多種 Agent 選擇
"""

import streamlit as st
from typing import Optional
import uuid

# Agent 模組
from src.agent import LangGraphAgent, SYSTEM_PROMPT
from src.meeting_room import create_meeting_room_agent


# ============================================================
# Agent 配置
# ============================================================
AGENT_OPTIONS = {
    "general": {
        "name": "🤖 通用 AI 助理",
        "description": "多功能助理，支援時間查詢、計算、天氣、單位轉換等",
        "examples": [
            "現在幾點？",
            "計算 sqrt(144) + 25 * 2",
            "台北今天天氣如何？",
            "把 100 公里轉換成英里",
            "什麼是 LangChain？",
        ],
    },
    "meeting_room": {
        "name": "🏢 會議室預約 Agent",
        "description": "專門處理會議室查詢、預約、管理的智能助理（Supervisor 多 Agent 架構）",
        "examples": [
            "有哪些大樓可以預約？",
            "查詢 A 棟明天的會議室",
            "我要預約 A-101，明天 9:00-10:00，週會",
            "查詢我的預約",
            "取消預約 RES-001",
        ],
    },
}


def init_session_state():
    """初始化 Streamlit session state"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_agent_type" not in st.session_state:
        st.session_state.current_agent_type = "general"
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user001"


def get_or_create_agent(agent_type: str, user_id: str = "user001"):
    """取得或建立 Agent 實例"""
    # 如果 Agent 類型改變，重新建立
    if (st.session_state.agent is None or 
        st.session_state.current_agent_type != agent_type):
        
        try:
            if agent_type == "general":
                st.session_state.agent = LangGraphAgent()
            elif agent_type == "meeting_room":
                st.session_state.agent = create_meeting_room_agent(
                    mode="multi", 
                    user_id=user_id
                )
            st.session_state.current_agent_type = agent_type
        except Exception as e:
            st.error(f"建立 Agent 失敗: {str(e)}")
            return None
    
    return st.session_state.agent


def send_message_direct(message: str, agent_type: str, user_id: str = "user001") -> Optional[str]:
    """直接透過 Agent 發送訊息（不經過 API）"""
    try:
        agent = get_or_create_agent(agent_type, user_id)
        if agent is None:
            return None
        
        # 根據 Agent 類型調用對應方法
        if agent_type == "meeting_room":
            response = agent.chat(message, user_id=user_id)
        else:
            response = agent.chat(message)
        
        return response
    except Exception as e:
        st.error(f"發生錯誤: {str(e)}")
        return None


def clear_conversation():
    """清除對話歷史"""
    st.session_state.messages = []
    # 重置 Agent 以清除內部狀態
    st.session_state.agent = None


def get_agent_tools(agent_type: str) -> list:
    """取得 Agent 的工具列表"""
    if agent_type == "general":
        from src.tools import ALL_TOOLS
        tools = ALL_TOOLS
    elif agent_type == "meeting_room":
        from src.meeting_room.tools import MEETING_ROOM_TOOLS
        tools = MEETING_ROOM_TOOLS
    else:
        return []
    
    return [{"name": t.name, "description": t.description} for t in tools]


def main():
    """主程式"""
    st.set_page_config(
        page_title="LangAgent - AI 助理",
        page_icon="🤖",
        layout="wide",
    )
    
    init_session_state()
    
    # 標題
    st.title("🤖 LangAgent - AI 助理")
    st.markdown("基於 LangChain & LangGraph 的智能代理")
    
    # 側邊欄
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # ========================================
        # Agent 選擇
        # ========================================
        st.subheader("🎯 選擇 Agent")
        
        selected_agent = st.selectbox(
            "Agent 類型",
            options=list(AGENT_OPTIONS.keys()),
            format_func=lambda x: AGENT_OPTIONS[x]["name"],
            key="agent_selector",
        )
        
        # 顯示 Agent 描述
        st.info(AGENT_OPTIONS[selected_agent]["description"])
        
        # 如果切換了 Agent，清除對話
        if selected_agent != st.session_state.current_agent_type:
            st.session_state.messages = []
            st.session_state.agent = None
            st.session_state.current_agent_type = selected_agent
        
        st.divider()
        
        # ========================================
        # 用戶設定（會議室 Agent 專用）
        # ========================================
        if selected_agent == "meeting_room":
            st.subheader("👤 用戶設定")
            user_id = st.text_input(
                "用戶 ID",
                value=st.session_state.user_id,
                help="用於識別預約記錄的用戶ID"
            )
            if user_id != st.session_state.user_id:
                st.session_state.user_id = user_id
                # 重建 Agent 以使用新的 user_id
                st.session_state.agent = None
            
            st.divider()
        
        # ========================================
        # 會話管理
        # ========================================
        st.subheader("📝 會話資訊")
        st.text(f"會話 ID: {st.session_state.session_id}")
        if selected_agent == "meeting_room":
            st.text(f"用戶 ID: {st.session_state.user_id}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清除", use_container_width=True):
                clear_conversation()
                st.rerun()
        with col2:
            if st.button("🔄 新會話", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())[:8]
                clear_conversation()
                st.rerun()
        
        st.divider()
        
        # ========================================
        # 可用工具
        # ========================================
        st.subheader("🛠️ 可用工具")
        tools = get_agent_tools(selected_agent)
        if tools:
            for tool in tools:
                with st.expander(f"📌 {tool['name']}"):
                    st.write(tool['description'])
        
        st.divider()
        
        # ========================================
        # 範例問題
        # ========================================
        st.subheader("💡 試試這些問題")
        examples = AGENT_OPTIONS[selected_agent]["examples"]
        
        for q in examples:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                st.session_state.pending_message = q
                st.rerun()
    
    # ========================================
    # 主要對話區域
    # ========================================
    
    # 顯示當前使用的 Agent
    agent_info = AGENT_OPTIONS[selected_agent]
    st.caption(f"當前使用: {agent_info['name']}")
    
    # 顯示對話歷史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 處理待發送的範例訊息
    if hasattr(st.session_state, 'pending_message'):
        pending = st.session_state.pending_message
        del st.session_state.pending_message
        
        # 添加用戶訊息
        st.session_state.messages.append({"role": "user", "content": pending})
        
        with st.chat_message("user"):
            st.markdown(pending)
        
        # 取得 AI 回應
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = send_message_direct(
                    pending, 
                    selected_agent,
                    st.session_state.user_id
                )
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("無法取得回應")
    
    # 輸入框
    if prompt := st.chat_input("輸入您的問題..."):
        # 添加用戶訊息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 取得 AI 回應
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = send_message_direct(
                    prompt, 
                    selected_agent,
                    st.session_state.user_id
                )
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("無法取得回應")
    
    # 頁尾
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔗 [API 文件](http://localhost:8000/docs)")
    with col2:
        st.caption("📊 基於 LangChain & LangGraph")
    with col3:
        st.caption("🛠️ 支援多 Agent 系統")


if __name__ == "__main__":
    main()
