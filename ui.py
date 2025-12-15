"""
Streamlit UI
提供互動式介面來測試 Agent，支援多種 Agent 選擇
"""

import streamlit as st
from typing import Optional
import uuid
import io

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
    if "trace_mode" not in st.session_state:
        st.session_state.trace_mode = False
    if "execution_traces" not in st.session_state:
        st.session_state.execution_traces = {}


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


def send_message_with_trace(message: str, agent_type: str, user_id: str = "user001", trace_container=None):
    """
    發送訊息並即時追蹤執行過程
    
    Returns:
        tuple: (最終回應, 執行步驟列表)
    """
    try:
        agent = get_or_create_agent(agent_type, user_id)
        if agent is None:
            return None, []
        
        steps = []
        final_response = None
        
        # 建立執行追蹤的 UI 元素
        if trace_container:
            trace_container.markdown("### 🔄 執行追蹤")
        
        # 使用 streaming 方法
        if agent_type == "meeting_room":
            stream = agent.chat_stream(message, user_id=user_id)
        else:
            stream = agent.chat_stream(message)
        
        step_count = 0
        for step in stream:
            step_count += 1
            steps.append(step)
            
            # 更新最終回應
            if step.get("output") and step["output"] != "(調用工具中...)":
                final_response = step["output"]
            
            # 即時顯示在 trace_container
            if trace_container:
                with trace_container:
                    render_step(step, step_count)
        
        return final_response, steps
    except Exception as e:
        st.error(f"發生錯誤: {str(e)}")
        return None, []


def render_step(step: dict, step_number: int):
    """渲染單一執行步驟"""
    node_name = step.get("node", "unknown")
    timestamp = step.get("timestamp", "")
    
    # 節點圖示對應
    node_icons = {
        "agent": "🤖",
        "tools": "🔧",
        "supervisor": "👨‍💼",
        "booking_agent": "📅",
        "query_agent": "🔍",
        "management_agent": "📋",
        "booking_tools": "🔧",
        "query_tools": "🔧",
        "management_tools": "🔧",
    }
    icon = node_icons.get(node_name, "⚙️")
    
    # 使用 expander 顯示步驟詳情
    with st.expander(f"{icon} Step {step_number}: **{node_name}**", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col2:
            st.caption(f"⏱️ {timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp}")
        
        # 顯示意圖（如果有）
        if step.get("intent"):
            st.info(f"🎯 意圖: `{step['intent']}`")
        
        if step.get("current_agent"):
            st.info(f"➡️ 路由至: `{step['current_agent']}`")
        
        # 顯示工具調用
        if step.get("tool_calls"):
            st.markdown("**📤 工具調用 (Input):**")
            for tc in step["tool_calls"]:
                st.code(f"🔧 {tc['name']}\n📥 參數: {tc['args']}", language="yaml")
        
        # 顯示工具結果
        if step.get("tool_results"):
            st.markdown("**📥 工具結果 (Output):**")
            for tr in step["tool_results"]:
                st.code(f"🔧 {tr['name']}\n📤 結果: {tr['result']}", language="yaml")
        
        # 顯示 AI 輸出
        if step.get("output") and step["output"] != "(調用工具中...)":
            st.markdown("**💬 AI 回應:**")
            st.markdown(step["output"])


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
                st.session_state.execution_traces = {}
                st.rerun()
        with col2:
            if st.button("🔄 新會話", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())[:8]
                clear_conversation()
                st.session_state.execution_traces = {}
                st.rerun()
        
        st.divider()
        
        # ========================================
        # 執行追蹤模式
        # ========================================
        st.subheader("🔍 執行追蹤")
        st.session_state.trace_mode = st.toggle(
            "啟用即時追蹤",
            value=st.session_state.trace_mode,
            help="顯示 LangGraph 每個節點的執行狀態、輸入輸出"
        )
        
        if st.session_state.trace_mode:
            st.success("✅ 追蹤模式已啟用")
            st.caption("將顯示每個步驟的詳細執行過程")
        
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
        # LangGraph 視覺化
        # ========================================
        st.subheader("📊 LangGraph 視覺化")
        
        if st.button("🗺️ 顯示工作流程圖", use_container_width=True):
            st.session_state.show_graph = True
        
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
    
    # ========================================
    # 顯示 LangGraph 工作流程圖
    # ========================================
    if st.session_state.get("show_graph", False):
        st.subheader("🗺️ LangGraph 工作流程圖")
        
        try:
            agent = get_or_create_agent(selected_agent, st.session_state.user_id)
            if agent:
                # 建立分頁顯示不同格式
                tab1, tab2, tab3 = st.tabs(["📷 PNG 圖片", "📝 Mermaid 程式碼", "🔤 ASCII"])
                
                with tab1:
                    try:
                        # 嘗試取得 PNG 圖片
                        png_data = agent.get_graph_image("png")
                        st.image(png_data, caption=f"{agent_info['name']} 工作流程圖")
                    except Exception as e:
                        st.warning(f"無法生成 PNG 圖片: {str(e)}")
                        st.info("提示：PNG 渲染需要安裝額外套件，請使用 Mermaid 或 ASCII 格式查看")
                
                with tab2:
                    try:
                        # Mermaid 格式
                        mermaid_code = agent.get_graph_mermaid()
                        st.code(mermaid_code, language="mermaid")
                        
                        # 嘗試用 Streamlit 的 mermaid 功能渲染
                        st.markdown("**渲染預覽：**")
                        st.markdown(f"```mermaid\n{mermaid_code}\n```")
                    except Exception as e:
                        st.error(f"無法生成 Mermaid: {str(e)}")
                
                with tab3:
                    try:
                        # ASCII 格式
                        ascii_graph = agent.get_graph_image("ascii")
                        st.code(ascii_graph, language="text")
                    except Exception as e:
                        st.error(f"無法生成 ASCII: {str(e)}")
        except Exception as e:
            st.error(f"建立 Agent 失敗: {str(e)}")
        
        if st.button("❌ 關閉圖形", use_container_width=True):
            st.session_state.show_graph = False
            st.rerun()
        
        st.divider()
    
    # 顯示對話歷史
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # 如果有對應的執行追蹤，顯示展開按鈕
            if msg["role"] == "assistant" and idx in st.session_state.execution_traces:
                with st.expander("🔍 查看執行追蹤", expanded=False):
                    for i, step in enumerate(st.session_state.execution_traces[idx], 1):
                        render_step(step, i)
    
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
            if st.session_state.trace_mode:
                # 追蹤模式：即時顯示執行過程
                trace_container = st.container()
                response, steps = send_message_with_trace(
                    pending, 
                    selected_agent,
                    st.session_state.user_id,
                    trace_container
                )
                if response:
                    st.divider()
                    st.markdown("### 💬 最終回應")
                    st.markdown(response)
                    msg_idx = len(st.session_state.messages)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.execution_traces[msg_idx] = steps
                else:
                    st.error("無法取得回應")
            else:
                # 一般模式
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
            if st.session_state.trace_mode:
                # 追蹤模式：即時顯示執行過程
                trace_container = st.container()
                response, steps = send_message_with_trace(
                    prompt, 
                    selected_agent,
                    st.session_state.user_id,
                    trace_container
                )
                if response:
                    st.divider()
                    st.markdown("### 💬 最終回應")
                    st.markdown(response)
                    msg_idx = len(st.session_state.messages)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.execution_traces[msg_idx] = steps
                else:
                    st.error("無法取得回應")
            else:
                # 一般模式
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
