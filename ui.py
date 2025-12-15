"""
Streamlit UI
提供互動式介面來測試 Agent
"""

import streamlit as st
import requests
from typing import Optional
import uuid

# API 設定
API_BASE_URL = "http://localhost:8000"


def init_session_state():
    """初始化 Streamlit session state"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_connected" not in st.session_state:
        st.session_state.api_connected = False


def check_api_health() -> bool:
    """檢查 API 是否可用"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("agent_ready", False)
    except:
        return False


def send_message(message: str, use_session: bool = True) -> Optional[str]:
    """發送訊息到 API"""
    try:
        endpoint = "/chat" if use_session else "/chat/simple"
        payload = {"message": message}
        
        if use_session:
            payload["session_id"] = st.session_state.session_id
        
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=60,
        )
        
        if response.status_code == 200:
            return response.json().get("response")
        else:
            st.error(f"API 錯誤: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("無法連接到 API 伺服器。請確保 API 正在運行。")
        return None
    except Exception as e:
        st.error(f"發生錯誤: {str(e)}")
        return None


def get_available_tools() -> list:
    """取得可用工具列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/tools", timeout=5)
        if response.status_code == 200:
            return response.json().get("tools", [])
    except:
        pass
    return []


def clear_conversation():
    """清除對話歷史"""
    st.session_state.messages = []
    try:
        requests.delete(f"{API_BASE_URL}/sessions/{st.session_state.session_id}", timeout=5)
    except:
        pass


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
        
        # API 狀態
        api_status = check_api_health()
        st.session_state.api_connected = api_status
        
        if api_status:
            st.success("✅ API 連線正常")
        else:
            st.error("❌ API 未連線")
            st.info("請先啟動 API 伺服器:\n```\npython api.py\n```")
        
        st.divider()
        
        # 會話資訊
        st.subheader("📝 會話資訊")
        st.text(f"會話 ID: {st.session_state.session_id}")
        
        # 清除對話按鈕
        if st.button("🗑️ 清除對話", use_container_width=True):
            clear_conversation()
            st.rerun()
        
        # 新會話按鈕
        if st.button("🔄 開始新會話", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # 可用工具
        st.subheader("🛠️ 可用工具")
        if api_status:
            tools = get_available_tools()
            if tools:
                for tool in tools:
                    with st.expander(f"📌 {tool['name']}"):
                        st.write(tool['description'])
            else:
                st.info("無法取得工具列表")
        else:
            st.info("請先連接 API")
        
        st.divider()
        
        # 範例問題
        st.subheader("💡 試試這些問題")
        example_questions = [
            "現在幾點？",
            "計算 sqrt(144) + 25 * 2",
            "台北今天天氣如何？",
            "把 100 公里轉換成英里",
            "什麼是 LangChain？",
            "幫我把攝氏 30 度轉成華氏",
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}", use_container_width=True):
                st.session_state.pending_message = q
                st.rerun()
    
    # 主要對話區域
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
                response = send_message(pending)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("無法取得回應")
    
    # 輸入框
    if prompt := st.chat_input("輸入您的問題...", disabled=not st.session_state.api_connected):
        # 添加用戶訊息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 取得 AI 回應
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = send_message(prompt)
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
        st.caption("🛠️ 支援 Function Tools / MCP")


if __name__ == "__main__":
    main()
