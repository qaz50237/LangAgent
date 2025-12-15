"""
Function Tools 模組
這些 tools 可以被 LangGraph agent 調用
設計為可以輕鬆替換為 MCP (Model Context Protocol) tools
"""

from langchain_core.tools import tool
from typing import Optional
from datetime import datetime
import math


@tool
def get_current_time() -> str:
    """取得目前的日期和時間。當用戶詢問現在幾點或今天日期時使用此工具。"""
    now = datetime.now()
    return f"目前時間: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"


@tool
def calculator(expression: str) -> str:
    """
    執行數學計算。支援基本運算（+、-、*、/）和進階函數（sqrt、pow、sin、cos、tan）。
    
    Args:
        expression: 要計算的數學表達式，例如 "2 + 2" 或 "sqrt(16)"
    
    Returns:
        計算結果的字串表示
    """
    try:
        # 安全的數學函數白名單
        safe_dict = {
            "sqrt": math.sqrt,
            "pow": pow,
            "abs": abs,
            "round": round,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"計算結果: {expression} = {result}"
    except Exception as e:
        return f"計算錯誤: {str(e)}"


@tool
def search_knowledge_base(query: str) -> str:
    """
    搜尋內部知識庫。這是一個範例工具，可以替換為實際的資料庫查詢或 MCP 工具。
    
    Args:
        query: 搜尋查詢字串
    
    Returns:
        搜尋結果
    """
    # 模擬知識庫 - 實際使用時可以連接資料庫或 MCP
    knowledge_base = {
        "langchain": "LangChain 是一個用於開發 LLM 應用程式的框架，提供了模組化的元件來建構 AI 應用。",
        "langgraph": "LangGraph 是基於 LangChain 的圖形化工作流程框架，用於建構有狀態的多步驟 AI 應用程式。",
        "mcp": "MCP (Model Context Protocol) 是一個開放協議，允許 AI 模型與外部工具和資料來源互動。",
        "agent": "AI Agent 是一種能夠自主執行任務、使用工具並做出決策的人工智慧系統。",
    }
    
    query_lower = query.lower()
    results = []
    for key, value in knowledge_base.items():
        if key in query_lower or query_lower in key:
            results.append(f"【{key}】: {value}")
    
    if results:
        return "\n".join(results)
    return f"在知識庫中找不到與 '{query}' 相關的資訊。"


@tool
def weather_info(city: str) -> str:
    """
    取得城市的天氣資訊。這是一個模擬工具，實際使用時可以連接真實的天氣 API 或 MCP。
    
    Args:
        city: 城市名稱
    
    Returns:
        天氣資訊
    """
    # 模擬天氣資料 - 實際使用時可以連接天氣 API 或 MCP
    mock_weather = {
        "台北": {"temp": 25, "condition": "多雲", "humidity": 75},
        "taipei": {"temp": 25, "condition": "多雲", "humidity": 75},
        "高雄": {"temp": 28, "condition": "晴天", "humidity": 70},
        "kaohsiung": {"temp": 28, "condition": "晴天", "humidity": 70},
        "東京": {"temp": 18, "condition": "晴天", "humidity": 55},
        "tokyo": {"temp": 18, "condition": "晴天", "humidity": 55},
    }
    
    city_lower = city.lower()
    for key, data in mock_weather.items():
        if key.lower() == city_lower or city_lower in key.lower():
            return f"{city}的天氣: 溫度 {data['temp']}°C, {data['condition']}, 濕度 {data['humidity']}%"
    
    return f"找不到 {city} 的天氣資訊。支援的城市: 台北、高雄、東京"


@tool  
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    單位轉換工具。支援長度、重量、溫度等常見單位轉換。
    
    Args:
        value: 要轉換的數值
        from_unit: 來源單位 (如: km, m, cm, kg, g, lb, celsius, fahrenheit)
        to_unit: 目標單位
    
    Returns:
        轉換結果
    """
    conversions = {
        # 長度
        ("km", "m"): lambda x: x * 1000,
        ("m", "km"): lambda x: x / 1000,
        ("m", "cm"): lambda x: x * 100,
        ("cm", "m"): lambda x: x / 100,
        ("mile", "km"): lambda x: x * 1.60934,
        ("km", "mile"): lambda x: x / 1.60934,
        # 重量
        ("kg", "g"): lambda x: x * 1000,
        ("g", "kg"): lambda x: x / 1000,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x / 2.20462,
        # 溫度
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    
    return f"不支援從 {from_unit} 轉換到 {to_unit}。請確認單位名稱。"


# 所有可用的工具列表
ALL_TOOLS = [
    get_current_time,
    calculator,
    search_knowledge_base,
    weather_info,
    unit_converter,
]


def get_tools_description() -> str:
    """取得所有工具的描述"""
    descriptions = []
    for tool in ALL_TOOLS:
        descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(descriptions)
