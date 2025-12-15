"""
LLM 工具模組
提供統一的 LLM 初始化和配置
"""

import os
from typing import Optional, List
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()


def get_llm(
    model_name: str = None,
    temperature: float = 0.7,
    azure_api_key: str = None,
    azure_endpoint: str = None,
    azure_deployment: str = None,
    api_version: str = "2025-01-01-preview",
) -> BaseChatModel:
    """
    根據環境變數或參數自動選擇 LLM
    優先使用 Azure OpenAI，若無則使用 OpenAI
    
    Args:
        model_name: OpenAI 模型名稱（非 Azure 時使用）
        temperature: 生成溫度
        azure_api_key: Azure API Key（覆蓋環境變數）
        azure_endpoint: Azure Endpoint（覆蓋環境變數）
        azure_deployment: Azure Deployment Name（覆蓋環境變數）
        api_version: Azure API 版本
    
    Returns:
        LLM 實例
    """
    # 優先使用參數，其次環境變數
    _azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
    _azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    _azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if _azure_api_key and _azure_endpoint and _azure_deployment:
        # 使用 Azure OpenAI
        base_endpoint = _azure_endpoint.split("/openai/")[0] if "/openai/" in _azure_endpoint else _azure_endpoint
        
        return AzureChatOpenAI(
            azure_endpoint=base_endpoint,
            azure_deployment=_azure_deployment,
            api_key=_azure_api_key,
            api_version=api_version,
        )
    else:
        # 使用 OpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=temperature,
        )


def create_llm_with_tools(
    tools: List[BaseTool],
    model_name: str = None,
    temperature: float = 0.7,
    **llm_kwargs,
) -> BaseChatModel:
    """
    建立綁定工具的 LLM
    
    Args:
        tools: 工具列表
        model_name: 模型名稱
        temperature: 生成溫度
        **llm_kwargs: 其他 LLM 參數
    
    Returns:
        綁定工具的 LLM 實例
    """
    llm = get_llm(model_name=model_name, temperature=temperature, **llm_kwargs)
    
    if tools:
        return llm.bind_tools(tools)
    return llm
