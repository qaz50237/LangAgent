"""
Agent 配置模組
提供統一的配置管理
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from langchain_core.tools import BaseTool


@dataclass
class LLMConfig:
    """LLM 配置"""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: str = "2025-01-01-preview"
    
    def to_dict(self) -> dict:
        """轉換為字典"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "azure_api_key": self.azure_api_key,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "api_version": self.api_version,
        }


@dataclass  
class AgentConfig:
    """
    Agent 配置類別
    
    統一管理 Agent 的所有配置項目，讓開發者專注於核心設計。
    
    使用範例:
    ```python
    config = AgentConfig(
        name="booking_agent",
        description="會議室預約 Agent",
        system_prompt="你是一個會議室預約助手...",
        tools=my_tools,  # 從 MCP 載入
        llm_config=LLMConfig(model_name="gpt-4o"),
    )
    ```
    """
    # 基本資訊
    name: str
    description: str = ""
    
    # Prompt 設計
    system_prompt: str = ""
    system_prompt_template: str = ""  # 支援動態 prompt
    prompt_variables: Dict[str, Any] = field(default_factory=dict)
    
    # 工具配置（從 MCP 取得）
    tools: List[BaseTool] = field(default_factory=list)
    
    # LLM 配置
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    
    # 用戶相關
    default_user_id: str = "default_user"
    
    # 額外配置
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_system_prompt(self, **kwargs) -> str:
        """
        取得系統提示詞
        
        支援動態變數替換
        
        Args:
            **kwargs: 動態變數
        
        Returns:
            完整的系統提示詞
        """
        if self.system_prompt_template:
            variables = {**self.prompt_variables, **kwargs}
            return self.system_prompt_template.format(**variables)
        return self.system_prompt
    
    def with_tools(self, tools: List[BaseTool]) -> "AgentConfig":
        """
        建立帶有新工具的配置副本
        
        Args:
            tools: 新工具列表
        
        Returns:
            新的 AgentConfig 實例
        """
        return AgentConfig(
            name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
            system_prompt_template=self.system_prompt_template,
            prompt_variables=self.prompt_variables.copy(),
            tools=tools,
            llm_config=self.llm_config,
            default_user_id=self.default_user_id,
            metadata=self.metadata.copy(),
        )


@dataclass
class MultiAgentConfig:
    """
    多 Agent 系統配置
    
    用於 Supervisor 模式的多 Agent 系統。
    
    使用範例:
    ```python
    config = MultiAgentConfig(
        name="meeting_system",
        supervisor_config=AgentConfig(...),
        sub_agents={
            "booking": AgentConfig(...),
            "query": AgentConfig(...),
        },
        routing_rules={...},
    )
    ```
    """
    # 系統資訊
    name: str
    description: str = ""
    
    # Supervisor 配置
    supervisor_config: AgentConfig = None
    
    # 子 Agent 配置
    sub_agents: Dict[str, AgentConfig] = field(default_factory=dict)
    
    # 路由規則
    routing_rules: Dict[str, str] = field(default_factory=dict)
    
    # 預設用戶 ID
    default_user_id: str = "default_user"
    
    def get_all_tools(self) -> List[BaseTool]:
        """取得所有 Agent 的工具"""
        all_tools = []
        for config in self.sub_agents.values():
            all_tools.extend(config.tools)
        return all_tools
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """取得指定 Agent 的配置"""
        return self.sub_agents.get(agent_name)
