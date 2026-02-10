#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Agent Class - 所有 Agent 的基类
提供统一的接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate


class BaseAgent(ABC):
    """Agent 基类 - 定义所有 Agent 的通用接口"""
    
    def __init__(
        self,
        agent_name: str,
        description: str,
        instructions: str = "",
        verbose: bool = False,
        llm_config: Dict[str, Any] = None,
        ctx: Optional['RuntimeContext'] = None
    ):
        """
        初始化 Agent
        
        Args:
            agent_name: Agent 名称
            description: Agent 描述
            instructions: 额外的任务指令
            verbose: 是否打印详细日志
            llm_config: LLM 配置（可选）
            ctx: RuntimeContext 实例（用于共享模型和 LLM client）
        """
        self.agent_name = agent_name
        self.description = description
        self.instructions = instructions
        self.verbose = verbose
        self.ctx = ctx
        
        # 默认 LLM 配置（DeepSeek）
        self.llm_config = llm_config or {
            "model": "deepseek-chat",
            "openai_api_key": "sk-8978af3f624a46a5845c9a597b46ee4b",
            "openai_api_base": "https://api.deepseek.com",
            "temperature": 0
        }
        
        # 初始化模型资源（延迟加载）
        self._model = None
        self._llm = None
        self._agent_executor = None
        
        print(f"✓ {self.agent_name} 初始化完成")
    
    @abstractmethod
    def load_model(self):
        """
        加载 Agent 所需的模型（如 TrOCR, PaddleOCR 等）
        子类必须实现此方法
        """
        pass
    
    @abstractmethod
    def get_tool_function(self) -> Callable:
        """
        返回 Agent 的工具函数（用 @tool 装饰）
        子类必须实现此方法
        
        Returns:
            带 @tool 装饰器的函数
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        返回 Agent 的 System Prompt
        子类必须实现此方法
        
        Returns:
            System prompt 字符串
        """
        pass
    
    def get_llm(self) -> ChatOpenAI:
        """
        获取或创建 LLM 实例（支持共享）
        
        如果提供了 RuntimeContext，则所有 Agent 共享同一个 LLM client（同配置）
        否则退回到实例内缓存模式
        
        Returns:
            ChatOpenAI 实例
        """
        # 如果有 ctx，从 ctx 获取共享的 LLM client
        if self.ctx is not None:
            from runtime_context import make_llm_key
            key = make_llm_key(
                base_url=self.llm_config.get("openai_api_base", ""),
                model=self.llm_config.get("model", ""),
                temperature=self.llm_config.get("temperature", 0.0)
            )
            return self.ctx.get(key, lambda: ChatOpenAI(**self.llm_config))
        
        # 退化模式：实例内缓存（向后兼容）
        if self._llm is None:
            self._llm = ChatOpenAI(**self.llm_config)
        return self._llm
    
    def build_agent(self) -> AgentExecutor:
        """
        构建 LangChain Agent Executor
        
        Returns:
            AgentExecutor 实例
        """
        if self._agent_executor is not None:
            return self._agent_executor
        
        # 获取 LLM
        llm = self.get_llm()
        
        # 获取工具函数
        tool_function = self.get_tool_function()
        tools = [tool_function]
        
        # 构建 System Prompt
        system_prompt = self.get_system_prompt()
        
        # 如果有额外的 instructions，追加到 system prompt
        if self.instructions:
            system_prompt += f"\n\n额外要求：\n{self.instructions}"
        
        # 创建 Prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # 创建 Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # 创建 Agent Executor
        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.verbose,
            handle_parsing_errors=True
        )
        
        return self._agent_executor
    
    def execute(self, image_path: str, query: str = None) -> Dict[str, Any]:
        """
        执行 Agent 任务
        
        Args:
            image_path: 图片路径
            query: 用户查询（可选）
        
        Returns:
            执行结果字典
        """
        # 确保模型已加载
        if self._model is None:
            self.load_model()
        
        # 构建 Agent（如果还没有）
        agent_executor = self.build_agent()
        
        # 构建输入提示
        if query:
            input_text = f"{query}。图片路径：{image_path}"
        else:
            input_text = f"请处理这张图片：{image_path}"
        
        # 执行
        result = agent_executor.invoke({"input": input_text})
        
        return {
            "agent_name": self.agent_name,
            "input": image_path,
            "query": query,
            "output": result.get('output', ''),
            "status": "success"
        }
    
    def invoke(self, input_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        LangChain 标准的 invoke 接口
        
        Args:
            input_dict: 包含 'input' 键的字典
        
        Returns:
            包含 'output' 键的字典
        """
        # 确保模型已加载
        if self._model is None:
            self.load_model()
        
        # 构建 Agent
        agent_executor = self.build_agent()
        
        # 执行
        return agent_executor.invoke(input_dict)
    
    def __call__(self, image_path: str, query: str = None) -> Dict[str, Any]:
        """
        使 Agent 实例可以像函数一样调用
        
        Args:
            image_path: 图片路径
            query: 用户查询（可选）
        
        Returns:
            执行结果字典
        """
        return self.execute(image_path, query)
    
    def get_info(self) -> Dict[str, str]:
        """
        获取 Agent 信息
        
        Returns:
            Agent 信息字典
        """
        return {
            "name": self.agent_name,
            "description": self.description,
            "type": self.__class__.__name__,
            "instructions": self.instructions
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.agent_name}')"


# 示例：如何使用基类创建具体的 Agent
class ExampleAgent(BaseAgent):
    """示例 Agent - 展示如何继承基类"""
    
    def __init__(self, instructions: str = "", verbose: bool = False):
        super().__init__(
            agent_name="ExampleAgent",
            description="这是一个示例 Agent",
            instructions=instructions,
            verbose=verbose
        )
    
    def load_model(self):
        """加载模型（示例）"""
        print("加载示例模型...")
        self._model = "example_model"
    
    def get_tool_function(self) -> Callable:
        """返回工具函数"""
        @tool
        def example_tool(image_path: str) -> str:
            """示例工具函数"""
            return f"处理图片: {image_path}"
        
        return example_tool
    
    def get_system_prompt(self) -> str:
        """返回 System Prompt"""
        return """你是一个示例 Agent。
        
当用户提供图片路径时，使用 example_tool 工具进行处理。"""


if __name__ == "__main__":
    print("=" * 60)
    print("Base Agent 架构测试")
    print("=" * 60)
    
    # 测试示例 Agent
    agent = ExampleAgent(verbose=True)
    print(f"\nAgent 信息: {agent.get_info()}")
    print(f"Agent 表示: {agent}")

