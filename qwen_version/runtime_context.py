#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RuntimeContext / ModelRegistry
安全共享底座：只共享无状态资源（模型、processor、tokenizer、client等）
不共享任何 messages/history/past_key_values 等推理状态
"""

import threading
from typing import Any, Callable, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class RuntimeContext:
    """
    运行时上下文：管理共享的无状态资源（模型、processor、client等）
    
    特性：
    1. 懒加载：首次访问时才初始化
    2. 线程安全：支持多线程环境
    3. 只缓存无状态资源：不存储 messages/history 等推理状态
    4. 清晰日志：记录资源的创建和复用
    
    示例：
        ctx = RuntimeContext()
        
        # 获取或创建模型
        model = ctx.get("phi_model::path/to/model", 
                       lambda: AutoModel.from_pretrained("path/to/model"))
        
        # 再次获取时返回缓存的实例
        model2 = ctx.get("phi_model::path/to/model", 
                        lambda: AutoModel.from_pretrained("path/to/model"))
        # model2 is model -> True
    """
    
    def __init__(self):
        self._registry: Dict[str, Any] = {}
        self._lock = threading.Lock()
        logger.info("🔧 RuntimeContext initialized")
    
    def get(self, key: str, factory: Callable[[], Any]) -> Any:
        """
        获取或创建资源（懒加载 + 缓存）
        
        Args:
            key: 资源唯一标识（建议格式：资源类型::配置标识）
                 例如：
                 - "phi_model::microsoft/Phi-3.5-vision-instruct"
                 - "phi_processor::microsoft/Phi-3.5-vision-instruct"
                 - "llm::http://localhost:11434::qwen2.5:7b::0.7"
                 - "trocr::microsoft/trocr-large-handwritten"
                 - "paddleocr::en"
            factory: 工厂函数，当资源不存在时调用创建
            
        Returns:
            缓存的或新创建的资源实例
        """
        # 快速路径：无锁检查
        if key in self._registry:
            logger.debug(f"✓ Reuse cached resource: {key}")
            return self._registry[key]
        
        # 需要创建资源：加锁
        with self._lock:
            # 双重检查（避免竞争条件）
            if key in self._registry:
                logger.debug(f"✓ Reuse cached resource: {key}")
                return self._registry[key]
            
            # 创建新资源
            logger.info(f"⚡ Init new resource: {key}")
            resource = factory()
            self._registry[key] = resource
            return resource
    
    def contains(self, key: str) -> bool:
        """检查资源是否已缓存"""
        return key in self._registry
    
    def pop(self, key: str) -> Optional[Any]:
        """移除并返回资源（用于手动释放显存）"""
        with self._lock:
            if key in self._registry:
                logger.info(f"🗑️  Remove resource: {key}")
                return self._registry.pop(key)
            return None
    
    def clear(self):
        """清空所有缓存资源"""
        with self._lock:
            count = len(self._registry)
            self._registry.clear()
            logger.info(f"🧹 Cleared {count} cached resources")
    
    def keys(self):
        """返回所有已缓存资源的 key"""
        return list(self._registry.keys())
    
    def __len__(self):
        """返回已缓存资源数量"""
        return len(self._registry)
    
    def __repr__(self):
        return f"RuntimeContext(cached_resources={len(self._registry)})"


# 便捷函数：生成标准化的资源 key
def make_model_key(model_type: str, model_path: str) -> str:
    """生成模型资源的标准 key"""
    return f"{model_type}::{model_path}"


def make_llm_key(base_url: str, model: str, temperature: float) -> str:
    """生成 LLM client 的标准 key"""
    return f"llm::{base_url}::{model}::{temperature}"


def make_ocr_key(ocr_type: str, lang_or_config: str) -> str:
    """生成 OCR 引擎的标准 key"""
    return f"{ocr_type}::{lang_or_config}"
