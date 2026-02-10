#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片格式转换工具
用于将 PaddleOCR 不支持的格式（如 .tif）转换为支持的格式
"""

import os
import tempfile
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional


def is_paddleocr_supported(image_path: str) -> bool:
    """
    检查图片格式是否被 PaddleOCR 支持
    
    Args:
        image_path: 图片路径
        
    Returns:
        True if supported, False otherwise
    """
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.pdf'}
    ext = Path(image_path).suffix.lower()
    return ext in supported_exts


def convert_to_paddleocr_format(image_path: str, output_format: str = 'png') -> Tuple[str, bool]:
    """
    将图片转换为 PaddleOCR 支持的格式（如果需要）
    
    Args:
        image_path: 原始图片路径
        output_format: 输出格式（默认 'png'）
        
    Returns:
        (转换后的图片路径, 是否进行了转换)
        - 如果原格式已支持，返回 (原路径, False)
        - 如果进行了转换，返回 (临时文件路径, True)
    """
    # 检查是否需要转换
    if is_paddleocr_supported(image_path):
        return image_path, False
    
    # 需要转换
    try:
        # 打开图片
        img = Image.open(image_path)
        
        # 转换为 RGB 模式（如果不是）
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix=f'.{output_format}')
        os.close(temp_fd)  # 关闭文件描述符
        
        # 保存为新格式
        img.save(temp_path, format=output_format.upper())
        
        return temp_path, True
        
    except Exception as e:
        raise RuntimeError(f"转换图片格式失败: {e}")


def cleanup_temp_file(file_path: str, is_temp: bool):
    """
    清理临时文件（如果是临时文件）
    
    Args:
        file_path: 文件路径
        is_temp: 是否为临时文件
    """
    if is_temp and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            pass  # 静默失败
