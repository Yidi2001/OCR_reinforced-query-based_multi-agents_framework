#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target Detection Module
使用分类模型验证 Phi3.5 的文字类型判断
"""

import sys
import torch
from pathlib import Path
from PIL import Image
from typing import Dict

# 添加父目录到路径以导入 case1 的模块
sys.path.insert(0, str(Path(__file__).parent.parent))


class TargetDetector:
    """目标检测器：手写体 vs 印刷体"""
    
    def __init__(self, ckpt_path: str = "checkpoints/printed_vs_hand_best.pth"):
        """
        初始化目标检测器
        
        Args:
            ckpt_path: 分类模型 checkpoint 路径
        """
        from printed_vs_hand_main import load_model_from_ckpt, get_device
        
        print(f"加载 Target Detector: {ckpt_path}")
        
        self.device = get_device()
        self.model, self.idx_to_class, self.eval_tf = load_model_from_ckpt(
            ckpt_path, 
            self.device
        )
        
        print("✓ Target Detector 加载完成")
    
    def detect(self, image_path: str, phi35_prediction: str = None) -> Dict:
        """
        检测图片的文字类型
        
        Args:
            image_path: 图片路径
            phi35_prediction: Phi3.5 的预测结果 (可选)
            
        Returns:
            检测结果 dict
        """
        # 加载图片
        img = Image.open(image_path).convert('RGB')
        
        # 预处理
        x = self.eval_tf(img).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)
            pred_idx = int(prob.argmax(dim=1).item())
            pred_label = self.idx_to_class[pred_idx]
            confidence = float(prob[0, pred_idx].item())
        
        result = {
            "image_path": str(image_path),
            "label": pred_label,
            "confidence": confidence,
            "probabilities": {
                self.idx_to_class[i]: float(prob[0, i].item()) 
                for i in range(len(self.idx_to_class))
            }
        }
        
        # 如果有 Phi3.5 的预测，进行比对
        if phi35_prediction:
            phi35_pred_normalized = self._normalize_text_type(phi35_prediction)
            model_pred_normalized = pred_label
            
            result["phi35_prediction"] = phi35_prediction
            result["phi35_normalized"] = phi35_pred_normalized
            result["agreement"] = phi35_pred_normalized == model_pred_normalized
            
            if not result["agreement"]:
                result["conflict_info"] = f"Phi3.5预测 '{phi35_prediction}' vs 模型预测 '{pred_label}' (conf: {confidence:.3f})"
        
        return result
    
    def _normalize_text_type(self, text_type: str) -> str:
        """
        标准化文字类型名称
        
        Args:
            text_type: 原始类型字符串
            
        Returns:
            标准化后的类型 (hand/printed/mixed)
        """
        text_type_lower = text_type.lower()
        
        if "hand" in text_type_lower or "手写" in text_type_lower:
            return "hand"
        elif "print" in text_type_lower or "印刷" in text_type_lower:
            return "printed"
        elif "mix" in text_type_lower or "混合" in text_type_lower:
            return "mixed"
        else:
            return text_type_lower
    
    def batch_detect(self, image_paths: list, phi35_predictions: list = None) -> list:
        """
        批量检测
        
        Args:
            image_paths: 图片路径列表
            phi35_predictions: Phi3.5 预测列表（可选）
            
        Returns:
            检测结果列表
        """
        results = []
        
        for i, img_path in enumerate(image_paths):
            phi35_pred = phi35_predictions[i] if phi35_predictions else None
            result = self.detect(img_path, phi35_pred)
            results.append(result)
        
        return results


def test_target_detector():
    """测试 Target Detector"""
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("用法: python target_detection.py <image_path> [phi35_prediction]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    phi35_pred = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 创建检测器
    detector = TargetDetector()
    
    # 检测
    print(f"\n{'='*60}")
    print("Target Detection")
    print(f"{'='*60}")
    print(f"图片: {image_path}")
    if phi35_pred:
        print(f"Phi3.5 预测: {phi35_pred}")
    
    result = detector.detect(image_path, phi35_pred)
    
    print(f"\n{'='*60}")
    print("检测结果")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 保存结果
    output_path = Path("detection_output.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到: {output_path}")


if __name__ == "__main__":
    test_target_detector()

