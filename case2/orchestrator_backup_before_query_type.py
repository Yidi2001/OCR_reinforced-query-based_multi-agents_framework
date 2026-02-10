#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator - 主流程编排器
整合 Task Planner -> Target Detection -> Prompt Generator -> Agent Execution
"""

import json
import time
from pathlib import Path
from typing import Dict
from PIL import Image

from runtime_context import RuntimeContext
from task_planner import Phi35TaskPlanner
from target_detection import TargetDetector
from prompt_generator import PromptGenerator

# 导入所有 Agents
from preprocessing_agent import PreprocessingAgent
from LayoutDetectionAgent import LayoutDetectionAgent
from trocr import HandOCRAgent
from printed_ocr_agent import PrintedOCRAgent


class MultiAgentOrchestrator:
    """多智能体框架编排器"""
    
    def __init__(
        self,
        phi35_model_path: str = "models/phi-3_5_vision",
        classifier_ckpt_path: str = "checkpoints/printed_vs_hand_best.pth",
        execute_agents: bool = False,
        verbose: bool = False
    ):
        """
        初始化编排器
        
        Args:
            phi35_model_path: Phi3.5-Vision 模型路径
            classifier_ckpt_path: 分类器模型路径
            execute_agents: 是否执行 Agents（True=完整流程，False=仅规划）
            verbose: 是否打印详细日志
        """
        print("\n" + "="*60)
        print("初始化 Multi-Agent Orchestrator (共享模型底座模式)")
        print("="*60)
        
        self.execute_agents = execute_agents
        self.verbose = verbose
        
        # 🆕 创建 RuntimeContext（共享模型底座）
        print("\n⚡ 初始化 RuntimeContext (模型注册中心)...")
        self.ctx = RuntimeContext()
        
        # 初始化规划模块（注入 ctx）
        print("\n初始化规划模块...")
        self.task_planner = Phi35TaskPlanner(phi35_model_path, ctx=self.ctx)
        self.target_detector = TargetDetector(classifier_ckpt_path)
        self.prompt_generator = PromptGenerator()
        
        # 初始化执行 Agents（如果需要，注入 ctx）
        self.agents = {}
        if execute_agents:
            print("\n初始化执行 Agents (共享 LLM client)...")
            self.agents = {
                'PreprocessingAgent': PreprocessingAgent(verbose=verbose, ctx=self.ctx),
                'LayoutDetectionAgent': LayoutDetectionAgent(verbose=verbose, ctx=self.ctx),
                'HandOCRAgent': HandOCRAgent(verbose=verbose, ctx=self.ctx),
                'PrintedOCRAgent': PrintedOCRAgent(verbose=verbose, ctx=self.ctx)
            }
            print(f"✓ 已初始化 {len(self.agents)} 个 Agents")
        
        print(f"\n✓ 所有模块初始化完成 (RuntimeContext 已缓存 {len(self.ctx)} 个资源)")
    
    def run(self, image_path: str, query: str, output_path: str = None, sample_output_dir: str = None) -> Dict:
        """
        运行完整的规划+执行流程
        
        Args:
            image_path: 图片路径
            query: 用户查询
            output_path: 输出文件路径（可选）
            sample_output_dir: 样本专属输出目录（用于保存可视化图片等）
            
        Returns:
            Structured Execution Plan + Execution Results (dict)
        """
        start_time = time.time()
        
        print("\n" + "="*60)
        print("开始任务规划流程")
        print("="*60)
        print(f"图片: {image_path}")
        print(f"查询: {query}")
        
        # ============================
        # 步骤 1: Target Detection（分类模型判断手写/印刷）
        # ============================
        print("\n" + "="*60)
        print("步骤 1/3: Target Detection (手写/印刷体分类)")
        print("="*60)
        
        classification = self.target_detector.detect(image_path, phi35_prediction=None)
        
        print(f"\n✓ Target Detection 完成")
        print(f"  - 分类结果: {classification.get('label', 'unknown')}")
        print(f"  - 置信度: {classification.get('confidence', 0.0):.3f}")
        
        # ============================
        # 步骤 2: Phi3.5-Vision Task Planning（预处理规划）
        # ============================
        print("\n" + "="*60)
        print("步骤 2/3: Phi3.5-Vision 预处理规划")
        print("="*60)
        
        task_plan = self.task_planner.plan(image_path, query)
        
        print(f"\n✓ Task Plan 生成完成")
        print(f"  - 需要超分: {task_plan.get('needs_super_resolution', False)}")
        print(f"  - 需要布局检测: {task_plan.get('needs_layout_detection', False)}")
        print(f"  - 文字复杂度: {task_plan.get('text_complexity', 'unknown')}")
        
        # ============================
        # 步骤 3: Prompt Generation
        # ============================
        print("\n" + "="*60)
        print("步骤 3/3: Prompt Generation")
        print("="*60)
        
        execution_plan = self.prompt_generator.generate(
            task_plan=task_plan,
            classification=classification,
            query=query
        )
        
        print(f"\n✓ Execution Plan 生成完成")
        print(f"  - 总 Agent 数: {execution_plan['execution_plan']['total_agents']}")
        
        # 添加分类信息到结果
        execution_plan["classification"] = classification
        
        # 添加规划时间
        planning_time = time.time() - start_time
        execution_plan["planning_time"] = planning_time
        
        # ============================
        # 步骤 4: 执行 Agents（如果启用）
        # ============================
        if self.execute_agents:
            execution_results = self._execute_agents(
                execution_plan=execution_plan,
                image_path=image_path,
                classification=classification,
                sample_output_dir=sample_output_dir
            )
            execution_plan["execution_results"] = execution_results
            execution_plan["execution_time"] = execution_results["total_execution_time"]
            execution_plan["total_time"] = planning_time + execution_results["total_execution_time"]
        
        # ============================
        # 保存结果
        # ============================
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(execution_plan, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ 结果已保存到: {output_file}")
        
        # ============================
        # 打印完整的执行计划
        # ============================
        self._print_execution_plan(execution_plan)
        
        return execution_plan
    
    def _ocr_with_layout(self, ocr_agent, image_path: str, layout_result: Dict, text_type: str) -> str:
        """
        基于布局检测结果对每个框进行 OCR 识别
        
        Args:
            ocr_agent: OCR Agent 实例
            image_path: 图片路径
            layout_result: 布局检测结果
            text_type: 文字类型（手写体/印刷体）
            
        Returns:
            格式化的识别结果
        """
        import tempfile
        import os
        
        # 读取原始图片
        image = Image.open(image_path)
        boxes = layout_result.get('boxes', [])
        
        print(f"基于布局检测对 {len(boxes)} 个区域进行 {text_type} OCR 识别...")
        
        ocr_results = []
        
        for idx, box in enumerate(boxes, 1):
            label = box['label']
            coordinate = box['coordinate']
            score = box['score']
            
            # 裁剪图片
            x1, y1, x2, y2 = map(int, coordinate)
            cropped = image.crop((x1, y1, x2, y2))
            
            # 保存裁剪后的图片到临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cropped.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # 调用 OCR Agent
                result = ocr_agent.invoke({
                    "input": f"请识别这张{text_type}图片中的文字：{tmp_path}\n\n重要提示：只输出识别结果原文，不要翻译、不要解释、不要添加任何说明文字。"
                })
                ocr_text = result['output']
                
                ocr_results.append({
                    "region_id": idx,
                    "label": label,
                    "confidence": score,
                    "coordinate": coordinate,
                    "ocr_result": ocr_text
                })
                
                print(f"  区域 {idx} ({label}): 识别完成")
                
            finally:
                # 删除临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        # 格式化输出结果
        output_lines = [f"基于布局检测的 OCR 结果（共 {len(ocr_results)} 个区域）：\n"]
        
        for result in ocr_results:
            output_lines.append(f"区域 {result['region_id']}: {result['label']} (置信度: {result['confidence']:.3f})")
            output_lines.append(f"位置: {result['coordinate']}")
            output_lines.append(f"文字内容:\n{result['ocr_result']}")
            output_lines.append("-" * 60)
        
        return "\n".join(output_lines)
    
    def _execute_agents(self, execution_plan: Dict, image_path: str, classification: Dict, 
                       sample_output_dir: str = None) -> Dict:
        """
        执行 Agents
        
        Args:
            execution_plan: 执行计划
            image_path: 原始图片路径
            classification: 分类结果
            sample_output_dir: 样本专属输出目录（用于保存可视化图片等）
            
        Returns:
            执行结果字典
        """
        print("\n" + "="*60)
        print("步骤 4: 执行 Agents")
        print("="*60)
        
        execution_start = time.time()
        agents_to_execute = execution_plan['execution_plan']['agents']
        
        results = []
        current_image_path = image_path  # 当前使用的图片路径（可能会被超分后替换）
        layout_result = None  # 布局检测结果
        
        for agent_info in agents_to_execute:
            agent_name = agent_info['name']
            agent_order = agent_info['order']
            
            print(f"\n{'─'*60}")
            print(f"执行 Agent {agent_order}: {agent_name}")
            print(f"{'─'*60}")
            
            agent_start = time.time()
            
            try:
                # 根据 Agent 名称执行对应的 Agent
                if agent_name == "PreprocessingAgent":
                    agent = self.agents['PreprocessingAgent']
                    # 执行超分辨率增强
                    enhanced_path = agent.enhance(current_image_path)
                    output = f"图片增强成功: {enhanced_path}"
                    current_image_path = enhanced_path  # 更新当前图片路径
                    
                elif agent_name == "LayoutDetectionAgent":
                    agent = self.agents['LayoutDetectionAgent']
                    # 执行布局检测，传递 sample_output_dir
                    layout_result = agent.detect(current_image_path, output_dir=sample_output_dir)
                    output = f"检测到 {layout_result['detected_regions']} 个布局区域"
                    
                elif agent_name == "HandOCRAgent":
                    agent = self.agents['HandOCRAgent']
                    if layout_result and layout_result['detected_regions'] > 0:
                        # 有布局检测结果，对每个框进行识别
                        output = self._ocr_with_layout(agent, current_image_path, layout_result, "手写体")
                    else:
                        # 没有布局检测，直接识别整张图
                        result = agent.invoke({
                            "input": f"请识别这张手写图片中的文字：{current_image_path}\n\n重要提示：只输出识别结果原文，不要翻译、不要解释、不要添加任何说明文字。"
                        })
                        output = result['output']
                    
                elif agent_name == "PrintedOCRAgent":
                    agent = self.agents['PrintedOCRAgent']
                    if layout_result and layout_result['detected_regions'] > 0:
                        # 有布局检测结果，对每个框进行识别
                        output = self._ocr_with_layout(agent, current_image_path, layout_result, "印刷体")
                    else:
                        # 没有布局检测，直接识别整张图
                        result = agent.invoke({
                            "input": f"请识别这张印刷体图片中的文字：{current_image_path}\n\n重要提示：只输出识别结果原文，不要翻译、不要解释、不要添加任何说明文字。"
                        })
                        output = result['output']
                    
                else:
                    output = f"未知 Agent: {agent_name}"
                
                agent_time = time.time() - agent_start
                
                results.append({
                    "agent_name": agent_name,
                    "order": agent_order,
                    "status": "success",
                    "output": output,
                    "execution_time": agent_time
                })
                
                print(f"✓ {agent_name} 完成 (耗时: {agent_time:.2f}秒)")
                
            except Exception as e:
                agent_time = time.time() - agent_start
                error_msg = f"执行失败: {str(e)}"
                
                results.append({
                    "agent_name": agent_name,
                    "order": agent_order,
                    "status": "failed",
                    "error": error_msg,
                    "execution_time": agent_time
                })
                
                print(f"✗ {agent_name} 失败: {error_msg}")
        
        total_execution_time = time.time() - execution_start
        
        # 提取最终的 OCR 结果
        final_output = ""
        for result in results:
            if result['agent_name'] in ['HandOCRAgent', 'PrintedOCRAgent']:
                final_output = result.get('output', '')
        
        print(f"\n{'─'*60}")
        print(f"✓ 所有 Agents 执行完成")
        print(f"总执行时间: {total_execution_time:.2f}秒")
        print(f"{'─'*60}")
        
        return {
            "agents_executed": results,
            "total_execution_time": total_execution_time,
            "final_image_path": current_image_path,
            "layout_result": layout_result,
            "final_output": final_output
        }
    
    def _print_execution_plan(self, execution_plan: Dict):
        """打印结构化的执行计划"""
        print("\n" + "="*60)
        print("📋 Structured Execution Plan")
        print("="*60)
        
        agents = execution_plan['execution_plan']['agents']
        
        print(f"\n图片: {execution_plan['image_path']}")
        print(f"查询: {execution_plan['query']}")
        print(f"执行流程: {execution_plan['execution_plan']['execution_flow']}")
        
        print(f"\n{'─'*60}")
        print("Agent 调用链:")
        print(f"{'─'*60}")
        
        for agent in agents:
            print(f"\n{agent['order']}. {agent['name']}")
            print(f"   描述: {agent['description']}")
            print(f"   任务提示:")
            
            # 打印 task prompt（缩进）
            for line in agent['task_prompt'].split('\n'):
                print(f"      {line}")
        
        print(f"\n{'─'*60}")
        print("元数据:")
        print(f"{'─'*60}")
        metadata = execution_plan['metadata']
        print(f"  Phi3.5 推理: {metadata.get('phi35_reasoning', 'N/A')}")
        print(f"  验证后的文字类型: {metadata.get('text_type_verified', 'unknown')}")
        print(f"  分类置信度: {metadata.get('classification_confidence', 0.0):.3f}")
        
        agreement = metadata.get('agreement')
        if agreement is not None:
            print(f"  Phi3.5 与分类器一致: {'✓' if agreement else '✗'}")
        
        print(f"\n规划耗时: {execution_plan.get('planning_time', 0):.2f}秒")
        
        # 如果有执行结果，也打印出来
        if 'execution_results' in execution_plan:
            exec_results = execution_plan['execution_results']
            print(f"执行耗时: {exec_results['total_execution_time']:.2f}秒")
            print(f"总耗时: {execution_plan.get('total_time', 0):.2f}秒")
            
            print(f"\n{'─'*60}")
            print("执行结果:")
            print(f"{'─'*60}")
            print(f"\nOCR 识别结果:")
            print(exec_results['final_output'])
        
        print("\n" + "="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Framework Orchestrator")
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--query', type=str, default='识别图片中的文字', help='用户查询')
    parser.add_argument('--output', type=str, default='execution_plan.json', help='输出文件路径')
    parser.add_argument('--phi35-model', type=str, default='models/phi-3_5_vision', 
                        help='Phi3.5-Vision 模型路径')
    parser.add_argument('--classifier-ckpt', type=str, default='checkpoints/printed_vs_hand_best.pth',
                        help='分类器模型路径')
    
    args = parser.parse_args()
    
    # 创建编排器
    orchestrator = MultiAgentOrchestrator(
        phi35_model_path=args.phi35_model,
        classifier_ckpt_path=args.classifier_ckpt
    )
    
    # 运行规划流程
    execution_plan = orchestrator.run(
        image_path=args.image,
        query=args.query,
        output_path=args.output
    )
    
    print("\n✓ 规划流程完成！")
    print(f"执行计划已生成: {args.output}")


if __name__ == "__main__":
    main()

