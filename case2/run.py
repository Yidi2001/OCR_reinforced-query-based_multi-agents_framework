#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case2 Multi-Agent Framework - 入口脚本
Phi3.5-Vision 作为任务规划器的多智能体框架
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from case2.orchestrator import MultiAgentOrchestrator


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="Case2: Phi3.5-Vision Multi-Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  
  1. 基础使用:
     python case2/run.py --image path/to/image.jpg
  
  2. 自定义查询:
     python case2/run.py --image path/to/image.jpg --query "提取图片中的所有数字"
  
  3. 指定输出路径:
     python case2/run.py --image path/to/image.jpg --output my_plan.json
  
  4. 使用自定义模型:
     python case2/run.py --image path/to/image.jpg \\
                         --phi35-model /path/to/phi35 \\
                         --classifier checkpoints/my_classifier.pth
        """
    )
    
    parser.add_argument(
        '--image', 
        type=str, 
        required=True, 
        help='输入图片路径'
    )
    
    parser.add_argument(
        '--query', 
        type=str, 
        default='识别图片中的文字', 
        help='用户查询/任务描述（默认: "识别图片中的文字"）'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='case2_output/execution_plan.json', 
        help='输出文件路径（默认: case2_output/execution_plan.json）'
    )
    
    parser.add_argument(
        '--phi35-model', 
        type=str, 
        default='models/phi-3_5_vision', 
        help='Phi3.5-Vision 模型路径（默认: models/phi-3_5_vision）'
    )
    
    parser.add_argument(
        '--classifier-ckpt', 
        type=str, 
        default='checkpoints/printed_vs_hand_best.pth',
        help='分类器模型路径（默认: checkpoints/printed_vs_hand_best.pth）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出'
    )
    
    args = parser.parse_args()
    
    # 检查图片文件是否存在
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ 错误: 图片文件不存在 - {args.image}")
        sys.exit(1)
    
    try:
        # 创建编排器
        print("\n" + "="*70)
        print(" "*15 + "Case2: Multi-Agent Framework")
        print(" "*10 + "Phi3.5-Vision 任务规划 + 多智能体协作")
        print("="*70)
        
        orchestrator = MultiAgentOrchestrator(
            phi35_model_path=args.phi35_model,
            classifier_ckpt_path=args.classifier_ckpt
        )
        
        # 运行规划流程
        execution_plan = orchestrator.run(
            image_path=str(image_path),
            query=args.query,
            output_path=args.output
        )
        
        # 成功提示
        print("\n" + "="*70)
        print("✓ 任务规划完成！")
        print("="*70)
        print(f"\n📄 执行计划已保存到: {args.output}")
        print(f"📊 计划包含 {execution_plan['execution_plan']['total_agents']} 个 Agent")
        print(f"⏱️  规划耗时: {execution_plan.get('planning_time', 0):.2f} 秒")
        
        # 简要显示 Agent 链
        agent_names = [a['name'] for a in execution_plan['execution_plan']['agents']]
        print(f"\n🔗 Agent 调用链:")
        print(f"   {' → '.join(agent_names)}")
        
        print("\n" + "="*70)
        print("💡 提示: 执行计划已生成，可用于后续的 Agent 执行阶段")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

