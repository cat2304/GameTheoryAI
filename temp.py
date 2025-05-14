#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用官方RF-DETR包进行扑克牌检测
"""

print("正在导入官方RF-DETR库...")
from rfdetr import RFDETRBase

# 测试图像路径
TEST_IMAGE = "/Users/mac/ai/adb/test11.png"

# 1. 创建模型实例并指定预训练权重
print("创建模型并加载预训练权重...")
model = RFDETRBase(pretrain_weights="rf-detr-base.pth")

# 2. 执行推理
print(f"对图像进行推理: {TEST_IMAGE}")
detections = model.predict(TEST_IMAGE)

# 3. 输出结果
print("\n检测结果:")
for i, det in enumerate(detections):
    print(f"  {i+1}. 类别: {det['class']}, 置信度: {det['confidence']:.2f}")
    print(f"     位置: {det['box']}")

print("\n推理完成!")