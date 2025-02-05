# 睿抗机器人开发者大赛(RAICOM)[算法调优] 国家一等奖方案 遥感图像分类

## 项目描述
本项目使用PyTorch框架和EfficientNet-B7模型，实现遥感图像的五分类任务（飞机、桥梁、宫殿、船舶、体育场）。包含完整的数据预处理、数据增强、模型训练和保存流程。

## 数据集准备
### 数据集结构
```
root/
├── datasets/
│   ├── airplane/
│   ├── bridge/
│   ├── palace/
│   ├── ship/
│   └── stadium/
└── train.ipynb
```

### 数据划分
- 自动分割训练集/测试集（90%/10%）
- 保持类别分布均衡
- 随机种子固定（random_state=42）

## 环境要求
### 依赖库
```bash
pip install torch torchvision scikit-learn pillow tqdm
```

### 硬件建议
- NVIDIA GPU (推荐使用CUDA)
- 至少16GB内存（训练时）

## 数据增强策略
| 增强类型                | 参数设置                     |
|-----------------------|---------------------------|
| 随机裁剪                | 224x224 (缩放0.8-1.0)       |
| 颜色抖动                | 亮度/对比度/饱和度=0.4，色相=0.2 |
| 仿射变换                | 平移10%，缩放0.8-1.2，剪切10°  |
| 高斯模糊                | 内核5x5，σ=0.1-2.0          |
| 随机擦除                | 概率50%，面积2%-40%         |

## 模型配置
### 网络架构
- 基于EfficientNet-B7预训练模型
- 修改最后一层全连接层：
  ```python
  model.classifier[1] = nn.Linear(num_ftrs, 5)
  ```

### 训练参数
| 参数                 | 设置        |
|---------------------|-----------|
| 损失函数              | 交叉熵 + 标签平滑（α=0.9） |
| 优化器               | Adam (lr=0.001) |
| 学习率调度            | StepLR (step=7, γ=0.1) |
| Batch Size          | 32        |
| Epochs              | 25        |

## 使用方法
1. 准备数据集（按上述结构组织）
2. 运行Jupyter训练脚本：
3. 训练完成后会生成模型文件：

## 注意事项
1. 首次运行会自动创建训练集/测试集目录
2. 修改`data_dir`路径需同步调整数据集结构
3. 随机增强参数可能导致不同设备间结果微小差异
4. 完整训练需要约5GB GPU显存

## 参考文献
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [EfficientNet论文](https://arxiv.org/abs/1905.11946)
- [Label Smoothing研究](https://arxiv.org/abs/1512.00567)
