# 睿抗算法调优赛道 国一方案（遥感图像分类）

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
2. 运行Jupyter训练脚本
3. 训练完成后会生成模型文件

## 注意事项
1. 首次运行会自动创建训练集/测试集目录
2. 修改`data_dir`路径需同步调整数据集结构
3. 随机增强参数可能导致不同设备间结果微小差异
4. 完整训练需要约5GB GPU显存

## 关于蒸馏
- 在比赛的过程中，由于需要在平台上跑程序并且通过接口测试，所以考虑通过蒸馏来实现小权重，节省本地上传时间
- b0蒸馏出来的效果不如b7效果好，针对比赛，性能优先，所以最后放弃蒸馏方案
- 若后续仍希望尝试蒸馏方案以提高`EfficientNet - B0`的性能，使其更接近`EfficientNet - B7`，可尝试以下改进措施：

#### （一）调整蒸馏参数
蒸馏损失函数中的温度参数`temperature`和权重参数`alpha`对蒸馏效果有重要影响。
- **温度参数`temperature`**：控制软标签的平滑程度。过高的温度会使软标签过于平滑，导致信息丢失；过低的温度则使软标签接近硬标签，无法充分发挥蒸馏的优势。
- **权重参数`alpha`**：控制硬损失（交叉熵损失）和软损失的权重分配。不合理的`alpha`设置可能无法充分利用教师模型的知识。

建议通过实验对这两个参数进行调优，找到最优值。示例代码如下：
```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    soft_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                         F.softmax(teacher_logits / temperature, dim=1),
                         reduction='batchmean') * (temperature ** 2)
    hard_loss = cross_entropy_loss(student_logits, labels)
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

#### （二）增加训练轮数
适当增加训练轮数，让学生模型有更多的时间学习教师模型的知识。由于学生模型需要学习教师模型的知识，特别是在模型架构差异较大的情况下，可能需要更多的训练轮数来收敛。示例代码如下：
```python
# 增加训练轮数
num_epochs = 50

# 训练学生模型
student_model = train_model(teacher_model, student_model, distillation_loss, optimizer, scheduler, num_epochs=num_epochs)
```

#### （三）调整学习率
学习率是影响模型训练的重要因素。尝试不同的学习率和学习率调度策略，如使用`CosineAnnealingLR`，可以提高模型的收敛速度和性能。示例代码如下：
```python
from torch.optim.lr_scheduler import CosineAnnealingLR
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

## 参考文献
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [EfficientNet论文](https://arxiv.org/abs/1905.11946)
- [Label Smoothing研究](https://arxiv.org/abs/1512.00567)
