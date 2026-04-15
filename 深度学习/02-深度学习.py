# ==============================================
#  📘 Python 学习笔记：深度神经网络基础
# ==============================================

# ------------------------------
# 第一步：理解 Tensor（张量）
# ------------------------------
import torch

# 创造数据
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

# 张量 = 可以自动求导的数组
print(x)
print(y)


# ------------------------------
# 第二步：定义模型（必须彻底懂）
# ------------------------------

class LinearModel(torch.nn.Module):
    # torch.nn.Module 是 PyTorch 所有模型的父类
    # 提供了参数管理、前向传播、求导等功能

    def __init__(self):
        super().__init__()  # 固定写法，不用深究

        # 线性层：输入1个，输出1个
        self.linear = torch.nn.Linear(1, 1)

        # Linear(1,1) = 一个最简单的直线拟合模型
        # 一次函数 y = kx + b，k和b自动学习

    def forward(self, x):
        # 前向传播固定函数名，不能改
        # 功能：输入x → 经过网络 → 输出预测值

        # 前向传播：给输入，返回预测值
        # 简单理解就是输入x，输出y
        # 这个self.linear已经被定义成一个计算工具
        # 输出 = self.linear(输入)
        return self.linear(x)


# ------------------------------
# 第三步：损失函数 & 优化器（核心中的核心）
# ------------------------------

# 生成一个模型对象
model = LinearModel()

# 损失函数：衡量预测值和真实值差多远
criterion = torch.nn.MSELoss()  # 中文名：均方误差损失

# MSELoss 是一个类（class），就像 Linear、Module 一样
# torch.nn.MSELoss   # 这是类（图纸）
# torch.nn.MSELoss() # 这是实例（造出来的计算器）
# criterion 就是一个：均方误差损失计算器
# 使用：loss = criterion(预测值y_pred, 真实值y_true)

# 优化器：负责更新 w 和 b，让 loss 变小
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 中文名：随机梯度下降

# 创建了一个 SGD 优化器实例
# 告诉它：要优化的参数是 model.parameters()（所有 w、b）
# 学习率：lr=0.01
# 新w = 旧w - 学习率 × 梯度

# 透彻解释：
# loss 越小，模型越准
# optimizer 就是 “改错的工具”
# lr=0.01 是学习率：每次改一点点，不能太大


# ------------------------------
# 第四步：训练循环（最重要！）
# ------------------------------

for epoch in range(100):
    # 1. 前向传播：给 x，得到预测 y_pred
    y_pred = model(x.view(-1, 1))  # 调整形状，把一维变成二维（列向量）

    # 因为：
    # nn.Linear(1, 1)
    # 要求输入必须是 2 维：(样本数，特征数)
    # 正确：(4, 1) → 4 个样本，每个样本 1 个数
    # 错误：(4) → 1 维，模型不认识

    # 2. 计算损失
    loss = criterion(y_pred, y.view(-1, 1))

    # 3. 清空梯度（必须写）
    optimizer.zero_grad()

    # 作用：清空上一轮的梯度
    # 必须写，否则梯度会累加

    # 4. 反向传播：求 w 和 b 怎么改
    loss.backward()

    # 反向传播 = 从损失往回算，把每个参数的梯度求出来
    # 梯度是 loss 函数对 w 求导
    # 导数值大于0则loss随w增大而增大，反之减小
    # 所以梯度大于0，就减小w，梯度小于0，就增大w
    # 公式：w = w - lr × grad

    # 5. 更新 w 和 b
    optimizer.step()

    # 直接执行公式 w = w - lr × w.grad，参数内部获取

    # 打印看效果
    if epoch % 10 == 0:
        print(f"epoch: {epoch:3d} | loss: {loss.item():.6f}")


# 训练完看最终参数
print("\n=== 训练完成 ===")
print("最终权重 w =", model.linear.weight.item())
print("最终偏置 b =", model.linear.bias.item())

# 测试：x=6 → 应该预测≈12
test_x = torch.tensor([6.0])
pred_y = model(test_x.view(-1, 1))
print("x=6 → 预测 y =", pred_y.item())