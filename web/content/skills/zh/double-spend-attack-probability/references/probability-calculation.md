# 双花攻击概率：详细计算

## 数学基础

### 等待时间的泊松分布

收款人等待 z 个区块时，攻击者可能取得的进展服从泊松分布，其期望值为：

$$\lambda = z \frac{q}{p}$$

### 概率公式

在给定泊松分布的情况下，攻击者仍然能够追上的概率 P 为：

$$P = 1 - \sum_{k=0}^{z} \frac{\lambda^{k} e^{-\lambda}}{k!} \left(1 - \frac{q}{p}\right)^{(z-k)}$$

其中：
- k 表示攻击者可能已经取得的进展数量
- 求和范围为 k=0 到 k=z

## 算法实现（C 风格）

```c
double CalculateAttackProbability(double p, double q, int z) {
    // Calculate lambda (expected value of attacker's progress)
    double lambda = z * (q / p);
    
    // Initialize sum to 1 (we'll subtract terms)
    double sum = 1.0;
    
    // Loop k from 0 to z
    for (int k = 0; k <= z; k++) {
        // Calculate Poisson term: e^(-λ) × (λ^k / k!)
        double poisson = exp(-lambda) * pow(lambda, k) / factorial(k);
        
        // Calculate catch-up term: (1 - (q/p)^(z-k))
        double catch_up = 1.0 - pow(q / p, z - k);
        
        // Update sum
        sum = sum - (poisson * catch_up);
    }
    
    return sum;
}

// Helper function for factorial
long factorial(int n) {
    if (n <= 1) return 1;
    long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
```

## 计算示例

### 示例 1：攻击者拥有 10% 算力，等待 5 次确认
- p = 0.9，表示诚实节点
- q = 0.1，表示攻击者
- z = 5，表示确认次数
- λ = 5 × (0.1/0.9) ≈ 0.556
- P ≈ 0.0009，即成功概率为 0.09%

### 示例 2：攻击者拥有 30% 算力，等待 6 次确认
- p = 0.7
- q = 0.3
- z = 6
- λ = 6 × (0.3/0.7) ≈ 2.571
- P ≈ 0.041，即成功概率为 4.1%

### 示例 3：攻击者拥有 45% 算力，等待 10 次确认
- p = 0.55
- q = 0.45
- z = 10
- λ = 10 × (0.45/0.55) ≈ 8.182
- P ≈ 0.132，即成功概率为 13.2%

## 边界情况

### p ≤ q 时
如果攻击者算力等于或高于诚实节点，即 p ≤ q，最终追上的概率趋近于 1.0。此时：
- 系统的安全假设不再成立
- 无论等待多少次确认，都无法提供有意义的安全保障
- 网络已受到攻击者控制

### z = 0 时
没有确认时，概率取决于相对算力：
- 如果 q < p：概率为 q/p
- 如果 q ≥ p：概率趋近于 1.0

## 数学依据

1. **二项随机游走**：对诚实链与攻击链之间的竞赛建模
2. **赌徒破产问题**：研究从落后状态追上的经典概率问题
3. **泊松分布**：对固定时间间隔内的事件次数，即找到的区块数建模

## 限制条件

- 节点不会接受无效交易，也不会接受包含无效交易的区块
- 攻击者只能更改自己的交易
- 假设 p > q，即诚实方占多数，拥有更高算力
- 收款人在签名前生成新密钥对，以防范预计算攻击
