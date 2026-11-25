# 默克尔树实现细节

## 结构

默克尔树是一种二叉树，其中：
- 叶节点包含单笔交易的哈希
- 非叶节点包含其子节点的哈希
- 根节点（默克尔根）代表区块中的全部交易

## 构造默克尔分支

为某笔特定交易创建默克尔分支时：

1. 从叶层的交易哈希开始
2. 逐层向上，直到根节点：
   - 纳入兄弟节点的哈希
   - 对组合后的哈希对（左 + 右）计算哈希
3. 最终哈希应与默克尔根一致

## 验证算法

```
function verifyMerkleBranch(txHash, merkleBranch, merkleRoot):
    currentHash = txHash
    for siblingHash in merkleBranch:
        if currentHash < siblingHash:
            currentHash = hash(currentHash + siblingHash)
        else:
            currentHash = hash(siblingHash + currentHash)
    return currentHash == merkleRoot
```

## 存储优化示例

### 裁剪前
- 一个包含 2000 笔交易的区块
- 每笔交易：约 250 字节
- 总计：2000 × 250 = 500,000 字节（约 500KB）

### 裁剪后
- 区块头：80 字节
- 默克尔根：32 字节
- 总计：112 字节

**节省量**：每个被裁剪区块的存储需求减少约 99.98%
