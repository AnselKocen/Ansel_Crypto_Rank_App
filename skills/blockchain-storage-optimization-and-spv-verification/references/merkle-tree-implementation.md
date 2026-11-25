# Merkle Tree Implementation Details

## Structure

A Merkle Tree is a binary tree where:
- Leaf nodes contain hashes of individual transactions
- Non-leaf nodes contain hashes of their children
- The root node (Merkle root) represents all transactions in the block

## Merkle Branch Construction

To create a Merkle branch for a specific transaction:

1. Start with the transaction hash at the leaf level
2. For each level up to the root:
   - Include the sibling node's hash
   - Hash the combined pair (left + right)
3. The final hash should match the Merkle root

## Verification Algorithm

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

## Storage Optimization Example

### Before Pruning
- Block with 2000 transactions
- Each transaction: ~250 bytes
- Total: 2000 × 250 = 500,000 bytes (~500KB)

### After Pruning
- Block header: 80 bytes
- Merkle root: 32 bytes
- Total: 112 bytes

**Savings**: ~99.98% reduction per pruned block