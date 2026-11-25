---
name: proof-of-work-consensus
description: Use this skill when implementing blockchain consensus mechanisms, understanding how nodes reach agreement on transaction history, studying the mining process, or analyzing chain security and immutability.
---

# Proof-of-Work and Distributed Consensus

## When to Use
- Implementing or analyzing blockchain consensus
- Understanding how nodes agree on transaction history
- Studying the mining process and block validation
- Analyzing chain security and attack resistance
- Adjusting mining difficulty

## Core Concept

Proof-of-Work (PoW) is a mechanism that requires computational effort to add blocks to the blockchain, making it expensive to modify past transactions while keeping verification cheap.

## Proof-of-Work Implementation

1. **Find a valid nonce**:
   - Increment a nonce value in the block
   - Hash the block with the nonce using SHA-256
   - Continue until the hash has the required number of leading zero bits

2. **Work characteristics**:
   - Average work required is exponential in the number of zero bits
   - Verification is computationally cheap (single hash operation)

## Chain Immutability

1. **Block immutability**: Once CPU effort is expended to satisfy PoW, the block cannot be changed without redoing the work
2. **Chain reinforcement**: As later blocks are chained after it, changing a block requires redoing PoW for that block AND all subsequent blocks

## Distributed Consensus (One-CPU-One-Vote)

1. **Majority decision**: Represented by the longest chain (greatest proof-of-work effort invested)
2. **IP protection**: Prevents subversion by allocating many IPs (unlike one-IP-address-one-vote)
3. **Security condition**: If a majority of CPU power is controlled by honest nodes, the honest chain grows fastest

## Attack Difficulty

To modify a past block, an attacker must:
1. Redo the proof-of-work of the target block
2. Redo the proof-of-work of all subsequent blocks
3. Catch up with and surpass the work of honest nodes
4. The probability of catching up diminishes exponentially as more blocks are added

## Difficulty Adjustment

1. **Target**: Average number of blocks per hour
2. **Mechanism**: Moving average of recent block times
3. **Adjustment**:
   - If blocks are generated too fast → increase difficulty
   - If blocks are generated too slow → decrease difficulty

## Key Variables

- **Nonce**: Value incremented to find a valid hash
- **ZeroBitsTarget**: Required number of leading zero bits in the hash
- **BlockHash**: Result of hashing the block with the nonce

## Output

A timestamped block added to the longest chain, representing the consensus of the network.

## Security Assumption

This mechanism requires that a majority of CPU power is controlled by honest nodes for security.