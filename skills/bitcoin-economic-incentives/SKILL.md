---
name: bitcoin-economic-incentives
description: Use this skill when analyzing Bitcoin's economic structure, understanding mining rewards and transaction fees, evaluating game-theoretic aspects of network security, or studying attacker motivations and defenses.
---

# Bitcoin Economic Incentive Structure

## When to Use
- Understanding how miners are rewarded
- Analyzing transaction fee economics
- Evaluating the game theory of network security
- Studying attacker motivations and deterrence
- Planning for Bitcoin's inflation transition

## Core Principle

Bitcoin uses economic incentives to encourage honest participation and make attacks economically irrational.

## Coin Creation (Coinbase)

1. **First transaction in a block**: A special transaction that creates new coins
2. **Ownership**: New coins are owned by the creator of the block
3. **Purpose**:
   - Incentivizes nodes to support the network
   - Provides initial coin distribution without central authority
4. **Analogy**: Similar to gold miners expending resources (CPU time and electricity) to add gold to circulation

## Transaction Fees

1. **Calculation**: If output value < input value, the difference is the transaction fee
2. **Allocation**: The fee is added to the incentive value of the block containing the transaction
3. **Purpose**: Encourages miners to include transactions in blocks

## Inflation Transition

1. **Current state**: New coins are created with each block (inflationary)
2. **Future state**: Once a predetermined number of coins have entered circulation:
   - Incentive transitions entirely to transaction fees
   - System becomes completely inflation-free
3. **Limit**: Total number of coins is mathematically capped

## Honesty Game Theory (Attacker Dilemma)

A greedy attacker with more CPU power than all honest nodes faces a choice:

**Option A**: Use power to defraud people by stealing back payments
- Undermines the system
- Destroys value of attacker's own wealth
- Short-term gain, long-term loss

**Option B**: Use power to generate new coins
- Follows the rules
- Yields more new coins than everyone else combined
- Preserves system value and attacker's wealth

**Rational Choice**: The attacker should find it more profitable to play by the rules (generating new coins) than to undermine the system.

## Key Variables

- **input_value**: Total value of coins entering the transaction
- **output_value**: Total value of coins leaving the transaction
- **transaction_fee**: Difference between input and output value (input_value - output_value)

## Output

Block creator receives new coins (coinbase) + transaction fees from all transactions in the block.

## Security Assumption

This incentive structure assumes an honest node majority and relies on the economic rationality of participants.