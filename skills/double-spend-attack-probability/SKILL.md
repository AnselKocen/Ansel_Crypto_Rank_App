---
name: double-spend-attack-probability
description: Use this skill when calculating the security risk of Bitcoin transactions, determining how many confirmations are needed for a transaction, or analyzing the probability of an attacker successfully double-spending by catching up from a deficit.
---

# Double-Spend Attack Probability

## When to Use
- Calculating transaction security based on confirmations
- Determining how many blocks to wait before accepting a payment
- Analyzing the risk of double-spend attacks
- Evaluating blockchain security parameters
- Studying probability theory in consensus systems

## Core Problem

Calculate the probability that an attacker, who is behind by z blocks, can catch up and override the honest chain to double-spend a transaction.

## Key Parameters

- **p**: Probability an honest node finds the next block
- **q**: Probability the attacker finds the next block
- **z**: Number of blocks the attacker is behind (number of confirmations)
- **Assumption**: p > q (honest majority has more computing power)

## Calculation Approach

1. **Model the race**: Characterize as a Binomial Random Walk
   - Success event: Honest chain extends (+1 lead)
   - Failure event: Attacker's chain extends (-1 lead)

2. **Gambler's Ruin**: Calculate probability of catching up from deficit
   - If p ≤ q: Probability is 1 (attacker will eventually catch up)
   - If p > q: Probability is (q/p)^z

3. **Poisson Distribution**: Account for waiting time (z confirmations)
   - Calculate expected value λ = z × (q/p)
   - Attacker's potential progress follows Poisson distribution

4. **Final probability**: Compute using the summation formula (see references)

## Interpretation

- The probability drops off exponentially as z (confirmations) increases
- Example: If q=0.1 (attacker has 10% hash power) and z=5, P ≈ 0.0009
- More confirmations = exponentially higher security

## Security Best Practices

1. **For small transactions**: Wait for fewer confirmations (e.g., 1-3)
2. **For large transactions**: Wait for more confirmations (e.g., 6+)
3. **Generate new key pair**: Receiver should generate a new key pair before signing to prevent pre-computation attacks

## Key Variables

- **p**: Probability honest node finds next block (float)
- **q**: Probability attacker finds next block (float)
- **z**: Number of blocks attacker is behind (integer)
- **λ**: Expected value of attacker's potential progress (float)
- **k**: Variable representing attacker's progress amount (integer)

## Output

A probability value between 0.0 and 1.0 representing the likelihood the attacker successfully overrides the transaction. Lower values indicate higher security.

## References

For detailed mathematical formulas and implementation, see `references/probability-calculation.md`.