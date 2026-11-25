# Double-Spend Attack Probability - Detailed Calculation

## Mathematical Foundation

### Poisson Distribution for Waiting Time

When a recipient waits for z blocks, the attacker's potential progress follows a Poisson distribution with expected value:

$$\lambda = z \frac{q}{p}$$

### Probability Formula

The probability P that the attacker could still catch up given the Poisson distribution:

$$P = 1 - \sum_{k=0}^{z} \frac{\lambda^{k} e^{-\lambda}}{k!} \left(1 - \frac{q}{p}\right)^{(z-k)}$$

Where:
- k is the amount of potential progress the attacker has made
- The summation runs from k=0 to k=z

## Algorithm Implementation (C-style)

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

## Example Calculations

### Example 1: Attacker with 10% hash power, 5 confirmations
- p = 0.9 (honest nodes)
- q = 0.1 (attacker)
- z = 5 (confirmations)
- λ = 5 × (0.1/0.9) ≈ 0.556
- P ≈ 0.0009 (0.09% chance of success)

### Example 2: Attacker with 30% hash power, 6 confirmations
- p = 0.7
- q = 0.3
- z = 6
- λ = 6 × (0.3/0.7) ≈ 2.571
- P ≈ 0.041 (4.1% chance of success)

### Example 3: Attacker with 45% hash power, 10 confirmations
- p = 0.55
- q = 0.45
- z = 10
- λ = 10 × (0.45/0.55) ≈ 8.182
- P ≈ 0.132 (13.2% chance of success)

## Edge Cases

### When p ≤ q
If the attacker has equal or more hash power than honest nodes (p ≤ q), the probability of eventually catching up approaches 1.0. In this case:
- The system's security assumptions are violated
- No number of confirmations provides meaningful security
- The network is compromised

### When z = 0
With zero confirmations, the probability depends on the relative hash power:
- If q < p: Probability is q/p
- If q ≥ p: Probability approaches 1.0

## Mathematical Basis

1. **Binomial Random Walk**: Models the race between honest and attacker chains
2. **Gambler's Ruin Problem**: Classic probability problem about catching up from a deficit
3. **Poisson Distribution**: Models the number of events (blocks found) in a fixed interval

## Constraints

- Nodes will not accept invalid transactions or blocks containing them
- Attacker can only change their own transactions
- Assumes p > q (honest majority has more computing power)
- Receiver generates new key pair before signing to prevent pre-computation