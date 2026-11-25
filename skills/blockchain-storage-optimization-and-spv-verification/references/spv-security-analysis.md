# SPV Security Analysis

## Attack Vectors

### 51% Attack

If an attacker controls >50% of network hash power:
1. They can create an alternative chain with fabricated transactions
2. SPV clients may accept this as the longest chain
3. The attacker can double-spend transactions

### Prevention Strategies

1. **Wait for confirmations**: More confirmations = exponentially harder to reverse
   - 6 confirmations: Requires ~6 blocks of work to override
   - Probability decreases with each additional confirmation

2. **Connect to multiple nodes**: Query several independent nodes to detect inconsistencies

3. **Alert system**: Implement peer-to-peer alerts for invalid blocks

## Confirmation Requirements

| Transaction Value | Recommended Confirmations |
|-------------------|---------------------------|
| Small (<$10)      | 0-1                       |
| Medium ($10-$1000)| 3-6                       |
| Large (>$1000)    | 6+                        |

## Memory Requirements

### Historical Growth
- 2008: 2GB RAM sufficient for headers
- Growth rate: ~1.2GB/year (Moore's Law)

### Calculation
```
headers_per_year = 6 blocks/hour × 24 hours × 365 days = 52,560
header_storage = 52,560 × 80 bytes = 4.2MB/year
```

### Practical Limits
- 10 years of headers: ~42MB
- 20 years of headers: ~84MB
- Easily fits in modern RAM constraints