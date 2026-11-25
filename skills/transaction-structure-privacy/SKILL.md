---
name: transaction-structure-privacy
description: Guides the creation of cryptocurrency transactions with proper value handling (combining/splitting) and privacy considerations. Use when designing transaction structures, implementing payment systems, or ensuring pseudonymity in blockchain-based transactions.
---

# Transaction Structure and Privacy

## When to Use This Skill
Use this skill when:
- Creating or designing cryptocurrency transactions
- Implementing value transfer mechanisms in blockchain systems
- Designing privacy-preserving payment protocols
- Analyzing transaction privacy implications
- Implementing key management for transaction anonymity

## Value Handling Procedures

### Input Structure
Transactions should follow these input patterns:
- **Single input**: Use one input from a larger previous transaction when sufficient
- **Multiple inputs**: Combine smaller amounts from multiple previous transactions when needed

### Output Structure
Limit transaction outputs to at most two:
1. **Payment output**: Transfers value to the recipient
2. **Change output**: Returns any remaining value back to the sender (if applicable)

### Fan-Out Handling
- Transactions can depend on multiple previous transactions
- Those transactions can depend on many more
- Never need to extract a complete standalone copy of transaction history
- The dependency graph can grow arbitrarily complex

## Privacy Model Implementation

### Public Visibility Model
Unlike traditional banking (which limits access to involved parties), this model:
- Announces all transactions publicly
- Allows anyone to see value transfers between addresses
- Maintains privacy through anonymous public keys
- Operates like a stock exchange tape—visible but not personally identifiable

### Key Rotation for Privacy
To prevent transaction linkage:
- Generate a new key pair for each transaction
- Never reuse public keys across different transactions
- Keep public keys anonymous and unlinked to identities

## Privacy Constraints and Risks

### Multi-Input Linking Risk
**Critical Constraint**: Multi-input transactions necessarily reveal that their inputs were owned by the same owner.

**Implications**:
- If one key's owner is revealed, all other inputs in that transaction can be linked to the same owner
- This linking can cascade to reveal other transactions belonging to the same owner
- Some privacy loss is unavoidable when combining multiple inputs

**Mitigation**:
- Be aware of this inherent limitation
- Consider privacy implications when deciding to combine inputs
- Balance transaction efficiency against privacy needs