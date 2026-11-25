---
name: bitcoin-proof-of-work-consensus
description: Implement and understand Bitcoin's proof-of-work consensus mechanism for distributed ledgers. Use when building or analyzing peer-to-peer electronic transaction systems that require trustless consensus, preventing double-spending, and maintaining network integrity without central authority.
---

# Bitcoin Proof-of-Work Consensus

Implement Bitcoin's consensus mechanism that enables trustless electronic payments through proof-of-work and CPU-based voting.

## When to Use

Use this skill when:
- Implementing a peer-to-peer electronic transaction network
- Designing a distributed ledger without central authority
- Preventing double-spending in digital currency systems
- Understanding how Bitcoin nodes reach consensus
- Analyzing network security and integrity in blockchain systems

## Prerequisites

- Digital signatures framework for ownership control
- Proof-of-work algorithm implementation
- Majority of honest CPU power in the network
- Nodes connected to peer-to-peer network

## Network Operation

### Trigger Conditions
Execute network steps when:
- A new transaction is received
- A proof-of-work solution is found

### Transaction Broadcast
1. Broadcast new transactions to all nodes
2. Transaction propagation does not require immediate reach to all nodes
3. Reaching many nodes is sufficient for eventual inclusion

### Block Processing Workflow

1. **Collect Transactions**
   - Each node collects new transactions into a block
   - Prepare block for proof-of-work computation

2. **Find Proof-of-Work**
   - Work on finding a difficult proof-of-work for the block
   - Use CPU power to solve the cryptographic puzzle

3. **Broadcast Block**
   - When proof-of-work is found, broadcast block to all nodes
   - Block broadcasts tolerate dropped messages

4. **Validate Block**
   - Accept block only if all transactions are valid
   - Verify no transactions are double-spent
   - Reject invalid blocks

5. **Extend Chain**
   - Express acceptance by working on the next block
   - Use accepted block's hash as previous hash
   - This "votes with CPU power" for the chain

## Consensus Rules

### Longest Chain Rule
- Always consider the longest chain to be correct
- "Longest" = greatest cumulative proof-of-work
- Continuously work on extending the longest chain

### Fork Resolution

When two nodes broadcast different blocks simultaneously:

1. **Save Both Branches**
   - Work on the first received block
   - Save the other branch in case it becomes longer

2. **Switch on Longer Chain**
   - When next proof-of-work extends one branch
   - Nodes on shorter branch switch to longer one

3. **Recover Missing Blocks**
   - If a node misses a block, request it upon receiving next block
   - Detect gaps by comparing previous hashes

## Node Participation

### Leaving and Rejoining
- Nodes can leave and rejoin at will
- Upon rejoining, accept proof-of-work chain as proof of events
- No need for full history synchronization during absence

### Consensus Voting
- Vote with CPU power by extending valid chains
- Reject invalid blocks by refusing to work on them
- Use consensus mechanism to enforce rules and incentives

## Security Constraints

- Attacker must not control majority of CPU power
- Network latency may cause temporary forks
- Messages may be dropped (system is fault-tolerant)
- History becomes immutable as proof-of-work accumulates

## Output

Network reaches consensus on a valid block and extends the blockchain with a secure, immutable public history of transactions.