# Bitcoin Consensus Security Model

## Theoretical Foundation

### Computational Impracticality
The security of Bitcoin's consensus relies on the principle that it is "computationally impractical for an attacker to change" the transaction history, provided that "honest nodes control a majority of CPU power."

### Unstructured Simplicity
The network operates on principles of minimal coordination:
- Nodes work simultaneously with little coordination
- Nodes do not need to be identified
- Messages are not routed to any particular place
- Delivery is on a "best effort basis"

## Ownership and Double-Spending

### Digital Signatures Framework
- Coins are made from digital signatures
- Provides strong control of ownership
- Eliminates need for trusted third parties

### Double-Spending Prevention
- Peer-to-peer network records public history
- Proof-of-work makes history tamper-resistant
- Majority honest CPU power ensures integrity

## Consensus as CPU Voting

### Voting Mechanism
Nodes express consensus through their CPU power:
- **Accept**: Work on extending valid blocks
- **Reject**: Refuse to work on invalid blocks

### Rule Enforcement
The consensus mechanism can enforce any needed rules and incentives through the same voting process.

## Security Properties

### Immutability
- Transaction history becomes more secure over time
- Each new block adds to the cumulative proof-of-work
- Changing older blocks requires redoing all subsequent work

### Fault Tolerance
- System tolerates message drops
- Nodes can recover missing blocks
- Network latency handled through fork resolution

### Attack Resistance
- Attacker needs majority CPU power to succeed
- Honest majority ensures network integrity
- Cost of attack increases with network growth