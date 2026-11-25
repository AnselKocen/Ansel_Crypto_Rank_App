---
name: bitcoin-transaction-model
description: Use this skill when working with electronic coin transfers, verifying transaction chains, understanding Bitcoin's transaction structure, or analyzing the double-spending problem and its solutions.
---

# Bitcoin Transaction Model

## When to Use
- Creating or verifying electronic coin transfers
- Understanding the chain of ownership in Bitcoin
- Analyzing the double-spending problem
- Designing transaction verification systems

## Core Concepts

An electronic coin is defined as a chain of digital signatures. Each transfer adds a new signature to the chain, establishing ownership from one owner to the next.

## Transfer Procedure

To transfer a coin to a new owner:

1. **Obtain the previous transaction hash** - The hash of the transaction that transferred the coin to the current owner
2. **Obtain the next owner's public key** - The recipient's public key to establish ownership
3. **Create digital signatures**:
   - Sign the hash of the previous transaction
   - Sign the public key of the next owner
4. **Add signatures to the coin** - Append these signatures to the end of the coin's signature chain

## Verification

The payee can verify the chain of ownership by:
1. Checking each digital signature in the chain
2. Verifying that each signature was created by the owner at that point in the chain
3. Confirming the chain is unbroken and properly linked

## The Double-Spending Problem

**Limitation**: The payee cannot verify that one of the owners did not double-spend the coin by signing an earlier transaction to a different recipient.

**Solution Requirements** (without a trusted party):
1. Transactions must be publicly announced
2. Participants need a system to agree on a single history of the order transactions were received
3. The payee needs proof that at the time of the transaction, the majority of nodes agreed it was the first received

## Key Variables

- **PreviousTransactionHash**: Hash of the transaction that transferred the coin to the current owner
- **NextOwnerPublicKey**: Public key of the recipient to establish ownership
- **DigitalSignature**: Cryptographic signature proving current owner's authorization

## Output

A valid transfer of ownership recorded in the coin's signature chain.