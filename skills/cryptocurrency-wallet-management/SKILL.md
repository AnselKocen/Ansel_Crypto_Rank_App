---
name: cryptocurrency-wallet-management
description: Use when selecting storage methods for cryptocurrency holdings or securing existing wallets. Covers choosing between exchange wallets, hot wallets, cold storage, and paper wallets based on trading frequency and asset value, plus mandatory security protocols including backups, updates, and 2FA.
---

# Cryptocurrency Wallet Management

## When to Use
- Holding cryptocurrency and need to select storage method
- Setting up a new wallet or exchange account
- Securing existing cryptocurrency holdings

## Prerequisites
- Ownership of cryptocurrency
- Understanding of public/private keys

## Wallet Selection Decision Tree

### Evaluate Trading Frequency and Asset Value

| Trading Frequency | Asset Value | Recommended Wallet Type |
|-------------------|-------------|------------------------|
| Active / Short-term | Any | Exchange Wallet |
| Medium-term | Moderate | Online Wallet (Hot Wallet) |
| Long-term | >$25,000 | Offline Wallet (Cold Storage) |
| Any | Lowest Budget | Paper Wallet |

### Wallet Type Details

#### Exchange Wallet
- **Pros**: Easiest for trading money where it is kept
- **Cons**: Unregulated entity; highest risk of hacks; worst place to keep money long-term
- **Security**: Enable 2FA, prefer Google Authenticator over SMS, check if exchange uses cold storage

#### Online Wallet (Hot Wallet)
- **Pros**: More difficult to hack than an exchange
- **Cons**: Still online, so can be hacked. Requires managing multiple wallets for different coins
- **Types**:
  - Internet-connected software wallet (e.g., MetaMask): A traditional locally created MetaMask wallet encrypts its recovery phrase and private keys on the user's device; an internet connection does not mean its keys are stored publicly online
  - Desktop/Mobile Wallet (e.g., Exodus): Stores private keys on PC/Mobile. Syncs desktop and phone for backup

#### Offline Wallet (Cold Storage)
- **Threshold**: Recommended for coins worth more than $25,000
- **Mechanism**: Private key stored on a special USB stick (off-grid)
- **Pros**: Almost impossible to hack when offline
- **Cons**: A lost or damaged device does not by itself mean permanent loss: a valid wallet backup/recovery phrase can restore access on a compatible replacement. Losing every usable device and recovery backup can cause permanent loss of access
- **Examples**: Trezor (original, expensive), Ledger Nano S (popular, cheaper), KeepKey (larger screen)

#### Paper Wallet
- **Mechanism**: Literally write down the private key on paper
- **Risk**: If the only usable private-key copy is destroyed and there is no valid backup, access can be lost permanently. Protect any backups against both loss and disclosure

## Mandatory Security Protocols

Apply these rules regardless of wallet type:

### 1. Backup Wallet
Always back up your wallet to ensure recovery if the device is lost or damaged.

### 2. Software Updates
Keep wallet software up to date if using software wallets. This patches security vulnerabilities.

### 3. Enable Extra Security
- **Two-Factor Authentication (2FA)**: Enable it immediately
- **Authenticator App Preference**: Use 'Google Authenticator' where possible instead of text messaging (SMS)
- **Reasoning**: SMS is vulnerable to phone cloning attacks

## Risk Factors
- Hacking (highest for exchange wallets)
- Physical loss (cold storage, paper wallets)
- Device viruses (desktop/mobile wallets)

## Verified clarifications (2026-11-25)

- Device recovery conditions: [Trezor wallet backup and recovery](https://trezor.io/support/troubleshooting/trezor-suite-issues/troubleshoot-wallet-backup-and-recovery-problems).
- Traditional MetaMask local key storage: [MetaMask FAQ](https://metamask.com/faqs). Other account and recovery options can differ; consult the product's current documentation.
- The source's value thresholds and product comparisons are historical educational examples, not universal safety guarantees or current purchase recommendations.
