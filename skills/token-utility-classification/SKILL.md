---
name: token-utility-classification
description: Classify the utility and function of tokens issued in ICOs, STOs, or IEOs based on the issuing company's stated goals. Use this skill when analyzing cryptocurrency tokens to determine if they function as a product/medium of exchange, profit-sharing instrument (like shares), or debt instrument (like corporate bonds).
---

# Token Utility Classification

## When to Use
Use this skill when:
- Analyzing tokens issued in Initial Coin Offerings (ICOs)
- Evaluating tokens from Security Token Offerings (STOs)
- Reviewing tokens from Initial Exchange Offerings (IEOs)
- Determining the regulatory classification or financial nature of a blockchain token

## Prerequisites
- A startup or company issuing a token
- Defined goals and documentation from the issuing company
- Understanding of the token's intended use case

## Classification Procedure

### Step 1: Determine if Token Represents Company Product
Check if the token functions as:
- A medium of exchange within the project ecosystem
- A redeemable unit for specific amounts of the company's products or services
- A trading instrument within the project's platform

**If yes**: Classify as **Product Token**

### Step 2: Determine if Token Represents Profit Sharing Rights
Check if the token functions as:
- An ownership stake in the company (analogous to traditional shares)
- A right to receive a percentage of company profits
- A voting or governance right over company decisions

**If yes**: Classify as **Profit Sharing Token**

### Step 3: Determine if Token Represents Corporate Bond
Check if the token functions as:
- A loan to the issuing company
- An entitlement to receive interest payments
- A debt instrument with a pre-set interest rate

**If yes**: Classify as **Corporate Bond Token**

## Important Notes
- Token roles are defined by the company's stated goals, not necessarily market perception
- This classification does not apply to traditional financial instruments unless they are tokenized
- A token may exhibit characteristics of multiple categories; prioritize the primary function based on company documentation

## Output Format
Provide the classification result in the following format:

```
The token represents [Role Name] based on the company's goals
```

Where [Role Name] is one of:
- **Product Token** (medium of exchange)
- **Profit Sharing Token** (ownership instrument)
- **Corporate Bond Token** (debt instrument)