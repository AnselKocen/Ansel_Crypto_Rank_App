---
name: bull-market-trading-strategy
description: A passive trading strategy for bull markets that involves buying on dips and taking partial profits (20%) on rallies. Use when market trend is bullish and you believe the asset will continue appreciating in value. Not suitable for assets believed to go to zero or when price stops going up.
---

# Bull Market Trading Strategy

## When to Use
- Market trend is bullish (price consistently moving upward)
- You believe the asset will continue to appreciate
- Asset is not believed to go to zero

## When NOT to Use
- Price stops going up (switch to sideways strategy)
- Asset is believed to go to zero

## Prerequisites
- Open trading account
- Belief that the asset will continue to go up

## Execution Steps

1. **Adopt Passive Approach**: Do as little as possible and allow the market to do the work

2. **Entry**: Buy on the dips (pull-backs) as the price fluctuates upward

3. **Profit Taking**: As the price goes back up, sell 20% of your profit

4. **Position Holding**: Hold on to the remaining 80% (and principal) for when the 'big move' comes

5. **Repetition**: Rinse and repeat this cycle of buying dips and selling partial profits

6. **Exit Condition**: If the asset stops going up, stop this strategy and switch to a sideways/bear market strategy

## Key Variables
- `market_trend`: Current state of the market (Bull, Sideways, Bear)
- `profit_sell_percentage`: Fixed percentage of profit to realize (default: 20%)