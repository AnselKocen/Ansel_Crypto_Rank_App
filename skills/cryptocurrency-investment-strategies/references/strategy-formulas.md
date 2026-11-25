# Value Cost Average™ Formula Details

## Variables
- **sideways_drop_threshold**: -20% (Trigger drop for sideways market)
- **bear_drop_threshold**: -30% (Trigger drop for bear market)
- **subsequent_drop_threshold**: -10% (Additional drop required for next buy order)
- **investment_increment**: 25% (Max increase in position size per buy)

## Example Calculation

Recent High: $10,000

### Sideways Market Scenario
- Buy 1: $8,000 (-20% from high) - Invest $1,000
- Buy 2: $7,000 (-30% from high, -10% from Buy 1) - Invest $1,250 (25% increase)
- Buy 3: $6,000 (-40% from high, -10% from Buy 2) - Invest $1,562.50 (25% increase)

### Bear Market Scenario
- Buy 1: $7,000 (-30% from high) - Invest $1,000
- Buy 2: $6,000 (-40% from high, -10% from Buy 1) - Invest $1,250 (25% increase)
- Buy 3: $5,000 (-50% from high, -10% from Buy 2) - Invest $1,562.50 (25% increase)