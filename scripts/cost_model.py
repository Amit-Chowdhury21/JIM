"""
Cost-Aware Evaluation Model (V2.0 Overhaul)
===========================================
Simulates realistic exchange trading costs (Spread, Slippage, Commission)
to ensure models are profitable after broker fees.
"""

class CostModel:
    def __init__(self, spread_bps=2.0, slippage_bps=1.0, commission_bps=0.5):
        """
        Gold typically trades with:
        - 0.2 to 0.5 pips spread (2-5 bps)
        - 0.1 to 0.3 pips slippage
        - Small commission per lot
        """
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps

    def estimate_entry_cost(self, price, side="LONG"):
        # Entry pays half the spread, full commission, and incurs slippage
        cost_bps = (self.spread_bps / 2.0) + self.commission_bps + self.slippage_bps
        cost_pct = cost_bps / 10000.0
        return price * cost_pct

    def estimate_exit_cost(self, price, side="LONG"):
        # Exit pays half the spread, full commission, and incurs slippage
        cost_bps = (self.spread_bps / 2.0) + self.commission_bps + self.slippage_bps
        cost_pct = cost_bps / 10000.0
        return price * cost_pct

    def total_trade_cost(self, entry_price, exit_price, side="LONG", size=1.0):
        entry_fee = self.estimate_entry_cost(entry_price, side)
        exit_fee = self.estimate_exit_cost(exit_price, side)
        return (entry_fee + exit_fee) * size

class TradeEvaluator:
    def __init__(self, cost_model: CostModel):
        self.cost_model = cost_model

    def evaluate_trade(self, entry_price, exit_price, side="LONG"):
        """Returns the net PnL percentage after all costs."""
        raw_pnl = (exit_price - entry_price) / entry_price
        if side == "SHORT":
            raw_pnl = -raw_pnl
            
        cost_pnl = self.cost_model.total_trade_cost(entry_price, exit_price, side) / entry_price
        net_pnl = raw_pnl - cost_pnl
        return net_pnl
