"""
v2.1 Execution & Policy Engine
==============================
Decouples raw prediction from live execution. Uses asymmetric probability
thresholds and MCDropout epistemic uncertainty to filter low-conviction signals.
"""

from typing import Dict
from loguru import logger

class SignalPolicy:
    """
    Acts as a strict gatekeeper for the v2.1 Execution Engine.
    Filters out trades that don't meet asymmetric probability thresholds
    or exhibit high model uncertainty (variance across stochastic passes).
    """
    def __init__(self, long_threshold: float = 0.45, short_threshold: float = 0.65, max_uncertainty: float = 0.05):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.max_uncertainty = max_uncertainty
        
    def evaluate(self, mcd_result: Dict) -> Dict:
        """
        Evaluates the raw MCDropout output against the execution policy.
        Returns a modified result dict that defaults to HOLD if it fails the gates,
        and includes a dynamic sizing scalar based on uncertainty.
        """
        signal = mcd_result.get("signal", "HOLD")
        uncertainty = mcd_result.get("uncertainty", 0.0)
        probs = mcd_result.get("probabilities", {})
        
        long_prob = probs.get("LONG", 0.0)
        short_prob = probs.get("SHORT", 0.0)
        
        # 1. Epistemic Uncertainty Gate
        if uncertainty > self.max_uncertainty:
            return self._reject("HOLD", f"High uncertainty ({uncertainty:.4f} > {self.max_uncertainty})")
            
        # 2. Asymmetric Probability Thresholds
        if signal == "LONG" and long_prob < self.long_threshold:
            return self._reject("HOLD", f"LONG prob ({long_prob:.4f}) below threshold ({self.long_threshold})")
            
        if signal == "SHORT" and short_prob < self.short_threshold:
            return self._reject("HOLD", f"SHORT prob ({short_prob:.4f}) below threshold ({self.short_threshold})")
            
        # 3. Dynamic Sizing Scalar (inverse to uncertainty)
        # Maps uncertainty [0, max_uncertainty] -> size multiplier [1.0, 0.5]
        sizing_scalar = 1.0
        if uncertainty > 0 and self.max_uncertainty > 0:
            sizing_scalar = max(0.5, 1.0 - (uncertainty / self.max_uncertainty) * 0.5)
            
        # 4. If originally HOLD, pass through
        if signal == "HOLD":
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "uncertainty": uncertainty,
                "sizing_scalar": 0.0,
                "reasoning": "Model predicts HOLD"
            }
            
        # Passed all gates
        return {
            "signal": signal,
            "confidence": mcd_result.get("confidence", 0.0),
            "uncertainty": uncertainty,
            "sizing_scalar": round(sizing_scalar, 3),
            "reasoning": f"v2.1 Policy Passed (Unc: {uncertainty:.4f}, Size: {sizing_scalar:.2f}x)"
        }
        
    def _reject(self, new_signal: str, reason: str) -> Dict:
        return {
            "signal": new_signal,
            "confidence": 0.0,
            "uncertainty": 0.0,
            "sizing_scalar": 0.0,
            "reasoning": f"v2.1 Policy Rejected: {reason}"
        }
