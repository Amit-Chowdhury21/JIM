import os

with open(r"e:\PRO\JIMxNik\jim_new\src\models\tft_pro_max.py", "w", encoding="utf-8") as f:
    f.write('''"""
TFT_Pro_Max_V5 — Institutional Production Temporal Fusion Transformer
====================================================================

Architecture: True Decoder TFT with Causal Masking and Monotonic Quantiles
Horizons: 3, 6, 12, 24 bars (5-minute bars)
Quantiles: 10th, 25th, 50th, 75th, 90th percentile
"""

import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from loguru import logger

# ============================================================================
# REPRODUCIBILITY SEEDING
# ============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_TIMEZONE = "Asia/Kolkata"

N_OBSERVED_PAST = 9    
N_KNOWN_FUTURE = 7     
N_STATIC = 3           

HORIZONS = [3, 6, 12, 24]
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

ENCODER_LENGTH = 128   
MAX_HORIZON = max(HORIZONS) 
HIDDEN_SIZE = 64       
N_HEADS = 4
DROPOUT = 0.15

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "tft_pro_max_checkpoint.pt")

# ============================================================================
# UNIFIED TARGET HELPER
# ============================================================================
def compute_returns(prices: np.ndarray, anchor_idx: int, horizons: List[int]) -> np.ndarray:
    """Unified target construction anchoring all future returns to exactly end-1"""
    current_price = prices[anchor_idx]
    returns = []
    for h in horizons:
        if anchor_idx + h < len(prices):
            future_price = prices[anchor_idx + h]
            ret = (future_price - current_price) / (current_price + 1e-10)
        else:
            ret = 0.0
        returns.append(ret)
    return np.array(returns, dtype=np.float32)

# ============================================================================
# COST MODELING (V5)
# ============================================================================
def effective_cost(static_cost: float, gvz: float, session_overlap_flag: float, utc_hour: float) -> float:
    """Scale execution cost if volatility (gvz) is extremely high, during high-volume overlap, or rollover vacuum."""
    vol_factor = 1.0 + 0.5 * max(0, gvz - 10) / 10.0
    session_factor = 1.2 if session_overlap_flag > 0.5 else 1.0
    
    # Rollover penalty (21:00 to 22:00 UTC) represents 17:00 EST daily exchange settlement.
    if 21.0 <= utc_hour < 22.0:
        rollover_factor = 5.0
    else:
        rollover_factor = 1.0
        
    return static_cost * vol_factor * session_factor * rollover_factor

# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int = None, context_size: int = None,
                 dropout: float = DROPOUT):
        super().__init__()
        output_size = output_size or input_size
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size * 2)

        self.context_proj = None
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        self.skip_proj = None
        if input_size != output_size:
            self.skip_proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            if context.dim() == 2 and h.dim() == 3:
                context = context.unsqueeze(1).expand(-1, h.size(1), -1)
            h = h + self.context_proj(context)
        h = self.elu(h)
        h = self.fc2(h)

        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)
        h = self.dropout(h)

        skip = x if self.skip_proj is None else self.skip_proj(x)
        return self.layer_norm(skip + h)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size: int, n_vars: int, hidden_size: int,
                 var_dim: int = 1, context_size: int = None, dropout: float = DROPOUT):
        super().__init__()
        self.n_vars = n_vars
        self.var_dim = var_dim
        self.hidden_size = hidden_size

        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(var_dim, hidden_size,
                                 output_size=hidden_size,
                                 context_size=context_size,
                                 dropout=dropout)
            for _ in range(n_vars)
        ])

        self.selection_grn = GatedResidualNetwork(
            input_size, hidden_size, output_size=n_vars,
            context_size=context_size, dropout=dropout
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        selection_weights = self.softmax(
            self.selection_grn(x, context)
        )

        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_input = x[..., i * self.var_dim:(i + 1) * self.var_dim]
            var_outputs.append(grn(var_input, context))

        var_stack = torch.stack(var_outputs, dim=-2)
        weights_expanded = selection_weights.unsqueeze(-1)
        selected = (var_stack * weights_expanded).sum(dim=-2)

        return selected, selection_weights


class MonotonicQuantileHead(nn.Module):
    """Guarantees strictly monotonically increasing quantiles (q10 <= q25 <= q50 <= q75 <= q90)"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.base = nn.Linear(hidden_size, 1) # Median (q50)
        self.pos_inc = nn.Sequential(nn.Linear(hidden_size, 2), nn.Softplus()) # For q75, q90
        self.neg_inc = nn.Sequential(nn.Linear(hidden_size, 2), nn.Softplus()) # For q25, q10
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q50 = self.base(x)
        p_inc = self.pos_inc(x)
        n_inc = self.neg_inc(x)
        
        q75 = q50 + p_inc[..., 0:1]
        q90 = q75 + p_inc[..., 1:2]
        
        q25 = q50 - n_inc[..., 0:1]
        q10 = q25 - n_inc[..., 1:2]
        
        return torch.cat([q10, q25, q50, q75, q90], dim=-1)

# ============================================================================
# MAIN TFT_PRO_MAX MODEL
# ============================================================================

class TFTProMaxModel(nn.Module):
    def __init__(
        self,
        n_observed: int = N_OBSERVED_PAST,
        n_known_future: int = N_KNOWN_FUTURE,
        n_static: int = N_STATIC,
        hidden_size: int = HIDDEN_SIZE,
        n_heads: int = N_HEADS,
        horizons: List[int] = None,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.n_observed = n_observed
        self.n_known_future = n_known_future
        self.n_static = n_static
        self.hidden_size = hidden_size
        self.horizons = horizons or HORIZONS
        self.max_horizon = max(self.horizons)

        self.static_encoder = GatedResidualNetwork(
            n_static, hidden_size, output_size=hidden_size, dropout=dropout
        )

        self.observed_vsn = VariableSelectionNetwork(
            input_size=n_observed, n_vars=n_observed, var_dim=1,
            hidden_size=hidden_size, context_size=hidden_size,
            dropout=dropout
        )

        self.enc_known_future_vsn = VariableSelectionNetwork(
            input_size=n_known_future, n_vars=n_known_future, var_dim=1,
            hidden_size=hidden_size, context_size=hidden_size,
            dropout=dropout
        )
        
        self.dec_known_future_vsn = VariableSelectionNetwork(
            input_size=n_known_future, n_vars=n_known_future, var_dim=1,
            hidden_size=hidden_size, context_size=hidden_size,
            dropout=dropout
        )
        
        self.encoder_fusion = GatedResidualNetwork(
            hidden_size * 2, hidden_size, output_size=hidden_size, dropout=dropout
        )

        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False, 
        )

        self.post_lstm_grn = GatedResidualNetwork(
            hidden_size, hidden_size, dropout=dropout
        )
        
        self.decoder_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_self_attn_grn = GatedResidualNetwork(
            hidden_size, hidden_size, dropout=dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_layer_norm = nn.LayerNorm(hidden_size)

        self.post_attn_grn = GatedResidualNetwork(
            hidden_size, hidden_size, dropout=dropout
        )

        self.quantile_head = MonotonicQuantileHead(hidden_size)

    def forward(
        self,
        observed_past: torch.Tensor,
        encoder_known_future: torch.Tensor,
        decoder_known_future: torch.Tensor,
        static_covariates: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        B = observed_past.size(0)

        static_context = self.static_encoder(static_covariates)
        
        selected_obs, obs_weights = self.observed_vsn(observed_past, static_context)
        selected_enc_kf, enc_kf_weights = self.enc_known_future_vsn(encoder_known_future, static_context)
        
        encoder_features = torch.cat([selected_obs, selected_enc_kf], dim=-1)
        encoder_features = self.encoder_fusion(encoder_features)
        
        lstm_out, _ = self.lstm_encoder(encoder_features)
        encoder_states = self.post_lstm_grn(lstm_out)

        decoder_features, dec_kf_weights = self.dec_known_future_vsn(decoder_known_future, static_context)
        
        B, L, _ = decoder_features.shape
        causal_mask = torch.triu(torch.ones(L, L, device=decoder_features.device), diagonal=1).bool()

        self_attn_out, self_attn_weights = self.decoder_self_attn(
            query=decoder_features,
            key=decoder_features,
            value=decoder_features,
            attn_mask=causal_mask,
            need_weights=True,
            average_attn_weights=False 
        )

        dec_self = self.self_attn_layer_norm(decoder_features + self_attn_out)
        dec_self = self.post_self_attn_grn(dec_self)
        
        attn_out, attn_weights = self.attention(
            query=dec_self,
            key=encoder_states,
            value=encoder_states,
            need_weights=True,
            average_attn_weights=False
        )
        
        dec_out = self.attn_layer_norm(dec_self + attn_out)
        dec_out = self.post_attn_grn(dec_out)

        horizon_indices = [h - 1 for h in self.horizons]
        selected_dec_out = dec_out[:, horizon_indices, :] 
        
        predictions = self.quantile_head(selected_dec_out) 

        interpretability = {
            "observed_var_weights": obs_weights,
            "encoder_kf_weights": enc_kf_weights,
            "decoder_kf_weights": dec_kf_weights,
            "self_attention_weights": self_attn_weights,
            "attention_weights": attn_weights,
        }

        return predictions, interpretability


# ============================================================================
# QUANTILE LOSS
# ============================================================================

def quantile_loss(predictions: torch.Tensor, targets: torch.Tensor,
                  quantiles: List[float] = None) -> torch.Tensor:
    quantiles = quantiles or QUANTILES
    losses = []
    for q_idx, q in enumerate(quantiles):
        pred_q = predictions[:, :, q_idx]
        residual = targets - pred_q
        loss_q = torch.mean(torch.max(q * residual, (q - 1) * residual))
        losses.append(loss_q)
    return sum(losses) / len(losses)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features_from_df(
    df: pd.DataFrame,
    regime: str = "NORMAL",
    require_timezone: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    required_columns = ["close", "high", "low", "gvz_zscore_20", "dxy", "us10y", "gold_silver_ratio"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        logger.error(f"FEATURE CONTRACT VIOLATION: Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns for TFT inference: {missing_cols}")
        
    closes = df["close"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    T = len(closes)

    log_ret = np.zeros(T)
    log_ret[1:] = np.log(closes[1:] / (closes[:-1] + 1e-10))

    ema21 = pd.Series(closes).ewm(span=21, adjust=False).mean().values
    dist_ema21 = (closes - ema21) / (ema21 + 1e-10)

    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss_arr = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(7, min_periods=1).mean().values + 1e-10
    avg_loss = pd.Series(loss_arr).rolling(7, min_periods=1).mean().values + 1e-10
    rsi_7 = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    fast_rsi = (rsi_7 - 50.0) / 50.0

    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
    macd = ema12 - ema26
    signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    macd_hist = (macd - signal) / (closes + 1e-10)

    tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
    tr = np.maximum(tr, np.abs(lows - np.roll(closes, 1)))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr_ratio = atr / (closes + 1e-10)

    gvz_zscore_20 = df["gvz_zscore_20"].values.astype(np.float64)

    dxy_momentum = np.zeros(T)
    dxy = df["dxy"].values
    dxy_momentum[3:] = (dxy[3:] - dxy[:-3]) / (dxy[:-3] + 1e-10)

    us10y_momentum = np.zeros(T)
    us10y = df["us10y"].values
    us10y_momentum[3:] = (us10y[3:] - us10y[:-3]) / (us10y[:-3] + 1e-10)

    gsr_dev = np.zeros(T)
    gsr = df["gold_silver_ratio"].values
    gsr_mean = pd.Series(gsr).rolling(20, min_periods=1).mean().values
    gsr_dev = (gsr - gsr_mean) / (gsr_mean + 1e-10)

    observed_past = np.column_stack([
        log_ret, dist_ema21, fast_rsi, macd_hist, atr_ratio,
        gvz_zscore_20, dxy_momentum, us10y_momentum, gsr_dev
    ]).astype(np.float32)

    observed_past = np.nan_to_num(observed_past, nan=0.0, posinf=0.0, neginf=0.0)

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            if require_timezone:
                logger.error(f"FEATURE CONTRACT VIOLATION: DatetimeIndex must be timezone-aware (expected {DATA_TIMEZONE})")
                raise ValueError("DatetimeIndex must be timezone-aware")
            idx = idx.tz_localize(DATA_TIMEZONE)
        if str(idx.tz) != DATA_TIMEZONE:
            idx = idx.tz_convert(DATA_TIMEZONE)
    else:
        logger.error("FEATURE CONTRACT VIOLATION: df.index must be a DatetimeIndex")
        raise ValueError("df.index must be a DatetimeIndex")

    # Timezone Bug Fix: Use actual DST-aware localized times
    london_time = idx.tz_convert("Europe/London")
    ny_time = idx.tz_convert("America/New_York")

    float_hour_ist = idx.hour.values.astype(np.float64) + (idx.minute.values.astype(np.float64) / 60.0)
    dows = idx.dayofweek.values.astype(np.float64)

    hour_sin = np.sin(2 * np.pi * float_hour_ist / 24.0)
    hour_cos = np.cos(2 * np.pi * float_hour_ist / 24.0)
    dow_sin = np.sin(2 * np.pi * dows / 7.0)
    dow_cos = np.cos(2 * np.pi * dows / 7.0)

    # DST-Aware London: 08:00 to 17:00 local
    london_float = london_time.hour.values.astype(np.float64) + (london_time.minute.values.astype(np.float64) / 60.0)
    session_london = ((london_float >= 8.0) & (london_float < 17.0)).astype(np.float64)
    
    # DST-Aware NY Open: 09:30 to 16:00 local
    ny_float = ny_time.hour.values.astype(np.float64) + (ny_time.minute.values.astype(np.float64) / 60.0)
    session_ny = ((ny_float >= 9.5) & (ny_float < 16.0)).astype(np.float64)
    
    # Overlap
    session_overlap = (session_london * session_ny).astype(np.float64)
    
    known_future = np.column_stack([
        hour_sin, hour_cos, dow_sin, dow_cos, session_london, session_ny, session_overlap
    ]).astype(np.float32)
    known_future = np.nan_to_num(known_future, nan=0.0, posinf=0.0, neginf=0.0)

    regime_map = {"GROWTH": [1, 0, 0], "NORMAL": [0, 1, 0], "CRISIS": [0, 0, 1]}
    static = np.array(regime_map.get(regime, [0, 1, 0]), dtype=np.float32)

    return observed_past, known_future, static


def generate_live_decoder_known_future(last_timestamp: pd.Timestamp, max_horizon: int = MAX_HORIZON) -> np.ndarray:
    freq = pd.Timedelta("5min")
    current_time = last_timestamp
    future_times = []
    
    # Stale-Bar Defense: Fast forward through weekends (Saturday=5, Sunday=6)
    while len(future_times) < max_horizon:
        current_time += freq
        if current_time.dayofweek < 5: 
            future_times.append(current_time)
            
    future_index = pd.DatetimeIndex(future_times)
    
    if future_index.tz is None:
        future_index = future_index.tz_localize(DATA_TIMEZONE)
    elif str(future_index.tz) != DATA_TIMEZONE:
        future_index = future_index.tz_convert(DATA_TIMEZONE)
        
    london_time = future_index.tz_convert("Europe/London")
    ny_time = future_index.tz_convert("America/New_York")

    float_hour_ist = future_index.hour.values.astype(np.float64) + (future_index.minute.values.astype(np.float64) / 60.0)
    dows = future_index.dayofweek.values.astype(np.float64)

    hour_sin = np.sin(2 * np.pi * float_hour_ist / 24.0)
    hour_cos = np.cos(2 * np.pi * float_hour_ist / 24.0)
    dow_sin = np.sin(2 * np.pi * dows / 7.0)
    dow_cos = np.cos(2 * np.pi * dows / 7.0)

    london_float = london_time.hour.values.astype(np.float64) + (london_time.minute.values.astype(np.float64) / 60.0)
    session_london = ((london_float >= 8.0) & (london_float < 17.0)).astype(np.float64)
    
    ny_float = ny_time.hour.values.astype(np.float64) + (ny_time.minute.values.astype(np.float64) / 60.0)
    session_ny = ((ny_float >= 9.5) & (ny_float < 16.0)).astype(np.float64)
    
    session_overlap = (session_london * session_ny).astype(np.float64)
    
    known_future = np.column_stack([
        hour_sin, hour_cos, dow_sin, dow_cos, session_london, session_ny, session_overlap
    ]).astype(np.float32)
    return known_future

# ============================================================================
# VALIDATION & CALIBRATION
# ============================================================================

def validate_quantile_coverage(
    model: TFTProMaxModel,
    df: pd.DataFrame,
    device: torch.device,
    horizons: List[int] = None,
    regime: str = "NORMAL",
) -> Dict:
    """Check whether predicted quantiles achieve their nominal coverage rates on a holdout block."""
    horizons = horizons or HORIZONS
    observed, known_future, static = build_features_from_df(df, regime, require_timezone=False)
    prices = df["close"].values
    T = len(observed)
    
    predictions = {h: [] for h in horizons}
    realized_returns = {h: [] for h in horizons}
    
    model.eval()
    
    for i in range(ENCODER_LENGTH, T - MAX_HORIZON):
        obs_t = torch.FloatTensor(observed[i - ENCODER_LENGTH:i]).unsqueeze(0).to(device)
        enc_kf_t = torch.FloatTensor(known_future[i - ENCODER_LENGTH:i]).unsqueeze(0).to(device)
        static_t = torch.FloatTensor(static).unsqueeze(0).to(device)
        dec_kf_t = torch.FloatTensor(known_future[i:i + MAX_HORIZON]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds, _ = model(obs_t, enc_kf_t, dec_kf_t, static_t)
            preds = preds[0].cpu().numpy()
            
        # Unified Target Anchor
        actual_returns = compute_returns(prices, i - 1, horizons)
            
        for h_idx, h in enumerate(horizons):
            q_list = preds[h_idx]
            predictions[h].append(q_list)
            realized_returns[h].append(actual_returns[h_idx])
    
    results = {}
    for h in horizons:
        if len(predictions[h]) < 50:
            results[h] = {"coverage_errors": None, "n_samples": len(predictions[h])}
            continue
        
        preds_arr = np.array(predictions[h]) 
        actuals_arr = np.array(realized_returns[h])
        
        coverage_errors = {}
        for q_idx, q in enumerate(QUANTILES):
            pred_q = preds_arr[:, q_idx]
            actual_below = (actuals_arr <= pred_q).mean()
            error = actual_below - q
            coverage_errors[f"q{int(q*100)}"] = {
                "actual_coverage": float(actual_below),
                "expected_coverage": q,
                "error": float(error),
            }
        
        results[h] = {
            "coverage_errors": coverage_errors,
            "n_samples": len(actuals_arr),
        }
    
    return results


def calibrate_decision_thresholds(
    model: TFTProMaxModel,
    df: pd.DataFrame,
    device: torch.device,
    regime: str = "NORMAL",
    base_static_cost: float = 0.00015,
) -> Dict:
    forecasts = []
    realized = []
    costs = []
    
    try:
        observed, known_future, static = build_features_from_df(df, regime, require_timezone=False)
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return {"base_threshold": 0.0010, "max_uncertainty": 0.008, "cost_estimate": base_static_cost}
        
    prices = df["close"].values
    T = len(observed)
    
    # Extract UTC floating hours for the rollover penalty
    df_utc = df.index.tz_convert('UTC') if df.index.tz is not None else df.index.tz_localize('UTC')
    utc_hours = df_utc.hour.values + df_utc.minute.values / 60.0
    
    model.eval()
    
    for i in range(ENCODER_LENGTH, T - MAX_HORIZON):
        obs_t = torch.FloatTensor(observed[i - ENCODER_LENGTH:i]).unsqueeze(0).to(device)
        enc_kf_t = torch.FloatTensor(known_future[i - ENCODER_LENGTH:i]).unsqueeze(0).to(device)
        static_t = torch.FloatTensor(static).unsqueeze(0).to(device)
        dec_kf_t = torch.FloatTensor(known_future[i:i + MAX_HORIZON]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds, _ = model(obs_t, enc_kf_t, dec_kf_t, static_t)
            preds = preds[0].cpu().numpy()
            
        h6_preds = preds[1] # h=6 is index 1
        q10, q25, q50, q75, q90 = h6_preds
        median = q50
        width = q90 - q10
        
        # Unified Target Anchor
        actual_return = compute_returns(prices, i - 1, [6])[0]
        
        # Dynamic Cost (V5 with Rollover)
        gvz_val = observed[i - 1, 5] 
        session_overlap_flag = known_future[i - 1, 6]
        utc_h = utc_hours[i - 1]
        c = effective_cost(base_static_cost, gvz_val, session_overlap_flag, utc_h)
        
        forecasts.append((median, width))
        realized.append(actual_return)
        costs.append(c)
    
    if len(forecasts) < 50:
        return {"base_threshold": 0.0010, "max_uncertainty": 0.008, "cost_estimate": base_static_cost}
    
    forecasts = np.array(forecasts)
    realized = np.array(realized)
    costs = np.array(costs)
    
    candidates = np.percentile(np.abs(forecasts[:, 0]), [10, 20, 30, 40, 50])
    best_threshold = 0.0010
    best_sharpe = -1e9
    
    for thresh in candidates:
        pos_idx = forecasts[:, 0] > thresh
        neg_idx = forecasts[:, 0] < -thresh
        
        pnl_pos = realized[pos_idx] - costs[pos_idx]
        pnl_neg = -realized[neg_idx] - costs[neg_idx]
        
        all_pnl = np.concatenate([pnl_pos, pnl_neg])
        if len(all_pnl) < 10:
            continue
        
        sharpe = np.mean(all_pnl) / (np.std(all_pnl) + 1e-10)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = thresh
    
    # Hit rate Boolean Fix
    widths = forecasts[:, 1]
    hit_rates = []
    for w in np.percentile(widths, [50, 60, 70, 75, 80]):
        idx = widths <= w
        count = idx.sum()
        if count < 30: # Trust threshold
            continue
        hit = ((realized[idx] - costs[idx]) > 0).mean()
        hit_rates.append((w, hit))
    
    max_uncertainty = hit_rates[-1][0] if hit_rates else 0.01
    global_hit_rate = 0.50
    for w, hit in hit_rates:
        if hit >= 0.55:
            max_uncertainty = w
            global_hit_rate = hit
            break
            
    width_stats = {
        "p25": float(np.percentile(widths, 25)),
        "p50": float(np.percentile(widths, 50)),
        "p75": float(np.percentile(widths, 75)),
        "realized_hit_rate": float(global_hit_rate)
    }
    
    return {
        "base_threshold": float(best_threshold),
        "max_uncertainty": float(max_uncertainty),
        "cost_estimate": base_static_cost,
        "width_stats": width_stats,
    }

# ============================================================================
# TFT_PRO_MAX FORECASTER
# ============================================================================

class TFTProMaxForecaster:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[TFTProMaxModel] = None
        self.model_loaded = False
        self.calibration = None
        self._load_model()

    def _load_model(self):
        try:
            if os.path.exists(CHECKPOINT_PATH):
                # Secure loading
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
                self.model = TFTProMaxModel()
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                
                # Auto-load calibration if saved in checkpoint
                self.calibration = checkpoint.get("calibration", None)
                logger.info(f"TFT_Pro_Max loaded safely from {CHECKPOINT_PATH} (weights_only=True)")
            else:
                logger.info("TFT_Pro_Max: No checkpoint found, heuristic fallback.")
        except Exception as e:
            logger.warning(f"TFT_Pro_Max load failed: {e}")

    def set_calibration(self, calibration: Dict):
        self.calibration = calibration

    def predict(
        self,
        df: pd.DataFrame,
        regime: str = "NORMAL",
    ) -> Dict:
        if not self.model_loaded or self.model is None:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": "model_not_loaded",
                "quantiles": None,
                "status": "ERROR",
                "error_code": "MODEL_NOT_LOADED",
            }

        try:
            observed, known_future, static = build_features_from_df(
                df, regime
            )

            T = len(observed)
            if T < ENCODER_LENGTH:
                logger.warning(f"TFT_Pro_Max: insufficient history ({T} < {ENCODER_LENGTH})")
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reasoning": "insufficient_history",
                    "quantiles": None,
                    "status": "STALE",
                    "error_code": "INSUFFICIENT_HISTORY",
                }

            enc_len = ENCODER_LENGTH

            obs_t = torch.FloatTensor(observed[-enc_len:]).unsqueeze(0).to(self.device)
            enc_kf_t = torch.FloatTensor(known_future[-enc_len:]).unsqueeze(0).to(self.device)
            static_t = torch.FloatTensor(static).unsqueeze(0).to(self.device)
            
            dec_kf = generate_live_decoder_known_future(df.index[-1], MAX_HORIZON)
            dec_kf_t = torch.FloatTensor(dec_kf).unsqueeze(0).to(self.device)

            with torch.no_grad():
                preds, interp = self.model(obs_t, enc_kf_t, dec_kf_t, static_t)

            preds = preds[0].cpu().numpy()
            quantile_output = {}
            for h_idx, h in enumerate(HORIZONS):
                quantile_output[f"h{h}"] = {
                    "q10": float(preds[h_idx, 0]),
                    "q25": float(preds[h_idx, 1]),
                    "q50": float(preds[h_idx, 2]),
                    "q75": float(preds[h_idx, 3]),
                    "q90": float(preds[h_idx, 4]),
                }

            h6 = quantile_output["h6"]
            median_forecast = h6["q50"]
            interval_width = h6["q90"] - h6["q10"]
            
            downside = h6["q50"] - h6["q25"]
            upside = h6["q75"] - h6["q50"]

            # UTC extraction for Rollover penalty
            df_utc = df.index[-1].tz_convert('UTC') if df.index[-1].tz is not None else df.index[-1].tz_localize('UTC')
            utc_h = df_utc.hour + df_utc.minute / 60.0

            if self.calibration is None:
                base_threshold = 0.0010  
                max_uncertainty = 0.008  
                p25, p75 = 0.002, 0.006
                realized_hit_rate = 0.50
                c = effective_cost(0.00015, observed[-1, 5], known_future[-1, 6], utc_h)
            else:
                base_threshold = self.calibration.get("base_threshold", 0.0010)
                max_uncertainty = self.calibration.get("max_uncertainty", 0.008)
                width_stats = self.calibration.get("width_stats", {"p25": 0.002, "p75": 0.006, "realized_hit_rate": 0.50})
                p25, p75 = width_stats["p25"], width_stats["p75"]
                realized_hit_rate = width_stats.get("realized_hit_rate", 0.50)
                
                base_cost = self.calibration.get("cost_estimate", 0.00015)
                gvz_val = observed[-1, 5]
                overlap_flag = known_future[-1, 6]
                c = effective_cost(base_cost, gvz_val, overlap_flag, utc_h)
                
            required_move = base_threshold + c
            
            if median_forecast > required_move and upside > downside:
                signal = "LONG"
            elif median_forecast < -required_move and downside > upside:
                signal = "SHORT"
            else:
                signal = "HOLD"
                
            # Empirical Confidence Scoring (scaled by realized holdout hit rate)
            if interval_width <= p25:
                base_conf = 0.90
            elif interval_width >= p75:
                base_conf = 0.30
            else:
                ratio = (interval_width - p25) / (p75 - p25 + 1e-10)
                base_conf = 0.90 - ratio * 0.60
                
            # Scale by actual edge
            edge_scalar = max(0.5, realized_hit_rate) / 0.55
            confidence = float(np.clip(base_conf * edge_scalar, 0.0, 1.0))

            if interval_width > max_uncertainty:
                signal = "HOLD"
                confidence *= 0.5 

            obs_importance = interp["observed_var_weights"][0, -1].cpu().numpy()
            top_vars = np.argsort(obs_importance)[-3:][::-1]
            var_names = [
                "log_ret", "d_ema21", "rsi7", "macd", "atr",
                "gvz", "dxy", "us10y", "gsr"
            ]
            top_var_str = ",".join(f"{var_names[i]}={obs_importance[i]:.2f}" for i in top_vars)

            reasoning = (
                f"TFT_Pro_Max: h6_med={median_forecast:+.4f}, "
                f"band={interval_width:.4f}, cost={c:.5f}, "
                f"h12={quantile_output['h12']['q50']:+.4f} "
                f"[top: {top_var_str}]"
            )

            return {
                "signal": signal,
                "confidence": round(confidence, 3),
                "reasoning": reasoning,
                "quantiles": quantile_output,
                "status": "OK",
                "error_code": "NONE",
            }

        except Exception as e:
            logger.warning(f"TFT_Pro_Max inference error: {e}")
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": str(e),
                "quantiles": None,
                "status": "ERROR",
                "error_code": "INFERENCE_ERROR",
            }

# ============================================================================
# DATASET & TRAINING LOGIC
# ============================================================================

class TFTProMaxDataset(Dataset):
    def __init__(
        self,
        observed: np.ndarray,
        known_future: np.ndarray,
        static: np.ndarray,
        prices: np.ndarray,
        encoder_length: int = ENCODER_LENGTH,
        horizons: List[int] = None,
    ):
        self.observed = torch.FloatTensor(observed)
        self.known_future = torch.FloatTensor(known_future)
        self.static = torch.FloatTensor(static)
        self.prices = prices.astype(np.float64)
        self.encoder_length = encoder_length
        self.horizons = horizons or HORIZONS
        self.max_horizon = max(self.horizons)
        self.valid_indices = range(self.encoder_length, len(self.observed) - self.max_horizon)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end = self.valid_indices[idx]
        start = end - self.encoder_length
        
        obs = self.observed[start:end]
        enc_kf = self.known_future[start:end]
        dec_kf = self.known_future[end:end + self.max_horizon]
        
        # Unified Target Construction
        targets = compute_returns(self.prices, end - 1, self.horizons)

        return {
            "observed": obs,
            "encoder_known_future": enc_kf,
            "decoder_known_future": dec_kf,
            "static": self.static,
            "targets": torch.FloatTensor(targets),
        }

def train_tft_pro_max(
    df: pd.DataFrame,
    regime: str = "NORMAL",
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-3,
    val_split: float = 0.15,
    device: str = "auto",
) -> Dict:
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    logger.info(f"Training TFT_Pro_Max_V5 on {dev} | {len(df)} bars | epochs={epochs}")

    observed, known_future, static = build_features_from_df(df, regime)
    prices = df["close"].values

    warmup = 26 
    observed = observed[warmup:]
    known_future = known_future[warmup:]
    prices = prices[warmup:]

    n = len(observed)
    
    # 75% for CV / Training, 15% Calibration, 10% Strict Holdout Promotion
    cv_end = int(n * 0.75)
    calib_end = int(n * 0.90)
    
    n_cv = cv_end
    n_folds = 3
    fold_size = n_cv // (n_folds + 1)
    
    # Improve Purge/Embargo Logic
    purge_window = max(HORIZONS) + 6 # max_label_span = max(horizons) + holding period
    embargo_window = 26           

    logger.info(f"Using Purged Walk-Forward CV (75% Split): {n_folds} folds, Purge={purge_window}, Embargo={embargo_window}")

    fold_losses = []
    
    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        val_start = train_end + purge_window + embargo_window
        val_end = val_start + fold_size
        
        if val_end > n_cv:
            val_end = n_cv
            
        train_ds = TFTProMaxDataset(
            observed[:train_end], known_future[:train_end], static, prices[:train_end]
        )
        val_ds = TFTProMaxDataset(
            observed[val_start - ENCODER_LENGTH : val_end], 
            known_future[val_start - ENCODER_LENGTH : val_end],
            static, prices[val_start - ENCODER_LENGTH : val_end]
        )

        if len(train_ds) < 10 or len(val_ds) < 5:
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = TFTProMaxModel().to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_fold_val = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                obs = batch["observed"].to(dev)
                enc_kf = batch["encoder_known_future"].to(dev)
                dec_kf = batch["decoder_known_future"].to(dev)
                stat = batch["static"].to(dev)
                targets = batch["targets"].to(dev)

                preds, _ = model(obs, enc_kf, dec_kf, stat)
                loss = quantile_loss(preds, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    obs = batch["observed"].to(dev)
                    enc_kf = batch["encoder_known_future"].to(dev)
                    dec_kf = batch["decoder_known_future"].to(dev)
                    stat = batch["static"].to(dev)
                    targets = batch["targets"].to(dev)

                    preds, _ = model(obs, enc_kf, dec_kf, stat)
                    loss = quantile_loss(preds, targets)
                    val_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            
            logger.info(f"Fold {fold+1}/{n_folds} | Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
            
            if avg_val < best_fold_val:
                best_fold_val = avg_val
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        fold_losses.append(best_fold_val)

    if not fold_losses:
        return {"status": "error", "message": "No folds completed."}
        
    avg_cv_loss = float(np.mean(fold_losses))
    logger.info(f"CV Complete. Avg Val Loss: {avg_cv_loss:.6f}. Training Evaluation Model (75%).")

    # Train Evaluation model strictly up to the 75% boundary
    eval_ds = TFTProMaxDataset(
        observed[:cv_end], known_future[:cv_end], static, prices[:cv_end]
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True)

    model_eval = TFTProMaxModel().to(dev)
    optimizer_eval = torch.optim.Adam(model_eval.parameters(), lr=lr)

    for epoch in range(epochs):
        model_eval.train()
        train_losses = []
        for batch in eval_loader:
            obs = batch["observed"].to(dev)
            enc_kf = batch["encoder_known_future"].to(dev)
            dec_kf = batch["decoder_known_future"].to(dev)
            stat = batch["static"].to(dev)
            targets = batch["targets"].to(dev)

            preds, _ = model_eval(obs, enc_kf, dec_kf, stat)
            loss = quantile_loss(preds, targets)

            optimizer_eval.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_eval.parameters(), 1.0)
            optimizer_eval.step()
            train_losses.append(loss.item())
            
        avg_train = np.mean(train_losses)
        logger.info(f"Eval Model | Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.6f}")
            
    logger.info("Executing Calibration on 15% Block.")
    df_calib = df.iloc[cv_end + warmup - ENCODER_LENGTH : calib_end + warmup].copy()
    calib = calibrate_decision_thresholds(model_eval, df_calib, dev, regime)
    
    logger.info("Executing Promotion Gates on 10% Strict Holdout Block.")
    
    df_holdout = df.iloc[calib_end + warmup - ENCODER_LENGTH :].copy()
    
    holdout_ds = TFTProMaxDataset(
        observed[calib_end - ENCODER_LENGTH:], 
        known_future[calib_end - ENCODER_LENGTH:], 
        static, 
        prices[calib_end - ENCODER_LENGTH:]
    )
    holdout_loader = DataLoader(holdout_ds, batch_size=batch_size, shuffle=False)
    
    # Pre-calculate UTC hours for the holdout block
    df_holdout_utc = df_holdout.index.tz_convert('UTC') if df_holdout.index.tz is not None else df_holdout.index.tz_localize('UTC')
    utc_hours_holdout = df_holdout_utc.hour.values + df_holdout_utc.minute.values / 60.0
    
    model_eval.eval()
    holdout_losses = []
    naive_losses = []
    
    forecasts = []
    realized = []
    costs = []
    
    base_cost = calib.get("cost_estimate", 0.00015)
    
    # Manually track index offset for UTC hours extraction
    idx_counter = ENCODER_LENGTH
    
    with torch.no_grad():
        for batch in holdout_loader:
            obs = batch["observed"].to(dev)
            enc_kf = batch["encoder_known_future"].to(dev)
            dec_kf = batch["decoder_known_future"].to(dev)
            stat = batch["static"].to(dev)
            targets = batch["targets"].to(dev)

            preds, _ = model_eval(obs, enc_kf, dec_kf, stat)
            loss = quantile_loss(preds, targets)
            holdout_losses.append(loss.item())
            
            naive_preds = torch.zeros_like(preds).to(dev)
            naive_loss = quantile_loss(naive_preds, targets)
            naive_losses.append(naive_loss.item())
            
            preds_cpu = preds.cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            enc_kf_cpu = enc_kf.cpu().numpy()
            obs_cpu = obs.cpu().numpy()
            
            for b in range(preds_cpu.shape[0]):
                h6_preds = preds_cpu[b, 1]
                q10, q25, q50, q75, q90 = h6_preds
                median = q50
                width = q90 - q10
                forecasts.append((median, width))
                realized.append(targets_cpu[b, 1])
                
                # Extract V5 Dynamic cost using exact holdout UTC hour
                utc_h = utc_hours_holdout[idx_counter - 1]
                c = effective_cost(base_cost, obs_cpu[b, -1, 5], enc_kf_cpu[b, -1, 6], utc_h)
                costs.append(c)
                idx_counter += 1

    avg_holdout_loss = float(np.mean(holdout_losses))
    avg_naive_loss = float(np.mean(naive_losses))
    logger.info(f"Holdout Loss: {avg_holdout_loss:.6f} | Naive Loss: {avg_naive_loss:.6f}")
    
    if avg_holdout_loss >= avg_naive_loss:
        logger.error(f"PROMOTION GATE FAILED: Model ({avg_holdout_loss:.6f}) failed to beat Naive baseline ({avg_naive_loss:.6f})")
        raise ValueError("Model failed naive baseline benchmark. Will not promote to production.")

    coverage_results = validate_quantile_coverage(model_eval, df_holdout, dev, regime=regime)
    
    for h in [6, 12]:  
        if h in coverage_results and coverage_results[h].get("coverage_errors"):
            errors = coverage_results[h]["coverage_errors"]
            if abs(errors["q10"]["error"]) > 0.035 or abs(errors["q90"]["error"]) > 0.035:
                logger.error(f"PROMOTION GATE FAILED: Coverage tolerance exceeded on H{h}. Errors: {errors}")
                raise ValueError(f"Quantile coverage off by > 3.5% on H{h}. Model is uncalibrated.")

    logger.info("Evaluating V5 FROZEN Trade Count & Net PnL Gates...")
    best_thresh = calib["base_threshold"]
    max_unc = calib["max_uncertainty"]
    
    trades_pnl = []
    for f, r, c in zip(forecasts, realized, costs):
        med, wid = f[0], f[1]
        if wid > max_unc:
            continue
        if med > best_thresh + c: # Long
            trades_pnl.append(r - c)
        elif med < -best_thresh - c: # Short
            trades_pnl.append(-r - c)
            
    total_trades = len(trades_pnl)
    if total_trades < 50:
        logger.error(f"PROMOTION GATE FAILED: Trade Count Gate. Only {total_trades} trades found on 10% strict holdout. Minimum 50 required.")
        raise ValueError("Insufficient trade frequency on strictly unseen data.")
        
    net_pnl = sum(trades_pnl)
    if net_pnl <= 0:
        logger.error(f"PROMOTION GATE FAILED: Net PnL Gate. Net PnL is {net_pnl:.6f}.")
        raise ValueError("Model has negative expectancy when tested on frozen thresholds.")
        
    win_rate = np.mean([1 if x > 0 else 0 for x in trades_pnl])
    logger.info(f"Holdout PnL: {net_pnl:.4f} | Win Rate: {win_rate:.2%} | Trades: {total_trades}")
    
    if win_rate < 0.50:
        logger.error(f"PROMOTION GATE FAILED: Win Rate Gate. Win rate is {win_rate:.2%}.")
        raise ValueError("Model win rate is below 50% on frozen test block.")

    logger.info("All Promotion Gates Passed. V5 Deployment Authorised.")
        
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    torch.save({
        "model_state_dict": model_eval.state_dict(),
        "epoch": epochs,
        "avg_cv_loss": avg_cv_loss,
        "avg_holdout_loss": avg_holdout_loss,
        "calibration": calib,  
        "model_version": "TFT_Pro_Max_V5_StrictPromotion",
        "feature_schema": {
            "required_columns": ["close", "high", "low", "gvz_zscore_20", "dxy", "us10y", "gold_silver_ratio"],
            "observed_features": ["log_ret", "d_ema21", "rsi7", "macd", "atr", "gvz", "dxy", "us10y", "gsr"],
            "known_future_features": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_london", "session_ny", "session_overlap"],
            "static_features": ["regime_GROWTH", "regime_NORMAL", "regime_CRISIS"],
        },
        "config": {
            "encoder_length": ENCODER_LENGTH,
            "horizons": HORIZONS,
            "quantiles": QUANTILES,
            "hidden_size": HIDDEN_SIZE,
            "n_heads": N_HEADS,
            "dropout": DROPOUT,
        },
    }, CHECKPOINT_PATH)

    return {
        "status": "success",
        "avg_cv_loss": avg_cv_loss,
        "avg_holdout_loss": avg_holdout_loss,
        "epochs_trained": epochs,
        "checkpoint_path": CHECKPOINT_PATH,
        "calibration": calib,
        "coverage_results": coverage_results,
        "device": str(dev),
    }
''')
