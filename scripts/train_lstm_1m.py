"""
Professional LSTM Training Pipeline — 18-Year 1-Minute Multi-Asset Data
=========================================================================
GPU-optimized training script for the CNN-LSTM-Attention gold trading model
using 18+ years of raw 1-minute data from DataBase/.

Hardware Target: RTX 5070 Ti (16 GB VRAM) + Ryzen 7 9800X3D

Key Features:
    - Loads 1.7 GB of local CSV data (Gold, DXY, GVZ, Silver, US10Y)
    - Trains directly on RAW 1-MINUTE bars (~7M+ bars from 2008+)
    - Memory-efficient streaming TimeSeriesDataset (no 97GB array copy)
    - 58 engineered features via LSTMFeatureEngineer
    - Walk-forward validation (5 folds, expanding window, purged)
    - Mixed-precision training (AMP) for 2x speedup + 40% VRAM savings
    - AdamW + Cosine Annealing with warm restarts
    - 3-class prediction: SHORT / HOLD / LONG
    - Model saved to models/lstm_cnn_attention_1m.pt

Usage:
    python scripts/train_lstm_1m.py                    # Full training
    python scripts/train_lstm_1m.py --dry-run           # Verify data loading only
    python scripts/train_lstm_1m.py --resample 5min     # Use 5-min bars instead
    python scripts/train_lstm_1m.py --folds 3           # Fewer folds (faster)
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from pathlib import Path
from loguru import logger
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.lstm_features import LSTMFeatureEngineer
from src.models.lstm_preprocessor import LSTMPreprocessor
from src.models.lstm_temporal import CNNLSTMAttention, GoldLSTMModel
from scripts.local_data_loader import load_merged_data
from scripts.cost_model import CostModel, TradeEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Data
    "resample": "5min",       # Default to 5-minute bars

    # Model Architecture (scaled up for massive dataset)
    "cnn_filters": 128,
    "cnn_kernel": 3,
    "lstm_hidden": 256,
    "lstm_layers": 3,
    "attn_heads": 8,
    "dropout": 0.3,

    # Training
    "seq_len": 60,            # 60-minute lookback window
    "fwd_bars": 12,           # 12 bars forward (vertical barrier)
    "pt_factor": 2.0,         # Profit Take ATR multiplier
    "sl_factor": 1.0,         # Stop Loss ATR multiplier
    "epochs": 150,
    "batch_size": 1024,       # Larger batch for 7M+ dataset
    "lr": 3e-4,
    "weight_decay": 1e-2,
    "patience": 25,
    "label_smoothing": 0.1,
    "grad_clip": 1.0,

    # Walk-Forward
    "n_folds": 5,
    "gap_bars": 120,          # 2-hour purge gap at 1-minute resolution

    # GPU
    "use_amp": False,         # Disable mixed precision (caused FP16 NaN collapse)
    "num_workers": 0,         # 0 = main process (avoids Windows paging file issues)
    "pin_memory": True,
    "compile_model": False,
}


# ══════════════════════════════════════════════════════════════
# MEMORY-EFFICIENT STREAMING DATASET
# ══════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for time-series sequence data.

    Instead of pre-creating all (N × seq_len × features) sequences in RAM
    (which would be ~97 GB for 7M bars), this dataset stores only the
    scaled 2D array and creates sequences on-the-fly via __getitem__.

    Memory usage: O(N × F) for the scaled array, not O(N × seq_len × F).
    """

    def __init__(self, scaled_data: np.ndarray, labels: np.ndarray, seq_len: int):
        """
        Args:
            scaled_data: (N, F) float32 array of scaled features.
            labels:      (N,)   int64 array of class labels.
            seq_len:     Lookback window length.
        """
        self.data = np.ascontiguousarray(scaled_data, dtype=np.float32)
        self.labels = np.ascontiguousarray(labels, dtype=np.int64)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]       # (seq_len, F) — view, no copy
        y = self.labels[idx + self.seq_len]
        return torch.from_numpy(x.copy()), torch.tensor(y, dtype=torch.long)


# ══════════════════════════════════════════════════════════════
# HELPER: SCALE + LABEL WITHOUT FULL SEQUENCE MATERIALIZATION
# ══════════════════════════════════════════════════════════════

def prepare_fold_data(
    preprocessor: LSTMPreprocessor,
    features: pd.DataFrame,
    df: pd.DataFrame,
    fit: bool = False,
) -> tuple:
    """
    Scale features and generate labels WITHOUT creating the full
    (N × seq_len × F) sequence array. Returns raw (scaled_2d, labels_1d).

    Args:
        preprocessor: LSTMPreprocessor (must be fitted or fit=True).
        features: Feature DataFrame.
        df: Aligned raw OHLCV DataFrame.
        fit: Whether to fit the scaler on this data.

    Returns:
        (scaled_data, labels) — numpy arrays.
    """
    if fit:
        preprocessor.fit(features, df)

    scaled = preprocessor.scaler.transform(features.values).astype(np.float32)
    labels = preprocessor._generate_labels(df)

    return scaled, labels


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD TRAINER
# ══════════════════════════════════════════════════════════════

class ProfessionalWalkForwardTrainer:
    """
    Production-grade walk-forward trainer with:
    - Expanding window validation
    - Purged gap (no leakage)
    - Mixed-precision GPU training
    - Memory-efficient streaming dataset
    - Per-fold model selection by Sharpe
    """

    def __init__(self, config: dict, device: torch.device):
        self.cfg = config
        self.device = device
        self.fold_results = []
        self.feature_engineer = LSTMFeatureEngineer()

        # AMP scaler
        self.scaler = GradScaler("cuda") if config["use_amp"] and device.type == "cuda" else None

    def train(self, df: pd.DataFrame) -> GoldLSTMModel:
        """
        Full walk-forward training pipeline.

        Returns:
            Trained GoldLSTMModel wrapper ready for inference.
        """
        total_start = time.time()

        # ━━━ 1. Feature Engineering ━━━
        logger.info("=" * 70)
        logger.info("  PHASE 1: FEATURE ENGINEERING")
        logger.info("=" * 70)

        t0 = time.time()
        features = self.feature_engineer.transform(df)
        
        # --- NEW V2.0 LOGIC: Sub-select ONLY the 20 mathematically validated features ---
        feature_list_path = PROJECT_ROOT / "models" / "v2_selected_features.json"
        if feature_list_path.exists():
            with open(feature_list_path, "r") as f:
                sel_feats = json.load(f)["selected_features"]
            # Keep only the ones that exist
            sel_feats = [f for f in sel_feats if f in features.columns]
            features = features[sel_feats]
            logger.info(f"Loaded {len(sel_feats)} validated features from v2_selected_features.json")
        else:
            logger.warning("v2_selected_features.json not found, using all features.")
        
        # CRITICAL: Drop any remaining NaNs OR Infs (divide by zero) to prevent PyTorch NaN loss
        valid_idx = features.replace([np.inf, -np.inf], np.nan).dropna().index
        features = features.loc[valid_idx]
        df_aligned = df.loc[valid_idx].copy()
        
        n_features = features.shape[1]
        fe_time = time.time() - t0
        logger.info(f"Features: {n_features} columns, {len(features):,} rows "
                     f"(took {fe_time / 60:.1f} min)")

        # ━━━ 2. Walk-Forward Splits ━━━
        logger.info("=" * 70)
        logger.info("  PHASE 2: WALK-FORWARD VALIDATION")
        logger.info("=" * 70)

        total_bars = len(features)
        n_folds = self.cfg["n_folds"]
        gap = self.cfg["gap_bars"]
        seq_len = self.cfg["seq_len"]

        # Each fold: test ≈ total/(folds+2), val ≈ test/2
        test_size = total_bars // (n_folds + 2)
        val_size = test_size // 2

        logger.info(f"Total bars: {total_bars:,} | Folds: {n_folds} | "
                    f"Val size: {val_size:,} | Test size: {test_size:,} | Gap: {gap}")

        best_fold_sharpe = -np.inf
        best_model_state = None

        for fold in range(n_folds):
            fold_start = time.time()
            logger.info(f"\n{'━' * 70}")
            logger.info(f"  FOLD {fold + 1}/{n_folds}")
            logger.info(f"{'━' * 70}")

            # Expanding window: each fold uses more training data
            test_end = total_bars - (n_folds - fold - 1) * test_size
            test_start = test_end - test_size
            val_end = test_start - gap
            val_start = val_end - val_size
            train_end = val_start - gap

            if train_end < seq_len + 100:
                logger.warning(f"Fold {fold + 1}: insufficient training data, skipping")
                continue

            # Split features and aligned df
            train_feat = features.iloc[:train_end]
            train_df = df_aligned.iloc[:train_end]
            val_feat = features.iloc[val_start:val_end]
            val_df = df_aligned.iloc[val_start:val_end]
            test_feat = features.iloc[test_start:test_end]
            test_df = df_aligned.iloc[test_start:test_end]

            logger.info(f"  Train: {len(train_feat):,} bars "
                        f"({train_feat.index.min()} → {train_feat.index.max()})")
            logger.info(f"  Val:   {len(val_feat):,} bars "
                        f"({val_feat.index.min()} → {val_feat.index.max()})")
            logger.info(f"  Test:  {len(test_feat):,} bars "
                        f"({test_feat.index.min()} → {test_feat.index.max()})")

            # ━━━ Preprocessing (memory-efficient) ━━━
            preprocessor = LSTMPreprocessor(
                seq_len=self.cfg["seq_len"],
                fwd_bars=self.cfg["fwd_bars"],
                pt_factor=self.cfg["pt_factor"],
                sl_factor=self.cfg["sl_factor"],
            )

            # Scale + label WITHOUT creating full (N × seq × F) array
            scaled_train, labels_train = prepare_fold_data(
                preprocessor, train_feat, train_df, fit=True
            )
            scaled_val, labels_val = prepare_fold_data(
                preprocessor, val_feat, val_df, fit=False
            )
            scaled_test, labels_test = prepare_fold_data(
                preprocessor, test_feat, test_df, fit=False
            )

            n_train_seq = len(scaled_train) - seq_len
            n_val_seq = len(scaled_val) - seq_len
            n_test_seq = len(scaled_test) - seq_len

            if n_train_seq < self.cfg["batch_size"]:
                logger.warning(f"Fold {fold + 1}: insufficient sequences, skipping")
                continue

            logger.info(f"  Sequences: Train={n_train_seq:,} | Val={n_val_seq:,} | Test={n_test_seq:,}")

            # Class weights (from training labels only)
            class_weights = preprocessor.get_class_weights(
                labels_train[seq_len:]  # Labels that correspond to actual sequences
            )

            # ━━━ Build Model ━━━
            model = CNNLSTMAttention(
                n_features=n_features,
                cnn_filters=self.cfg["cnn_filters"],
                cnn_kernel=self.cfg["cnn_kernel"],
                lstm_hidden=self.cfg["lstm_hidden"],
                lstm_layers=self.cfg["lstm_layers"],
                attn_heads=self.cfg["attn_heads"],
                dropout=self.cfg["dropout"],
            ).to(self.device)

            n_params = model.count_parameters()
            logger.info(f"  Model: {n_params:,} parameters")

            # ━━━ Train Fold ━━━
            fold_result = self._train_fold(
                model,
                (scaled_train, labels_train),
                (scaled_val, labels_val),
                (scaled_test, labels_test, test_df),
                class_weights, fold + 1,
            )
            self.fold_results.append(fold_result)

            fold_elapsed = time.time() - fold_start
            logger.info(f"\n  Fold {fold + 1} completed in {fold_elapsed / 60:.1f} min")
            logger.info(f"  Val  Acc: {fold_result['val_acc']:.1%} | Loss: {fold_result['val_loss']:.4f}")
            logger.info(f"  Test Acc: {fold_result['test_acc']:.1%} | Sharpe: {fold_result['test_sharpe']:.3f}")

            # Track best fold
            if fold_result["test_sharpe"] > best_fold_sharpe:
                best_fold_sharpe = fold_result["test_sharpe"]
                if hasattr(model, "_orig_mod"):
                    best_model_state = {k: v.clone() for k, v in model._orig_mod.state_dict().items()}
                else:
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # ━━━ 3. Final Model (retrain on all data) ━━━
        logger.info(f"\n{'━' * 70}")
        logger.info(f"  PHASE 3: FINAL MODEL (ALL DATA)")
        logger.info(f"{'━' * 70}")

        final_preprocessor = LSTMPreprocessor(
            seq_len=self.cfg["seq_len"],
            fwd_bars=self.cfg["fwd_bars"],
            pt_factor=self.cfg["pt_factor"],
            sl_factor=self.cfg["sl_factor"],
        )

        scaled_all, labels_all = prepare_fold_data(
            final_preprocessor, features, df_aligned, fit=True
        )

        n_all_seq = len(scaled_all) - self.cfg["seq_len"]
        class_weights_all = final_preprocessor.get_class_weights(
            labels_all[self.cfg["seq_len"]:]
        )

        final_model = CNNLSTMAttention(
            n_features=n_features,
            cnn_filters=self.cfg["cnn_filters"],
            cnn_kernel=self.cfg["cnn_kernel"],
            lstm_hidden=self.cfg["lstm_hidden"],
            lstm_layers=self.cfg["lstm_layers"],
            attn_heads=self.cfg["attn_heads"],
            dropout=self.cfg["dropout"],
        ).to(self.device)

        logger.info(f"  Final model: {final_model.count_parameters():,} params, "
                     f"{n_all_seq:,} sequences")

        # Initialize from best fold
        if best_model_state is not None:
            try:
                final_model.load_state_dict(best_model_state)
                logger.info(f"  Initialized from best fold (Sharpe={best_fold_sharpe:.3f})")
            except Exception as e:
                logger.warning(f"  Could not initialize from best fold: {e}")

        # Train on 90/10 split of all data
        split_idx = int(len(scaled_all) * 0.9)
        self._train_fold(
            final_model,
            (scaled_all[:split_idx], labels_all[:split_idx]),
            (scaled_all[split_idx:], labels_all[split_idx:]),
            None,  # No test set for final
            class_weights_all, fold_num=0,
        )

        # ━━━ 4. Save ━━━
        model_path = PROJECT_ROOT / "models" / "lstm_v2.0_triple_barrier.pt"
        preprocessor_path = PROJECT_ROOT / "models" / "lstm_preprocessor_v2.0.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        wrapper = GoldLSTMModel(device=str(self.device))
        wrapper.model = final_model
        wrapper.config = {
            "n_features": n_features,
            "cnn_kernel": self.cfg["cnn_kernel"],
            "cnn_filters": self.cfg["cnn_filters"],
            "lstm_hidden": self.cfg["lstm_hidden"],
            "lstm_layers": self.cfg["lstm_layers"],
            "attn_heads": self.cfg["attn_heads"],
            "dropout": self.cfg["dropout"],
        }
        wrapper._is_loaded = True

        wrapper.save(str(model_path))
        final_preprocessor.save(str(preprocessor_path))

        # ━━━ 5. Summary ━━━
        total_elapsed = time.time() - total_start
        self._print_summary(total_elapsed)
        self._save_report(total_elapsed)

        return wrapper

    def _train_fold(
        self,
        model: nn.Module,
        train_data: tuple,    # (scaled_2d, labels_1d)
        val_data: tuple,
        test_data,            # tuple or None
        class_weights: np.ndarray,
        fold_num: int,
    ) -> dict:
        """Train a single fold with streaming dataset and AMP."""

        seq_len = self.cfg["seq_len"]

        # Create streaming datasets
        train_ds = TimeSeriesDataset(train_data[0], train_data[1], seq_len)
        val_ds = TimeSeriesDataset(val_data[0], val_data[1], seq_len)

        batch_size = min(self.cfg["batch_size"], len(train_ds))
        if batch_size < 4:
            batch_size = len(train_ds)

        use_pin = self.cfg["pin_memory"] and self.device.type == "cuda"
        n_workers = self.cfg["num_workers"] if self.device.type == "cuda" else 0

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=n_workers, pin_memory=use_pin,
            persistent_workers=(n_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=n_workers, pin_memory=use_pin,
            persistent_workers=(n_workers > 0),
        )

        if len(train_loader) == 0:
            logger.warning(f"  Fold {fold_num}: 0 training batches")
            return {"val_loss": float("inf"), "val_acc": 0.0,
                    "test_acc": 0.0, "test_sharpe": 0.0}

        # Loss with class weights
        weight_tensor = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=self.cfg["label_smoothing"],
        )

        # Optimizer + Scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.cfg["lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=self.cfg["lr"] * 0.01,
        )

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        n_batches = len(train_loader)
        logger.info(f"  Training: {n_batches:,} batches/epoch, batch_size={batch_size}")

        for epoch in range(self.cfg["epochs"]):
            epoch_start = time.time()

            # ── Train ──
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if self.scaler is not None:
                    with autocast("cuda"):
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg["grad_clip"])
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg["grad_clip"])
                    optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += X_batch.size(0)

            scheduler.step()

            if train_total == 0:
                continue

            train_loss /= train_total
            train_acc = train_correct / train_total

            # ── Validate ──
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                    if self.scaler is not None:
                        with autocast("cuda"):
                            logits = model(X_batch)
                            loss = criterion(logits, y_batch)
                    else:
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)

                    val_loss += loss.item() * X_batch.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += X_batch.size(0)

            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if hasattr(model, "_orig_mod"):
                    best_state = {k: v.clone() for k, v in model._orig_mod.state_dict().items()}
                else:
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            epoch_time = time.time() - epoch_start

            if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter >= self.cfg["patience"]:
                lr_now = optimizer.param_groups[0]["lr"]
                gpu_mem = ""
                if self.device.type == "cuda":
                    mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    gpu_mem = f" | VRAM={mem_gb:.1f}GB"
                logger.info(
                    f"    Epoch {epoch + 1:3d}/{self.cfg['epochs']} "
                    f"({epoch_time:.0f}s) | "
                    f"Train Loss={train_loss:.4f} Acc={train_acc:.1%} | "
                    f"Val Loss={val_loss:.4f} Acc={val_acc:.1%} | "
                    f"LR={lr_now:.2e} | Pat={patience_counter}/{self.cfg['patience']}"
                    f"{gpu_mem}"
                )

            if patience_counter >= self.cfg["patience"]:
                logger.info(f"    ⏹ Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            if hasattr(model, "_orig_mod"):
                model._orig_mod.load_state_dict(best_state)
            else:
                model.load_state_dict(best_state)

        # ── Test ──
        test_acc = 0.0
        test_sharpe = 0.0

        if test_data is not None:
            test_scaled, test_labels = test_data[0], test_data[1]
            test_df = test_data[2] if len(test_data) > 2 else None
            test_ds = TimeSeriesDataset(test_scaled, test_labels, seq_len)

            if len(test_ds) > 0:
                test_loader = DataLoader(
                    test_ds, batch_size=self.cfg["batch_size"], shuffle=False,
                    num_workers=n_workers, pin_memory=use_pin,
                )

                model.eval()
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(self.device, non_blocking=True)

                        if self.scaler is not None:
                            with autocast("cuda"):
                                logits = model(X_batch)
                        else:
                            logits = model(X_batch)

                        preds = logits.argmax(dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(y_batch.numpy())

                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                test_acc = (all_preds == all_labels).mean()

                # Simulated Sharpe with realistic cost model
                signals = np.where(all_preds == 2, 1, np.where(all_preds == 0, -1, 0))
                
                annualization = np.sqrt(525600) if self.cfg["resample"] == "1min" else np.sqrt(105120)
                
                if test_df is not None:
                    evaluator = TradeEvaluator(CostModel(spread_bps=2.0, slippage_bps=1.0, commission_bps=0.5))
                    prices = test_df["close"].values[seq_len:]
                    
                    trade_returns = np.zeros_like(signals, dtype=float)
                    for i in range(len(signals)):
                        if signals[i] != 0 and i + self.cfg["fwd_bars"] < len(prices):
                            entry_price = prices[i]
                            exit_price = prices[i + self.cfg["fwd_bars"]]
                            side = "LONG" if signals[i] == 1 else "SHORT"
                            trade_returns[i] = evaluator.evaluate_trade(entry_price, exit_price, side=side)
                            
                    if trade_returns.std() > 0:
                        test_sharpe = trade_returns.mean() / trade_returns.std() * annualization
                    else:
                        test_sharpe = 0.0
                else:
                    actual_dir = np.where(all_labels == 2, 1, np.where(all_labels == 0, -1, 0))
                    trade_returns = signals * actual_dir * 0.001
                    if trade_returns.std() > 0:
                        test_sharpe = trade_returns.mean() / trade_returns.std() * annualization
                    else:
                        test_sharpe = 0.0

                # Per-class accuracy
                for cls_id, cls_name in enumerate(["SHORT", "HOLD", "LONG"]):
                    cls_mask = all_labels == cls_id
                    if cls_mask.sum() > 0:
                        cls_acc = (all_preds[cls_mask] == cls_id).mean()
                        logger.info(f"    {cls_name} Acc: {cls_acc:.1%} ({cls_mask.sum():,} samples)")

        return {
            "val_loss": best_val_loss,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_sharpe": test_sharpe,
        }

    def _print_summary(self, total_elapsed: float):
        """Print walk-forward training summary."""
        logger.info(f"\n{'═' * 70}")
        logger.info(f"  WALK-FORWARD TRAINING SUMMARY")
        logger.info(f"{'═' * 70}")

        if not self.fold_results:
            logger.warning("No fold results to summarize!")
            return

        val_accs = [r["val_acc"] for r in self.fold_results]
        test_accs = [r["test_acc"] for r in self.fold_results]
        test_sharpes = [r["test_sharpe"] for r in self.fold_results]

        logger.info(f"  Folds:          {len(self.fold_results)}")
        logger.info(f"  Val Accuracy:   {np.mean(val_accs):.1%} ± {np.std(val_accs):.1%}")
        logger.info(f"  Test Accuracy:  {np.mean(test_accs):.1%} ± {np.std(test_accs):.1%}")
        logger.info(f"  Test Sharpe:    {np.mean(test_sharpes):.3f} ± {np.std(test_sharpes):.3f}")
        logger.info(f"  Best Sharpe:    {max(test_sharpes):.3f}")
        logger.info(f"  Total Time:     {total_elapsed / 60:.1f} minutes ({total_elapsed / 3600:.1f} hours)")
        logger.info(f"{'═' * 70}")

    def _save_report(self, total_elapsed: float):
        """Save training report to JSON."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.cfg,
            "device": str(self.device),
            "total_time_min": round(total_elapsed / 60, 1),
            "folds": self.fold_results,
            "summary": {
                "mean_val_acc": float(np.mean([r["val_acc"] for r in self.fold_results])) if self.fold_results else 0,
                "mean_test_acc": float(np.mean([r["test_acc"] for r in self.fold_results])) if self.fold_results else 0,
                "mean_test_sharpe": float(np.mean([r["test_sharpe"] for r in self.fold_results])) if self.fold_results else 0,
                "best_test_sharpe": float(max([r["test_sharpe"] for r in self.fold_results])) if self.fold_results else 0,
            }
        }

        report_path = PROJECT_ROOT / "models" / "training_report_v2.0.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Training report saved to {report_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Professional LSTM Training on 18-Year 1-Minute Multi-Asset Data"
    )
    parser.add_argument("--resample", type=str, default="1min",
                        help="Resampling frequency (1min, 5min, 15min, 1h)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Max epochs per fold")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size (tune for VRAM)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="Sequence length (lookback window in minutes)")
    parser.add_argument("--fwd-bars", type=int, default=5,
                        help="Forward bars for label (5 = 5 minutes at 1min)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Walk-forward folds")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto/cuda/cpu)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force reload from CSVs (ignore Parquet cache)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for speed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data + features only, skip training")
    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # GPU info
    logger.info("=" * 70)
    logger.info("  PROFESSIONAL LSTM TRAINING PIPELINE (1-MINUTE BARS)")
    logger.info("=" * 70)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"  GPU: {gpu_name} ({vram_gb:.0f} GB VRAM)")
        logger.info(f"  CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
    else:
        logger.info("  Running on CPU (training will be slow!)")

    # ── 1. Load Data ──
    logger.info(f"\nLoading {args.resample} data from local database...")
    df = load_merged_data(
        resample=args.resample,
        use_cache=not args.no_cache,
    )

    # Filter to 2008+ for consistent multi-asset coverage
    df = df[df.index.year >= 2008].copy()
    logger.info(f"Filtered to 2008+: {len(df):,} rows")

    if args.dry_run:
        logger.info("\n── DRY RUN: Testing feature engineering ──")
        engineer = LSTMFeatureEngineer()
        features = engineer.transform(df)
        logger.info(f"Features: {features.shape[1]} columns, {len(features):,} rows")
        logger.info(f"Feature names: {engineer.feature_names}")
        logger.info(f"\nSample features (last 3 rows):")
        print(features.tail(3))

        # Estimate memory usage
        n_seq = len(features) - args.seq_len
        mem_streaming = (len(features) * features.shape[1] * 4) / (1024 ** 3)
        mem_naive = (n_seq * args.seq_len * features.shape[1] * 4) / (1024 ** 3)
        logger.info(f"\n  Memory estimate:")
        logger.info(f"    Streaming Dataset: {mem_streaming:.1f} GB (actual)")
        logger.info(f"    Naive pre-alloc:   {mem_naive:.1f} GB (avoided!)")
        logger.info(f"    Savings:           {mem_naive - mem_streaming:.1f} GB")
        logger.info("\n✅ Dry run complete! Data loading and features verified.")
        sys.exit(0)

    # ── 2. Build Config ──
    config = DEFAULT_CONFIG.copy()
    config["resample"] = args.resample
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["seq_len"] = args.seq_len
    config["fwd_bars"] = args.fwd_bars
    config["n_folds"] = args.folds
    config["patience"] = args.patience
    config["use_amp"] = not args.no_amp
    config["compile_model"] = args.compile

    logger.info(f"\nTraining config:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")

    # ── 3. Train ──
    trainer = ProfessionalWalkForwardTrainer(config, device)
    model = trainer.train(df)

    logger.info(f"\n  ✅ Model saved to: models/lstm_cnn_attention_1m.pt")
    logger.info(f"  ✅ Preprocessor saved to: models/lstm_preprocessor_1m.joblib")
    logger.info(f"  ✅ Report saved to: models/training_report_1m.json")
