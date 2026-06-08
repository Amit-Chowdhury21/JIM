# TFT_Pro_Max V5 Training Guide

This document outlines the steps to re-initiate the Institutional V5 training pipeline for the `TFT_Pro_Max` model.

## Prerequisites
- Ensure your GPU (RTX 5070 Ti) is not being heavily utilized by other applications (e.g., heavy gaming, other deep learning trainings).
- Ensure your machine is set to not go to sleep for at least 3 hours.

## Step 1: Open Your Terminal
Open a new terminal or command prompt window. 

## Step 2: Navigate to the Project Workspace
Navigate to the root directory of the project where the script is located.
```bash
cd e:\PRO\JIMxNik\jim_new
```

## Step 3: Launch the Training Script
Run the training script using Python. 
```bash
python scripts/train_tft_pro_max.py
```

> [!TIP]
> If you are using a virtual environment (like conda or venv), make sure to activate it before running the script.

## Step 4: Monitor the Progress
Once launched, the script will execute the following automated sequence:
1. **Data Prep:** Load and merge ~18 years of 5-minute Gold and Macro data (takes ~5 minutes).
2. **Cross Validation (75% of data):** Train the neural network across 3 Purged Walk-Forward folds.
3. **Evaluation Retrain:** Retrain the master Evaluation Model on the entire 75% block.
4. **Calibration (15% of data):** Sweep and freeze the optimal trading thresholds.
5. **Strict Promotion Gate (10% of data):** Simulate live trading on the final 1.8 years of unseen data using the frozen thresholds.

**Expected Training Time:** ~1.5 to 2.5 hours total.

## Step 5: Verify the Output
When you return, look at the end of the terminal log. 
- If the model survives the execution spread (5x rollover penalty) and prints a Win Rate > 50% and a Net PnL > 0, you will see:
  `"All Promotion Gates Passed. V5 Deployment Authorised."`
- The strictly calibrated, live-ready `.pt` model weights will be safely exported to:
  `models/tft_pro_max_checkpoint.pt`

If it fails the promotion gates, the script will deliberately crash with an error explaining which gate failed (PnL, Trade Count, or Coverage). This protects your capital.
