import os
import pytz
from datetime import datetime

print("Testing LSTM CSV Log Creation...")

log_dir = r"E:\PRO\JIMxNik\LSTMLogs"
os.makedirs(log_dir, exist_ok=True)

ist = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(pytz.utc).astimezone(ist)

filename = f"lstm_logs_{now_ist.strftime('%Y%m%d_%H')}.csv"
filepath = os.path.join(log_dir, filename)

file_exists = os.path.exists(filepath)

# Mock signals
v1_sig = "HOLD"
v2_sig = "LONG"
v21_sig = "SHORT"
exec_str = "TRADED"
pnl_val = -12.50

try:
    with open(filepath, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("timestamp(ist),v1.0,v2.0,v2.1,Exec,PnL\n")
        f.write(f"{now_ist.strftime('%Y-%m-%d %H:%M:%S')},{v1_sig},{v2_sig},{v21_sig},{exec_str},{pnl_val:.2f}\n")
    print(f"✅ Successfully wrote mock log to {filepath}")
except Exception as e:
    print(f"❌ Failed to write log: {e}")

# Read it back
print("\nReading recent logs via logic used by endpoint:")
try:
    with open(filepath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        
    header = all_lines[0] if len(all_lines) > 0 else ""
    data_lines = all_lines[1:] if len(all_lines) > 1 else []
    recent_lines = data_lines[-5:] if len(data_lines) > 0 else []
    
    print("Header:", header.strip())
    for line in recent_lines:
        print("Data  :", line.strip())
    print("✅ Successfully verified read logic.")
except Exception as e:
    print(f"❌ Failed to read log: {e}")
