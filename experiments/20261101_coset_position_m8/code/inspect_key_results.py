"""提取正式报告所需的少量关键分层指标。"""
from pathlib import Path
import pandas as pd

p = Path(__file__).resolve().parent / "server_export" / "three_seed_analysis"
m = pd.read_csv(p / "逐种子逐SNR指标.csv")
c = pd.read_csv(p / "逐类指标.csv")
pair = pd.read_csv(p / "逐源配对差.csv")
print("METRICS")
print(m[m.snr_db.isin([-10,-6,-2,4,10])].to_string(index=False))
print("CLASSES")
print(c[(c.snr_db.isin([-2,10])) & (c.label.isin(['PSK8','PSK16','QAM64','QAM256']))].to_string(index=False))
print("PAIR")
print(pair[pair.snr_db.isin([-10,-6,-2,10])].to_string(index=False))
