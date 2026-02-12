import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "n_liparteliani23_48213_server.log")
LOG_FILENAME = "n_liparteliani23_48213_server.log"

# ============================================================
# 1. PARSE THE LOG FILE
# ============================================================
print("=" * 60)
print("  DDoS ATTACK DETECTION - REGRESSION ANALYSIS")
print("=" * 60)

# Log format example:
# 4.131.115.232 - - [2024-03-22 18:00:32+04:00] "DELETE /usr/admin HTTP/1.0" 303 5048 "ref" "ua" 2026
log_pattern = re.compile(
    r'(?P<ip>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<datetime>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<url>\S+)\s+\S+"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\d+)'
)

records = []
with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        match = log_pattern.match(line)
        if match:
            d = match.groupdict()
            records.append(d)

print(f"\nTotal log entries parsed: {len(records)}")

df = pd.DataFrame(records)
df['timestamp'] = pd.to_datetime(df['datetime'], format='mixed', utc=True)
df['status'] = df['status'].astype(int)
df['size'] = df['size'].astype(int)

print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Unique IPs: {df['ip'].nunique()}")

# ============================================================
# 2. AGGREGATE REQUESTS PER MINUTE
# ============================================================
df['minute'] = df['timestamp'].dt.floor('min')
requests_per_minute = df.groupby('minute').agg(
    request_count=('ip', 'count'),
    unique_ips=('ip', 'nunique'),
    error_count=('status', lambda x: (x >= 400).sum())
).reset_index()

requests_per_minute = requests_per_minute.sort_values('minute').reset_index(drop=True)

requests_per_minute['minute_num'] = (
    (requests_per_minute['minute'] - requests_per_minute['minute'].min())
    .dt.total_seconds() / 60
).astype(int)

print(f"\nRequests per minute statistics:")
print(f"  Mean:   {requests_per_minute['request_count'].mean():.1f}")
print(f"  Median: {requests_per_minute['request_count'].median():.1f}")
print(f"  Max:    {requests_per_minute['request_count'].max()}")
print(f"  Std:    {requests_per_minute['request_count'].std():.1f}")

# ============================================================
# 3. REGRESSION ANALYSIS
# ============================================================
X = requests_per_minute['minute_num'].values.reshape(-1, 1)
y = requests_per_minute['request_count'].values

# --- Linear Regression ---
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)
r2_linear = lin_reg.score(X, y)

# --- Polynomial Regression (degree 3) ---
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
r2_poly = poly_reg.score(X_poly, y)

# --- Residuals and anomaly detection ---
residuals = y - y_pred_poly
residual_mean = residuals.mean()
residual_std = residuals.std()

threshold = residual_mean + 2 * residual_std
threshold_absolute = y_pred_poly + residual_mean + 2 * residual_std

requests_per_minute['predicted'] = y_pred_poly
requests_per_minute['residual'] = residuals
requests_per_minute['is_ddos'] = residuals > threshold

print(f"\n--- Regression Analysis ---")
print(f"  Linear R²:     {r2_linear:.4f}")
print(f"  Polynomial R²: {r2_poly:.4f}")
print(f"  Residual Mean: {residual_mean:.2f}")
print(f"  Residual Std:  {residual_std:.2f}")
print(f"  Anomaly Threshold: {threshold:.2f}")

# ============================================================
# 4. IDENTIFY DDoS TIME INTERVALS
# ============================================================
ddos_minutes = requests_per_minute[requests_per_minute['is_ddos']].copy()

intervals = []
if len(ddos_minutes) > 0:
    ddos_times = ddos_minutes['minute'].sort_values().tolist()
    start = ddos_times[0]
    prev = ddos_times[0]
    for t in ddos_times[1:]:
        if (t - prev).total_seconds() <= 120:
            prev = t
        else:
            intervals.append((start, prev))
            start = t
            prev = t
    intervals.append((start, prev))

print(f"\n{'=' * 60}")
print(f"  DDoS ATTACK INTERVALS DETECTED: {len(intervals)}")
print(f"{'=' * 60}")
for i, (s, e) in enumerate(intervals):
    duration = (e - s).total_seconds() / 60
    iv = requests_per_minute[(requests_per_minute['minute'] >= s) & (requests_per_minute['minute'] <= e)]
    print(f"\n  Interval {i+1}:")
    print(f"    Start:       {s}")
    print(f"    End:         {e}")
    print(f"    Duration:    {duration:.0f} minutes")
    print(f"    Avg req/min: {iv['request_count'].mean():.0f}")
    print(f"    Max req/min: {iv['request_count'].max()}")

# ============================================================
# 5. TOP ATTACKER IPs
# ============================================================
ddos_mask = pd.Series(False, index=df.index)
if len(intervals) > 0:
    for s, e in intervals:
        ddos_mask |= (df['minute'] >= s) & (df['minute'] <= e)
    ddos_traffic = df[ddos_mask]
    top_ips = ddos_traffic['ip'].value_counts().head(15)
    print(f"\n--- Top 15 Attacker IPs during DDoS ---")
    for ip, count in top_ips.items():
        print(f"    {ip:20s} -> {count} requests")
else:
    ddos_traffic = pd.DataFrame()
    top_ips = pd.Series(dtype=int)

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\nGenerating visualizations...")

# --- Figure 1: Requests Per Minute + Regression ---
fig, ax = plt.subplots(figsize=(16, 7))
ax.bar(requests_per_minute['minute'], requests_per_minute['request_count'],
       width=pd.Timedelta(seconds=50), alpha=0.5, color='steelblue', label='Requests/min')
ax.plot(requests_per_minute['minute'], y_pred_linear,
        color='green', linewidth=2, linestyle='--', label=f'Linear Regression (R²={r2_linear:.3f})')
ax.plot(requests_per_minute['minute'], y_pred_poly,
        color='orange', linewidth=2.5, label=f'Polynomial Regression deg=3 (R²={r2_poly:.3f})')
ax.plot(requests_per_minute['minute'], threshold_absolute,
        color='red', linewidth=1.5, linestyle=':', label='DDoS Threshold (mean + 2σ)')
for idx_iv, (s, e) in enumerate(intervals):
    ax.axvspan(s, e, alpha=0.2, color='red', label='DDoS Interval' if idx_iv == 0 else '')
ax.set_xlabel('Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Requests per Minute', fontsize=13, fontweight='bold')
ax.set_title('Web Server Traffic - Requests per Minute with Regression Analysis', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'requests_per_minute.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: requests_per_minute.png")

# --- Figure 2: Residual Analysis ---
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
colors_res = ['red' if d else 'steelblue' for d in requests_per_minute['is_ddos']]
axes[0].bar(requests_per_minute['minute'], requests_per_minute['residual'],
            width=pd.Timedelta(seconds=50), color=colors_res, alpha=0.7)
axes[0].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.1f}')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0].set_xlabel('Time', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12, fontweight='bold')
axes[0].set_title('Residual Analysis - Red Bars Indicate DDoS Traffic', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

axes[1].fill_between(requests_per_minute['minute'], requests_per_minute['unique_ips'], alpha=0.4, color='purple')
axes[1].plot(requests_per_minute['minute'], requests_per_minute['unique_ips'], color='purple', linewidth=1.5, label='Unique IPs/min')
for idx_iv, (s, e) in enumerate(intervals):
    axes[1].axvspan(s, e, alpha=0.2, color='red', label='DDoS' if idx_iv == 0 else '')
axes[1].set_xlabel('Time', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Unique IPs per Minute', fontsize=12, fontweight='bold')
axes[1].set_title('Unique Source IPs per Minute', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'residual_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: residual_analysis.png")

# --- Figure 3: Top Attacker IPs ---
if len(top_ips) > 0:
    fig, ax = plt.subplots(figsize=(12, 7))
    top_plot = top_ips.head(15)
    bars = ax.barh(range(len(top_plot)), top_plot.values, color='crimson', alpha=0.8)
    ax.set_yticks(range(len(top_plot)))
    ax.set_yticklabels(top_plot.index, fontsize=10)
    ax.set_xlabel('Number of Requests', fontsize=13, fontweight='bold')
    ax.set_title('Top 15 Attacker IPs During DDoS Intervals', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, top_plot.values):
        ax.text(val + max(top_plot.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'top_attackers.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: top_attackers.png")

# --- Figure 4: HTTP Status Codes ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
normal_traffic = df[~ddos_mask]
normal_status = normal_traffic['status'].value_counts().head(8)
axes[0].pie(normal_status.values, labels=[str(s) for s in normal_status.index],
            autopct='%1.1f%%', colors=plt.cm.Greens(np.linspace(0.3, 0.8, len(normal_status))))
axes[0].set_title('HTTP Status Codes - Normal Traffic', fontsize=13, fontweight='bold')
if len(ddos_traffic) > 0:
    ddos_status = ddos_traffic['status'].value_counts().head(8)
    axes[1].pie(ddos_status.values, labels=[str(s) for s in ddos_status.index],
                autopct='%1.1f%%', colors=plt.cm.Reds(np.linspace(0.3, 0.8, len(ddos_status))))
    axes[1].set_title('HTTP Status Codes - DDoS Traffic', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'status_codes.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: status_codes.png")
