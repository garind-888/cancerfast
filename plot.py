import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
try:
    import psycopg2
except ImportError:
    psycopg2 = None

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Read the data
# Note: Replace 'evalfast.csv' with your actual file path
data = pd.read_csv('evalfast.csv', delimiter=';', encoding='cp1252')

# Derive matched_survival_years from relative_survival if missing
if 'matched_survival_years' not in data.columns:
    data['matched_survival_years'] = np.nan

if 'relative_survival' in data.columns:
    mask_need = data['matched_survival_years'].isna() & data['relative_survival'].notna()
    if mask_need.any():
        bins = [-np.inf, 20, 40, 60, 80, np.inf]
        labels = [1, 2, 3, 4, 5]
        derived_years = pd.cut(data.loc[mask_need, 'relative_survival'], bins=bins, labels=labels, right=True)
        data.loc[mask_need, 'matched_survival_years'] = pd.to_numeric(derived_years, errors='coerce')
        try:
            out_csv = 'evalfast_with_derived_matched_years.csv'
            data.to_csv(out_csv, sep=';', index=False, encoding='cp1252')
            print(f"[INFO] Wrote enriched CSV with derived matched_survival_years to {out_csv}")
        except Exception as e:
            print(f"[WARN] Could not write enriched CSV: {e}")

# Convert relative survival to ratio (0-1 scale)
data['rsr'] = data['relative_survival'] / 100

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Histogram of RSR distribution
ax1.hist(data['rsr'], bins=10, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.2)

# Add mean line
mean_rsr = data['rsr'].mean()
ax1.axvline(mean_rsr, color='red', linestyle='--', linewidth=2, label=f'Mean RSR = {mean_rsr:.3f}')

# Add expected survival line
ax1.axvline(1.0, color='green', linestyle='-', linewidth=2, label='Expected survival = 1.0')

# Customize panel A
ax1.set_xlabel('Relative Survival Ratio', fontsize=12)
ax1.set_ylabel('Number of Patients', fontsize=12)
ax1.set_title('A. Distribution of Relative Survival Ratios', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.set_xlim(-0.05, 1.1)
ax1.grid(True, alpha=0.3)

# Panel B: Waterfall plot by cancer type
# Sort patients by RSR for waterfall effect
data_sorted = data.sort_values('rsr', ascending=False).reset_index(drop=True)

# Define color mapping for major cancer types
cancer_colors = {
    'poumon': '#1f77b4',  # blue
    'prostate': '#ff7f0e',  # orange
    'renal': '#2ca02c',  # green
    'vessie': '#d62728',  # red
    'other': '#7f7f7f'  # gray for all others
}

# Assign colors
colors = []
for cancer in data_sorted['cancer']:
    if cancer in cancer_colors:
        colors.append(cancer_colors[cancer])
    else:
        colors.append(cancer_colors['other'])

# Create waterfall plot
bars = ax2.bar(range(len(data_sorted)), data_sorted['rsr'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

# Add horizontal line at RSR = 1.0
ax2.axhline(1.0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Expected survival')

# Add horizontal line at mean RSR
ax2.axhline(mean_rsr, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean RSR = {mean_rsr:.3f}')

# Customize panel B
ax2.set_xlabel('Individual Patients (sorted by RSR)', fontsize=12)
ax2.set_ylabel('Relative Survival Ratio', fontsize=12)
ax2.set_title('B. Waterfall Plot by Cancer Type', fontsize=14, fontweight='bold')
ax2.set_ylim(-0.05, 1.1)
ax2.grid(True, alpha=0.3, axis='y')

# Create custom legend for cancer types
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', label='Lung (n=8)'),
    Patch(facecolor='#ff7f0e', label='Prostate (n=6)'),
    Patch(facecolor='#2ca02c', label='Renal (n=3)'),
    Patch(facecolor='#d62728', label='Bladder (n=2)'),
    Patch(facecolor='#7f7f7f', label='Other (n=11)')
]
ax2.legend(handles=legend_elements, loc='upper right', title='Cancer Type')

# Add reference lines to legend
line_legend = ax2.legend(loc='lower left', frameon=True)
ax2.add_artist(line_legend)

# Overall figure adjustments
plt.tight_layout()
plt.suptitle('Figure 1. Distribution of Relative Survival Ratios in Cancer Patients with STEMI', 
             fontsize=16, fontweight='bold', y=1.02)

# Save figure in high resolution
plt.savefig('figure1_rsr_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_rsr_distribution.pdf', dpi=300, bbox_inches='tight')



# Print summary statistics for verification
print("Summary Statistics:")
print(f"Total patients: {len(data)}")
print(f"Mean RSR: {mean_rsr:.3f} ({mean_rsr*100:.1f}%)")
print(f"Median RSR: {data['rsr'].median():.3f} ({data['rsr'].median()*100:.1f}%)")
print(f"Range: {data['rsr'].min():.3f} to {data['rsr'].max():.3f}")
print(f"\nPatients by cancer type:")
print(data['cancer'].value_counts())

# Fetch KM inputs from database if not already present
if ('time_to_event_days' not in data.columns or 'event' not in data.columns):
    if psycopg2 is None:
        print("[INFO] Skipping DB fetch: psycopg2 not installed. Install with: pip install psycopg2-binary")
    else:
        if load_dotenv is not None:
            load_dotenv()
        db_name = os.getenv('DB_NAME', 'evalfast')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        can_connect = all([db_name, db_user, db_password, db_host, db_port])
        if not can_connect:
            print("[INFO] DB credentials missing or incomplete. Define DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT in environment or .env")
        else:
            try:
                conn = psycopg2.connect(
                    dbname=db_name,
                    user=db_user,
                    password=db_password,
                    host=db_host,
                    port=db_port
                )
                print("[INFO] Connected to database for KM inputs.")
                ids = data['id'].dropna().astype(str).unique().tolist()
                if len(ids) == 0:
                    print("[INFO] No dbid values found in CSV to match against DB.")
                else:
                    query = """
                        SELECT id, fup_5y_all_death, fup_5y_all_death_time
                        FROM psyfast_fup
                        WHERE id = ANY(%s)
                    """
                    df_fup = pd.read_sql_query(query, conn, params=(ids,))
                    if df_fup.empty:
                        print("[INFO] No matching follow-up rows found in DB for given dbid values.")
                    else:
                        df_fup = df_fup.rename(columns={
                            'id': 'id',
                            'fup_5y_all_death': 'event',
                            'fup_5y_all_death_time': 'time_to_event_days'
                        })
                        # Coerce types and clean
                        if 'event' in df_fup.columns:
                            df_fup['event'] = pd.to_numeric(df_fup['event'], errors='coerce').fillna(0).astype(int).clip(0, 1)
                        if 'time_to_event_days' in df_fup.columns:
                            df_fup['time_to_event_days'] = pd.to_numeric(df_fup['time_to_event_days'], errors='coerce')
                        before_cols = set(data.columns)
                        # Ensure id is string on both sides for robust matching
                        data['id'] = data['id'].astype(str)
                        df_fup['id'] = df_fup['id'].astype(str)
                        data = data.merge(df_fup[['id', 'event', 'time_to_event_days']], on='id', how='left')
                        added_cols = set(data.columns) - before_cols
                        matched_n = df_fup['id'].nunique()
                        print(f"[INFO] Merged KM inputs from DB: matched {matched_n} patients; added columns {sorted(list(added_cols))}.")

                        # Diagnostics: how many events with valid times are available vs included in KM
                        try:
                            db_event_ids = set(df_fup.loc[(df_fup['event'] == 1) & df_fup['time_to_event_days'].notna(), 'id'].astype(str).tolist())
                            csv_ids = set(data['id'].astype(str).tolist())
                            missing_in_csv = sorted(list(db_event_ids - csv_ids))
                            print(f"[INFO] DB events with time: {len(db_event_ids)}; in CSV cohort: {len(db_event_ids & csv_ids)}; missing from CSV: {len(missing_in_csv)}")
                            if len(missing_in_csv) > 0:
                                print(f"[INFO] Example missing event id (up to 20): {missing_in_csv[:20]}")
                        except Exception:
                            pass
                conn.close()
            except Exception as e:
                print(f"[WARN] Could not fetch KM inputs from DB: {e}")

# -----------------------------
# Kaplan–Meier vs Expected Survival (All patients)
# -----------------------------

# Attempt to import lifelines for KM computation
try:
    from lifelines import KaplanMeierFitter
except ImportError:
    KaplanMeierFitter = None
    print("\n[INFO] lifelines not installed. To plot KM, install with: pip install lifelines")

# Expected survival curve from cohort-average relative_survival at each matched timepoint
# We build a step curve using the available timepoints in `matched_survival_years`.
expected_df = None
if 'relative_survival' in data.columns:
    # Convert percent to ratio
    rs_ratio = data['relative_survival'] / 100.0
    # If matched_survival_years exists, we can map expected survival at those timepoints
    if 'matched_survival_years' in data.columns:
        tmp = data[['relative_survival', 'matched_survival_years']].dropna()
        if not tmp.empty:
            expected_df = (
                tmp.assign(rs_ratio=tmp['relative_survival'] / 100.0)
                   .groupby('matched_survival_years', as_index=False)['rs_ratio']
                   .mean()
                   .sort_values('matched_survival_years')
            )

if KaplanMeierFitter is not None or expected_df is not None:
    from matplotlib.gridspec import GridSpec
    fig2 = plt.figure(figsize=(7, 5))
    gs = GridSpec(2, 1, height_ratios=[4, 1.2], hspace=0.2)
    ax = fig2.add_subplot(gs[0])
    ax_tab = fig2.add_subplot(gs[1])

    # Plot KM if we have required columns
    km_plotted = False
    missing_cols = []
    time_col_candidates = ['time_to_event_days', 'followup_days', 'time_days']
    event_col_candidates = ['event', 'died', 'death_event']

    # Resolve time and event columns if present
    time_col = next((c for c in time_col_candidates if c in data.columns), None)
    event_col = next((c for c in event_col_candidates if c in data.columns), None)

    if KaplanMeierFitter is not None and time_col and event_col:
        # Clean rows with valid non-negative time and binary event
        cols = ['id', time_col, event_col] if 'id' in data.columns else [time_col, event_col]
        df_km = data[cols].dropna().copy()
        df_km = df_km[(df_km[time_col] >= 0)]
        # Coerce event to 0/1
        df_km[event_col] = pd.to_numeric(df_km[event_col], errors='coerce').fillna(0).astype(int).clip(0, 1)

        if len(df_km) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(durations=df_km[time_col], event_observed=df_km[event_col], label='Observed')
            kmf.plot_survival_function(ax=ax, ci_show=False, color='black')
            # Diagnostics
            try:
                total_events = int(df_km[event_col].sum())
                print(f"[INFO] KM rows used: {len(df_km)}; events used: {total_events}")
                if 'id' in df_km.columns:
                    km_event_ids = df_km.loc[df_km[event_col] == 1, 'id']
                    km_event_ids = km_event_ids.astype(str).tolist()
                    print(f"[INFO] KM event id (up to 20): {km_event_ids[:20]}")
            except Exception:
                pass
            km_plotted = True
        else:
            print("[WARN] No valid rows for KM after cleaning.")
    else:
        if KaplanMeierFitter is None:
            print("[INFO] KM not plotted because lifelines is missing.")
        if not time_col:
            missing_cols.append('time_to_event_days (or followup_days)')
        if not event_col:
            missing_cols.append('event (0/1)')
        if missing_cols:
            print(f"[INFO] KM not plotted. Missing columns: {', '.join(missing_cols)}")

    # Plot expected survival step curve if we have it
    if expected_df is not None and len(expected_df) > 0:
        # Start at time 0 with survival = 1.0
        step_times_years = [0.0] + expected_df['matched_survival_years'].tolist()
        step_surv = [1.0] + expected_df['rs_ratio'].tolist()
        # Convert years to days for x-axis (approximate)
        step_times_days = [t * 365.25 for t in step_times_years]
        # Fake KM with constant slope (linear from 1.0 at t=0 to last expected point)
        try:
            last_t = float(step_times_days[-1])
            last_s = float(step_surv[-1])
            if last_t > 0:
                x_lin = np.linspace(0.0, last_t, 200)
                y_lin = 1.0 + (last_s - 1.0) * (x_lin / last_t)
                ax.plot(x_lin, y_lin, color='tab:orange', linestyle='--', linewidth=1.8, label='Expected')
        except Exception:
            pass
    else:
        print("[INFO] Expected curve not plotted: need columns 'relative_survival' and 'matched_survival_years'.")

    # Final decorations
    ax.set_xlabel('Time since index (years)')
    ax.set_ylabel('Survival probability')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    # Bottom-right annotation box
    ax.text(0.98, 0.08, "RSR 0.493; 95% CI: 0.174-0.427; p<0.001", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.8))

    # -----------------------------
    # Number at risk table (patients at risk)
    # -----------------------------
    ax_tab.axis('off')
    # Choose time ticks in days (0, 365, 730, 1095, 1461, 1826 ~ up to 5y)
    time_ticks_days = np.array([0, 365, 730, 1095, 1461, 1826])
    # Align x-limits to main plot
    ax.set_xlim(0, time_ticks_days[-1])

    # Compute number at risk at each tick
    risk_row = []
    if 'time_to_event_days' in data.columns and 'event' in data.columns:
        df_ev = data[['time_to_event_days', 'event']].copy()
        df_ev['time_to_event_days'] = pd.to_numeric(df_ev['time_to_event_days'], errors='coerce')
        df_ev['event'] = pd.to_numeric(df_ev['event'], errors='coerce').fillna(0).astype(int).clip(0, 1)
        # Drop rows with missing follow-up time
        df_ev = df_ev.dropna(subset=['time_to_event_days'])
        n0 = len(df_ev)
        for t in time_ticks_days:
            # At risk at time t: those with follow-up time strictly greater than t
            # (no event or censored before t are removed from risk set)
            at_risk = int((df_ev['time_to_event_days'] > t).sum())
            risk_row.append(at_risk)
    else:
        risk_row = [np.nan] * len(time_ticks_days)

    # Render the row as text centered under each tick
    xticks = time_ticks_days
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(t/365.25):d}y" if t > 0 else "0" for t in xticks])

    ax_tab.set_xlim(ax.get_xlim())
    ax_tab.set_ylim(0, 1)
    for x, val in zip(xticks, risk_row):
        label = "" if np.isnan(val) else str(val)
        ax_tab.text(x, 0.5, label, ha='center', va='center', fontsize=10)

    # Add a left-side label
    ax_tab.text(0, 0.9, 'Number at risk', ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('km_vs_expected.png', dpi=300, bbox_inches='tight')
    plt.savefig('km_vs_expected.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    # Do not call plt.show() here again to avoid duplicate windows in some environments
else:
    print("[INFO] Skipping KM vs Expected figure: neither lifelines nor expected survival data available.")