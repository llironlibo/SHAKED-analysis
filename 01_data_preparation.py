"""
Stage 01 -- Data preparation (provenance).

Builds the analytic cohort data/cleaned/unified_pilot_cohort.csv from the raw
de-identified hospital exports. Documents every cleaning decision: pilot-period
filter, SHAKED usage flags, comparison-group labels, outcome derivation, and the
consultation-timing merge.

Reproduces: the analytic dataset underlying Table 1 and all downstream analyses.

Inputs : data/raw/*.xlsx  (controlled access; NOT included in this repository)
Outputs: data/cleaned/unified_pilot_cohort.csv  (and wing_a_shaked_analysis.csv,
         consultation_data.csv, dataset_summary.md)

NOTE: The raw exports are not shipped (protected health information). This stage
is provided for transparency; it cannot run without controlled-access raw files.
The downstream stages 02-16 load the already-built cohort CSV directly.

The constants below (COL_* and VAL_*) are neutral English placeholders that map
to the original Hebrew electronic-health-record field names and field values in
the controlled-access raw exports. They are listed here in one place; the holder
of the controlled-access dataset substitutes the verbatim source labels. Code
logic is unaffected -- these strings are only used to address spreadsheet columns
and to match categorical values.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Neutral aliases for the controlled-access raw EMR column labels and values.
# (Original labels are Hebrew; replaced here with English placeholders.)
# =============================================================================
# --- Column labels (raw spreadsheet headers) ---
COL_ADMIT_DT        = 'ed_admission_datetime'          # ED registration date/time
COL_DAY_TYPE        = 'admission_day_type'             # weekday / weekend marker
COL_SHIFT           = 'admission_shift'                # shift (morning / etc.)
COL_CASE_ED         = 'ed_case_id'                     # ED case number (Wing A/B exports)
COL_CASE_CONSULT    = 'consult_case_id'                # case number (consultation export)
COL_PATIENT_ID      = 'patient_id_enc'                 # encrypted patient identifier
COL_AGE             = 'age_raw'                         # age
COL_SEX             = 'sex_raw'                         # sex
COL_FIRST_PHYS_DT   = 'first_physician_datetime'       # first physician exam date/time
COL_DECISION_DT     = 'disposition_decision_datetime'  # disposition decision date/time (min)
COL_LOS_HOURS       = 'los_hours_raw'                  # length of stay (hours)
COL_TRIAGE          = 'triage_acuity_raw'              # triage acuity label
COL_DISPOSITION     = 'disposition_decision'           # admit/discharge decision
COL_FIRST_WARD      = 'first_ward'                     # first ward in ED
COL_ADMIT_REASON    = 'admission_reason'               # reason for admission to hospital
COL_CHIEF_COMPLAINT = 'chief_complaint'                # presenting chief complaint
COL_ICD9_DESC       = 'icd9_diagnosis_desc'            # ICD-9 diagnosis description
COL_ICD9_CODE       = 'icd9_diagnosis_code'            # ICD-9 diagnosis code
COL_CONSULT_ORDER_DT = 'consult_order_datetime'        # consultation order date/time
COL_CONSULT_START_DT = 'consult_start_datetime'        # consultation start date/time
COL_CONSULT_END_DT   = 'consult_end_datetime'          # consultation end date/time
COL_CONSULT_STATION  = 'consult_station'               # consultation station/service

# --- Categorical field VALUES (matched against the columns above) ---
VAL_WEEKDAY           = 'weekday'                       # day-type value for a weekday
VAL_MORNING           = 'morning'                       # shift value for the morning shift
VAL_FEMALE            = 'female'                         # sex value for female
VAL_DISP_WARD         = 'admit_to_ward'                 # disposition: admitted to a ward
VAL_DISP_OTHER        = 'transfer_other_facility'       # disposition: transferred elsewhere
VAL_RADIOLOGY_STATION = 'emergency_radiology'           # station value: emergency radiology
VAL_MDA               = 'mda'                            # ambulance-service substring
VAL_AMBULANCE         = 'ambulance'                      # ambulance substring
# Triage acuity label values (immediate -> non-urgent):
VAL_TRIAGE_IMMEDIATE  = 'triage_immediate'
VAL_TRIAGE_URGENT     = 'triage_urgent'
VAL_TRIAGE_LESS_URGENT = 'triage_less_urgent'
VAL_TRIAGE_NON_URGENT = 'triage_non_urgent'

print("=" * 60)
print("SHAKED DATA PREPARATION PIPELINE")
print("=" * 60)

# =============================================================================
# STEP 1: Load raw data
# =============================================================================
print("\n[Step 1] Loading raw data...")

# Repo root resolved from this file's location.
BASE = os.path.dirname(os.path.abspath(__file__))
RAW_A = os.path.join(BASE, 'data', 'raw', 'wing_a_export.xlsx')
RAW_B = os.path.join(BASE, 'data', 'raw', 'wing_b_export.xlsx')

# The raw hospital exports are controlled-access and are not shipped with this
# repository. If they are absent (the usual public case), skip this provenance
# stage cleanly; stages 02-16 load the already-built cohort CSV directly.
if not (os.path.exists(RAW_A) and os.path.exists(RAW_B)):
    print("[SKIP] Raw exports not found under data/raw/.")
    print("       Stage 01 documents how the cohort CSV is built from the raw")
    print("       hospital exports, which are controlled-access (not shipped).")
    print("       Downstream stages use data/cleaned/unified_pilot_cohort.csv.")
    raise SystemExit(0)

xlsx_a = pd.ExcelFile(RAW_A)
xlsx_b = pd.ExcelFile(RAW_B)

ed_a = pd.read_excel(xlsx_a, sheet_name='ED Patient Data')
ed_b = pd.read_excel(xlsx_b, sheet_name='ED Patient Data')
shaked_actual = pd.read_excel(xlsx_a, sheet_name='Shaked Actual')
shaked_results = pd.read_excel(xlsx_a, sheet_name='Shaked Results')
consult_a = pd.read_excel(xlsx_a, sheet_name='Consultations')
consult_b = pd.read_excel(xlsx_b, sheet_name='Consultations')

print(f"  - Wing A ED: {len(ed_a):,} rows")
print(f"  - Wing B ED: {len(ed_b):,} rows")
print(f"  - Shaked Actual: {len(shaked_actual):,} rows")
print(f"  - Shaked Results: {len(shaked_results):,} rows")

# =============================================================================
# STEP 2: Filter to the pilot period (weekday morning shift)
# =============================================================================
print("\n[Step 2] Filtering to pilot period...")

date_col = COL_ADMIT_DT
ed_a[date_col] = pd.to_datetime(ed_a[date_col])
ed_b[date_col] = pd.to_datetime(ed_b[date_col])

pilot_start = pd.Timestamp('2024-11-09')
pilot_end = pd.Timestamp('2024-12-07 23:59:59')

# The exported timestamps carry the data-warehouse year; align the pilot window
# to whatever year the data actually uses.
actual_min = ed_a[date_col].min()
actual_max = ed_a[date_col].max()
print(f"  - Actual date range: {actual_min} to {actual_max}")

if actual_min.year == 2025:
    pilot_start = pd.Timestamp('2025-11-09')
    pilot_end = pd.Timestamp('2025-12-07 23:59:59')
    print(f"  - Adjusted pilot period: {pilot_start.date()} to {pilot_end.date()}")

# Filter: pilot period + weekday + morning shift.
def filter_pilot(df, date_col):
    initial = len(df)
    df_date = df[(df[date_col] >= pilot_start) & (df[date_col] <= pilot_end)]
    excl_date = initial - len(df_date)

    df_weekday = df_date[df_date[COL_DAY_TYPE] == VAL_WEEKDAY]
    excl_weekend = len(df_date) - len(df_weekday)

    df_final = df_weekday[df_weekday[COL_SHIFT] == VAL_MORNING]
    excl_shift = len(df_weekday) - len(df_final)

    print(f"    - Initial: {initial}")
    print(f"    - Excluded date range: {excl_date}")
    print(f"    - Excluded weekend: {excl_weekend}")
    print(f"    - Excluded shift: {excl_shift}")
    print(f"    - Final: {len(df_final)}")

    return df_final.copy()

ed_a_pilot = filter_pilot(ed_a, date_col)
ed_b_pilot = filter_pilot(ed_b, date_col)

print(f"  - Wing A pilot cohort: {len(ed_a_pilot):,} patients")
print(f"  - Wing B pilot cohort: {len(ed_b_pilot):,} patients")

# =============================================================================
# STEP 3: Create SHAKED usage flags
# =============================================================================
print("\n[Step 3] Creating SHAKED usage flags...")

case_col = COL_CASE_ED
shaked_actual_cases = set(shaked_actual['case_num'].dropna().astype(int).astype(str))
shaked_results_cases = set(shaked_results['Case num'].dropna().astype(int).astype(str))

ed_a_pilot['case_num_str'] = ed_a_pilot[case_col].astype(int).astype(str)
ed_a_pilot['is_shaked_opened'] = ed_a_pilot['case_num_str'].isin(shaked_actual_cases)
ed_a_pilot['has_shaked_results'] = ed_a_pilot['case_num_str'].isin(shaked_results_cases)
ed_a_pilot['is_technical_failure'] = ed_a_pilot['is_shaked_opened'] & ~ed_a_pilot['has_shaked_results']

print(f"  - SHAKED opened: {ed_a_pilot['is_shaked_opened'].sum():,}")
print(f"  - Has results: {ed_a_pilot['has_shaked_results'].sum():,}")
print(f"  - Technical failures: {ed_a_pilot['is_technical_failure'].sum():,}")
adoption_rate = ed_a_pilot['is_shaked_opened'].mean() * 100
print(f"  - Adoption rate: {adoption_rate:.1f}%")

# =============================================================================
# STEP 4: Define comparison groups
# =============================================================================
print("\n[Step 4] Defining comparison groups...")

# Intention-to-treat labels.
ed_a_pilot['wing'] = 'A'
ed_a_pilot['group_itt'] = 'Intervention'
ed_b_pilot['wing'] = 'B'
ed_b_pilot['group_itt'] = 'Control'

# Per-protocol labels (Wing A only).
ed_a_pilot['group_pp'] = np.where(ed_a_pilot['is_shaked_opened'], 'SHAKED User', 'Non-User')

# Placeholder columns on Wing B for a consistent schema.
ed_b_pilot['is_shaked_opened'] = False
ed_b_pilot['has_shaked_results'] = False
ed_b_pilot['is_technical_failure'] = False
ed_b_pilot['group_pp'] = 'Control'
ed_b_pilot['case_num_str'] = ed_b_pilot[case_col].astype(int).astype(str)

print("  - ITT: Intervention (Wing A) vs Control (Wing B)")
print("  - PP1: SHAKED Users (Wing A) vs Control (Wing B)")
print("  - PP2: SHAKED Users vs Non-Users (within Wing A)")

# =============================================================================
# STEP 5: Merge datasets
# =============================================================================
print("\n[Step 5] Merging datasets...")

common_cols = [
    case_col, COL_PATIENT_ID, COL_AGE, COL_SEX,
    COL_ADMIT_DT,
    COL_FIRST_PHYS_DT,
    COL_DECISION_DT,
    COL_LOS_HOURS,
    COL_TRIAGE,
    COL_DISPOSITION,
    COL_FIRST_WARD,
    COL_SHIFT,
    COL_DAY_TYPE,
    COL_ADMIT_REASON, COL_CHIEF_COMPLAINT
]

flag_cols = ['wing', 'group_itt', 'group_pp', 'is_shaked_opened', 'has_shaked_results', 'is_technical_failure', 'case_num_str']

cols_a = [c for c in common_cols if c in ed_a_pilot.columns] + flag_cols
cols_b = [c for c in common_cols if c in ed_b_pilot.columns] + flag_cols

unified = pd.concat([ed_a_pilot[cols_a], ed_b_pilot[cols_b]], ignore_index=True)
print(f"  - Unified dataset (raw rows): {len(unified):,} rows")

# Deduplicate cases that appear in both wings (patient moved); keep the row that
# records SHAKED engagement so the ITT label is preserved.
unified.sort_values(by=['is_shaked_opened', 'wing'], ascending=[False, True], inplace=True)
unified.drop_duplicates(subset=[case_col], keep='first', inplace=True)
unified.sort_values(by=['wing', 'group_itt'], inplace=True)

print(f"  - Unified dataset (unique cases): {len(unified):,} patients")

# =============================================================================
# STEP 5.5: Merge diagnosis data
# =============================================================================
print("\n[Step 5.5] Merging diagnosis data...")
diag_a = pd.read_excel(xlsx_a, sheet_name='Diagnosis')
diag_b = pd.read_excel(xlsx_b, sheet_name='Diagnosis')

diag_cols = {COL_ICD9_DESC: 'icd9_desc', COL_ICD9_CODE: 'icd9_code', COL_CASE_ED: case_col}
diag_a.rename(columns=diag_cols, inplace=True)
diag_b.rename(columns=diag_cols, inplace=True)

cohort_cases = unified[case_col].unique()
diag_a = diag_a[diag_a[case_col].isin(cohort_cases)]
diag_b = diag_b[diag_b[case_col].isin(cohort_cases)]

diag_all = pd.concat([diag_a, diag_b])

# Per patient: any diagnosis present, plus the list of ICD-9 codes.
diag_agg = diag_all.groupby(case_col).agg({
    'icd9_code': lambda x: list(x.dropna().unique()),
    'icd9_desc': lambda x: ' | '.join(x.dropna().unique().astype(str))
}).reset_index()

diag_agg['has_diagnosis_code'] = diag_agg['icd9_code'].apply(lambda x: len(x) > 0)
diag_agg['diagnosis_count'] = diag_agg['icd9_code'].apply(len)

unified = unified.merge(diag_agg, on=case_col, how='left')
unified['has_diagnosis_code'].fillna(False, inplace=True)
unified['diagnosis_count'].fillna(0, inplace=True)

print(f"  - Merged diagnosis data. Patients with >=1 code: {unified['has_diagnosis_code'].sum()} ({unified['has_diagnosis_code'].mean()*100:.1f}%)")

# =============================================================================
# STEP 6: Derive outcome variables
# =============================================================================
print("\n[Step 6] Deriving outcome variables...")

time_cols = [
    COL_ADMIT_DT,
    COL_FIRST_PHYS_DT,
    COL_DECISION_DT
]

for col in time_cols:
    if col in unified.columns:
        unified[col] = pd.to_datetime(unified[col], errors='coerce')

# Time to first physician (minutes).
unified['time_to_first_physician_min'] = (
    unified[COL_FIRST_PHYS_DT] -
    unified[COL_ADMIT_DT]
).dt.total_seconds() / 60

# Validity flag: drop negative times and extreme outliers (> 3 days).
unified['time_to_first_physician_valid'] = (unified['time_to_first_physician_min'] >= 0) & (unified['time_to_first_physician_min'] < 4320)
unified.loc[~unified['time_to_first_physician_valid'], 'time_to_first_physician_min'] = np.nan

# Time to disposition decision (minutes).
unified['time_to_decision_min'] = (
    unified[COL_DECISION_DT] -
    unified[COL_ADMIT_DT]
).dt.total_seconds() / 60

# Length of stay in hours (already provided; rename).
unified['los_hours'] = unified[COL_LOS_HOURS]

# Triage acuity mapping. Raw data mixes categorical acuity labels with P1-P5
# codes; '-1' (unknown) is left unmapped (NaN).
acuity_map = {
    VAL_TRIAGE_IMMEDIATE: 1, VAL_TRIAGE_URGENT: 2, VAL_TRIAGE_LESS_URGENT: 3, VAL_TRIAGE_NON_URGENT: 4,
    'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4, 'P5': 5
}
unified['triage_acuity_numeric'] = unified[COL_TRIAGE].map(acuity_map)

# Disposition flag (admitted to a ward or transferred to another facility).
unified['is_admitted'] = unified[COL_DISPOSITION].isin([VAL_DISP_WARD, VAL_DISP_OTHER])

# Temporal variables. admission_hour = ED arrival hour, NOT hospital admission.
unified['pilot_week'] = unified[COL_ADMIT_DT].dt.isocalendar().week
unified['admission_hour'] = unified[COL_ADMIT_DT].dt.hour
unified['is_weekend'] = unified[COL_ADMIT_DT].dt.dayofweek.isin([4, 5])

# Study weeks: merge the 1-day phantom ISO Week 45 into Week 46. ISO weeks
# 47/48/49 map to study weeks 2/3/4 unchanged. pilot_week is retained for the
# C(pilot_week) regression covariate.
unified['study_week'] = unified['pilot_week'].map({45: 1, 46: 1, 47: 2, 48: 3, 49: 4})

# Arrival mode (ambulance vs walk-in), if the source field is present.
if COL_ADMIT_REASON in unified.columns:
    unified['is_ambulance_arrival'] = unified[COL_ADMIT_REASON].str.contains(VAL_MDA, na=False) | \
                                      unified[COL_ADMIT_REASON].str.contains(VAL_AMBULANCE, na=False)
else:
    unified['is_ambulance_arrival'] = False

print(f"  - Time to first physician: mean={unified['time_to_first_physician_min'].mean():.1f} min")
print(f"  - Time to decision: mean={unified['time_to_decision_min'].mean():.1f} min")
print(f"  - LOS: mean={unified['los_hours'].mean():.2f} hours")

# =============================================================================
# STEP 7: Consultation timing metrics
# =============================================================================
print("\n[Step 7] Processing consultation data...")

consult_a['wing'] = 'A'
consult_b['wing'] = 'B'
consultations = pd.concat([consult_a, consult_b], ignore_index=True)

pilot_cases = set(unified['case_num_str'])
consultations['case_num_str'] = consultations[COL_CASE_CONSULT].astype(str)
consultations_pilot = consultations[consultations['case_num_str'].isin(pilot_cases)].copy()

print(f"  - Consultations in pilot period: {len(consultations_pilot):,}")

consult_time_cols = [COL_CONSULT_ORDER_DT, COL_CONSULT_START_DT, COL_CONSULT_END_DT]
for col in consult_time_cols:
    if col in consultations_pilot.columns:
        consultations_pilot[col] = pd.to_datetime(consultations_pilot[col], errors='coerce')

if COL_CONSULT_ORDER_DT in consultations_pilot.columns and COL_CONSULT_START_DT in consultations_pilot.columns:
    consultations_pilot['order_to_start_min'] = (
        consultations_pilot[COL_CONSULT_START_DT] -
        consultations_pilot[COL_CONSULT_ORDER_DT]
    ).dt.total_seconds() / 60

if COL_CONSULT_ORDER_DT in consultations_pilot.columns and COL_CONSULT_END_DT in consultations_pilot.columns:
    consultations_pilot['order_to_end_min'] = (
        consultations_pilot[COL_CONSULT_END_DT] -
        consultations_pilot[COL_CONSULT_ORDER_DT]
    ).dt.total_seconds() / 60

# Radiology flag (formal emergency-radiology consultation station).
consultations_pilot['is_radiology'] = consultations_pilot[COL_CONSULT_STATION] == VAL_RADIOLOGY_STATION

# Aggregate per case.
consult_summary = consultations_pilot.groupby('case_num_str').agg({
    COL_CONSULT_STATION: 'count',
    'order_to_start_min': 'mean',
    'order_to_end_min': 'mean',
    'is_radiology': 'any'
}).rename(columns={
    COL_CONSULT_STATION: 'n_consultations',
    'order_to_start_min':    'mean_order_to_start_min',
    'order_to_end_min': 'mean_order_to_end_min',
    'is_radiology': 'has_radiology_consult'
})

unified = unified.merge(consult_summary, left_on='case_num_str', right_index=True, how='left')
unified['n_consultations'] = unified['n_consultations'].fillna(0).astype(int)
unified['has_radiology_consult'].fillna(False, inplace=True)

# Primary exploratory outcome: mean order-to-end consultation cycle time.
unified['consultation_cycle_min'] = unified['mean_order_to_end_min']

print(f"  - Patients with consultations: {(unified['n_consultations'] > 0).sum():,}")
print(f"  - Mean consultations per patient: {unified['n_consultations'].mean():.2f}")

# =============================================================================
# STEP 8: Missing data and outliers
# =============================================================================
print("\n[Step 8] Handling missing data and outliers...")

unified['los_outlier'] = unified['los_hours'] > 24
unified['time_decision_outlier'] = unified['time_to_decision_min'] > 1440

print(f"  - LOS outliers (>24h): {unified['los_outlier'].sum():,}")
print(f"  - Time-to-decision outliers: {unified['time_decision_outlier'].sum():,}")

missing_summary = {}
for col in ['time_to_first_physician_min', 'time_to_decision_min', 'los_hours', 'triage_acuity_numeric']:
    if col in unified.columns:
        missing_summary[col] = unified[col].isna().sum()
print(f"  - Missing values: {missing_summary}")

# =============================================================================
# STEP 8b: Derive analysis convenience columns
# =============================================================================
print("\n[Step 8b] Deriving analysis convenience columns...")

cols = unified.columns.tolist()

# English aliases for the leading raw columns (addressed by position).
unified['case_id'] = unified[cols[0]].astype(str)
unified['age'] = pd.to_numeric(unified[cols[2]], errors='coerce')
unified['is_female'] = (unified[cols[3]].astype(str).str.strip() == VAL_FEMALE).astype(int)
unified['arrival_dt'] = pd.to_datetime(unified[cols[4]], errors='coerce')
unified['arrival_date'] = unified['arrival_dt'].dt.date.astype(str)

unified['is_user'] = unified['is_shaked_opened'].astype(int)
unified['is_wing_a'] = (unified['wing'] == 'A').astype(int)
unified['triage'] = unified['triage_acuity_numeric']
unified['has_radiology'] = unified['has_radiology_consult'].astype(int)
unified['is_ambulance'] = unified['is_ambulance_arrival'].astype(int)
unified['arrival_hour'] = unified['admission_hour']
unified['tfp_min'] = unified['time_to_first_physician_min']

unified['log_consult'] = np.log(unified['consultation_cycle_min'].clip(lower=0.01))
unified['consult_le_120'] = (unified['consultation_cycle_min'] <= 120).astype(int)
unified['consult_le_180'] = (unified['consultation_cycle_min'] <= 180).astype(int)

unified['day_index'] = (unified['arrival_dt'] - unified['arrival_dt'].min()).dt.days
unified['daily_volume'] = unified.groupby('arrival_date')['arrival_date'].transform('count')

# Mechanism-based engagement level.
# The published manuscript reports a 4-tier convention: NoAccess (control),
# Passive, TechFailure, and FullResults. A distinct non-user/per-protocol-
# excluded stratum (Wing A patients for whom SHAKED was neither opened nor
# produced output) is kept SEPARATE from Passive (it has has_shaked_results=False,
# whereas Passive has has_shaked_results=True), so Passive group sizes and all
# downstream Passive contrasts are unaffected. It is labelled neutrally below and
# is not reported as a standalone published tier.
def _engagement(row):
    if row['wing'] == 'B':
        return '0_NoAccess'
    if not row['is_shaked_opened']:
        if row['has_shaked_results']:
            return '2_Passive'
        return '1_NonUserExcluded'
    if row['is_technical_failure']:
        return '3_TechFailure'
    if row['has_shaked_results']:
        return '4_FullResults'
    return '2_Passive'

unified['engagement_level'] = unified.apply(_engagement, axis=1)
eng_map = {'0_NoAccess': 0, '1_NonUserExcluded': 1, '2_Passive': 2, '3_TechFailure': 3, '4_FullResults': 4}
unified['engagement_num'] = unified['engagement_level'].map(eng_map)

print(f"  - Engagement levels: {unified['engagement_level'].value_counts().sort_index().to_dict()}")
print(f"  - day_index range: {unified['day_index'].min()}-{unified['day_index'].max()}")
print(f"  - daily_volume range: {unified['daily_volume'].min()}-{unified['daily_volume'].max()}")

# =============================================================================
# STEP 9: Export analysis-ready datasets
# =============================================================================
print("\n[Step 9] Exporting datasets...")

output_dir = os.path.join(BASE, 'data', 'cleaned')
os.makedirs(output_dir, exist_ok=True)

unified.to_csv(f'{output_dir}/unified_pilot_cohort.csv', index=False, encoding='utf-8-sig')
print(f"  - unified_pilot_cohort.csv: {len(unified):,} rows")

wing_a_analysis = unified[unified['wing'] == 'A'].copy()
wing_a_analysis.to_csv(f'{output_dir}/wing_a_shaked_analysis.csv', index=False, encoding='utf-8-sig')
print(f"  - wing_a_shaked_analysis.csv: {len(wing_a_analysis):,} rows")

consultations_pilot.to_csv(f'{output_dir}/consultation_data.csv', index=False, encoding='utf-8-sig')
print(f"  - consultation_data.csv: {len(consultations_pilot):,} rows")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETE - SUMMARY")
print("=" * 60)

summary = f"""
## Dataset Summary

### Pilot Cohort
- **Total patients**: {len(unified):,}
- **Wing A (Intervention)**: {len(unified[unified['wing'] == 'A']):,}
- **Wing B (Control)**: {len(unified[unified['wing'] == 'B']):,}

### SHAKED Usage (Wing A only)
- **SHAKED opened**: {unified[unified['wing'] == 'A']['is_shaked_opened'].sum():,}
- **Has results**: {unified[unified['wing'] == 'A']['has_shaked_results'].sum():,}
- **Technical failures**: {unified[unified['wing'] == 'A']['is_technical_failure'].sum():,}
- **Adoption rate**: {unified[unified['wing'] == 'A']['is_shaked_opened'].mean()*100:.1f}%

### Key Metrics (All Patients)
- **Mean LOS**: {unified['los_hours'].mean():.2f} hours
- **Mean time to decision**: {unified['time_to_decision_min'].mean():.1f} min
- **Mean time to first physician**: {unified['time_to_first_physician_min'].mean():.1f} min
- **Patients with consultations**: {(unified['n_consultations'] > 0).sum():,} ({(unified['n_consultations'] > 0).mean()*100:.1f}%)

### Comparison Groups Ready
- **ITT**: {len(unified[unified['group_itt'] == 'Intervention']):,} vs {len(unified[unified['group_itt'] == 'Control']):,}
- **PP1**: {len(unified[(unified['wing'] == 'A') & (unified['is_shaked_opened'])]):,} SHAKED users vs {len(unified[unified['wing'] == 'B']):,} controls
- **PP2**: {len(unified[(unified['wing'] == 'A') & (unified['is_shaked_opened'])]):,} users vs {len(unified[(unified['wing'] == 'A') & (~unified['is_shaked_opened'])]):,} non-users (within Wing A)
"""

print(summary)

with open(f'{output_dir}/dataset_summary.md', 'w', encoding='utf-8') as f:
    f.write(summary)

print("\nDone!")
