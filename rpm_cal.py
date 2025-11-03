# app_fertilizer_rpm_from_kraphor.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from math import ceil
import os
import json
from pathlib import Path

st.set_page_config(
    page_title="โปรแกรมคำนวณสูตรปุ๋ยและแนะนำการปรับค่าเครื่องจ่ายเกลียวลำเลียง AGN03", 
    layout="wide",
    page_icon="🌾"
)

# Custom CSS for agricultural theme
st.markdown("""
<style>
    /* Main theme colors - Agricultural/Farm theme */
    :root {
        --primary-green: #4a7c59;
        --light-green: #8fbc8f;
        --earth-brown: #8b7355;
        --cream: #f5f5dc;
        --soft-yellow: #f0e68c;
    }
    
    /* Header styling */
    h1 {
        color: #2d5016 !important;
        font-weight: 700 !important;
        padding: 1rem 0 !important;
        border-bottom: 3px solid #8fbc8f !important;
    }
    
    h2 {
        color: #4a7c59 !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    
    h3 {
        color: #5a8a6a !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f8f0 0%, #e8f5e8 100%) !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #2d5016 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #5a8a6a 0%, #4a7c59 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4a7c59 0%, #3a6c49 100%) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border: 2px solid #c8e6c9 !important;
        border-radius: 6px !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #4a7c59 !important;
        box-shadow: 0 0 0 0.2rem rgba(74, 124, 89, 0.25) !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #c8e6c9 0%, #4a7c59 100%) !important;
    }
    
    /* Table styling */
    table {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }
    
    thead tr th {
        background: linear-gradient(135deg, #5a8a6a 0%, #4a7c59 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
    }
    
    tbody tr:nth-child(odd) {
        background-color: #f9fdf9 !important;
    }
    
    tbody tr:hover {
        background-color: #e8f5e8 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f8f0 !important;
        border-radius: 6px !important;
        border: 1px solid #c8e6c9 !important;
        font-weight: 600 !important;
        color: #2d5016 !important;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess {
        background-color: #d4edda !important;
        border-left: 4px solid #4a7c59 !important;
    }
    
    .stWarning {
        background-color: #fff3cd !important;
        border-left: 4px solid #ffc107 !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        border-left: 4px solid #dc3545 !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px !important;
    }
    
    /* Divider */
    hr {
        border-color: #c8e6c9 !important;
        margin: 1.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_JSON = BASE_DIR / "data" / "calibration_default.json"
DEFAULT_EXCEL = BASE_DIR / "กระพ้อ.xlsx"  # ใช้เมื่อรันใน local เท่านั้น
# --------- HELPERS ----------
def load_calibration_from_json(path: Path = CALIBRATION_JSON):
    """Load embedded calibration data from JSON if available."""
    try:
        if path.exists():
            with path.open('r', encoding='utf-8') as fp:
                return json.load(fp)
    except Exception:
        return None
    return None

# --------- HELPERS ----------
def create_default_groups():
    """สร้างข้อมูลตัวอย่างเริ่มต้นสำหรับการทดสอบ (ใช้เมื่อไม่มีไฟล์ Excel)"""
    # พยายามโหลดจากไฟล์ JSON ที่ฝังอยู่ในโปรแกรมก่อน
    calibration_data = load_calibration_from_json()
    if not calibration_data:
        # Fallback ค่าเดิมถ้าไฟล์หาย
        calibration_data = {
            'N': {
                'rpm': [10, 500, 1000, 1500, 2000, 2500, 2750],
                'rate_gps': [0.2, 10, 20, 30, 40, 50, 55]
            },
            'P': {
                'rpm': [10, 500, 1000, 1500, 2000, 2500, 2750],
                'rate_gps': [0.15, 7.5, 15, 22.5, 30, 37.5, 41.25]
            },
            'K': {
                'rpm': [10, 500, 1000, 1500, 2000, 2500, 2750],
                'rate_gps': [0.1, 5, 10, 15, 20, 25, 27.5]
            }
        }

    groups = {}
    for hopper, data in calibration_data.items():
        rpm_actual = np.array(data.get('rpm') or data.get('rpms', []), dtype=float)
        if rpm_actual.size == 0:
            continue
        rpm_pct = np.array(data.get('rpm_pct', rpm_actual * 100.0 / 2750.0), dtype=float)
        rates = np.array(data.get('rate_gps') or data.get('rates', []), dtype=float)
        touts = np.array(data.get('t_out', np.zeros_like(rpm_actual)), dtype=float)
        talls = np.array(data.get('t_all', np.full_like(rpm_actual, 3600.0)), dtype=float)
        losses = np.array(data.get('loss', np.zeros_like(rpm_actual)), dtype=float)
        effs = np.array(data.get('eff', np.zeros_like(rpm_actual)), dtype=float)

        # สร้าง DataFrame เพื่อใช้แสดงผลใน UI
        data_df = pd.DataFrame({
            'hopper': hopper,
            'rpm': rpm_actual,
            'rpm_pct': rpm_pct,
            'g/s': rates,
            't_out': touts,
            't_all': talls,
            'loss': losses,
            'eff': effs
        })

        groups[hopper] = {
            'rpm_min': float(rpm_actual.min()), 
            'rpm_max': float(rpm_actual.max()),
            'is_percentage': False,
            'rate_func': interp1d(rpm_actual, rates, kind='linear', fill_value='extrapolate', bounds_error=False),
            'tout_func': interp1d(rpm_actual, touts, kind='linear', fill_value='extrapolate', bounds_error=False),
            'tall_func': interp1d(rpm_actual, talls, kind='linear', fill_value='extrapolate', bounds_error=False),
            'loss_func': interp1d(rpm_actual, losses, kind='linear', fill_value='extrapolate', bounds_error=False),
            'raw_rpms': rpm_actual,
            'raw_rpm_pct': rpm_pct,
            'raw_rates': rates,
            'raw_t_out': touts,
            'raw_t_all': talls,
            'raw_loss': losses,
            'raw_eff': effs,
            'data': data_df
        }
    return groups

def find_col(cols, target):
    for c in cols:
        if c is None:
            continue
        if c.lower().replace(' ', '') == target.lower().replace(' ', ''):
            return c
    return None

@st.cache_data
def load_testdata(path):
    """Load Excel and parse into groups per hopper (N,P,K). Return groups, proc_df, error"""
    if not os.path.exists(path):
        return None, None, f"File not found: {path}"
    try:
        df = pd.read_excel(path)
    except Exception as e:
        return None, None, f"Error reading Excel: {e}"
    # parse: column0 contains markers 'N','P','K' and rows with rpm numeric
    col0 = df.columns[0]
    col0vals = df[col0].astype(str).str.strip()
    rows = []
    current = None
    for idx, val in col0vals.items():
        if val in ['N','P','K']:
            current = val
            continue
        try:
            rpm = float(val)
        except:
            continue
        row_series = df.loc[idx].copy()
        row = row_series.to_dict()
        row['hopper'] = current
        row['rpm'] = rpm
        rows.append(row)
    if not rows:
        return None, None, "Parsed zero rows — ตรวจสอบโครงสร้างไฟล์ (คอลัมน์แรกต้องมี N/P/K และบรรทัด rpm)"
    proc = pd.DataFrame(rows)
    cols = proc.columns.tolist()
    col_rate = find_col(cols, 'g/s') or find_col(cols, 'gpersec') or find_col(cols, 'rate')
    col_tout = find_col(cols, 't out') or find_col(cols, 'tout') or find_col(cols, 't_out')
    col_tall = find_col(cols, 't all') or find_col(cols, 'tall') or find_col(cols, 't_all')
    col_loss = find_col(cols, 'loss') or find_col(cols, 'loss_g')
    keep = ['hopper', 'rpm']
    for c in [col_rate, col_tout, col_tall, col_loss]:
        if c and c not in keep:
            keep.append(c)
    proc = proc[[c for c in keep if c in proc.columns]].dropna().reset_index(drop=True)
    groups = {}
    for h in proc['hopper'].unique():
        sub = proc[proc['hopper'] == h].copy()
        xs_original = sub['rpm'].values.astype(float)
        order = np.argsort(xs_original)
        sub = sub.iloc[order].copy()
        xs_sorted = xs_original[order]

        is_percentage = xs_sorted.max() <= 100
        if is_percentage:
            rpm_actual = xs_sorted * 2750.0 / 100.0
            rpm_pct = xs_sorted
        else:
            rpm_actual = xs_sorted
            rpm_pct = xs_sorted * 100.0 / 2750.0

        sub['rpm'] = rpm_actual
        sub['rpm_actual'] = rpm_actual
        sub['rpm_pct'] = rpm_pct

        rates = sub[col_rate].values.astype(float) if col_rate in sub.columns else np.zeros_like(rpm_actual)
        touts = sub[col_tout].values.astype(float) if col_tout in sub.columns else np.zeros_like(rpm_actual)
        talls = sub[col_tall].values.astype(float) if col_tall in sub.columns else np.full_like(rpm_actual, 1e6)
        losses = sub[col_loss].values.astype(float) if col_loss in sub.columns else np.zeros_like(rpm_actual)

        try:
            groups[h] = {
                'rpm_min': float(rpm_actual.min()), 
                'rpm_max': float(rpm_actual.max()),
                'is_percentage': False,
                'rate_func': interp1d(rpm_actual, rates, kind='linear', fill_value='extrapolate', bounds_error=False),
                'tout_func': interp1d(rpm_actual, touts, kind='linear', fill_value='extrapolate', bounds_error=False),
                'tall_func': interp1d(rpm_actual, talls, kind='linear', fill_value='extrapolate', bounds_error=False),
                'loss_func': interp1d(rpm_actual, losses, kind='linear', fill_value='extrapolate', bounds_error=False),
                'raw_rpms': rpm_actual,
                'raw_rpm_pct': rpm_pct,
                'raw_rates': rates,
                'data': sub
            }
        except Exception as e:
            return None, None, f"Error building interpolators for hopper {h}: {e}"
    return groups, proc, None

# ---------- Composition calculation ----------
def calc_parents_from_formula(N_pct, P_pct, K_pct, total_kg):
    # parents: A 46-0-0, B 18-46-0, C 0-0-60
    A_N = 0.46
    B_N, B_P = 0.18, 0.46
    C_K = 0.60
    t = float(total_kg)
    n = float(N_pct) / 100.0
    p = float(P_pct) / 100.0
    k = float(K_pct) / 100.0
    # compute B from P, C from K, A from remaining N
    y = (p * t) / B_P  # kg of B
    z = (k * t) / C_K  # kg of C
    x = (n * t - B_N * y) / A_N  # kg of A
    sum_ferts = x + y + z
    filler = t - sum_ferts
    return {'A_46_0_0_kg': x, 'B_18_46_0_kg': y, 'C_0_0_60_kg': z, 'sum_ferts_kg': sum_ferts, 'filler_kg': filler}

# ---------- Planner: find t (equal) and rpm per hopper to match parent masses ----------
def evaluate_run_for_t_with_targets(groups, target_masses_g, t, tol=0.05, cap_by_tall=True, rpm_min_pct=20, rpm_max_pct=80):
    """
    target_masses_g: dict {'N': grams, 'P': grams, 'K': grams}
    t: run time in seconds (equal for all hoppers)
    rpm_min_pct, rpm_max_pct: ช่วง % ของ RPM ที่ต้องการ (20-80%)
    returns dict or error
    """
    rpm_choices = {}
    for h in ['N','P','K']:
        if groups is None or h not in groups:
            return {'ok': False, 'reason': f'No test data for hopper {h}'}
        funcs = groups[h]
        
        # จำกัดช่วง RPM ตาม % ที่กำหนด (ใช้ %RPM โดยตรง)
        rpm_range = funcs['rpm_max'] - funcs['rpm_min']
        rpm_min_search = funcs['rpm_min'] + (rpm_range * rpm_min_pct / 100.0)
        rpm_max_search = funcs['rpm_min'] + (rpm_range * rpm_max_pct / 100.0)
        
        rpms = np.linspace(rpm_min_search, rpm_max_search, 2000)  # %RPM
        rates = funcs['rate_func'](rpms)     # g/s
        touts = funcs['tout_func'](rpms)     # s
        talls = funcs['tall_func'](rpms)     # s
        losses = funcs['loss_func'](rpms)    # g
        
        if cap_by_tall:
            eff_times = np.maximum(0.0, np.minimum(t, talls) - touts)
        else:
            eff_times = np.maximum(0.0, t - touts)
        masses = rates * eff_times  # grams delivered
        usable_masses = masses - losses  # หักค่า loss ออก
        
        target = float(target_masses_g.get(h, 0.0))
        # pick rpm that gives usable mass closest to target
        idx = np.argmin(np.abs(usable_masses - target))
        mass_total = float(masses[idx])
        loss = float(losses[idx])
        mass_usable = float(usable_masses[idx])
        rel_err = abs(mass_usable - target) / (target + 1e-9)
        
        # ตรวจสอบว่าข้อมูลเป็น %RPM หรือ RPM เต็ม
        rpm_value = float(rpms[idx])
        if funcs.get('is_percentage', False):
            # ข้อมูลเป็น %RPM อยู่แล้ว
            rpm_pct = rpm_value
            rpm_actual = rpm_pct * 2750.0 / 100.0
        else:
            # ข้อมูลเป็น RPM เต็ม ต้องแปลงเป็น %
            rpm_actual = rpm_value
            rpm_pct = rpm_value * 100.0 / 2750.0
        
        rpm_choices[h] = {
            'rpm_actual': rpm_actual,
            'rpm_pct': rpm_pct,
            'rate_gps': float(rates[idx]),
            'tout_s': float(touts[idx]),
            'tall_s': float(talls[idx]),
            'mass_g': mass_usable,  # ใช้ค่าหลังหัก loss
            'mass_total_g': mass_total,  # เก็บค่าก่อนหัก loss ไว้ด้วย
            'rel_err': rel_err,
            'loss_g': loss
        }
    total_mass_g = sum([rpm_choices[h]['mass_g'] for h in rpm_choices])
    total_loss_g = sum([rpm_choices[h]['loss_g'] for h in rpm_choices])
    return {'ok': True, 't': t, 'settings': rpm_choices, 'total_mass_g': total_mass_g, 'total_loss_g': total_loss_g}

def find_t_for_parent_masses(groups, target_masses_g, t_min=1.0, t_max=3600.0, t_steps=800, tol=0.05, cap_by_tall=True, rpm_min_pct=20, rpm_max_pct=80):
    """
    Search t in [t_min, t_max] (equal for all hoppers) to find first t that yields per-hopper mass within tol.
    If not found, return best single-run (t that maximizes total_mass closeness) for diagnostics.
    rpm_min_pct, rpm_max_pct: ช่วง % ของ RPM (20-80%)
    """
    t_search = np.linspace(t_min, t_max, t_steps)
    feasible = []
    for t in t_search:
        res = evaluate_run_for_t_with_targets(groups, target_masses_g, t, tol=tol, cap_by_tall=cap_by_tall, rpm_min_pct=rpm_min_pct, rpm_max_pct=rpm_max_pct)
        if res.get('ok'):
            # check each hopper relative error within tol
            errs = [res['settings'][h]['rel_err'] for h in ['N','P','K']]
            if all(e <= tol for e in errs):
                return {'found': True, 'result': res}
            feasible.append((t, res))
    # not found: find best by total_mass closeness (or max total mass)
    best = None
    for t, res in feasible:
        if best is None or res['total_mass_g'] > best['total_mass_g']:
            best = res
    # if no feasible at all, compute t that gives max total_mass (ignoring tol)
    if best is None:
        best_overall = None
        for t in t_search:
            res = evaluate_run_for_t_with_targets(groups, target_masses_g, t, tol=tol, cap_by_tall=cap_by_tall, rpm_min_pct=rpm_min_pct, rpm_max_pct=rpm_max_pct)
            if best_overall is None or res['total_mass_g'] > best_overall['total_mass_g']:
                best_overall = res
        return {'found': False, 'best_single_run': best_overall}
    return {'found': False, 'best_single_run': best}

# ---------- Streamlit UI ----------
st.title("🌾 โปรแกรมคำนวณสูตรปุ๋ยและแนะนำการปรับค่าเครื่องจ่ายเกลียวลำเลียง AGN03")

# Sidebar with instructions
st.sidebar.header("📖 คู่มือการใช้งาน")
st.sidebar.markdown("""
### ขั้นตอนการใช้งาน
1. **อัปโหลดไฟล์ข้อมูลทดลอง** (ถ้ามี) หรือใช้ข้อมูลตัวอย่าง
2. **ป้อนสูตร N-P-K** ที่ต้องการ (%)
3. **ระบุน้ำหนักรวม** ที่ต้องการผลิต (kg)
4. **กดปุ่มคำนวณ** เพื่อดูผลลัพธ์

### ผลลัพธ์ที่ได้
- ปริมาณแม่ปุ๋ยแต่ละชนิด (kg)
  - 46-0-0 (ยูเรีย)
  - 18-46-0 (DAP)
  - 0-0-60 (โพแทช)
- ค่า %RPM ที่เหมาะสมสำหรับแต่ละ hopper
- เวลาที่ใช้ต่อรอบการผลิต
- ปริมาณที่ผลิตได้และการสูญเสียที่คาดการณ์

### การปรับแต่ง
- **tol**: ความคลาดเคลื่อนที่ยอมรับได้ (ค่าน้อย = แม่นยำมากขึ้น)
- **t_min/t_max**: ช่วงเวลาที่ใช้ค้นหา (วินาที)

### หมายเหตุ
1. กรณีที่สารตัวเติมติดลบ ให้ลองเลือกแม่ปุ๋ยที่มีธาตุอาหารสูงกว่า
2. ถ้าคำนวณได้สารตัวเติมติดลบ ปุ๋ยที่ได้จะมีสูตรต่ำกว่าที่ต้องการ และใช้ปริมาณมากกว่า จึงได้ธาตุอาหารเท่ากัน
""")

# File uploader
st.sidebar.header("📁 ข้อมูลทดลอง")
# ตัวเลือกเพื่อทำให้ผลลัพธ์ตรงกับข้อมูลที่ฝังในโปรแกรมทั้งบน Cloud และเครื่องตัวเอง
default_force_sample = True
force_sample = st.sidebar.checkbox(
    "ใช้ข้อมูลสอบเทียบที่มาพร้อมโปรแกรม",
    value=default_force_sample,
    help="ติ๊กเพื่อใช้ข้อมูลสอบเทียบที่บันทึกไว้ในโปรแกรม (ค่าเดียวกับที่ใช้บนเว็บไซต์)"
)

uploaded_file = st.sidebar.file_uploader(
    "อัปโหลดไฟล์ Excel (ถ้ามี)", 
    type=["xlsx", "xls"],
    help="อัปโหลดไฟล์ข้อมูลการทดลองของคุณ หรือใช้ข้อมูลตัวอย่างเริ่มต้น"
)

# Load data from uploaded file or default
data_source = ""
if uploaded_file is not None and not force_sample:
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        groups, proc_df, load_err = load_testdata(tmp_path)
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        if load_err:
            st.sidebar.error(f"ไม่สามารถอ่านไฟล์: {load_err}")
            st.sidebar.info("กำลังใช้ข้อมูลตัวอย่างแทน")
            groups = create_default_groups()
            proc_df = None
            data_source = "sample"
        else:
            st.sidebar.success("✅ ใช้ข้อมูลจากไฟล์ที่อัปโหลด")
            data_source = "uploaded"
    except Exception as e:
        st.sidebar.error(f"เกิดข้อผิดพลาด: {e}")
        st.sidebar.info("กำลังใช้ข้อมูลตัวอย่างแทน")
        groups = create_default_groups()
        proc_df = None
        data_source = "sample"
elif (not force_sample) and DEFAULT_EXCEL.exists():
    # ใช้ไฟล์ local ถ้ามี (สำหรับการรันใน local)
    groups, proc_df, load_err = load_testdata(DEFAULT_EXCEL)
    if load_err:
        st.sidebar.warning("ไม่พบไฟล์ข้อมูล - ใช้ข้อมูลตัวอย่าง")
        groups = create_default_groups()
        proc_df = None
        data_source = "sample"
    else:
        st.sidebar.info("📊 ใช้ข้อมูลจากไฟล์ local")
        data_source = "local-excel"
else:
    # ใช้ข้อมูลตัวอย่างเริ่มต้น
    st.sidebar.info("📊 ใช้ข้อมูลตัวอย่างเริ่มต้น")
    groups = create_default_groups()
    proc_df = None
    data_source = "sample"

# Initialize session state for results
if 'rpm_results' not in st.session_state:
    st.session_state.rpm_results = None
if 'comp_results' not in st.session_state:
    st.session_state.comp_results = None

st.header("1) ป้อนสูตร N-P-K และน้ำหนักที่ต้องการ")
col1, col2 = st.columns(2)
with col1:
    N_pct = st.number_input("Target N (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    P_pct = st.number_input("Target P (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
with col2:
    K_pct = st.number_input("Target K (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    total_kg = st.number_input("Total weight (kg)", min_value=0.01, value=100.0, step=0.1)

if st.button("คำนวณสูตรและหา %RPM/เวลา"):
    # composition
    comp = calc_parents_from_formula(N_pct, P_pct, K_pct, total_kg)
    
    # Save to session state
    st.session_state.comp_results = comp
    st.session_state.parent_targets_g = {
        'N': max(0.0, comp['A_46_0_0_kg']) * 1000.0,
        'P': max(0.0, comp['B_18_46_0_kg']) * 1000.0,
        'K': max(0.0, comp['C_0_0_60_kg']) * 1000.0
    }

# Display composition results if available
if st.session_state.comp_results is not None:
    comp = st.session_state.comp_results
    st.subheader("ผลการคำนวณแม่ปุ๋ย")
    # แสดงแหล่งที่มาของข้อมูล เพื่อความชัดเจนว่ากำลังใช้ข้อมูลใด
    source_label = {
        "uploaded": "ไฟล์ที่อัปโหลด",
        "local-excel": "ไฟล์ Excel ในเครื่อง",
        "sample": "ข้อมูลตัวอย่าง"
    }.get(data_source, data_source)
    if source_label:
        st.caption(f"แหล่งข้อมูล: {source_label}")

    with st.expander("📈 ข้อมูลสอบเทียบ (RPM ↔ g/s)", expanded=False):
        preview_rows = []
        if groups is not None:
            for hopper in ['N', 'P', 'K']:
                if hopper not in groups:
                    continue
                raw_rpms = np.asarray(groups[hopper].get('raw_rpms', []), dtype=float)
                raw_pct = np.asarray(groups[hopper].get('raw_rpm_pct', raw_rpms * 100.0 / 2750.0), dtype=float)
                raw_rates = np.asarray(groups[hopper].get('raw_rates', []), dtype=float)
                if raw_rates.size == 0 and raw_rpms.size:
                    raw_rates = np.zeros_like(raw_rpms)
                raw_tout = np.asarray(groups[hopper].get('raw_t_out', []), dtype=float)
                raw_tall = np.asarray(groups[hopper].get('raw_t_all', []), dtype=float)
                raw_loss = np.asarray(groups[hopper].get('raw_loss', []), dtype=float)
                raw_eff = np.asarray(groups[hopper].get('raw_eff', []), dtype=float)
                for i, (rpm_val, pct_val, rate_val) in enumerate(zip(raw_rpms, raw_pct, raw_rates)):
                    preview_rows.append({
                        'Hopper': hopper,
                        '%RPM': round(pct_val, 1),
                        'RPM': round(rpm_val, 1),
                        'g/s': round(rate_val, 3),
                        't_out (s)': round(raw_tout[i], 2) if raw_tout.size > i else None,
                        't_all (s)': round(raw_tall[i], 2) if raw_tall.size > i else None,
                        'loss (g)': round(raw_loss[i], 2) if raw_loss.size > i else None,
                        'efficiency': round(raw_eff[i], 4) if raw_eff.size > i else None
                    })
        if preview_rows:
            preview_df = pd.DataFrame(preview_rows).sort_values(['Hopper', 'RPM']).reset_index(drop=True)
            st.dataframe(preview_df, width='stretch')
        else:
            st.write("ไม่มีข้อมูลสอบเทียบให้แสดง")
    
    # Display in colored metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🟦 แม่ปุ๋ย 46-0-0 (N)", value=f"{comp['A_46_0_0_kg']:.2f} kg")
    with col2:
        st.metric(label="🟨 แม่ปุ๋ย 18-46-0 (N-P)", value=f"{comp['B_18_46_0_kg']:.2f} kg")
    with col3:
        st.metric(label="🟧 แม่ปุ๋ย 0-0-60 (K)", value=f"{comp['C_0_0_60_kg']:.2f} kg")
    
    st.divider()
    col4, col5 = st.columns(2)
    with col4:
        st.metric(label="💰 น้ำหนักแม่ปุ๋ยรวม", value=f"{comp['sum_ferts_kg']:.2f} kg")
    with col5:
        st.metric(label="➕ น้ำหนัก Filler", value=f"{comp['filler_kg']:.2f} kg")
    
    if comp['A_46_0_0_kg'] < 0 or comp['B_18_46_0_kg'] < 0 or comp['C_0_0_60_kg'] < 0:
        st.warning("⚠️ ค่าบางค่าเป็นลบ — สูตรนี้อาจทำไม่ได้ด้วยแม่ปุ๋ยชุดนี้")

    # Get parent_targets_g from session state
    parent_targets_g = st.session_state.parent_targets_g

    if groups is None:
        st.error("ไม่สามารถคำนวณ %RPM ได้ — ไม่มีข้อมูลทดลอง")
    else:
        # search parameters - ปรับค่าก่อนกดคำนวณ
        st.subheader("⚙️ การตั้งค่าสำหรับการค้นหา")
        
        with st.expander("🎚️ ปรับค่าพารามิเตอร์", expanded=True):
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                st.markdown("**ช่วงเวลาการทำงาน**")
                time_preset = st.selectbox(
                    "เลือกช่วงเวลาที่ต้องการ",
                    ["กำหนดเอง", "รวดเร็ว (1-10 นาที)", "ปานกลาง (10-30 นาที)", "ช้า (30-60 นาที)", "ยาวนาน (1-2 ชั่วโมง)"],
                    key="time_preset"
                )
                
                if time_preset == "รวดเร็ว (1-10 นาที)":
                    t_min_default, t_max_default = 60.0, 600.0
                elif time_preset == "ปานกลาง (10-30 นาที)":
                    t_min_default, t_max_default = 600.0, 1800.0
                elif time_preset == "ช้า (30-60 นาที)":
                    t_min_default, t_max_default = 1800.0, 3600.0
                elif time_preset == "ยาวนาน (1-2 ชั่วโมง)":
                    t_min_default, t_max_default = 3600.0, 7200.0
                else:  # กำหนดเอง
                    t_min_default, t_max_default = 1.0, 3600.0
                
                t_min = st.number_input("เวลาต่ำสุด (วินาที)", value=t_min_default, step=60.0, min_value=0.1, key="t_min_input")
                t_max = st.number_input("เวลาสูงสุด (วินาที)", value=t_max_default, step=60.0, min_value=1.0, max_value=86400.0, key="t_max_input")
                st.caption(f"ช่วง: {t_min/60:.1f} - {t_max/60:.1f} นาที")
                
            with col_param2:
                st.markdown("**ช่วงรอบเครื่อง (% RPM)**")
                rpm_min_pct = st.slider("รอบต่ำสุด (%)", 0, 100, 20, key="rpm_min_slider", 
                                       help="กำหนดรอบต่ำสุดที่ต้องการใช้ เพื่อลดภาระเครื่องจักร")
                rpm_max_pct = st.slider("รอบสูงสุด (%)", 0, 100, 80, key="rpm_max_slider", 
                                       help="กำหนดรอบสูงสุดที่ต้องการใช้ ควรอยู่ที่ 70-80% เพื่ออายุการใช้งานที่ดี")
                
                # Validate ว่า min < max
                if rpm_min_pct >= rpm_max_pct:
                    st.error("⚠️ ช่วงรอบต่ำสุดต้องน้อยกว่าช่วงรอบสูงสุด")
                
                st.markdown("**ความแม่นยำ**")
                tol = st.slider("ความคลาดเคลื่อนที่ยอมรับได้", 0.01, 0.5, 0.05, key="tol_slider",
                              help="ค่าน้อย = แม่นยำมากขึ้น แต่อาจหาคำตอบได้ยากขึ้น")

        # ปุ่มคำนวณ - กดเมื่อปรับค่าเสร็จแล้ว
        calculate_rpm = st.button("🔍 คำนวณหา RPM และเวลาที่เหมาะสม", type="primary", width='stretch')
        
        if calculate_rpm:
            if rpm_min_pct >= rpm_max_pct:
                st.error("❌ กรุณาตรวจสอบช่วงรอบให้ถูกต้อง (ต่ำสุด < สูงสุด)")
            else:
                # run search
                with st.spinner("กำลังค้นหาเวลาและ %RPM ที่เหมาะสม..."):
                    found = find_t_for_parent_masses(groups, parent_targets_g, t_min=float(t_min), t_max=float(t_max), t_steps=800, tol=float(tol), cap_by_tall=False, rpm_min_pct=rpm_min_pct, rpm_max_pct=rpm_max_pct)
                
                # Save to session state
                st.session_state.rpm_results = found
        
        # Display RPM results if available (outside the button click)
        if st.session_state.rpm_results is not None:
            found = st.session_state.rpm_results
            
            if found.get('found'):
                res = found['result']
                st.success(f"✅ พบการตั้งค่า: เวลา/รอบ = {res['t']:.1f} s ({res['t']/60.0:.2f} min)")
                
                # แนะนำการปรับรอบ
                st.write("**แนะนำการปรับรอบ:**")
                for h in ['N','P','K']:
                    s = res['settings'][h]
                    st.write(f"🔧 Hopper {h}: ปรับรอบที่ **{int(round(s['rpm_pct']))}%** (RPM = {s['rpm_actual']:.1f}) → ได้ {s['mass_g']/1000.0:.3f} kg")
                
                rows = []
                for h in ['N','P','K']:
                    s = res['settings'][h]
                    rows.append({
                        'hopper': h,
                        'ปรับรอบ (%)': int(round(s['rpm_pct'])),
                        'RPM': int(round(s['rpm_actual'])),
                        'กิโลกรัม': round(s['mass_g']/1000.0, 3)
                    })
                st.table(pd.DataFrame(rows))
                total_usable = res['total_mass_g']/1000.0
                total_loss = res['total_loss_g']/1000.0
                total_produced = total_usable + total_loss
                st.write(f"**รวม:** ผลิตได้ {total_produced:.3f} kg | ใช้งานได้ {total_usable:.3f} kg | สูญเสีย {total_loss:.3f} kg ({(total_loss/total_produced*100):.1f}%)")
            else:
                best = found.get('best_single_run')
                if best:
                    st.warning("ไม่พบเวลาเดียวที่พอ — แสดง best single-run ที่ใกล้เคียงที่สุด")
                    st.write(f"Best single-run: time = {best['t']:.1f} s → total_mass (kg) = {best['total_mass_g']/1000.0:.3f}")
                    rows = []
                    for h in ['N','P','K']:
                        r = best['settings'][h]
                        rows.append({
                            'hopper': h, 
                            'ปรับรอบ (%)': int(round(r['rpm_pct'])),
                            'RPM': int(round(r['rpm_actual'])),
                            'กิโลกรัม': round(r['mass_g']/1000.0, 3)
                        })
                    st.table(pd.DataFrame(rows))
                else:
                    st.error("ไม่พบการตั้งค่า — ลองเพิ่มช่วงเวลาหรือปรับความแม่นยำ")
