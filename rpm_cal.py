# app_fertilizer_rpm_from_kraphor.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from math import ceil
import os

st.set_page_config(page_title="โปรแกรมคำนวณสูตรปุ๋ยและแนะนำการปรับค่าเครื่องจ่ายเกลียวลำเลียง AGN03", layout="wide")

# --------- CONFIG ----------
DEFAULT_EXCEL = "C:\\Users\\Lenovo\\OneDrive\\Desktop\\NPK cal\\กระพ้อ.xlsx"  # ใช้เมื่อรันใน local เท่านั้น

# --------- HELPERS ----------
def create_default_groups():
    """สร้างข้อมูลตัวอย่างเริ่มต้นสำหรับการทดสอบ (ใช้เมื่อไม่มีไฟล์ Excel)"""
    # ข้อมูลตัวอย่างพื้นฐาน: rpm และ rate (g/s) สำหรับ N, P, K
    default_data = {
        'N': {'rpms': [10, 500, 1000, 1500, 2000, 2500, 2750], 
              'rates': [0.2, 10, 20, 30, 40, 50, 55]},
        'P': {'rpms': [10, 500, 1000, 1500, 2000, 2500, 2750], 
              'rates': [0.15, 7.5, 15, 22.5, 30, 37.5, 41.25]},
        'K': {'rpms': [10, 500, 1000, 1500, 2000, 2500, 2750], 
              'rates': [0.1, 5, 10, 15, 20, 25, 27.5]}
    }
    
    groups = {}
    for h, data in default_data.items():
        xs = np.array(data['rpms'], dtype=float)
        rates = np.array(data['rates'], dtype=float)
        touts = np.zeros_like(xs)
        talls = np.full_like(xs, 3600.0)  # 1 ชั่วโมง
        losses = np.zeros_like(xs)
        
        groups[h] = {
            'rpm_min': float(xs.min()), 
            'rpm_max': float(xs.max()),
            'rate_func': interp1d(xs, rates, kind='linear', fill_value='extrapolate', bounds_error=False),
            'tout_func': interp1d(xs, touts, kind='linear', fill_value='extrapolate', bounds_error=False),
            'tall_func': interp1d(xs, talls, kind='linear', fill_value='extrapolate', bounds_error=False),
            'loss_func': interp1d(xs, losses, kind='linear', fill_value='extrapolate', bounds_error=False),
            'data': pd.DataFrame({'hopper': h, 'rpm': xs, 'g/s': rates})
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
        xs = sub['rpm'].values.astype(float)
        order = np.argsort(xs)
        xs = xs[order]
        rates = sub[col_rate].values[order] if col_rate in sub.columns else np.zeros_like(xs)
        touts = sub[col_tout].values[order] if col_tout in sub.columns else np.zeros_like(xs)
        talls = sub[col_tall].values[order] if col_tall in sub.columns else np.full_like(xs, 1e6)
        losses = sub[col_loss].values[order] if col_loss in sub.columns else np.zeros_like(xs)
        try:
            groups[h] = {
                'rpm_min': float(xs.min()), 'rpm_max': float(xs.max()),
                'rate_func': interp1d(xs, rates, kind='linear', fill_value='extrapolate', bounds_error=False),
                'tout_func': interp1d(xs, touts, kind='linear', fill_value='extrapolate', bounds_error=False),
                'tall_func': interp1d(xs, talls, kind='linear', fill_value='extrapolate', bounds_error=False),
                'loss_func': interp1d(xs, losses, kind='linear', fill_value='extrapolate', bounds_error=False),
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
def evaluate_run_for_t_with_targets(groups, target_masses_g, t, tol=0.05, cap_by_tall=True):
    """
    target_masses_g: dict {'N': grams, 'P': grams, 'K': grams}
    t: run time in seconds (equal for all hoppers)
    returns dict or error
    """
    rpm_choices = {}
    for h in ['N','P','K']:
        if groups is None or h not in groups:
            return {'ok': False, 'reason': f'No test data for hopper {h}'}
        funcs = groups[h]
        rpms = np.linspace(funcs['rpm_min'], funcs['rpm_max'], 2000)
        rates = funcs['rate_func'](rpms)     # g/s
        touts = funcs['tout_func'](rpms)     # s
        talls = funcs['tall_func'](rpms)     # s
        if cap_by_tall:
            eff_times = np.maximum(0.0, np.minimum(t, talls) - touts)
        else:
            eff_times = np.maximum(0.0, t - touts)
        masses = rates * eff_times  # grams delivered
        target = float(target_masses_g.get(h, 0.0))
        # pick rpm that gives mass closest to target
        idx = np.argmin(np.abs(masses - target))
        mass = float(masses[idx])
        rel_err = abs(mass - target) / (target + 1e-9)
        rpm_choices[h] = {
            'rpm_pct': float(rpms[idx]),
            'rate_gps': float(rates[idx]),
            'tout_s': float(touts[idx]),
            'tall_s': float(talls[idx]),
            'mass_g': mass,
            'rel_err': rel_err,
            'loss_g': float(funcs['loss_func'](rpms[idx]))
        }
    total_mass_g = sum([rpm_choices[h]['mass_g'] for h in rpm_choices])
    total_loss_g = sum([rpm_choices[h]['loss_g'] for h in rpm_choices])
    return {'ok': True, 't': t, 'settings': rpm_choices, 'total_mass_g': total_mass_g, 'total_loss_g': total_loss_g}

def find_t_for_parent_masses(groups, target_masses_g, t_min=1.0, t_max=3600.0, t_steps=800, tol=0.05, cap_by_tall=True):
    """
    Search t in [t_min, t_max] (equal for all hoppers) to find first t that yields per-hopper mass within tol.
    If not found, return best single-run (t that maximizes total_mass closeness) for diagnostics.
    """
    t_search = np.linspace(t_min, t_max, t_steps)
    feasible = []
    for t in t_search:
        res = evaluate_run_for_t_with_targets(groups, target_masses_g, t, tol=tol, cap_by_tall=cap_by_tall)
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
            res = evaluate_run_for_t_with_targets(groups, target_masses_g, t, tol=tol, cap_by_tall=cap_by_tall)
            if best_overall is None or res['total_mass_g'] > best_overall['total_mass_g']:
                best_overall = res
        return {'found': False, 'best_single_run': best_overall}
    return {'found': False, 'best_single_run': best}

# ---------- Streamlit UI ----------
st.title("โปรแกรมคำนวณสูตรปุ๋ยและแนะนำการปรับค่าเครื่องจ่ายเกลียวลำเลียง AGN03")

# Sidebar with instructions
st.sidebar.header("📖 วิธีใช้งาน")
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
uploaded_file = st.sidebar.file_uploader(
    "อัปโหลดไฟล์ Excel (ถ้ามี)", 
    type=["xlsx", "xls"],
    help="อัปโหลดไฟล์ข้อมูลการทดลองของคุณ หรือใช้ข้อมูลตัวอย่างเริ่มต้น"
)

# Load data from uploaded file or default
if uploaded_file is not None:
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
        else:
            st.sidebar.success("✅ ใช้ข้อมูลจากไฟล์ที่อัปโหลด")
    except Exception as e:
        st.sidebar.error(f"เกิดข้อผิดพลาด: {e}")
        st.sidebar.info("กำลังใช้ข้อมูลตัวอย่างแทน")
        groups = create_default_groups()
        proc_df = None
elif os.path.exists(DEFAULT_EXCEL):
    # ใช้ไฟล์ local ถ้ามี (สำหรับการรันใน local)
    groups, proc_df, load_err = load_testdata(DEFAULT_EXCEL)
    if load_err:
        st.sidebar.warning("ไม่พบไฟล์ข้อมูล - ใช้ข้อมูลตัวอย่าง")
        groups = create_default_groups()
        proc_df = None
    else:
        st.sidebar.info("📊 ใช้ข้อมูลจากไฟล์ local")
else:
    # ใช้ข้อมูลตัวอย่างเริ่มต้น
    st.sidebar.info("📊 ใช้ข้อมูลตัวอย่างเริ่มต้น")
    groups = create_default_groups()
    proc_df = None

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
    st.subheader("ผลการคำนวณแม่ปุ๋ย")
    
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

    # prepare target masses per hopper (grams)
    parent_targets_g = {
        'N': max(0.0, comp['A_46_0_0_kg']) * 1000.0,
        'P': max(0.0, comp['B_18_46_0_kg']) * 1000.0,
        'K': max(0.0, comp['C_0_0_60_kg']) * 1000.0
    }

    if groups is None:
        st.error("ไม่สามารถคำนวณ %RPM ได้ — ไม่มีข้อมูลทดลอง")
    else:
        # search parameters
        st.subheader("การตั้งค่าสำหรับการค้นหา (search params)")
        tol = st.slider("Allowed relative error per hopper (tol)", 0.01, 0.5, 0.05)
        t_min = st.number_input("t_min (s)", value=1.0, step=1.0, min_value=0.1)
        t_max = st.number_input("t_max (s)", value=3600.0, step=100.0, min_value=1.0, max_value=86400.0)

        # run search
        with st.spinner("กำลังค้นหาเวลาและ %RPM ..."):
            found = find_t_for_parent_masses(groups, parent_targets_g, t_min=float(t_min), t_max=float(t_max), t_steps=800, tol=float(tol), cap_by_tall=False)
        if found.get('found'):
            res = found['result']
            st.success(f"พบการตั้งค่า: เวลา/รอบ = {res['t']:.1f} s ({res['t']/60.0:.2f} min)")
            rows = []
            for h in ['N','P','K']:
                s = res['settings'][h]
                rows.append({
                    'hopper': h,
                    'ปรับรอบ': int(round(s['rpm_pct'])),
                    'กิโลกรัม': round(s['mass_g']/1000.0, 3)
                })
            st.table(pd.DataFrame(rows))
            st.write(f"Total produced (kg): {res['total_mass_g']/1000.0:.3f} ; Predicted loss (kg): {res['total_loss_g']/1000.0:.3f}")
        else:
            best = found.get('best_single_run')
            if best:
                st.warning("ไม่พบเวลาเดียวที่พอ — แสดง best single-run ที่ใกล้เคียงที่สุด")
                st.write(f"Best single-run: time = {best['t']:.1f} s → total_mass (kg) = {best['total_mass_g']/1000.0:.3f}")
                rows = []
                for h in ['N','P','K']:
                    r = best['settings'][h]
                    rows.append({'hopper': h, 'ปรับรอบ': int(round(r['rpm_pct'])), 'กิโลกรัม': round(r['mass_g']/1000.0, 3)})
                st.table(pd.DataFrame(rows))
            else:
                st.error("ไม่พบการตั้งค่า — ลองเพิ่ม t_max หรือเพิ่ม tol")
