import streamlit as st
import numpy as np

def calc_fertilizers(targetN_pct, targetP_pct, targetK_pct, total_kg):
    A_N = 0.46
    B_N = 0.18; B_P = 0.46
    C_K = 0.60
    t = float(total_kg)
    n = float(targetN_pct) / 100.0
    p = float(targetP_pct) / 100.0
    k = float(targetK_pct) / 100.0

    y = (p * t) / B_P if B_P != 0 else float('inf')
    z = (k * t) / C_K if C_K != 0 else float('inf')
    x = (n * t - B_N * y) / A_N if A_N != 0 else float('inf')
    sum_ferts = x + y + z
    filler = t - sum_ferts
    return {'A': x, 'B': y, 'C': z, 'sum': sum_ferts, 'filler': filler}

# Page config
st.set_page_config(
    page_title="คำนวณสูตรปุ๋ย N-P-K",
    page_icon="🌾",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("ℹ️ คำแนะนำ")
    st.markdown("""
    ### วิธีใช้งาน
    1. ใส่ค่า N-P-K ที่ต้องการ (%)
    2. ใส่น้ำหนักรวมที่ต้องการ (kg)
    3. กดปุ่ม "คำนวณ"
    
    ### แม่ปุ๋ยที่ใช้
    - **46-0-0** (ยูเรีย)
    - **18-46-0** (ไดแอมโมเนียมฟอสเฟต)
    - **0-0-60** (โพแทสเซียมคลอไรด์)
    
    ### หมายเหตุ
    - ผลลัพธ์อาจมีค่าติดลบถ้าสูตรที่ต้องการไม่สามารถผสมได้
    - Filler คือส่วนเติมเต็มให้ได้น้ำหนักตามต้องการ
    """)

# Main content
st.title("🌾 โปรแกรมคำนวณสูตรปุ๋ย N-P-K")
st.markdown("""
<style>
div.stButton > button {
    width: 100%;
    height: 3em;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Input explanation
st.markdown("### 📝 กรอกข้อมูล")
st.markdown("ใส่ค่า N-P-K ที่ต้องการและน้ำหนักรวม")

# Input section with validation
input_col1, input_col2 = st.columns([2, 1])

with input_col1:
    col1, col2, col3 = st.columns(3)
    N = col1.number_input("🟩 N (%)", value=15.0, step=0.1, min_value=0.0, max_value=100.0,
                         help="ไนโตรเจน (N)")
    P = col2.number_input("🟨 P (%)", value=15.0, step=0.1, min_value=0.0, max_value=100.0,
                         help="ฟอสฟอรัส (P)")
    K = col3.number_input("🟧 K (%)", value=15.0, step=0.1, min_value=0.0, max_value=100.0,
                         help="โพแทสเซียม (K)")

with input_col2:
    T = st.number_input("⚖️ น้ำหนักรวม (kg)", value=100.0, step=0.1, min_value=0.1,
                       help="น้ำหนักปุ๋ยรวมที่ต้องการ")
    calculate = st.button("🧮 คำนวณ", use_container_width=True)

# Results section
if calculate:
    # Calculate for total
    res = calc_fertilizers(N, P, K, T)
    
    # Check for negative values or infinity for total
    has_error_total = any(v < 0 or v == float('inf') for v in [res['A'], res['B'], res['C']])

    if has_error_total:
        st.error("❌ ไม่สามารถคำนวณได้สำหรับน้ำหนักรวม: สูตรที่ต้องการอาจไม่สามารถผสมได้จากแม่ปุ๋ยที่มี")
    else:
        st.success("✅ ผลลัพธ์การคำนวณ (น้ำหนักรวม)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🟦 แม่ปุ๋ย 46-0-0\n### {res['A']:.2f} kg")
        with col2:
            st.info(f"🟨 แม่ปุ๋ย 18-46-0\n### {res['B']:.2f} kg")
        with col3:
            st.info(f"🟧 แม่ปุ๋ย 0-0-60\n### {res['C']:.2f} kg")
        st.divider()
        sum_col1, sum_col2 = st.columns(2)
        with sum_col1:
            st.metric(label="💰 น้ำหนักแม่ปุ๋ยรวม", value=f"{res['sum']:.2f} kg")
        with sum_col2:
            st.metric(label="➕ น้ำหนัก Filler", value=f"{res['filler']:.2f} kg")
        if res['filler'] > T * 0.3:
            st.warning("⚠️ ต้องใช้ Filler ในปริมาณมาก อาจต้องพิจารณาปรับสูตรหรือใช้แม่ปุ๋ยชนิดอื่นเพิ่มเติม")

    # (No split results — single calculation shown as before)

# --- BEGIN: Batch Planner (วางต่อจากการเตรียม groups/proc_all ที่มีอยู่แล้ว) ---
from math import ceil


def create_default_groups():
    """Return a simple default calibration for hoppers N, P, K.
    rate_func should accept a numpy array of rpms and return rate in g/s.
    tout_func should accept rpms and return startup delay in seconds (array-like).
    loss_func accepts a scalar rpm and returns loss in grams (we return 0 here).
    """
    groups = {}
    # default linear slopes (g/s per rpm) and zero intercept
    defaults = {
        'N': {'rpm_min': 10.0, 'rpm_max': 2750.0, 'slope': 0.02, 'intercept': 0.0},
        'P': {'rpm_min': 10.0, 'rpm_max': 2750.0, 'slope': 0.015, 'intercept': 0.0},
        'K': {'rpm_min': 10.0, 'rpm_max': 2750.0, 'slope': 0.01, 'intercept': 0.0},
    }
    for h, v in defaults.items():
        slope = v['slope']
        intercept = v['intercept']
        groups[h] = {
            'rpm_min': v['rpm_min'],
            'rpm_max': v['rpm_max'],
            'rate_func': (lambda s, i: (lambda rpms: (np.array(rpms) * s + i)))(slope, intercept),
            'tout_func': (lambda rpms: np.zeros_like(rpms)),
            'loss_func': (lambda rpm: 0.0)
        }
    return groups


# Prepare default groups if none exist (so batch planner can run)
if 'groups' not in globals():
    groups = create_default_groups()

def evaluate_run_for_t(groups, targetN_pct, targetP_pct, targetK_pct, total_kg, t, tol=0.10):
    needs = { 'N': total_kg*(targetN_pct/100.0)*1000.0, 'P': total_kg*(targetP_pct/100.0)*1000.0, 'K': total_kg*(targetK_pct/100.0)*1000.0 }
    rpm_choices = {}
    for h in ['N','P','K']:
        funcs = groups[h]
        rpms = np.linspace(funcs['rpm_min'], funcs['rpm_max'], 2000)
        rates = funcs['rate_func'](rpms)
        touts = funcs['tout_func'](rpms)
        eff_times = np.maximum(0.0, t - touts)
        masses = rates * eff_times
        idx = np.argmin(np.abs(masses - needs[h]))
        mass = float(masses[idx])
        rel_err = abs(mass - needs[h])/(needs[h] + 1e-9)
        if rel_err > tol:
            return {'ok': False, 'reason': f'Hopper {h} cannot meet required mass within tolerance at t={t:.1f}s (rel_err={rel_err:.3f})'}
        rpm_choices[h] = {
            'rpm_pct': float(rpms[idx]),
            'rate_gps': float(rates[idx]),
            'tout_s': float(touts[idx]),
            'mass_g': mass,
            'rel_err': rel_err,
            'loss_g': float(funcs['loss_func'](rpms[idx]))
        }
    total_mass_g = sum([rpm_choices[h]['mass_g'] for h in rpm_choices])
    total_loss_g = sum([rpm_choices[h]['loss_g'] for h in rpm_choices])
    return {'ok': True, 't': t, 'settings': rpm_choices, 'total_mass_g': total_mass_g, 'total_loss_g': total_loss_g}

def plan_batches(groups, N_pct, P_pct, K_pct, total_kg, t_min, t_max, t_steps, tol, max_batches):
    # (ฟังก์ชันนี้เหมือนกับในไฟล์ batch ที่ผมสร้างไว้)
    t_search = np.linspace(t_min, t_max, t_steps)
    best_run = None
    for t in t_search:
        res = evaluate_run_for_t(groups, N_pct, P_pct, K_pct, total_kg, t, tol=tol)
        if res['ok']:
            if (best_run is None) or (res['total_mass_g'] > best_run['total_mass_g']):
                best_run = res
    if best_run is None:
        return {'strategy': 'none', 'message': 'ไม่พบการตั้งค่าใดที่เป็นไปได้ภายในพารามิเตอร์ที่กำหนด (ลองเพิ่ม t_max หรือ tol)'}
    single_mass_kg = best_run['total_mass_g']/1000.0
    single_loss_kg = best_run['total_loss_g']/1000.0
    batches_needed = int(ceil(total_kg / single_mass_kg)) if single_mass_kg > 0 else None
    total_time = batches_needed * best_run['t']
    total_produced_kg = single_mass_kg * batches_needed
    total_loss_kg = single_loss_kg * batches_needed
    usable_mass_kg = total_produced_kg - total_loss_kg
    return {
        'strategy': 'multiple_batches_same_settings',
        'runs': batches_needed,
        'run': best_run,
        'single_mass_kg': single_mass_kg,
        'single_loss_kg': single_loss_kg,
        'total_produced_kg': total_produced_kg,
        'total_loss_kg': total_loss_kg,
        'usable_mass_kg': usable_mass_kg,
        'total_time_s': total_time
    }

