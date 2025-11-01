import streamlit as st

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
    res = calc_fertilizers(N, P, K, T)
    
    # Check for negative values or infinity
    has_error = any(v < 0 or v == float('inf') for v in [res['A'], res['B'], res['C']])
    
    if has_error:
        st.error("❌ ไม่สามารถคำนวณได้: สูตรที่ต้องการอาจไม่สามารถผสมได้จากแม่ปุ๋ยที่มี")
    else:
        st.success("✅ ผลลัพธ์การคำนวณ")
        
        # Results in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"🟦 แม่ปุ๋ย 46-0-0\n### {res['A']:.2f} kg")
        with col2:
            st.info(f"🟨 แม่ปุ๋ย 18-46-0\n### {res['B']:.2f} kg")
        with col3:
            st.info(f"🟧 แม่ปุ๋ย 0-0-60\n### {res['C']:.2f} kg")
        
        # Summary
        st.divider()
        sum_col1, sum_col2 = st.columns(2)
        with sum_col1:
            st.metric(label="💰 น้ำหนักแม่ปุ๋ยรวม", value=f"{res['sum']:.2f} kg")
        with sum_col2:
            st.metric(label="➕ น้ำหนัก Filler", value=f"{res['filler']:.2f} kg")
        
        # Warning if significant filler needed
        if res['filler'] > T * 0.3:  # If filler is more than 30% of total
            st.warning("⚠️ ต้องใช้ Filler ในปริมาณมาก อาจต้องพิจารณาปรับสูตรหรือใช้แม่ปุ๋ยชนิดอื่นเพิ่มเติม")
