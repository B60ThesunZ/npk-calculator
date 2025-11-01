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

st.title("🌾 โปรแกรมคำนวณสูตรปุ๋ย N-P-K")
st.markdown("แม่ปุ๋ยที่ใช้: **46-0-0**, **18-46-0**, **0-0-60**")

col1, col2, col3 = st.columns(3)
N = col1.number_input("N (%)", value=15.0, step=0.1)
P = col2.number_input("P (%)", value=15.0, step=0.1)
K = col3.number_input("K (%)", value=15.0, step=0.1)
T = st.number_input("น้ำหนักรวม (kg)", value=100.0, step=0.1)

if st.button("คำนวณ"):
    res = calc_fertilizers(N, P, K, T)
    st.success("ผลลัพธ์การคำนวณ (kg)")
    st.write(f"46-0-0 (A): {res['A']:.3f}")
    st.write(f"18-46-0 (B): {res['B']:.3f}")
    st.write(f"0-0-60 (C): {res['C']:.3f}")
    st.write(f"รวมแม่ปุ๋ย: {res['sum']:.3f}")
    st.write(f"Filler: {res['filler']:.3f}")
