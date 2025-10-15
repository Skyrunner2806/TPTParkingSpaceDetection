import sqlite3
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import json
import time

# === KONFIGURASI ===
DB_PATH = r"D:\Kuliah\Semester 6\TPT\Belajar Opencv_5\counter.db"
TABLE = "vehicle_counts"
REFRESH_SEC = 1  # perbarui setiap 1 detik
# ===================

st.set_page_config(page_title="Realtime Vehicle Counter", layout="wide")

st.title("🚗 Realtime Vehicle Counter Dashboard")
st.caption("Sumber: video_pagi_kanan.mp4 (simulasi kamera)")

# Fungsi ambil data terbaru
@st.cache_data(ttl=1.0, show_spinner=False)
def load_data():
    try:
        con = sqlite3.connect(DB_PATH, timeout=10)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE} ORDER BY id DESC LIMIT 200", con)
        con.close()
    except Exception as e:
        st.warning(f"Gagal membaca database: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.sort_values("id")
    df["class_counts"] = df["class_counts"].apply(
        lambda x: json.loads(x) if isinstance(x, str) and x else {}
    )
    return df


# ====== RENDER UI ======
df = load_data()

if df.empty:
    st.info("Menunggu data dari YOLO logger…")
    time.sleep(REFRESH_SEC)
    st.rerun()

latest = df.iloc[-1]
total = int(latest["vehicles_total"])
class_dict = latest["class_counts"]

col1, col2 = st.columns([0.3, 0.7])
with col1:
    st.metric("Total Kendaraan (Frame Terakhir)", total)
    st.write("**Per Kelas:**", class_dict)

with col2:
    # Buat grafik Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t_video_sec"],
        y=df["vehicles_total"],
        mode="lines+markers",
        name="Jumlah Kendaraan",
        line=dict(color="orange", width=3),
        marker=dict(size=6)
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Grafik Jumlah Kendaraan vs Waktu",
        xaxis_title="Waktu (detik video)",
        yaxis_title="Jumlah kendaraan",
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")

st.divider()
st.subheader("📋 Data Terbaru (20 Baris)")
st.dataframe(df.tail(20).iloc[:, 1:], use_container_width=True, key=f"table_{time.time()}")

# Auto-refresh
time.sleep(REFRESH_SEC)
st.rerun()
