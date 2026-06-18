import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste Clustering — Jawa Barat",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0e0 !important;
}

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
}

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    border-color: rgba(99, 102, 241, 0.3);
}
.metric-value {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 8px 0 4px;
}
.metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 4px;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}
.status-kritis {
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
}
.status-waspada {
    background: rgba(251,191,36,0.15);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.3);
}
.status-terkendali {
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
}

/* ── Section header ── */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Detail card ── */
.detail-row {
    display: flex;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #cbd5e1;
}
.detail-row:last-child {
    border-bottom: none;
}
.detail-label {
    font-weight: 500;
    color: #94a3b8;
}
.detail-value {
    font-weight: 600;
    color: #e2e8f0;
}

/* ── Hide default Streamlit styles ── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* ── Plotly chart container ── */
.stPlotlyChart {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    border-radius: 8px 8px 0 0;
    border: 1px solid rgba(255,255,255,0.06);
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.4) !important;
    color: #a5b4fc !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Koordinat dari data_kab.csv (otomatis, akurat, tidak perlu tulis manual) ─
# File data_kab.csv berisi koordinat resmi seluruh kab/kota Indonesia
# Difilter hanya Jawa Barat (province_id = 32) lalu dijadikan dict untuk lookup

# ─── Load Data ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    # Load hasil clustering
    df = pd.read_csv(os.path.join(BASE_DIR, "hasil_clustering.csv"))

    # Load koordinat dari data_kab.csv — filter Jawa Barat (province_id=32)
    df_kab = pd.read_csv(os.path.join(BASE_DIR, "data_kab.csv"))
    df_coord = (
        df_kab[df_kab["province_id"] == 32][["name", "latitude", "longitude"]]
        .copy()
    )
    # Samakan format nama agar bisa di-merge (uppercase & strip spasi)
    df_coord["name"] = df_coord["name"].str.upper().str.strip()
    df["nama_upper"] = df["nama_kabupaten_kota"].str.upper().str.strip()

    # Merge koordinat ke data clustering
    df = df.merge(
        df_coord.rename(columns={"name": "nama_upper"}),
        on="nama_upper",
        how="left",
    ).drop(columns=["nama_upper"])

    # Rename agar konsisten dengan kode peta
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    return df

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "kmeans_model.pkl"))

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

df = load_data()
model = load_model()
scaler = load_scaler()

# ─── Color palette ────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "Kritis":     "#ef4444",
    "Waspada":    "#f59e0b",
    "Terkendali": "#22c55e",
}
# Warna untuk marker folium (hex tanpa #)
STATUS_FOLIUM_COLORS = {
    "Kritis":     "red",
    "Waspada":    "orange",
    "Terkendali": "green",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#cbd5e1"),
    margin=dict(l=40, r=40, t=50, b=40),
)

DEFAULT_LEGEND = dict(
    bgcolor="rgba(0,0,0,0)",
    bordercolor="rgba(255,255,255,0.08)",
    borderwidth=1,
    font=dict(color="#94a3b8"),
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ♻️ Smart Waste")
    st.markdown("### Clustering Dashboard")
    st.markdown("---")

    st.markdown("#### 🎛️ Filter")
    status_filter = st.multiselect(
        "Status",
        options=sorted(df["status"].unique()),
        default=sorted(df["status"].unique()),
        help="Filter wilayah berdasarkan status clustering",
    )

    mean_range = st.slider(
        "Rata-rata Sampah (ton)",
        min_value=float(df["mean_sampah"].min()),
        max_value=float(df["mean_sampah"].max()),
        value=(float(df["mean_sampah"].min()), float(df["mean_sampah"].max())),
        format="%.0f",
    )

    st.markdown("---")
    st.markdown("#### ℹ️ Tentang")
    st.markdown(
        """
        Dashboard ini menampilkan hasil **K-Means Clustering**
        pengelolaan sampah di **Jawa Barat** berdasarkan
        volume produksi sampah harian per kabupaten/kota.
        """
    )
    st.markdown(
        """
        <div style='text-align:center; color:#64748b; font-size:0.75rem; margin-top:24px;'>
            Built with Streamlit & Plotly<br>
            © 2026 Smart Waste Jabar
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Filtered data ────────────────────────────────────────────────────────────
df_filtered = df[
    (df["status"].isin(status_filter))
    & (df["mean_sampah"] >= mean_range[0])
    & (df["mean_sampah"] <= mean_range[1])
]

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 24px 0 8px 0;'>
        <span style='font-size:3rem;'>♻️</span>
        <h1 style='font-size:2.2rem; font-weight:800;
            background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 8px 0 4px;'>
            Smart Waste Clustering
        </h1>
        <p style='color:#94a3b8; font-size:1.05rem; font-weight:400; margin:0;'>
            Analisis Pengelolaan Sampah Kabupaten/Kota di Jawa Barat
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

# ─── KPI Metrics ─────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

total_wilayah = len(df_filtered)
total_kritis   = (df_filtered["status"] == "Kritis").sum()
total_waspada  = (df_filtered["status"] == "Waspada").sum()
avg_sampah     = df_filtered["mean_sampah"].mean()

with k1:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-icon'>🏙️</div>
            <div class='metric-value' style='color:#818cf8;'>{total_wilayah}</div>
            <div class='metric-label'>Total Wilayah</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-icon'>🔴</div>
            <div class='metric-value' style='color:#ef4444;'>{total_kritis}</div>
            <div class='metric-label'>Wilayah Kritis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-icon'>🟡</div>
            <div class='metric-value' style='color:#f59e0b;'>{total_waspada}</div>
            <div class='metric-label'>Wilayah Waspada</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-icon'>📊</div>
            <div class='metric-value' style='color:#22c55e;'>{avg_sampah:,.0f}</div>
            <div class='metric-label'>Rata-rata Sampah</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

# ─── Tab layout ───────────────────────────────────────────────────────────────
tab_overview, tab_map, tab_analysis, tab_detail, tab_predict = st.tabs(
    ["📊 Overview", "🗺️ Peta Wilayah", "📈 Analisis", "🔍 Detail Wilayah", "🤖 Prediksi Cluster"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown(
            "<div class='section-header'>🍩 Distribusi Status Wilayah</div>",
            unsafe_allow_html=True,
        )
        status_count = df_filtered["status"].value_counts().reset_index()
        status_count.columns = ["Status", "Jumlah"]

        fig_donut = px.pie(
            status_count,
            values="Jumlah",
            names="Status",
            hole=0.55,
            color="Status",
            color_discrete_map=STATUS_COLORS,
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="label+value+percent",
            textfont=dict(size=13),
            pull=[0.03] * len(status_count),
            marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=2)),
        )
        fig_donut.update_layout(
            **PLOTLY_LAYOUT,
            height=420,
            showlegend=False,
            annotations=[
                dict(
                    text=f"<b>{total_wilayah}</b><br>Wilayah",
                    showarrow=False,
                    font=dict(size=18, color="#e2e8f0"),
                )
            ],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown(
            "<div class='section-header'>📊 Ranking Rata-rata Sampah</div>",
            unsafe_allow_html=True,
        )
        df_sorted = df_filtered.sort_values("mean_sampah", ascending=True)
        fig_bar = px.bar(
            df_sorted,
            x="mean_sampah",
            y="nama_kabupaten_kota",
            orientation="h",
            color="status",
            color_discrete_map=STATUS_COLORS,
            labels={"mean_sampah": "Rata-rata Sampah (ton)", "nama_kabupaten_kota": ""},
        )
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            height=420,
            yaxis=dict(tickfont=dict(size=10)),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(**DEFAULT_LEGEND, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_bar.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        "<div class='section-header'>📋 Tabel Data Clustering</div>",
        unsafe_allow_html=True,
    )

    df_display = df_filtered[
        ["nama_kabupaten_kota", "status", "mean_sampah", "max_sampah", "std_sampah", "growth_rate"]
    ].copy()
    df_display.columns = [
        "Kabupaten/Kota", "Status", "Rata-rata Sampah",
        "Maksimum Sampah", "Std Deviasi", "Growth Rate (%)",
    ]
    df_display = df_display.sort_values("Rata-rata Sampah", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_display.style.format(
            {
                "Rata-rata Sampah": "{:,.2f}",
                "Maksimum Sampah":  "{:,.2f}",
                "Std Deviasi":      "{:,.2f}",
                "Growth Rate (%)":  "{:.2f}",
            }
        ).map(
            lambda v: (
                "color: #ef4444" if v == "Kritis"
                else ("color: #f59e0b" if v == "Waspada" else "color: #22c55e")
            ),
            subset=["Status"],
        ),
        use_container_width=True,
        height=460,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PETA WILAYAH (BARU)
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown(
        "<div class='section-header'>🗺️ Sebaran Cluster Wilayah Jawa Barat</div>",
        unsafe_allow_html=True,
    )

    # Pilihan layer peta
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        tile_choice = st.selectbox(
            "🗂️ Tampilan Peta",
            options=["OpenStreetMap", "CartoDB Dark", "CartoDB Positron", "Satellite"],
            index=1,
        )
    with col_ctrl2:
        status_map_filter = st.multiselect(
            "🎯 Filter Status pada Peta",
            options=["Kritis", "Waspada", "Terkendali"],
            default=["Kritis", "Waspada", "Terkendali"],
        )

    # Map tile URLs
    tile_map = {
        "OpenStreetMap":    "OpenStreetMap",
        "CartoDB Dark":     "CartoDB dark_matter",
        "CartoDB Positron": "CartoDB positron",
        "Satellite":        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    }
    tile_attr = {
        "OpenStreetMap":    None,
        "CartoDB Dark":     None,
        "CartoDB Positron": None,
        "Satellite":        "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
    }

    # Buat peta Folium
    m = folium.Map(
        location=[-7.0, 107.6],
        zoom_start=8,
        tiles=tile_map[tile_choice],
        attr=tile_attr[tile_choice],
    )

    # Data untuk peta (gabungkan filter sidebar + filter peta)
    df_map = df_filtered[df_filtered["status"].isin(status_map_filter)].dropna(subset=["lat", "lon"])

    # Ukuran radius proporsional dengan mean_sampah
    max_mean = df["mean_sampah"].max()
    min_mean = df["mean_sampah"].min()

    for _, row in df_map.iterrows():
        # Normalisasi radius: 8–28px
        radius = 8 + 20 * (row["mean_sampah"] - min_mean) / (max_mean - min_mean + 1e-9)

        # Warna berdasarkan status
        color = STATUS_COLORS[row["status"]]

        # Popup HTML detail
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px; padding: 4px;">
            <h4 style="margin: 0 0 8px 0; color: #1a1a2e; font-size: 14px;">
                {row['nama_kabupaten_kota']}
            </h4>
            <table style="width:100%; font-size:12px; border-collapse:collapse;">
                <tr>
                    <td style="padding:3px 0; color:#555;">Status</td>
                    <td style="padding:3px 0; font-weight:bold;
                        color: {'#dc2626' if row['status']=='Kritis' else '#d97706' if row['status']=='Waspada' else '#16a34a'};">
                        {row['status']}
                    </td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#555;">Rata-rata</td>
                    <td style="padding:3px 0; font-weight:600;">{row['mean_sampah']:,.2f} ton/hari</td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#555;">Maksimum</td>
                    <td style="padding:3px 0; font-weight:600;">{row['max_sampah']:,.2f} ton/hari</td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#555;">Std Deviasi</td>
                    <td style="padding:3px 0; font-weight:600;">{row['std_sampah']:,.2f}</td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#555;">Growth Rate</td>
                    <td style="padding:3px 0; font-weight:600;">{row['growth_rate']:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding:3px 0; color:#555;">Cluster</td>
                    <td style="padding:3px 0; font-weight:600;">{int(row['cluster'])}</td>
                </tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=2,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=folium.Tooltip(
                f"<b>{row['nama_kabupaten_kota']}</b><br>"
                f"Status: <b>{row['status']}</b><br>"
                f"Rata-rata: <b>{row['mean_sampah']:,.0f} ton</b>",
                sticky=True,
            ),
        ).add_to(m)

    # Tambah legenda
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 9999;
        background: rgba(255,255,255,0.95); border-radius: 10px;
        padding: 12px 16px; font-family: Arial, sans-serif;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        border: 1px solid #e2e8f0;
    ">
        <div style="font-weight:700; margin-bottom:8px; font-size:13px; color:#1a1a2e;">
            🗺️ Legenda Cluster
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px;">
            <div style="width:14px;height:14px;border-radius:50%;background:#ef4444;"></div>
            <span style="font-size:12px; color:#333;">Kritis</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px;">
            <div style="width:14px;height:14px;border-radius:50%;background:#f59e0b;"></div>
            <span style="font-size:12px; color:#333;">Waspada</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:14px;height:14px;border-radius:50%;background:#22c55e;"></div>
            <span style="font-size:12px; color:#333;">Terkendali</span>
        </div>
        <div style="margin-top:8px; font-size:10px; color:#94a3b8;">
            * Ukuran lingkaran ∝ volume sampah
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Render peta di Streamlit
    st_folium(m, width="100%", height=520, returned_objects=[])

    # Ringkasan bawah peta
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-header'>📊 Ringkasan per Cluster</div>",
        unsafe_allow_html=True,
    )
    col_k, col_w, col_t = st.columns(3)
    for col, status, emoji, color in [
        (col_k, "Kritis",     "🔴", "#ef4444"),
        (col_w, "Waspada",    "🟡", "#f59e0b"),
        (col_t, "Terkendali", "🟢", "#22c55e"),
    ]:
        subset = df[df["status"] == status]
        with col:
            wilayah_list = "<br>".join(
                f"• {n}" for n in sorted(subset["nama_kabupaten_kota"].tolist())
            )
            st.markdown(
                f"""
                <div class='glass-card'>
                    <div style='font-size:1.1rem; font-weight:700; color:{color}; margin-bottom:12px;'>
                        {emoji} {status} ({len(subset)} wilayah)
                    </div>
                    <div style='font-size:0.82rem; color:#94a3b8; line-height:1.8;'>
                        {wilayah_list}
                    </div>
                    <div style='margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.06);
                                font-size:0.82rem; color:#cbd5e1;'>
                        Avg: <b>{subset['mean_sampah'].mean():,.1f} ton/hari</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALISIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    col_scatter, col_box = st.columns(2)

    with col_scatter:
        st.markdown(
            "<div class='section-header'>🔬 Scatter: Mean vs Max Sampah</div>",
            unsafe_allow_html=True,
        )
        fig_scatter = px.scatter(
            df_filtered,
            x="mean_sampah",
            y="max_sampah",
            color="status",
            size="std_sampah",
            hover_name="nama_kabupaten_kota",
            color_discrete_map=STATUS_COLORS,
            labels={
                "mean_sampah": "Rata-rata Sampah",
                "max_sampah":  "Maksimum Sampah",
                "std_sampah":  "Std Deviasi",
            },
        )
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            height=450,
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(**DEFAULT_LEGEND, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color="rgba(0,0,0,0.3)")))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_box:
        st.markdown(
            "<div class='section-header'>📦 Distribusi Sampah per Status</div>",
            unsafe_allow_html=True,
        )
        fig_box = px.box(
            df_filtered,
            x="status",
            y="mean_sampah",
            color="status",
            color_discrete_map=STATUS_COLORS,
            points="all",
            labels={"mean_sampah": "Rata-rata Sampah", "status": "Status"},
        )
        fig_box.update_layout(
            **PLOTLY_LAYOUT,
            height=450,
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        fig_box.update_traces(marker=dict(size=7, opacity=0.7))
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown(
        "<div class='section-header'>🔥 Heatmap Korelasi Fitur</div>",
        unsafe_allow_html=True,
    )

    numeric_cols = ["mean_sampah", "max_sampah", "std_sampah", "growth_rate"]
    corr = df_filtered[numeric_cols].corr()

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=["Mean", "Max", "Std Dev", "Growth Rate"],
            y=["Mean", "Max", "Std Dev", "Growth Rate"],
            colorscale=[
                [0, "#1e1b4b"], [0.25, "#312e81"],
                [0.5, "#4f46e5"], [0.75, "#818cf8"], [1, "#c7d2fe"],
            ],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=14, color="#e2e8f0"),
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Korelasi: %{z:.3f}<extra></extra>",
        )
    )
    fig_heatmap.update_layout(**PLOTLY_LAYOUT, height=400, xaxis=dict(side="bottom"))
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown(
        "<div class='section-header'>📊 Perbandingan Mean vs Max Sampah</div>",
        unsafe_allow_html=True,
    )
    df_compare = df_filtered.sort_values("mean_sampah", ascending=False).head(15)
    fig_compare = go.Figure()
    fig_compare.add_trace(
        go.Bar(
            name="Rata-rata",
            x=df_compare["nama_kabupaten_kota"],
            y=df_compare["mean_sampah"],
            marker_color="#818cf8",
            marker_line_width=0,
            opacity=0.85,
        )
    )
    fig_compare.add_trace(
        go.Bar(
            name="Maksimum",
            x=df_compare["nama_kabupaten_kota"],
            y=df_compare["max_sampah"],
            marker_color="#c084fc",
            marker_line_width=0,
            opacity=0.85,
        )
    )
    fig_compare.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        barmode="group",
        xaxis=dict(tickangle=-45, tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Volume Sampah (ton)"),
        legend=dict(**DEFAULT_LEGEND, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DETAIL WILAYAH
# ══════════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.markdown(
        "<div class='section-header'>🔍 Eksplorasi Wilayah</div>",
        unsafe_allow_html=True,
    )

    selected = st.selectbox(
        "Pilih Kabupaten/Kota",
        sorted(df_filtered["nama_kabupaten_kota"]),
        index=0,
    )

    detail     = df_filtered[df_filtered["nama_kabupaten_kota"] == selected].iloc[0]
    status     = detail["status"]
    badge_class = f"status-{status.lower()}"

    col_info, col_radar = st.columns([1, 1.3])

    with col_info:
        st.markdown(
            f"""
            <div class='glass-card'>
                <h2 style='color:#e2e8f0; margin:0 0 8px 0; font-size:1.4rem;'>{selected}</h2>
                <span class='status-badge {badge_class}'>{status}</span>
                <div style='margin-top: 24px;'>
                    <div class='detail-row'>
                        <span class='detail-label'>Cluster</span>
                        <span class='detail-value'>{int(detail['cluster'])}</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-label'>Rata-rata Sampah</span>
                        <span class='detail-value'>{detail['mean_sampah']:,.2f} ton</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-label'>Maksimum Sampah</span>
                        <span class='detail-value'>{detail['max_sampah']:,.2f} ton</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-label'>Standar Deviasi</span>
                        <span class='detail-value'>{detail['std_sampah']:,.2f}</span>
                    </div>
                    <div class='detail-row'>
                        <span class='detail-label'>Growth Rate</span>
                        <span class='detail-value'>{detail['growth_rate']:.2f}%</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rank_mean  = (df["mean_sampah"] <= detail["mean_sampah"]).sum()
        percentile = rank_mean / len(df) * 100
        st.markdown(
            f"""
            <div class='glass-card' style='text-align:center;'>
                <div style='color:#94a3b8; font-size:0.85rem; text-transform:uppercase;
                            letter-spacing:1.5px; margin-bottom:8px;'>
                    Peringkat Volume Sampah
                </div>
                <div style='font-size:2.4rem; font-weight:800; color:#818cf8;'>
                    #{rank_mean} <span style='font-size:1rem; color:#64748b;'>/ {len(df)}</span>
                </div>
                <div style='color:#94a3b8; font-size:0.8rem; margin-top:4px;'>
                    Persentil ke-{percentile:.0f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_radar:
        st.markdown(
            "<div class='section-header'>🕸️ Profil Radar</div>",
            unsafe_allow_html=True,
        )
        features_radar = ["mean_sampah", "max_sampah", "std_sampah"]
        feature_labels = ["Rata-rata", "Maksimum", "Std Deviasi"]
        values = []
        for f in features_radar:
            vmin, vmax = df[f].min(), df[f].max()
            norm = (detail[f] - vmin) / (vmax - vmin) if vmax > vmin else 0
            values.append(norm)
        values.append(values[0])
        labels = feature_labels + [feature_labels[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels,
                fill="toself",
                fillcolor="rgba(129, 140, 248, 0.15)",
                line=dict(color="#818cf8", width=2),
                marker=dict(size=6, color="#a5b4fc"),
                name=selected,
            )
        )
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(size=9, color="#64748b"),
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(size=12, color="#94a3b8"),
                ),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown(
        "<div class='section-header'>📊 Perbandingan dengan Seluruh Wilayah</div>",
        unsafe_allow_html=True,
    )
    df_highlight = df.sort_values("mean_sampah", ascending=False).copy()
    colors = [
        "#818cf8" if name == selected else "rgba(255,255,255,0.1)"
        for name in df_highlight["nama_kabupaten_kota"]
    ]
    fig_hl = go.Figure(
        go.Bar(
            x=df_highlight["nama_kabupaten_kota"],
            y=df_highlight["mean_sampah"],
            marker_color=colors,
            marker_line_width=0,
        )
    )
    fig_hl.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", title="Rata-rata Sampah"),
    )
    st.plotly_chart(fig_hl, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDIKSI CLUSTER
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(
        "<div class='section-header'>🤖 Prediksi Cluster Baru</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p style='color:#94a3b8; font-size:0.95rem; margin-bottom:24px;'>
            Masukkan data sampah untuk memprediksi cluster menggunakan model
            <b style='color:#a5b4fc'>K-Means</b> yang sudah dilatih.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.form("predict_form"):
        pc1, pc2 = st.columns(2)
        with pc1:
            inp_mean = st.number_input(
                "Rata-rata Sampah (ton/hari)",
                min_value=0.0,
                max_value=5000.0,
                value=500.0,
                step=10.0,
                format="%.2f",
                help="Masukkan nilai dalam satuan TON per hari",
            )
            inp_max = st.number_input(
                "Maksimum Sampah (ton/hari)",
                min_value=0.0,
                max_value=5000.0,
                value=800.0,
                step=10.0,
                format="%.2f",
            )
        with pc2:
            inp_std = st.number_input(
                "Standar Deviasi",
                min_value=0.0,
                max_value=1000.0,
                value=150.0,
                step=5.0,
                format="%.2f",
            )
            inp_growth = st.number_input(
                "Growth Rate (%)",
                min_value=-100.0,
                max_value=500.0,
                value=-50.0,
                step=1.0,
                format="%.2f",
            )

        submitted = st.form_submit_button(
            "🔮 Prediksi Cluster",
            use_container_width=True,
        )

    if submitted:
        input_data = np.array([[inp_mean, inp_max, inp_std, inp_growth]])

        try:
            scaled      = scaler.transform(input_data)
            prediction  = model.predict(scaled)[0]

            # Mapping cluster → status (sesuai hasil training)
            cluster_mean = {
                i: df[df["cluster"] == i]["mean_sampah"].mean()
                for i in df["cluster"].unique()
            }
            sorted_clusters = sorted(cluster_mean, key=cluster_mean.get)
            dynamic_map = {
                sorted_clusters[0]: "Terkendali",
                sorted_clusters[1]: "Waspada",
                sorted_clusters[2]: "Kritis",
            }
            pred_status = dynamic_map.get(prediction, f"Cluster {prediction}")
            badge_cls   = f"status-{pred_status.lower()}"
            status_emoji = {"Terkendali": "🟢", "Waspada": "🟡", "Kritis": "🔴"}.get(pred_status, "⚪")

            st.markdown(
                f"""
                <div class='glass-card' style='text-align:center; padding:40px;'>
                    <div style='font-size:3rem; margin-bottom:12px;'>{status_emoji}</div>
                    <div style='font-size:1rem; color:#94a3b8; text-transform:uppercase;
                                letter-spacing:2px; margin-bottom:8px;'>
                        Hasil Prediksi
                    </div>
                    <div style='font-size:2.2rem; font-weight:800; color:#e2e8f0; margin-bottom:16px;'>
                        Cluster {prediction}
                    </div>
                    <span class='status-badge {badge_cls}' style='font-size:1.1rem; padding:10px 28px;'>
                        {pred_status}
                    </span>
                    <div style='margin-top:28px; display:flex; justify-content:center; gap:40px;
                                color:#94a3b8; font-size:0.85rem;'>
                        <div>
                            <div style='font-weight:600; color:#cbd5e1;'>{inp_mean:,.0f}</div>
                            <div>Mean (ton)</div>
                        </div>
                        <div>
                            <div style='font-weight:600; color:#cbd5e1;'>{inp_max:,.0f}</div>
                            <div>Max (ton)</div>
                        </div>
                        <div>
                            <div style='font-weight:600; color:#cbd5e1;'>{inp_std:,.0f}</div>
                            <div>Std Dev</div>
                        </div>
                        <div>
                            <div style='font-weight:600; color:#cbd5e1;'>{inp_growth:.2f}%</div>
                            <div>Growth</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error saat prediksi: {e}")

    st.markdown(
        "<div class='section-header'>📐 Informasi Model</div>",
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-icon'>🎯</div>
                <div class='metric-value' style='color:#818cf8; font-size:2rem;'>{model.n_clusters}</div>
                <div class='metric-label'>Jumlah Cluster</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-icon'>🔄</div>
                <div class='metric-value' style='color:#a78bfa; font-size:2rem;'>{model.n_iter_}</div>
                <div class='metric-label'>Iterasi Konvergensi</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-icon'>📏</div>
                <div class='metric-value' style='color:#c084fc; font-size:2rem;'>{model.inertia_:,.0f}</div>
                <div class='metric-label'>Inertia (SSE)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-header'>📍 Centroid Cluster (Scaled)</div>",
        unsafe_allow_html=True,
    )
    centers_df = pd.DataFrame(
        model.cluster_centers_,
        columns=["Mean Sampah", "Max Sampah", "Std Deviasi", "Growth Rate"],
    )
    centers_df.index.name = "Cluster"
    st.dataframe(
        centers_df.style.format("{:.4f}").background_gradient(cmap="cool", axis=0),
        use_container_width=True,
    )
