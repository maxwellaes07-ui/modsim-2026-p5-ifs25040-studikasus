import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI APLIKASI STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Simulasi Monte Carlo - Pembangunan Gedung FITE",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling yang Adaptif untuk Dark Mode & Light Mode
st.markdown("""
<style>
    /* Header Utama */
    .main-header { 
        font-size: 2.5rem; 
        color: #3B82F6; 
        text-align: center; 
        margin-bottom: 1rem; 
        font-weight: 800;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 10px;
    }
    
    /* Sub Header */
    .sub-header { 
        font-size: 1.5rem; 
        color: #60A5FA; 
        margin-top: 2rem; 
        font-weight: bold; 
        margin-bottom: 1rem;
    }
    
    /* Kotak Informasi - Adaptif */
    .info-box { 
        padding: 1.2rem; 
        border-radius: 10px; 
        border: 1px solid #3B82F6;
        border-left: 6px solid #3B82F6; 
        margin-bottom: 1.5rem;
        background-color: rgba(59, 130, 246, 0.05);
        color: inherit; 
    }
    
    /* Kartu Metrik - Teks Putih Mutlak */
    .metric-card { 
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%); 
        padding: 1.5rem; 
        border-radius: 12px; 
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    
    .metric-card h3 { 
        color: #FFFFFF !important; 
        margin: 0; 
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-card p { 
        color: #E0E7FF !important; 
        margin: 5px 0 0 0; 
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. KELAS PEMODELAN SISTEM
# ============================================================================
class ProjectStage:
    def __init__(self, name, base_params, risk_factors=None, dependencies=None):
        self.name = name
        self.optimistic = base_params['optimistic']
        self.most_likely = base_params['most_likely']
        self.pessimistic = base_params['pessimistic']
        self.risk_factors = risk_factors or {}
        self.dependencies = dependencies or []
        
    def sample_duration(self, n_simulations, resource_multiplier=1.0):
        base_duration = np.random.triangular(
            self.optimistic, self.most_likely, self.pessimistic, n_simulations
        )
        base_duration = base_duration * resource_multiplier

        for risk_name, risk_params in self.risk_factors.items():
            if risk_params['type'] == 'discrete':
                occurrence = np.random.random(n_simulations) < risk_params['probability']
                base_duration = np.where(occurrence, base_duration * (1 + risk_params['impact']), base_duration)
            elif risk_params['type'] == 'continuous':
                factor = np.random.normal(risk_params['mean'], risk_params['std'], n_simulations)
                base_duration = base_duration / np.clip(factor, 0.6, 1.4)
        
        return base_duration

class MonteCarloProjectSimulation:
    def __init__(self, stages_config, num_simulations=10000, resource_boost=1.0):
        self.stages_config = stages_config
        self.num_simulations = num_simulations
        self.resource_boost = resource_boost
        self.stages = {name: ProjectStage(name, cfg['base_params'], cfg.get('risk_factors'), cfg.get('dependencies')) 
                       for name, cfg in stages_config.items()}
    
    def run_simulation(self):
        durations = pd.DataFrame(index=range(self.num_simulations))
        end_times = pd.DataFrame(index=range(self.num_simulations))
        
        for name, stage in self.stages.items():
            durations[name] = stage.sample_duration(self.num_simulations, self.resource_boost)
            start_time = 0 if not stage.dependencies else end_times[stage.dependencies].max(axis=1)
            end_times[name] = start_time + durations[name]
        
        results = durations.copy()
        results['Total_Duration'] = end_times.max(axis=1)
        for name in self.stages.keys():
            results[f'{name}_Finish'] = end_times[name]
        return results

# ============================================================================
# 3. FUNGSI VISUALISASI
# ============================================================================
def plot_completion_curve(results):
    sorted_durations = np.sort(results['Total_Duration'])
    p = np.linspace(0, 1, len(sorted_durations))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sorted_durations, y=p, mode='lines', name='Probabilitas', line=dict(color='#3B82F6', width=4)))
    
    for dl, clr in zip([16, 20, 24], ['#EF4444', '#F59E0B', '#10B981']):
        prob = np.mean(results['Total_Duration'] <= dl)
        fig.add_vline(x=dl, line_dash="dash", line_color=clr)
        fig.add_annotation(x=dl, y=prob, text=f"{dl} bln: {prob:.1%}", font=dict(color=clr))

    fig.update_layout(title="Kurva Probabilitas Penyelesaian Proyek", xaxis_title="Bulan", yaxis_title="Probabilitas",
                      template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white")
    return fig

# ============================================================================
# 4. MAIN INTERFACE
# ============================================================================
def main():
    # Header Utama Aplikasi
    st.markdown('<h1 class="main-header">📊 Simulasi Monte Carlo: Estimasi Waktu Proyek</h1>', unsafe_allow_html=True)
    
    # [PENGANTAR DIAWAL]
    st.markdown("""
    <div class="info-box">
    <b>Topik:</b> Estimasi Waktu Pembangunan Gedung Fakultas Informatika & Teknik Elektro (FITE).<br>
    <b>Kasus:</b> Proyek pembangunan gedung 5 lantai dengan fasilitas lengkap (ruang kelas, laboratorium komputer, 
    laboratorium elektro, laboratorium mobile, laboratorium VR/AR, laboratorium game, ruang dosen, toilet, dan ruang serbaguna).<br><br>
    Aplikasi ini menggunakan simulasi Monte Carlo untuk memodelkan ketidakpastian seperti cuaca buruk, 
    keterlambatan material teknis, dan variabilitas produktivitas pekerja untuk menghasilkan estimasi yang lebih akurat.
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR KONFIGURASI ---
    st.sidebar.header("⚙️ Parameter Simulasi")
    
    # Pengaturan Iterasi: Min 1000, Max 50000, Step 1000
    n_sim = st.sidebar.slider(
        "Jumlah Iterasi Simulasi",
        min_value=1000,
        max_value=50000,
        value=20000,
        step=1000,
        help="Semakin banyak iterasi, semakin akurat hasilnya tetapi lebih lama waktu prosesnya."
    )
    
    st.sidebar.subheader("🚀 Resource & Akselerasi")
    resource_option = st.sidebar.radio("Pengaruh Penambahan Resource:", ["Standar (1.0x)", "Menengah (0.9x)", "Maksimal (0.8x)"])
    boost_map = {"Standar (1.0x)": 1.0, "Menengah (0.9x)": 0.9, "Maksimal (0.8x)": 0.8}
    
    # Konfigurasi Tahapan (Bulan)
    config = {
        "Pekerjaan Persiapan & Struktur": {
            "base_params": {"optimistic": 4, "most_likely": 5, "pessimistic": 7},
            "risk_factors": {"Cuaca_Buruk": {"type": "discrete", "probability": 0.4, "impact": 0.2}},
            "dependencies": []
        },
        "Konstruksi Arsitektural Lantai 1-5": {
            "base_params": {"optimistic": 6, "most_likely": 8, "pessimistic": 11},
            "risk_factors": {"Produktivitas_Pekerja": {"type": "continuous", "mean": 1.0, "std": 0.15}},
            "dependencies": ["Pekerjaan Persiapan & Struktur"]
        },
        "Instalasi ME & Lab Khusus (VR/AR/Game)": {
            "base_params": {"optimistic": 4, "most_likely": 6, "pessimistic": 10},
            "risk_factors": {"Keterlambatan_Material_Khusus": {"type": "discrete", "probability": 0.3, "impact": 0.4}},
            "dependencies": ["Konstruksi Arsitektural Lantai 1-5"]
        },
        "Finishing & Interior Lab": {
            "base_params": {"optimistic": 2, "most_likely": 3, "pessimistic": 5},
            "risk_factors": {"Perubahan_Desain_Interior": {"type": "discrete", "probability": 0.2, "impact": 0.3}},
            "dependencies": ["Instalasi ME & Lab Khusus (VR/AR/Game)"]
        }
    }

    if st.sidebar.button("🚀 Jalankan Simulasi"):
        with st.spinner(f"Sedang menghitung {n_sim:,} kemungkinan skenario..."):
            sim = MonteCarloProjectSimulation(config, n_sim, boost_map[resource_option])
            results = sim.run_simulation()

            # --- BAGIAN 1: STATISTIK UTAMA ---
            st.markdown('<h2 class="sub-header">📈 Statistik Utama Proyek</h2>', unsafe_allow_html=True)
            avg_dur = results['Total_Duration'].mean()
            p95 = np.percentile(results['Total_Duration'], 95)
            risk_20 = (results['Total_Duration'] > 20).mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{avg_dur:.1f} Bulan</h3><p>Rata-rata Durasi Total</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{p95:.1f} Bulan</h3><p>Estimasi Terburuk (P95)</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>{risk_20:.1%}</h3><p>Risiko Melebihi 20 Bulan</p></div>', unsafe_allow_html=True)

            # --- BAGIAN 2: VISUALISASI ---
            st.markdown('<h2 class="sub-header">📊 Visualisasi Hasil Simulasi</h2>', unsafe_allow_html=True)
            st.plotly_chart(plot_completion_curve(results), use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<h2 class="sub-header">📅 Probabilitas Target Deadline</h2>', unsafe_allow_html=True)
                target_deadlines = [16, 20, 24]
                df_prob = pd.DataFrame({
                    "Skenario Deadline": [f"{d} Bulan" for d in target_deadlines],
                    "Probabilitas Selesai Tepat Waktu": [f"{np.mean(results['Total_Duration'] <= d):.2%}" for d in target_deadlines]
                })
                st.table(df_prob)
            with col_b:
                st.markdown('<h2 class="sub-header">🔍 Informasi Teknis</h2>', unsafe_allow_html=True)
                st.write(f"• Jumlah Iterasi: **{n_sim:,}**")
                st.write(f"• Pengaruh Resource: **{resource_option}**")
                st.write("• Faktor Risiko: **Cuaca, Material Teknis, Produktivitas**")
                st.write("• Metode Sampling: **Triangular & Normal Distribution**")
    else:
        # Tampilan Awal Sebelum Run
        st.info("Atur parameter di sidebar kiri dan klik tombol 'Jalankan Simulasi' untuk memulai analisis probabilitas.")
        
        st.markdown('<h2 class="sub-header">📋 Tahapan yang Akan Disimulasikan</h2>', unsafe_allow_html=True)
        for stage in config.keys():
            st.write(f"✔️ {stage}")

if __name__ == "__main__":
    main()