import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pycbc
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.coordinates import spherical_to_cartesian

def plot_gw(sample_rate, start_time, f_lower, mass1, mass2, distance, dec, ra, inclination, pol, coa_phase, spin1a, spin1az, spin1po, spin2a, spin2az, spin2po, h1, l1, v1, k1):
    spin1x, spin1y, spin1z = spherical_to_cartesian(spin1a, spin1az, spin1po)
    spin2x, spin2y, spin2z = spherical_to_cartesian(spin2a, spin2az, spin2po)
    hp, hc = get_td_waveform(approximant="IMRPhenomXPHM", mass1=mass1, mass2=mass2, distance=distance, coa_phase=coa_phase, inclination=inclination,
                            spin1x=spin1x, spin1y=spin1y, spin1z=spin1z, spin2x=spin2x, spin2y=spin2y, spin2z=spin2z, delta_t=1.0/sample_rate, f_lower=f_lower)
    hp.prepend_zeros(3*2048)
    hc.prepend_zeros(3*2048)
    plotDetector = (h1,l1,v1,k1)
    waves = {}
    colors = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728')
    
    for i, det in enumerate(('H1', 'L1', 'V1', 'K1')):
        if plotDetector[i]:
            wave = Detector(det).project_wave(hp, hc, ra, dec, pol)
            waves[det] = wave.time_slice(float(wave.end_time)+start_time, float(wave.end_time))

    fig = plt.figure(figsize=(20,6))
    ax = plt.axes()
    for i, det in enumerate(('H1', 'L1', 'V1', 'K1')):
        if plotDetector[i]:
            plt.plot(waves[det].sample_times, waves[det], color=colors[i], label=det)
    plt.title(f"mass1={mass1}M$_\odot$, mass2={mass2}M$_\odot$, spin1a={spin1a}, spin1az={spin1az}, spin1po={spin1po}, spin2a={spin2a}, spin2az={spin2az}, spin2po={spin2po}, \n distance={distance}Mpc, coa_phase={coa_phase}, inclination={inclination}, dec={dec}, ra={ra}, pol_angle={pol}, f_lower={f_lower}Hz")
    plt.xlabel("Time from merger (s)", fontsize=18)
    plt.ylabel("Strain", fontsize=18)
    plt.ylim(-2e-21, 2e-21)
    if 1 in plotDetector:
        plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    return fig, ax


st.set_page_config(layout="wide")
hide_menu = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
# plt.style.use('ggplot')
st.title("GW Plotting App")
st.markdown("[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![made-with-streamlit](https://img.shields.io/badge/Made%20with-Streamlit-1f425f.svg)](https://streamlit.io/) [![made-with-pycbc](https://img.shields.io/badge/Made%20with-Pycbc-1f425f.svg)](https://pycbc.org/) [![made-with-matplotlib](https://img.shields.io/badge/Made%20with-Matplotlib-1f425f.svg)](https://matplotlib.org/)")
st.markdown("Waveform model: [`IMRPhenomXPHM`](https://doi.org/10.48550/arXiv.2004.06503)")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.markdown("Detectors:")
with col2:
    h1= st.checkbox('LIGO H1', value=True)
with col3:
    l1= st.checkbox('LIGO L1')
with col4:
    v1= st.checkbox('Virgo')
with col5:
    k1= st.checkbox('KAGRA')

with col1:
    sample_rate = st.selectbox("Sampling rate (Hz)", (2048, 4096))
with col2:
    f_lower = st.slider("f_lower (Hz)", 8, 25, 20)
with col3:
    start_time = st.slider("Start time (s)", -5.0, -1.0, -3.0, step=0.5)
# st.text("Parameters:")
# st.text("Intrinsic parameters:")
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
with col1:
    mass1 = st.slider("Mass1 (solar mass)", 5, 50,30)
with col2:
    mass2 = st.slider("Mass2 (solar mass)", 5, 50,30)
with col3:
    spin1a = st.slider("Spin1 amplitude", 0.0, 0.99, 0.0)
with col4:
    spin1az = st.slider("Spin1 azimuthal", 0.0, 2*np.pi, 0.0)
with col5:
    spin1po = st.slider("Spin1 polar", 0.0, np.pi, 0.0)
with col6:
    spin2a = st.slider("Spin2 amplitude", 0.0, 0.99, 0.0)
with col7:
    spin2az = st.slider("Spin2 azimuthal", 0.0, 2*np.pi, 0.0)
with col8:
    spin2po = st.slider("Spin2 polar", 0.0, np.pi, 0.0)

# st.text("Extrinsic parameters:")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    distance = st.slider("Distance (Mpc)", 500, 1500, 1000)
with col2:
    coa_phase = st.slider("Coalescence phase", 0.0, 2*np.pi, 0.0)
with col3:
    inclination = st.slider("Inclination", 0.0, 2*np.pi, 0.0)
with col4:
    dec = st.slider("Declination", 0.0, np.pi, 0.0)
with col5:
    ra = st.slider("Right ascension", 0.0, 2*np.pi, 0.0)
with col6:
    pol = st.slider("Polarization angle", 0.0, 2*np.pi, 0.0)

fig, ax = plot_gw(sample_rate, start_time, f_lower, mass1, mass2, distance, dec, ra, inclination, pol, coa_phase, spin1a, spin1az, spin1po, spin2a, spin2az, spin2po, h1, l1, v1, k1)
st.pyplot(fig)

filename1 = "gw.pdf"
plt.savefig(filename1)
filename2 = "gw.png"
plt.savefig(filename2)

col1, col2, _, _, _  = st.columns(5)
with col1:
    with open(filename1, "rb") as img:
        btn = st.download_button(
            label="Download image as pdf",
            data=img,
            file_name=filename1,
            mime="image/png"
        )
with col2:
    with open(filename2, "rb") as img:
        btn = st.download_button(
            label="Download image as png",
            data=img,
            file_name=filename2,
            mime="image/png"
        )
