import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from estrategia_opciones import evaluar_estrategias, graficar_simulaciones, graficar_payoffs, graficar_historico_y_simulaciones

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="Cobertura con Opciones", layout="wide")
st.title("🛡️ Simulador de Estrategias de Cobertura con Opciones")

st.sidebar.header("⚙️ Configuración")
tickers_disponibles = sorted([
    '005930.KS', 'AA', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACI', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEE',
    'AEP', 'AFL', 'ALLE', 'ALB', 'AM', 'AMAT', 'AMD', 'AMGN', 'AMT', 'AMX', 'AMZN', 'APA', 'APD', 'APO', 'APP', 'AR',
    'ARTNA', 'ATLN', 'ATO', 'AVB', 'AVGO', 'AVA', 'AXP', 'BA', 'BAC', 'BAP', 'BAX', 'BBY', 'BE', 'BF-B', 'BHP', 'BIIB',
    'BJ', 'BK', 'BKNG', 'BLK', 'BKH', 'BMY', 'BP', 'BRK-B', 'BRX', 'BSX', 'BTI', 'BUD', 'BX', 'BXP', 'BZFD', 'C', 'CAH',
    'CAG', 'CAT', 'CB', 'CBRE', 'CDNS', 'CELH', 'CEG', 'CF', 'CHD', 'CHT', 'CHTR', 'CIO', 'CL', 'CMCSA', 'CMI', 'CNC',
    'CNP', 'COF', 'COLM', 'COP', 'CPB', 'CPRI', 'CRM', 'CRK', 'CRH', 'CSCO', 'CSX', 'CTVA', 'CUZ', 'CWCO', 'CWEN', 'CVS',
    'CVX', 'COST', 'CROX', 'CYL', 'D', 'DAL', 'DE', 'DELL', 'DEO', 'DGNX', 'DG', 'DHI', 'DHR', 'DINO', 'DIS', 'DLR',
    'DLTR', 'DOC', 'DOW', 'DUK', 'DVN', 'E', 'ECL', 'EC', 'ED', 'EIX', 'ELV', 'EME', 'EMN', 'EMR', 'ENB', 'EQIX',
    'EQNR', 'EOG', 'ES', 'ESS', 'ET', 'ETN', 'EQR', 'EVRG', 'EXC', 'EXP', 'F', 'FANG', 'FCX', 'FDX', 'FI', 'FMX',
    'FRT', 'G', 'GE', 'GEV', 'GD', 'GIS', 'GILD', 'GLNCY', 'GM', 'GOOG', 'GS', 'HAL', 'H', 'HCA', 'HD', 'HEI', 'HEINY',
    'HIW', 'HNHPF', 'HMC', 'HON', 'HPP', 'HR', 'HWM', 'IBM', 'ICE', 'IDXX', 'IFF', 'ILHMF', 'IMAX', 'IMBBY', 'INTC',
    'INTU', 'INV', 'IOBT', 'IP', 'IPG', 'IPI', 'IR', 'ISRG', 'ITT', 'JCI', 'JBHT', 'JNJ', 'JOYY', 'JPM', 'K', 'KHC',
    'KIM', 'KKR', 'KLAC', 'KMI', 'KO', 'KR', 'LHX', 'LI', 'LIN', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LYB', 'LYV',
    'LVMUY', 'MA', 'MAA', 'MAR', 'MCD', 'MC.PA', 'MCK', 'MGM', 'MMM', 'MNST', 'MO', 'MOS', 'MP', 'MPC', 'MPLX', 'MRK',
    'MRNA', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTNB', 'MU', 'NEE', 'NEM', 'NESN.SW', 'NI', 'NKE', 'NCLH', 'NDAQ', 'NFE',
    'NFLX', 'NIO', 'NOC', 'NOW', 'NOV', 'NTR', 'NSC', 'NSRGY', 'NVDA', 'NVO', 'NWN', 'ODFL', 'OGS', 'OKE', 'OMC',
    'ORCL', 'OVV', 'OHI', 'PANW', 'PARA', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGEN', 'PGR', 'PH',
    'PKG', 'PLL', 'PLD', 'PLTR', 'PLUG', 'PM', 'PNC', 'PPG', 'PPSI', 'PPL', 'PSX', 'PUBGY', 'PWR', 'QCOM', 'QBTS',
    'QUBT', 'REG', 'REGN', 'REXR', 'RGC', 'RGTI', 'RCL', 'RIO', 'RL', 'RIVN', 'ROKU', 'ROK', 'RSG', 'RTX', 'S', 'SAM',
    'SAP', 'SBUX', 'SCCO', 'SCHW', 'SHW', 'SHEL', 'SIE.DE', 'SIRI', 'SJW', 'SLB', 'SLG', 'SNAP', 'SNPS', 'SO', 'SONY',
    'SPG', 'SPGI', 'SPOT', 'SPY', 'SR', 'SSNLF', 'SRE', 'STAG', 'STZ', 'SUZ', 'SWX', 'SYK', 'T', 'TAP', 'TECK', 'TGEN', 'TGT',
    'TMO', 'TM', 'TMUS', 'TNX', 'TPR', 'TRNO', 'TRP', 'TRV', 'TSLA', 'TSM', 'TT', 'TTD', 'TTWO', 'TXN', 'TXT',
    'UBER', 'UDR', 'UAA', 'UL', 'UNH', 'UNP', 'UPS', 'USB', 'UVV', 'VALE', 'V', 'VFC', 'VIV', 'VLO', 'VMC', 'VMI',
    'VNO', 'VRTX', 'VTR', 'VZ', 'WAB', 'WBA', 'WBD', 'WEC', 'WELL', 'WFC', 'WMB', 'WMT', 'WPP', 'WY', 'XEL', 'XOM',
    'XYL', 'YARIY', 'YORW', 'ZBH', 'ZETA'
])

ticker = st.sidebar.selectbox("Selecciona un ticker:", tickers_disponibles, index=tickers_disponibles.index("SPY"))
monto = st.sidebar.slider("Monto de inversión (USD)", min_value=100, max_value=10000, value=1000, step=50)
dias = st.sidebar.slider("Horizonte de inversión (días)", min_value=30, max_value=252, value=90, step=15)

estrategia = st.sidebar.radio("Visualizar estrategia:", ["Sin cobertura", "Put protectora", "Collar"])

with st.spinner("Simulando precios y evaluando estrategias..."):
    payoffs, resumen, datos, ST = evaluar_estrategias(ticker, monto, dias)

# ============================
# Sección 1: Métricas
# ============================
st.subheader("📊 Métricas clave")
col1, col2, col3 = st.columns(3)

estrat = estrategia

col1.metric("🔻 VaR al 5%", f"${resumen['VaR 5%'][estrat]:,.2f}")
col2.metric("📈 Rentabilidad esperada", f"${resumen['Rentabilidad esperada'][estrat]:,.2f}")
col3.metric("💰 Spot actual", f"${resumen['Spot']:,.2f}")

# Extra KPIs
porc_perdidas = (payoffs[estrat] < -0.1 * monto).mean()
vol = payoffs[estrat].std()
hedge_ratio = 1 - (payoffs[estrat].var() / payoffs['Sin cobertura'].var()) if estrategia != "Sin cobertura" else 0

col4, col5, col6 = st.columns(3)
col4.metric("⚠️ Escenarios con pérdida > 10%", f"{porc_perdidas*100:.2f}%")
col5.metric("📉 Volatilidad estimada", f"${vol:.2f}")
if estrategia != "Sin cobertura":
    col6.metric("🛡️ Hedge effectiveness", f"{hedge_ratio*100:.2f}%")

# ============================
# Sección 2: Visualización de precios simulados
# ============================
st.subheader("📈 Distribución de precios simulados")
graficar_historico_y_simulaciones(datos, ST)

# ============================
# Sección 3: Gráfico de Payoffs
# ============================
st.subheader("💥 Distribución de payoff por estrategia")
graficar_payoffs(payoffs)

# ============================
# Sección 4: Detalles
# ============================
st.subheader("📋 Detalles de la estrategia seleccionada")
detalle_cols = [
    f"- Strike Put: ${resumen['Strike Put']}",
    f"- Prima Put: ${resumen['Prima Put']}",
    f"- Strike Call: ${resumen['Strike Call']}",
    f"- Prima Call: ${resumen['Prima Call']}"
]

if estrategia == "Put protectora":
    st.markdown("\n".join(detalle_cols[:2]))
elif estrategia == "Collar":
    st.markdown("\n".join(detalle_cols))
else:
    st.info("Esta estrategia no involucra derivados.")

st.caption("Proyecto desarrollado con 🐍 Python, Streamlit, y simulaciones estocásticas.")
