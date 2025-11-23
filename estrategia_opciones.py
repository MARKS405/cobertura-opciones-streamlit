import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import streamlit as st

# === Estilo de gráficos ===
plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({
    'axes.facecolor': 'white',
    'axes.edgecolor': '#d3d3d3',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#999999',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.frameon': False,
    'font.size': 9
})

# ===============================
# 1. Descargar datos y preparar entorno
# ===============================
def obtener_datos(ticker, dias_horizonte=90, ventana_vol=252*3):
    hoy = datetime.today()
    inicio = hoy - timedelta(days=ventana_vol * 1.5)
    datos = yf.download(ticker, start=inicio, end=hoy, progress=False)
    datos = datos[['Close']].dropna().rename(columns={'Close': 'Precio'})
    datos['Retornos_log'] = np.log(datos['Precio'] / datos['Precio'].shift(1))
    return datos

# ===============================
# 2. Volatilidad histórica (EWMA)
# ===============================
def volatilidad_ewma(retornos, lambda_=0.94):
    var = retornos.var()
    for r in retornos[::-1]:
        var = lambda_ * var + (1 - lambda_) * r**2
    return np.sqrt(var * 252)

# ===============================
# 3. Tasa libre de riesgo estimada desde ETF SHY (1-3Y Treasury)
# ===============================
def obtener_rf_shy():
    try:
        shy = yf.download('SHY', period='6mo')
        retornos = shy['Close'].pct_change().dropna()
        tasa_anualizada = retornos.mean() * 252  # días hábiles al año
        return float(tasa_anualizada)
    except:
        return 0.03  # tasa por defecto si falla la descarga

# ===============================
# 4. Cálculo de precios de opciones (Black–Scholes)
# ===============================
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ===============================
# 5. Simulación de precios futuros (GBM)
# ===============================
def simular_trayectorias_MB(S0, T, r, sigma, pasos=90, n_sim=10000):
    dt = T / pasos
    drift = (r - 0.5 * sigma**2) * dt
    shock = sigma * np.sqrt(dt) * np.random.randn(n_sim, pasos)
    log_returns = drift + shock
    precios_log = np.cumsum(log_returns, axis=1)
    precios = S0 * np.exp(precios_log)
    return precios

# ===============================
# 6. Cálculo de payoff por estrategia
# ===============================
def calcular_payoffs(S0, ST, T, r, sigma, monto):
    if ST.ndim > 1:
        ST = ST[:, -1]
    K_put = S0 * 0.95
    K_call = S0 * 1.05

    prima_put = black_scholes_put(S0, K_put, T, r, sigma)
    prima_call = black_scholes_call(S0, K_call, T, r, sigma)

    sin_cobertura = ST - S0
    put_protectora = (np.maximum(K_put - ST, 0) + ST - S0 - prima_put)
    collar = (
        np.maximum(K_put - ST, 0) - np.maximum(ST - K_call, 0) + ST - S0
        - prima_put + prima_call
    )

    resultados = pd.DataFrame({
        'Sin cobertura': sin_cobertura,
        'Put protectora': put_protectora,
        'Collar': collar
    })

    ajuste = monto / S0
    resultados *= ajuste

    resumen = {
        'Spot': S0,
        'Strike Put': round(K_put, 2),
        'Prima Put': round(prima_put, 2),
        'Strike Call': round(K_call, 2),
        'Prima Call': round(prima_call, 2),
        'Rentabilidad esperada': resultados.mean().round(2).to_dict(),
        'VaR 5%': resultados.quantile(0.05).round(2).to_dict(),
        '% pérdida >10%': (resultados < -0.10*S0).mean().round(3).to_dict(),
        'Volatilidad': resultados.std().round(2).to_dict(),
        'Hedge effectiveness': ((resultados['Sin cobertura'].var() - resultados.var()) / resultados['Sin cobertura'].var()).round(2).to_dict()
    }

    return resultados, resumen

# ===============================
# 7. Función maestra para correr todo el flujo
# ===============================
def evaluar_estrategias(ticker='SPY', monto=1000, dias=90):
    datos = obtener_datos(ticker)
    retornos = datos['Retornos_log'].dropna()
    sigma = volatilidad_ewma(retornos)
    r = obtener_rf_shy()
    S0 = float(datos['Precio'].iloc[-1])
    T = dias / 252

    ST = simular_trayectorias_MB(S0, T, r, sigma, pasos=dias, n_sim=10000) # Use dias as pasos and increase n_sim
    payoffs, resumen = calcular_payoffs(S0, ST, T, r, sigma, monto)
    
    return payoffs, resumen, datos, ST

# ===============================
# 8. Visualización de resultados
# ===============================
def graficar_simulaciones(ST, S0):
    if ST.ndim > 1:
        ST = ST[:, -1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ST, bins=50, color='skyblue', edgecolor='black')
    ax.axvline(S0, color='red', linestyle='--', label='Spot')
    ax.set_title('Distribución de precios simulados al vencimiento')
    ax.set_xlabel('Precio futuro')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

def graficar_payoffs(payoffs):
    fig, ax = plt.subplots(figsize=(8, 4))
    payoffs.plot.hist(bins=50, alpha=0.6, ax=ax)
    ax.axvline(0, color='black', linestyle='--', label='Break-even')
    ax.set_title('Distribución de Payoffs por Estrategia')
    ax.set_xlabel('Ganancia/Pérdida (USD)')
    ax.set_ylabel('Frecuencia')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

def graficar_historico_y_simulaciones(datos, ST, ticker='SPY'):
    fig, ax = plt.subplots(figsize=(10, 5))

    datos['Precio'].plot(ax=ax, color='blue', label=f'{ticker} - Histórico', linewidth=2)
    ultima_fecha = datos.index[-1]
    pasos = ST.shape[1]
    fechas_futuras = pd.bdate_range(start=ultima_fecha + pd.Timedelta(days=1), periods=pasos)
    for i in range(min(10, ST.shape[0])):
        ax.plot(fechas_futuras, ST[i], color='gray', alpha=0.3)
    promedio = ST.mean(axis=0)
    ax.plot(fechas_futuras, promedio, color='red', linewidth=2, label='Promedio simulado')
    ax.set_title('Precio histórico y escenarios simulados (MBG)', fontsize=12, weight='bold')
    ax.set_xlabel('Fecha', fontsize=10)
    ax.set_ylabel('Precio', fontsize=10)
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
