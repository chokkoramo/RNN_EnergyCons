import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

st.set_page_config(page_title="RNN: consumo de energía semanal", layout="wide")
BASE = Path(__file__).parent
plt.rcParams["figure.dpi"] = 110
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


# ------------------------------------------------------------------
# Carga del modelo y la escala generados en el Colab
# ------------------------------------------------------------------
@st.cache_resource
def cargar_modelo(huella, _datos):
    """Keras necesita una ruta: se escribe el archivo en una carpeta temporal."""
    from tensorflow import keras
    ruta = Path(tempfile.gettempdir()) / f"modelo_{huella}.keras"
    if not ruta.exists():
        ruta.write_bytes(_datos)
    return keras.models.load_model(ruta)


with st.sidebar:
    st.header("Modelo del Colab")
    st.caption("Sube los archivos de la sección 6. Despliegue. Si no subes nada, "
               "se usan los que están en el repositorio.")
    up_modelo = st.file_uploader("modelo_rnn.keras", type=["keras"])
    up_escala = st.file_uploader("escala.json", type=["json"])

if up_modelo is not None and up_escala is not None:
    datos_modelo = up_modelo.getvalue()
    escala = json.loads(up_escala.getvalue())
    origen = "archivos subidos"
elif (BASE / "modelo_rnn.keras").exists() and (BASE / "escala.json").exists():
    datos_modelo = (BASE / "modelo_rnn.keras").read_bytes()
    escala = json.loads((BASE / "escala.json").read_text())
    origen = "repositorio"
else:
    st.title("Redes neuronales recurrentes: consumo de energía con patrón semanal")
    st.warning(
        "Sube **modelo_rnn.keras** y **escala.json** en la barra lateral "
        "(los descargas del Colab, sección 6. Despliegue)."
    )
    st.stop()

faltan = [k for k in ("p_min", "p_max", "ventana") if k not in escala]
if faltan:
    st.error("El archivo escala.json no tiene: " + ", ".join(faltan))
    st.stop()

HUELLA = hashlib.md5(datos_modelo).hexdigest()[:10]
P_MIN, P_MAX, VENTANA = escala["p_min"], escala["p_max"], int(escala["ventana"])
try:
    with st.spinner("Cargando el modelo entrenado en el Colab…"):
        modelo = cargar_modelo(HUELLA, datos_modelo)
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()
st.sidebar.success(f"Modelo cargado desde: {origen}")
UNIDADES = modelo.layers[0].units

escalar = lambda v: (np.asarray(v) - P_MIN) / (P_MAX - P_MIN)
desescalar = lambda v: np.asarray(v) * (P_MAX - P_MIN) + P_MIN


# ------------------------------------------------------------------
# Los mismos datos del Colab (misma semilla)
# ------------------------------------------------------------------
@st.cache_data
def generar_datos():
    rng = np.random.default_rng(7)
    n = 52 * 7                        # 52 semanas, empieza en lunes
    t = np.arange(n)
    patron = np.array([120, 126, 124, 128, 118, 82, 64], dtype=float)  # kWh lun..dom
    return patron[t % 7] + 0.04 * t + rng.normal(0, 5, n)


consumo = generar_datos()
N = len(consumo)
CORTE = int(N * 0.8)
serie = escalar(consumo)
X = np.array([serie[i:i + VENTANA] for i in range(N - VENTANA)])[..., np.newaxis]
y = np.array([serie[i + VENTANA] for i in range(N - VENTANA)])
N_TRAIN = CORTE - VENTANA
X_test, y_test = X[N_TRAIN:], y[N_TRAIN:]


@st.cache_data
def evaluar(huella):
    est = desescalar(modelo.predict(X_test, verbose=0).ravel())
    real = desescalar(y_test)
    base = desescalar(X_test[:, -1, 0])
    return est, real, base


def que_probar(lista):
    st.info("**Qué probar**\n\n" + "\n".join(f"- {q}" for q in lista))


def cierre(conclusion, pregunta):
    st.success("**Conclusión**\n\n" + conclusion)
    st.markdown(f"**Pregunta:** {pregunta}")


def dibujar_rnn(unidades=8, ventana=7):
    TEAL, CORAL, GRIS = "#1D9E75", "#D85A30", "#888780"
    VERDE_CLARO, CORAL_CLARO = "#CBEBDD", "#F5D3C5"
    p_rnn = unidades + unidades * unidades + unidades
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 12), gridspec_kw={"height_ratios": [1, 1]})
    def flecha(ax, a, b, c="black", **k):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=c, **k))

    # ============ 1. La red como se construye ============
    ax = ax1
    ys = [0.5] if unidades == 1 else [0.85 - i * 0.7 / (unidades - 1) for i in range(unidades)]
    xin, xh, xd, xo = 0.12, 0.42, 0.72, 0.9
    for yy in ys:
        ax.plot([xin, xh], [0.5, yy], color=GRIS, lw=0.8, alpha=0.6, zorder=1)
        ax.plot([xh, xd], [yy, 0.5], color=CORAL, lw=1, alpha=0.7, ls="--", zorder=1)
    ax.add_patch(FancyBboxPatch((xh - 0.05, 0.08), 0.1, 0.84, boxstyle="round,pad=0.01", fc="none", ec=TEAL, ls="--"))
    ax.scatter([xh] * unidades, ys, s=380, c=VERDE_CLARO, edgecolors=TEAL, zorder=2)
    for yy in ys:
        ax.text(xh, yy, "n", ha="center", va="center", fontsize=8, zorder=3)
    ax.scatter([xin], [0.5], s=1200, c="lightgray", zorder=2)
    ax.text(xin, 0.5, "x", ha="center", va="center", fontsize=15)
    ax.scatter([xd], [0.5], s=1000, c=CORAL_CLARO, edgecolors=CORAL, zorder=2)
    ax.text(xd, 0.5, "Dense", ha="center", va="center", fontsize=9)
    flecha(ax, (xd + 0.035, 0.5), (xo - 0.03, 0.5))
    ax.text(xo, 0.5, "ŷ", ha="center", va="center", fontsize=16, bbox=dict(boxstyle="round", fc="lightgray", ec="gray"))
    ax.add_patch(FancyArrowPatch((xh + 0.035, 0.94), (xh - 0.035, 0.94), connectionstyle="arc3,rad=1.3",
                                 arrowstyle="-|>", color=TEAL, lw=2, mutation_scale=14))
    ax.text(xh + 0.07, 1.03, f"La salida h ({unidades} números) se guarda\ny vuelve como memoria al día siguiente",
            fontsize=10, color=TEAL, va="center")
    ax.text(xd, 0.72, "solo después\ndel último día", ha="center", fontsize=10, color=CORAL)
    ax.text(xin, 0.36, "consumo de UN día\n(entran de a uno)", ha="center", va="top", fontsize=10)
    ax.text(xin, -0.04, "Entrada", ha="center", va="top", fontsize=11, weight="bold")
    ax.text(xh, -0.04, f"SimpleRNN({unidades}): {unidades} neuronas\n{unidades} + {unidades*unidades} + {unidades} = {p_rnn} pesos",
            ha="center", va="top", fontsize=11)
    ax.text(xd, -0.04, f"Dense(1)\n{unidades} + 1 = {unidades+1} pesos", ha="center", va="top", fontsize=11)
    ax.set_title("1. La red como se construye: una sola capa de neuronas que se reutiliza", fontsize=13, loc="left")

    # ============ 2. La misma capa, día por día ============
    ax = ax2
    xs = [0.07 + i * 0.62 / max(ventana - 1, 1) for i in range(ventana)]
    w = min(0.1, 0.5 / ventana)
    fs = 10 if ventana <= 8 else 7
    n_dib = min(unidades, 8)
    cy = [0.575] if n_dib == 1 else [0.7 - i * 0.26 / (n_dib - 1) for i in range(n_dib)]
    for i, x in enumerate(xs):
        ultimo = i == ventana - 1
        ax.add_patch(FancyBboxPatch((x - w / 2, 0.36), w, 0.43, boxstyle="round,pad=0.01",
                                    fc=VERDE_CLARO if not ultimo else "#A6DCC4", ec=TEAL, lw=1.2 if not ultimo else 2.2))
        ax.text(x, 0.755, f"Día {i+1}", ha="center", va="center", fontsize=fs, weight="bold")
        ax.scatter([x] * n_dib, cy, s=45 if ventana <= 8 else 15, c="white", edgecolors=TEAL, zorder=3)
        if unidades > 8:
            ax.text(x, 0.4, "…", ha="center", va="center", fontsize=fs)
        ax.scatter([x], [0.16], s=550 if ventana <= 8 else 200, c="lightgray", zorder=2)
        ax.text(x, 0.16, f"x{i+1}", ha="center", va="center", fontsize=fs - 1, zorder=3)
        flecha(ax, (x, 0.2), (x, 0.345), c="gray")
        if not ultimo:
            flecha(ax, (x + w / 2 + 0.008, 0.55), (xs[i+1] - w / 2 - 0.008, 0.55), c=TEAL, lw=2)
            ax.text((x + xs[i+1]) / 2, 0.59, f"h{i+1}", ha="center", fontsize=fs - 1, color=TEAL, weight="bold")
    xd = xs[-1] + 0.14
    flecha(ax, (xs[-1] + w / 2 + 0.008, 0.55), (xd - 0.035, 0.55), c=CORAL, lw=2)
    ax.text((xs[-1] + xd) / 2, 0.59, f"h{ventana}", ha="center", fontsize=fs - 1, color=CORAL, weight="bold")
    ax.scatter([xd], [0.55], s=1000, c=CORAL_CLARO, edgecolors=CORAL, zorder=2)
    ax.text(xd, 0.55, "Dense", ha="center", va="center", fontsize=9, zorder=3)
    flecha(ax, (xd + 0.035, 0.55), (xd + 0.1, 0.55))
    ax.text(xd + 0.13, 0.55, "ŷ", ha="center", va="center", fontsize=16, bbox=dict(boxstyle="round", fc="lightgray", ec="gray"))
    ax.text(xd + 0.13, 0.44, f"consumo estimado\ndel día {ventana+1}", ha="center", va="top", fontsize=10)
    y0 = 0.85
    ax.plot([xs[0] - w / 2, xs[-1] + w / 2], [y0, y0], color=TEAL, lw=1.5)
    for xx in (xs[0] - w / 2, xs[-1] + w / 2):
        ax.plot([xx, xx], [y0, y0 - 0.03], color=TEAL, lw=1.5)
    ax.text((xs[0] + xs[-1]) / 2, y0 + 0.04,
            f"Son las MISMAS {unidades} neuronas (los mismos {p_rnn} pesos) en cada día — no son {ventana*unidades} neuronas distintas",
            ha="center", fontsize=10.5, color=TEAL)
    ax.text(0.0, -0.02,
            f"Días 1 a {ventana-1}: la capa solo produce la memoria h y se la pasa al día siguiente. No hay predicción todavía.\n"
            f"Día {ventana}: el último h es el único que llega a Dense, que produce la única predicción ŷ.",
            ha="left", va="top", fontsize=11)
    ax.set_title("2. La misma red vista en el tiempo: un cuadro por día", fontsize=13, loc="left")

    for a in (ax1, ax2):
        a.set_xlim(-0.02, 1.08); a.set_ylim(-0.15, 1.1); a.axis("off")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------
st.title("Redes neuronales recurrentes: consumo de energía con patrón semanal")
st.caption("Compañera del Colab. Consumo diario simulado de un edificio durante 52 semanas. "
           "El modelo de las pestañas 4 y 5 es el que entrenaste y guardaste en el Colab.")

t1, t2, t3, t4, t5 = st.tabs([
    "1. Datos y ventanas", "2. Una neurona", "3. Estructura de la red",
    "4. Evaluación", "5. Estimar el día siguiente",
])

# ---------------- 1. Datos y ventanas ----------------
with t1:
    st.markdown(
        f"Son {N} días de consumo ({N // 7} semanas), los mismos del Colab. Los {CORTE} primeros son para entrenar "
        f"y los {N - CORTE} últimos para probar. Cada ejemplo es una **ventana** de {VENTANA} días con su "
        f"respuesta: el día siguiente."
    )
    inicio = st.slider("Mueve la ventana (día en que empieza)", 1, N - VENTANA, 1)
    i0 = inicio - 1
    dias = np.arange(1, N + 1)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(dias, consumo, color="#1F4E8C", lw=1)
    ax.axvspan(CORTE + 0.5, N + 0.5, color="gray", alpha=0.12, label="prueba")
    ax.axvspan(inicio - 0.5, inicio + VENTANA - 0.5, color="#FFD166", alpha=0.7, label=f"ventana ({VENTANA} días)")
    ax.scatter([inicio + VENTANA], [consumo[i0 + VENTANA]], color="#D85A30", zorder=3, s=45, label="respuesta")
    ax.set_xlabel("Día"); ax.set_ylabel("Consumo (kWh)"); ax.grid(alpha=0.3); ax.legend(loc="lower left")
    st.pyplot(fig); plt.close(fig)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(dias[:28], consumo[:28], marker="o", color="#1F4E8C")
        for s in range(0, 28, 7):
            ax.axvspan(s + 5.5, s + 7.5, color="#FFD166", alpha=0.35)
        ax.set_title("Primeras 4 semanas (en amarillo, los fines de semana)")
        ax.set_xlabel("Día"); ax.set_ylabel("kWh"); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)
    with c2:
        prom = [consumo[:CORTE][np.arange(CORTE) % 7 == d].mean() for d in range(7)]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(DIAS, prom, color=["#1F4E8C"] * 5 + ["#D85A30"] * 2)
        ax.set_title("Consumo promedio por día de la semana")
        ax.set_ylabel("kWh"); ax.grid(alpha=0.3, axis="y")
        st.pyplot(fig); plt.close(fig)

    st.markdown(
        f"Esta ventana usa los días **{inicio} a {inicio + VENTANA - 1}** "
        f"({DIAS[i0 % 7]} a {DIAS[(i0 + VENTANA - 1) % 7]}) para estimar el día **{inicio + VENTANA}** "
        f"({DIAS[(i0 + VENTANA) % 7]}, {consumo[i0 + VENTANA]:.1f} kWh). "
        f"La forma de los datos de entrenamiento es `({N_TRAIN}, {VENTANA}, 1)`."
    )
    que_probar([
        "Mueve la ventana de a un día: el día de la respuesta va cambiando de lunes a domingo.",
        "Busca en el gráfico de 4 semanas dónde está la caída de cada fin de semana.",
    ])
    cierre(
        "El consumo tiene un **patrón que se repite cada 7 días**: es alto de lunes a viernes, baja el sábado y "
        "llega al mínimo el domingo, con algo de ruido encima. Como la ventana tiene 7 días, siempre contiene una "
        "semana completa y el modelo puede reconocer en qué parte del ciclo está. La parte de prueba va al final "
        "para simular el futuro.",
        "¿Qué día de la semana tiene el consumo más bajo?",
    )

# ---------------- 2. Una neurona ----------------
with t2:
    st.markdown(
        "Es la red de la parte A del Colab: **una sola neurona**, los primeros 12 días de consumo y ventanas de "
        "3 días. Mueve los pesos para ver cómo cambia la estimación."
    )
    st.latex(r"h_t = \tanh(w_x x_t + w_h h_{t-1} + b) \qquad \hat{y} = w_y h_3 + b_y")
    mini = np.round(consumo[:12], 1)
    mn, mx = mini.min(), mini.max()
    nm = (mini - mn) / (mx - mn)
    Xm = np.array([nm[i:i + 3] for i in range(9)]); ym = nm[3:]
    DEF = {"wx": 0.5, "wh": 0.3, "b": 0.0, "wy": 0.8, "by": 0.0}
    for k, v in DEF.items():
        st.session_state.setdefault("a_" + k, v)

    def restablecer():
        for k, v in DEF.items():
            st.session_state["a_" + k] = v

    cols = st.columns(5)
    etiquetas = {"wx": "w_x (entrada)", "wh": "w_h (memoria)", "b": "b (sesgo)", "wy": "w_y (salida)", "by": "b_y (sesgo salida)"}
    p = {k: cols[i].slider(etiquetas[k], -3.0, 3.0, step=0.01, key="a_" + k) for i, k in enumerate(DEF)}
    st.button("Restablecer pesos", on_click=restablecer)

    def adelante(x):
        h = [0.0]
        for t in range(3):
            h.append(np.tanh(p["wx"] * x[t] + p["wh"] * h[-1] + p["b"]))
        return p["wy"] * h[3] + p["by"], h

    v = st.selectbox("Ventana", range(9), format_func=lambda k: f"{k + 1}: consumos {mini[k:k + 3].tolist()} → real {mini[k + 3]}")
    yv, h = adelante(Xm[v])
    lineas = [f"h{t} = tanh({p['wx']:.2f}·{Xm[v][t-1]:.3f} + {p['wh']:.2f}·{h[t-1]:.3f} + {p['b']:.2f}) = {h[t]:.3f}" for t in (1, 2, 3)]
    lineas.append(f"ŷ  = {p['wy']:.2f}·{h[3]:.3f} + {p['by']:.2f} = {yv:.3f}  →  {yv * (mx - mn) + mn:.1f} kWh  (real {mini[v + 3]:.1f} kWh)")
    st.code("\n".join(lineas), language=None)

    est2 = [adelante(x)[0] * (mx - mn) + mn for x in Xm]
    mse = np.mean([(adelante(x)[0] - t) ** 2 for x, t in zip(Xm, ym)])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(range(1, 13), mini, marker="o", label="Real", color="#1F4E8C")
    ax.plot(range(4, 13), est2, marker="o", ls="--", label="Estimación de la neurona", color="#D85A30")
    ax.set_xlabel("Día"); ax.set_ylabel("Consumo (kWh)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"MSE (escala 0–1): {mse:.4f}")
    st.pyplot(fig); plt.close(fig)
    que_probar([
        "Pon w_h en 0: la neurona pierde la memoria y solo mira el último día.",
        "Intenta bajar el MSE a mano. Compáralo con el MSE que obtuvo el entrenamiento en el Colab (parte A).",
        "Pon w_x en 0: ¿qué información le queda a la neurona?",
    ])
    cierre(
        "La neurona recurrente lee **un día a la vez** y guarda en **h** un resumen de lo que ha visto: esa es su "
        "memoria. En cada día combina el dato nuevo con la memoria del día anterior, siempre con **los mismos "
        "pesos**. Encontrar buenos pesos a mano es difícil; el entrenamiento lo hace de forma automática.",
        "¿Qué pasa con la memoria de la neurona cuando w_h vale 0?",
    )

# ---------------- 3. Estructura ----------------
with t3:
    st.markdown(
        f"El modelo del Colab tiene **{UNIDADES} neuronas** recurrentes, ventana de **{VENTANA} días** y "
        f"**{modelo.count_params()} parámetros**. Cambia los valores para ver cómo sería la red con otra configuración."
    )
    c1, c2 = st.columns(2)
    n = c1.slider("Neuronas de la capa recurrente", 1, 32, UNIDADES)
    vent = c2.slider("Tamaño de la ventana (días)", 3, 20, VENTANA)
    fig = dibujar_rnn(n, vent)
    st.pyplot(fig); plt.close(fig)
    tabla = pd.DataFrame({
        "Parte": ["w_x (dato → neuronas)", "w_h (memoria de ayer → neuronas de hoy)", "b (sesgos)",
                  "Dense: pesos", "Dense: sesgo", "Total"],
        "Cálculo": [f"{n} × 1", f"{n} × {n}", f"{n}", f"{n}", "1", ""],
        "Parámetros": [n, n * n, n, n, 1, n + n * n + n + n + 1],
    })
    st.dataframe(tabla, hide_index=True, width="stretch")
    que_probar([
        "Pon 1 neurona y 3 días: es exactamente la red de la pestaña 2 (5 parámetros).",
        "Pasa de 8 a 16 neuronas: ¿cuánto crecen los parámetros?",
        f"Cambia la ventana de {VENTANA} a 14 días: ¿cambia el total de parámetros?",
    ])
    cierre(
        "Una RNN es **una sola capa de neuronas que se reutiliza** en cada día de la ventana. Por eso la cantidad "
        "de parámetros depende del número de neuronas, pero **no del tamaño de la ventana**. Durante la ventana la "
        "capa solo va actualizando su memoria, y únicamente la memoria del **último día** pasa a la capa Dense, "
        "que da la estimación.",
        f"Si la ventana pasa de {VENTANA} a 14 días, ¿cambia la cantidad de parámetros?",
    )

# ---------------- 4. Evaluación ----------------
with t4:
    est, real, base = evaluar(HUELLA)
    mae_rnn = np.mean(np.abs(est - real)); mae_base = np.mean(np.abs(base - real))
    st.markdown(
        f"Comparamos el modelo con una **línea base** muy simple: *el consumo de mañana será igual al de hoy*. "
        f"Los datos son los {len(real)} días de prueba, que el modelo nunca vio al entrenar."
    )
    hist = escala.get("historia")
    if hist and "loss" in hist:
        with st.expander("Ver cómo aprendió la red durante el entrenamiento en el Colab"):
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(hist["loss"], label="Entrenamiento", color="#1F4E8C")
            if "val_loss" in hist:
                ax.plot(hist["val_loss"], label="Validación", color="#D85A30")
            ax.set_xlabel("Época"); ax.set_ylabel("Error (MSE)"); ax.set_yscale("log")
            ax.grid(alpha=0.3); ax.legend()
            st.pyplot(fig); plt.close(fig)
            st.caption("El error baja a medida que la red ajusta sus pesos.")

    c1, c2 = st.columns(2)
    c1.metric("Error promedio de la RNN (MAE)", f"{mae_rnn:.1f} kWh")
    c2.metric("Error promedio de la línea base (MAE)", f"{mae_base:.1f} kWh")
    ver_base = st.checkbox("Mostrar también la línea base en el gráfico")
    dias_t = np.arange(CORTE + 1, N + 1)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(dias_t, real, label="Real", color="#1F4E8C")
    ax.plot(dias_t, est, "--", label="Estimación RNN", color="#D85A30")
    if ver_base:
        ax.plot(dias_t, base, ":", label="Línea base (mañana = hoy)", color="gray")
    ax.set_xlabel("Día"); ax.set_ylabel("Consumo (kWh)"); ax.grid(alpha=0.3); ax.legend(loc="lower left")
    st.pyplot(fig); plt.close(fig)

    dia_sem = np.arange(CORTE, N) % 7
    err_rnn = [np.mean(np.abs(est - real)[dia_sem == d]) for d in range(7)]
    err_base = [np.mean(np.abs(base - real)[dia_sem == d]) for d in range(7)]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    xx = np.arange(7)
    ax.bar(xx - 0.2, err_rnn, 0.4, label="RNN", color="#D85A30")
    ax.bar(xx + 0.2, err_base, 0.4, label="Línea base", color="gray")
    ax.set_xticks(xx, DIAS); ax.set_ylabel("Error promedio (kWh)")
    ax.set_title("Error promedio por día de la semana"); ax.grid(alpha=0.3, axis="y"); ax.legend()
    st.pyplot(fig); plt.close(fig)

    que_probar([
        "Activa la línea base: fíjate en qué días se despega de la curva real.",
        "Mira el gráfico de barras: ¿en qué días la línea base tiene errores muy grandes?",
    ])
    if mae_rnn < mae_base:
        mejora = 100 * (1 - mae_rnn / mae_base)
        texto = (f"Aquí la RNN **sí le gana a la línea base**: se equivoca en promedio {mae_rnn:.1f} kWh frente a "
                 f"{mae_base:.1f} kWh, es decir, reduce el error un {mejora:.0f} %. ")
    else:
        texto = (f"En este modelo la RNN no superó a la línea base ({mae_rnn:.1f} kWh frente a "
                 f"{mae_base:.1f} kWh); vale la pena revisar el entrenamiento en el Colab. ")
    cierre(
        texto + "La línea base falla sobre todo el **lunes** (el consumo sube y ella repite el del domingo) y el "
        "**sábado** (el consumo cae y ella repite el del viernes). La RNN aprendió el patrón semanal, así que "
        "**anticipa** esos cambios en vez de llegar un día tarde. Lo que ninguno puede predecir es el ruido "
        "aleatorio de cada día.",
        "¿En qué días de la semana se equivoca más la línea base?",
    )

# ---------------- 5. Estimar ----------------
with t5:
    sig = DIAS[N % 7]
    st.markdown(
        f"Escribe el consumo de **{VENTANA} días seguidos** y el modelo del Colab estima el día siguiente. "
        f"Los valores iniciales son la última semana de los datos, así que la estimación corresponde a un **{sig}**."
    )
    cols = st.columns(VENTANA)
    entradas = [cols[i].number_input(f"Día {i + 1} ({DIAS[(N - VENTANA + i) % 7]})",
                                     value=float(round(consumo[-VENTANA + i], 1)), step=1.0, format="%.1f")
                for i in range(VENTANA)]
    x = escalar(entradas).reshape(1, VENTANA, 1)
    valor = float(desescalar(modelo.predict(x, verbose=0)[0, 0]))
    st.metric("Estimación para el día siguiente", f"{valor:.1f} kWh", f"{valor - entradas[-1]:+.1f} frente al último día")
    if max(entradas) > P_MAX or min(entradas) < P_MIN:
        st.warning(
            f"Algunos valores están fuera del rango del entrenamiento ({P_MIN:.1f} a {P_MAX:.1f} kWh). "
            "El modelo nunca vio valores así, y su estimación puede ser poco confiable."
        )
    que_probar([
        "Con los valores iniciales, el último día es domingo (bajo): ¿el modelo estima un lunes bajo o alto?",
        "Cambia el domingo por un valor alto, como 125: ¿qué le pasa a la estimación?",
        "Escribe siete valores iguales, sin fin de semana: ¿qué estima ahora?",
    ])
    cierre(
        "El modelo no se limita a repetir el último valor: usa **la forma de la semana** para saber qué día viene. "
        "Por eso, después de un domingo bajo, estima un lunes alto, algo que la línea base no puede hacer. "
        "Si se cambian los valores y la semana deja de parecerse a las del entrenamiento, la estimación pierde "
        f"sentido. Además, solo es confiable dentro del rango de consumo del entrenamiento ({P_MIN:.1f} a {P_MAX:.1f} kWh).",
        "Si el último día de la ventana es un domingo con consumo bajo, ¿el modelo estima un lunes bajo o alto?",
    )
