import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --- Cargar archivo CSV ---
archivo = "resultados_pso_gridsearch.csv"
df = pd.read_csv(archivo)

# --- Agrupar por número de procesos (núcleos) y calcular estadísticas ---
agrupado = df.groupby("num_procesos")["tiempo"].mean().reset_index()
agrupado = agrupado.sort_values("num_procesos")

# --- Calcular speed-up y eficiencia ---
t1 = agrupado[agrupado["num_procesos"] == 1]["tiempo"].values[0]
agrupado["speed_up"] = t1 / agrupado["tiempo"]
agrupado["eficiencia"] = agrupado["speed_up"] / agrupado["num_procesos"]

# --- Graficar ---
plt.figure(figsize=(16, 5))

# Gráfica 1: Tiempo
plt.subplot(1, 3, 1)
plt.plot(agrupado["num_procesos"], agrupado["tiempo"], marker="o")
plt.title("Tiempo de ejecución")
plt.xlabel("Núcleos")
plt.ylabel("Tiempo (s)")
plt.grid(True)

# Gráfica 2: Speed-up
plt.subplot(1, 3, 2)
plt.plot(agrupado["num_procesos"], agrupado["speed_up"], marker="o", color="green")
plt.title("Speed-up")
plt.xlabel("Núcleos")
plt.ylabel("Speed-up")
plt.grid(True)

# Gráfica 3: Eficiencia
plt.subplot(1, 3, 3)
plt.plot(agrupado["num_procesos"], agrupado["eficiencia"], marker="o", color="orange")
plt.title("Eficiencia")
plt.xlabel("Núcleos")
plt.ylabel("Eficiencia")
plt.grid(True)

plt.tight_layout()

# Guardar la figura
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_archivo = f"graf_grid.png"
plt.savefig(nombre_archivo, dpi=300)

print(f"Gráficas guardadas como: {nombre_archivo}")

plt.show()
