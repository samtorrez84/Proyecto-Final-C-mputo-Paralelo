import pandas as pd
import matplotlib.pyplot as plt

# Cargar el CSV
archivo = "resultados_pso_gridsearch.csv"
df = pd.read_csv(archivo)

# Gráfico de dispersión de puntajes por procesador
plt.figure(figsize=(10, 6))
plt.scatter(df["num_procesos"], df["puntaje"], color='royalblue', alpha=0.6, label="Puntajes obtenidos")
plt.axhline(0, color='red', linestyle='--', label="Valor óptimo (0)")
plt.title("Puntajes obtenidos por procesador")
plt.xlabel("Número de procesos")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

# Guardar gráfico
nombre_archivo = "puntajes_grid.png"
plt.savefig(nombre_archivo, dpi=300)
print(f"Gráfico guardado como: {nombre_archivo}")
plt.show()
