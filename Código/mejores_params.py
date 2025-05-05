import pandas as pd
import matplotlib.pyplot as plt

# Cargar CSV con encabezados
archivo = "resultados_pso_gridsearch.csv"
df = pd.read_csv(archivo)

# Filtrar solo soluciones óptimas (puntaje == 0)
df_optimo = df[df["puntaje"] == 0]

# Hiperparámetros a analizar
param_cols = ["param_num_particulas", "param_w", "param_c1", "param_c2"]

# Calcular modas
print("Mejores hiperparámetros entre soluciones óptimas:")
for col in param_cols:
    if not df_optimo.empty:
        moda = df_optimo[col].mode().iloc[0]
        print(f"  {col}: {moda}")
    else:
        print(f"  {col}: No hay soluciones con puntaje 0")

# Graficar frecuencias
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(param_cols):
    ax = axs[i // 2, i % 2]
    df_optimo[col].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title(f"Frecuencia de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frecuencia")
    ax.grid(True)

plt.tight_layout()
nombre_grafico = "params_grid.png"
plt.savefig(nombre_grafico, dpi=300)
print(f"\nGráfico guardado como: {nombre_grafico}")
