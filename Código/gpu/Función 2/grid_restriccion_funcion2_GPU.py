import cupy as cp
import numpy as np
import itertools
import csv
import time
import os

# --- Función objetivo de dos variables
def funcion_objetivo_con_restriccion(x, r=1e5):
    f = x[:, 0]**2 + (x[:, 1] - 1)**2
    h = x[:, 1] - x[:, 0]**2
    penalizacion = r * (h**2)
    return f + penalizacion

# --- Algoritmo PSO adaptado para CuPy
def ejecutar_pso(funcion, limites_inf, limites_sup, dimensiones, parametros, max_iteraciones=1500):
    num_particulas, w, c1, c2 = parametros
    num_particulas = int(num_particulas)

    posiciones = cp.zeros((num_particulas, dimensiones))
    for d in range(dimensiones):
        posiciones[:, d] = cp.random.uniform(limites_inf[d], limites_sup[d], num_particulas)

    velocidades = cp.random.uniform(-1, 1, (num_particulas, dimensiones))
    mejor_personal = posiciones.copy()
    puntajes_personales = funcion(posiciones)
    mejor_global = mejor_personal[cp.argmin(puntajes_personales)]
    puntaje_global = cp.min(puntajes_personales)

    for _ in range(max_iteraciones):
        r1 = cp.random.rand(num_particulas, dimensiones)
        r2 = cp.random.rand(num_particulas, dimensiones)
        velocidades = w * velocidades + c1 * r1 * (mejor_personal - posiciones) + c2 * r2 * (mejor_global - posiciones)
        posiciones += velocidades
        for d in range(dimensiones):
            posiciones[:, d] = cp.clip(posiciones[:, d], limites_inf[d], limites_sup[d])

        nuevos_puntajes = funcion(posiciones)
        mejora = nuevos_puntajes < puntajes_personales
        mejor_personal[mejora] = posiciones[mejora]
        puntajes_personales[mejora] = nuevos_puntajes[mejora]

        if cp.min(nuevos_puntajes) < puntaje_global:
            mejor_global = posiciones[cp.argmin(nuevos_puntajes)]
            puntaje_global = cp.min(nuevos_puntajes)

    return puntaje_global.item(), mejor_global.get()

# --- Programa principal
if __name__ == "__main__":
    dimensiones = 2
    limites = [(-1, 1)] * dimensiones

    # Espacio de búsqueda (grid search)
    espacio_parametros = {
        'num_particulas': [10, 20, 30, 40, 50],
        'w': [0.4, 0.6, 0.8, 0.9],
        'c1': [1.0, 1.5, 2.0, 2.5],
        'c2': [0.5, 1.0, 1.5, 2.0, 2.5]
    }

    todas_combinaciones = list(itertools.product(
        espacio_parametros['num_particulas'],
        espacio_parametros['w'],
        espacio_parametros['c1'],
        espacio_parametros['c2']
    ))

    limites_inf = cp.array([lim[0] for lim in limites])
    limites_sup = cp.array([lim[1] for lim in limites])

    mejor_puntaje = float('inf')
    mejores_parametros = None
    mejor_solucion = None

    print("Iniciando búsqueda exhaustiva (grid search) con PSO en GPU usando CuPy...\n")
    inicio = time.time()

    for params in todas_combinaciones:
        try:
            score, solucion = ejecutar_pso(funcion_objetivo_con_restriccion, limites_inf, limites_sup, dimensiones, params)
            if score < mejor_puntaje:
                mejor_puntaje = score
                mejores_parametros = params
                mejor_solucion = solucion
        except Exception as e:
            print(f"Error con parámetros {params}: {e}")

    fin = time.time()
    duracion = fin - inicio

    print("\nResultados finales:")
    print(f"Tiempo total: {duracion:.2f} segundos")
    print(f"Mejor puntaje obtenido: {mejor_puntaje}")
    print("Mejores hiperparámetros encontrados:")
    for i, p in enumerate(mejores_parametros):
        print(f"  Parámetro {i+1}: {p}")
    print("Variables óptimas encontradas:")
    for i, val in enumerate(mejor_solucion):
        print(f"  x{i+1} = {val}")

    # --- Guardar en CSV ---
    nombre_csv = "resultados_pso_gridsearch_f2_cupy.csv"
    existe = os.path.exists(nombre_csv)

    with open(nombre_csv, mode='a', newline='') as archivo:
        writer = csv.writer(archivo)
        if not existe:
            writer.writerow(["tiempo", "puntaje", "param_num_particulas", "param_w", "param_c1", "param_c2"])
        writer.writerow([
            round(duracion, 4),
            mejor_puntaje,
            *mejores_parametros
        ])

    print(f"\nResultado agregado a: {nombre_csv}")
