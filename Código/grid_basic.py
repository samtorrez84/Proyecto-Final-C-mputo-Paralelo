from multiprocessing import Process, Lock, Value, Manager
import numpy as np
import time
import itertools
import csv
from datetime import datetime
import os

# --- Función objetivo de dos variables
def funcion_objetivo(x):
    return (x[0] - 3)**2 + (x[1] + 1)**2

# --- Algoritmo PSO
def ejecutar_pso(funcion, limites_inf, limites_sup, dimensiones, parametros, max_iteraciones=50):
    num_particulas, w, c1, c2 = parametros
    num_particulas = int(num_particulas)

    posiciones = np.zeros((num_particulas, dimensiones))
    for d in range(dimensiones):
        posiciones[:, d] = np.random.uniform(limites_inf[d], limites_sup[d], num_particulas)
    
    velocidades = np.random.uniform(-1, 1, (num_particulas, dimensiones))
    mejor_personal = posiciones.copy()
    puntajes_personales = np.array([funcion(p) for p in posiciones])
    mejor_global = mejor_personal[np.argmin(puntajes_personales)]
    puntaje_global = np.min(puntajes_personales)

    for _ in range(max_iteraciones):
        r1 = np.random.rand(num_particulas, dimensiones)
        r2 = np.random.rand(num_particulas, dimensiones)
        velocidades = w * velocidades + c1 * r1 * (mejor_personal - posiciones) + c2 * r2 * (mejor_global - posiciones)
        posiciones += velocidades
        for d in range(dimensiones):
            posiciones[:, d] = np.clip(posiciones[:, d], limites_inf[d], limites_sup[d])

        nuevos_puntajes = np.array([funcion(p) for p in posiciones])
        mejora = nuevos_puntajes < puntajes_personales
        mejor_personal[mejora] = posiciones[mejora]
        puntajes_personales[mejora] = nuevos_puntajes[mejora]

        if np.min(nuevos_puntajes) < puntaje_global:
            mejor_global = posiciones[np.argmin(nuevos_puntajes)]
            puntaje_global = np.min(nuevos_puntajes)

    return puntaje_global, mejor_global

# --- Función que corre en cada proceso
def busqueda_exhaustiva(lock, id_proceso, combinaciones, mejor_puntaje, mejores_parametros, mejor_solucion, dimensiones, limites, contador):
    print(f"[Proceso {id_proceso}] Evaluando {len(combinaciones)} combinaciones...")
    resultados = []
    limites_inf = [lim[0] for lim in limites]
    limites_sup = [lim[1] for lim in limites]

    for params in combinaciones:
        try:
            score, solucion = ejecutar_pso(funcion_objetivo, limites_inf, limites_sup, dimensiones, params)
            resultados.append((score, params, solucion))
        except Exception as e:
            print(f"[Proceso {id_proceso}] Error con parámetros {params}: {e}")

    if resultados:
        mejor_local = min(resultados, key=lambda x: x[0])
        with lock:
            contador.value += 1
            print(f"[Proceso {id_proceso}] Finalizado. Procesos terminados: {contador.value}")
            if mejor_local[0] < mejor_puntaje.value:
                mejor_puntaje.value = mejor_local[0]
                for i in range(len(mejores_parametros)):
                    mejores_parametros[i] = str(mejor_local[1][i])
                for i in range(dimensiones):
                    mejor_solucion[i] = mejor_local[2][i]

# --- Programa principal
if __name__ == "__main__":
    dimensiones = 2
    limites = [(-10, 10)] * dimensiones  # Cada variable en [-10, 10]

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

    num_procesos = 5
    cargas = np.array_split(todas_combinaciones, num_procesos)

    lock = Lock()
    procesos = []
    mejor_puntaje = Value('d', float('inf'))
    manager = Manager()
    mejores_parametros = manager.list([""] * 4)
    mejor_solucion = manager.list([0.0] * dimensiones)
    contador = Value('i', 0)

    print("Iniciando búsqueda exhaustiva (grid search) con PSO en paralelo...\n")
    inicio = time.time()

    for n in range(num_procesos):
        p = Process(target=busqueda_exhaustiva,
                    args=(lock, n, list(cargas[n]), mejor_puntaje, mejores_parametros, mejor_solucion, dimensiones, limites, contador))
        p.start()
        procesos.append(p)

    for p in procesos:
        p.join()

    fin = time.time()
    duracion = fin - inicio

    print("\nResultados finales:")
    print(f"Tiempo total: {duracion:.2f} segundos")
    print(f"Mejor puntaje obtenido: {mejor_puntaje.value}")
    print("Mejores hiperparámetros encontrados:")
    for i, p in enumerate(mejores_parametros):
        print(f"  Parámetro {i+1}: {p}")
    print("Variables óptimas encontradas:")
    for i, val in enumerate(mejor_solucion):
        print(f"  x{i+1} = {val}")

    # --- Guardar en CSV ---
    nombre_csv = "resultados_pso_gridsearch.csv"
    existe = os.path.exists(nombre_csv)

    with open(nombre_csv, mode='a', newline='') as archivo:
        writer = csv.writer(archivo)
        if not existe:
            writer.writerow(["num_procesos", "tiempo", "puntaje", "param_num_particulas", "param_w", "param_c1", "param_c2"])
        writer.writerow([
            num_procesos,
            round(duracion, 4),
            mejor_puntaje.value,
            *mejores_parametros
        ])

    print(f"\n📁 Resultado agregado a: {nombre_csv}")
