from multiprocessing import Process, Lock, Value, Manager
import numpy as np
import time
import itertools
import random
import csv
import os

# --- Función objetivo de dos variables
def funcion_objetivo_con_restricciones(x, r=1e5):
    x1, x2, x3, x4, x5 = x

    # Función objetivo
    f = 5.3578547 * x3**2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141

    # Restricciones (g ≤ 0)
    g1 = 85.334407 + 0.0056858*x2*x5 + 0.0006262*x1*x4 - 0.0022053*x3*x5 - 92
    g2 = -85.334407 - 0.0056858*x2*x5 - 0.0006262*x1*x4 + 0.0022053*x3*x5
    g3 = 80.51249 + 0.0071317*x2*x5 + 0.0029955*x1*x2 + 0.0021813*x3**2 - 110
    g4 = -80.51249 - 0.0071317*x2*x5 - 0.0029955*x1*x2 - 0.0021813*x3**2 + 90
    g5 = 9.300961 + 0.0047026*x3*x5 + 0.0012547*x1*x3 + 0.0019085*x3*x4 - 25
    g6 = -9.300961 - 0.0047026*x3*x5 - 0.0012547*x1*x3 - 0.0019085*x3*x4 + 20

    restricciones = [g1, g2, g3, g4, g5, g6]

    # Penalización por violación de restricciones
    penalizacion = r * sum(max(0, g)**2 for g in restricciones)
    
    return f + penalizacion

# --- Algoritmo PSO
def ejecutar_pso(funcion, limite_inferior, limite_superior, dimensiones, parametros, max_iteraciones=500):
    num_particulas, w, c1, c2 = parametros
    num_particulas = int(num_particulas)

    posiciones = np.random.uniform(limite_inferior, limite_superior, (num_particulas, dimensiones))
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
        posiciones = np.clip(posiciones, limite_inferior, limite_superior)

        nuevos_puntajes = np.array([funcion(p) for p in posiciones])
        mejora = nuevos_puntajes < puntajes_personales
        mejor_personal[mejora] = posiciones[mejora]
        puntajes_personales[mejora] = nuevos_puntajes[mejora]

        if np.min(nuevos_puntajes) < puntaje_global:
            mejor_global = posiciones[np.argmin(nuevos_puntajes)]
            puntaje_global = np.min(nuevos_puntajes)

    return puntaje_global, mejor_global

# --- Función que corre en cada proceso
def busqueda_aleatoria(lock, id_proceso, combinaciones, mejor_puntaje, mejores_parametros, mejor_solucion, dimensiones, limites, contador):
    print(f"[Proceso {id_proceso}] Evaluando {len(combinaciones)} combinaciones aleatorias...")
    resultados = []

    for params in combinaciones:
        try:
            score, solucion = ejecutar_pso(funcion_objetivo_con_restricciones, limites[0][0], limites[0][1], dimensiones, params)
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
    dimensiones = 5
    limites = [
    (78, 102),  # x1
    (33, 45),   # x2
    (27, 45),   # x3
    (27, 45),   # x4
    (27, 45)    # x5
]
  
    # Hiperparámetros que queremos ajustar
    espacio_parametros = {
        'num_particulas': [10, 20, 30, 40, 50],
        'w': [0.1, 0.3, 0.5, 0.7, 0.9],
        'c1': [0.5, 1.0, 1.5, 2.0, 2.5],
        'c2': [0.5, 1.0, 1.5, 2.0, 2.5]
    }

    # Generar combinaciones aleatorias
    todas = list(itertools.product(
        espacio_parametros['num_particulas'],
        espacio_parametros['w'],
        espacio_parametros['c1'],
        espacio_parametros['c2']
    ))

    num_muestras = 300
    combinaciones_aleatorias = random.sample(todas, min(num_muestras, len(todas)))

    num_procesos = 6
    
    combinaciones_con_peso = [(params, int(params[0]) ** 2) for params in combinaciones_aleatorias]
    cargas = [[] for _ in range(num_procesos)]
    pesos_cargas = [0 for _ in range(num_procesos)]
    combinaciones_ordenadas = sorted(combinaciones_con_peso, key=lambda x: -x[1])
    for combinacion, peso in combinaciones_ordenadas:
        idx = pesos_cargas.index(min(pesos_cargas))
        cargas[idx].append(combinacion)
        pesos_cargas[idx] += peso

# --- Verificación del reparto ---
    for i in range(num_procesos):
        total_peso = sum(int(c[0])**2 for c in cargas[i])
        print(f"Proceso {i}: {len(cargas[i])} combinaciones, peso total estimado: {total_peso}")


    # Variables compartidas entre procesos
    lock = Lock()
    procesos = []
    mejor_puntaje = Value('d', float('inf'))
    manager = Manager()
    mejores_parametros = manager.list([""] * 4)
    mejor_solucion = manager.list([0.0] * dimensiones)
    contador = Value('i', 0)

    print("Iniciando búsqueda aleatoria de hiperparámetros con PSO en paralelo...\n")
    inicio = time.time()

    for n in range(num_procesos):
        p = Process(target=busqueda_aleatoria,
                    args=(lock, n, list(cargas[n]), mejor_puntaje, mejores_parametros, mejor_solucion, dimensiones, limites, contador))
        p.start()
        procesos.append(p)

    for p in procesos:
        p.join()

    fin = time.time()
    duracion = round(fin - inicio, 4)

    print("\nResultados finales:")
    print(f"Tiempo total: {duracion:.2f} segundos")
    print(f"Mejor puntaje obtenido: {mejor_puntaje.value}")
    print("Mejores hiperparámetros encontrados:")
    for i, p in enumerate(mejores_parametros):
        print(f"  Parámetro {i+1}: {p}")
    print("Variables óptimas encontradas:")
    for i, val in enumerate(mejor_solucion):
        print(f"  x{i+1} = {val}")

    # --- Guardar en CSV acumulativo ---
    nombre_csv = "resultados_pso_randomsearch_funcion3.csv"
    existe = os.path.exists(nombre_csv)

    with open(nombre_csv, mode='a', newline='') as archivo:
        writer = csv.writer(archivo)
        if not existe:
            writer.writerow([
                "num_procesos", "tiempo", "puntaje", 
                "param_num_particulas", "param_w", "param_c1", "param_c2", 
                "x1", "x2"
            ])
        writer.writerow([
            num_procesos,
            round(duracion, 4),
            mejor_puntaje.value,
            *mejores_parametros,
            mejor_solucion[0],
            mejor_solucion[1]
        ])
