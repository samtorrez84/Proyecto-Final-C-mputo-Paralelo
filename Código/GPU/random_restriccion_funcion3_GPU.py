import cupy as cp
import itertools
import random
import csv
import os
import time

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

    posiciones = cp.random.uniform(limite_inferior, limite_superior, (num_particulas, dimensiones))
    velocidades = cp.random.uniform(-1, 1, (num_particulas, dimensiones))

    mejor_personal = posiciones.copy()
    puntajes_personales = cp.array([funcion(p) for p in posiciones])
    mejor_global = mejor_personal[cp.argmin(puntajes_personales)]
    puntaje_global = cp.min(puntajes_personales)

    for _ in range(max_iteraciones):
        r1 = cp.random.rand(num_particulas, dimensiones)
        r2 = cp.random.rand(num_particulas, dimensiones)

        velocidades = w * velocidades + c1 * r1 * (mejor_personal - posiciones) + c2 * r2 * (mejor_global - posiciones)
        posiciones += velocidades
        posiciones = cp.clip(posiciones, limite_inferior, limite_superior)

        nuevos_puntajes = cp.array([funcion(p) for p in posiciones])
        mejora = nuevos_puntajes < puntajes_personales
        mejor_personal[mejora] = posiciones[mejora]
        puntajes_personales[mejora] = nuevos_puntajes[mejora]

        if cp.min(nuevos_puntajes) < puntaje_global:
            mejor_global = posiciones[cp.argmin(nuevos_puntajes)]
            puntaje_global = cp.min(nuevos_puntajes)

    return puntaje_global, mejor_global

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

    print("Iniciando búsqueda de hiperparámetros...")
    inicio = time.time()

    mejor_puntaje = float('inf')
    mejores_parametros = None
    mejor_solucion = None

    for params in combinaciones_aleatorias:
        try:
            score, solucion = ejecutar_pso(funcion_objetivo_con_restricciones, limites[0][0], limites[0][1], dimensiones, params)
            if score < mejor_puntaje:
                mejor_puntaje = score
                mejores_parametros = params
                mejor_solucion = solucion
        except Exception as e:
            print(f"Error con parámetros {params}: {e}")

    fin = time.time()
    duracion = round(fin - inicio, 4)

    print("\nResultados finales:")
    print(f"Tiempo total: {duracion:.2f} segundos")
    print(f"Mejor puntaje obtenido: {mejor_puntaje}")
    print("Mejores hiperparámetros encontrados:")
    print(f"  num_particulas: {mejores_parametros[0]}")
    print(f"  w: {mejores_parametros[1]}")
    print(f"  c1: {mejores_parametros[2]}")
    print(f"  c2: {mejores_parametros[3]}")
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
                "num_particulas", "w", "c1", "c2", "x1", "x2", "x3", "x4", "x5", "puntaje"
            ])
        writer.writerow([
            *mejores_parametros,
            *mejor_solucion,
            mejor_puntaje
        ])
