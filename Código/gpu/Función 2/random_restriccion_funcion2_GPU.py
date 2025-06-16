import cupy as cp
import itertools
import random
import csv
import os
import time

# --- Función objetivo de dos variables
def funcion_objetivo_con_restriccion(x, r=1e5):
    f = x[0]**2 + (x[1] - 1)**2
    h = x[1] - x[0]**2
    penalizacion = r * (h**2)
    return f + penalizacion

# --- Algoritmo PSO
def ejecutar_pso(funcion, limite_inferior, limite_superior, dimensiones, parametros, max_iteraciones=1500):
    num_particulas, w, c1, c2 = parametros
    num_particulas = int(num_particulas)

    posiciones = cp.random.uniform(limite_inferior, limite_superior, (num_particulas, dimensiones))
    velocidades = cp.random.uniform(-1, 1, (num_particulas, dimensiones))

    mejor_personal = posiciones.copy()
    puntajes_personales = cp.array([funcion(p) for p in cp.asnumpy(posiciones)])
    mejor_global = mejor_personal[cp.argmin(puntajes_personales)]
    puntaje_global = cp.min(puntajes_personales)

    for _ in range(max_iteraciones):
        r1 = cp.random.rand(num_particulas, dimensiones)
        r2 = cp.random.rand(num_particulas, dimensiones)

        velocidades = w * velocidades + c1 * r1 * (mejor_personal - posiciones) + c2 * r2 * (mejor_global - posiciones)
        posiciones += velocidades
        posiciones = cp.clip(posiciones, limite_inferior, limite_superior)

        nuevos_puntajes = cp.array([funcion(p) for p in cp.asnumpy(posiciones)])
        mejora = nuevos_puntajes < puntajes_personales
        mejor_personal[mejora] = posiciones[mejora]
        puntajes_personales[mejora] = nuevos_puntajes[mejora]

        if cp.min(nuevos_puntajes) < puntaje_global:
            mejor_global = posiciones[cp.argmin(nuevos_puntajes)]
            puntaje_global = cp.min(nuevos_puntajes)

    return puntaje_global.get(), cp.asnumpy(mejor_global)

# --- Programa principal
if __name__ == "__main__":
    dimensiones = 2
    limites = [(-1, 1)] * dimensiones

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
    num_muestras = 250
    combinaciones_aleatorias = random.sample(todas, min(num_muestras, len(todas)))

    print("Iniciando búsqueda aleatoria de hiperparámetros con PSO...\n")
    inicio = time.time()

    mejor_puntaje = float('inf')
    mejores_parametros = None
    mejor_solucion = None

    for params in combinaciones_aleatorias:
        try:
            score, solucion = ejecutar_pso(funcion_objetivo_con_restriccion, limites[0][0], limites[0][1], dimensiones, params)
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
    print(f"  x1 = {mejor_solucion[0]}")
    print(f"  x2 = {mejor_solucion[1]}")

    # Guardar en CSV acumulativo
    nombre_csv = "resultados_pso_randomsearch_f2.csv"
    existe = os.path.exists(nombre_csv)

    with open(nombre_csv, mode='a', newline='') as archivo:
        writer = csv.writer(archivo)
        if not existe:
            writer.writerow([
                "num_particulas", "w", "c1", "c2", 
                "puntaje", "x1", "x2", "tiempo"
            ])
        writer.writerow([
            mejores_parametros[0], mejores_parametros[1], mejores_parametros[2], mejores_parametros[3],
            mejor_puntaje, mejor_solucion[0], mejor_solucion[1], duracion
        ])
