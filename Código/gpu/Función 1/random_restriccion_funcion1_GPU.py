import cupy as cp
import itertools
import random
import time
import csv
import os

print(f"Usando dispositivo: {'GPU (CuPy)' if cp.cuda.is_available() else 'CPU'}")

# --- Función objetivo vectorizada con restricciones
def funcion_objetivo_con_restriccion(x, r=1e5):
    # x es un tensor (N, 3)
    f = 1000 - x[:, 0]**2 - 2*x[:, 1]**2 - x[:, 2]**2 - x[:, 0]*x[:, 1] - x[:, 0]*x[:, 2]
    h1 = x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2 - 25
    h2 = 8*x[:, 0] + 14*x[:, 1] + 7*x[:, 2] - 56
    penalizacion = r * (h1**2 + h2**2)
    return f + penalizacion

# --- Algoritmo PSO usando CuPy
def ejecutar_pso(funcion, limites_inf, limites_sup, dimensiones, parametros, max_iteraciones=1500):
    num_particulas, w, c1, c2 = parametros
    num_particulas = int(num_particulas)

    # Inicializar partículas y velocidades en GPU
    posiciones = cp.zeros((num_particulas, dimensiones))
    for d in range(dimensiones):
        posiciones[:, d] = limites_inf[d] + (limites_sup[d] - limites_inf[d]) * cp.random.rand(num_particulas)

    velocidades = 2 * cp.random.rand(num_particulas, dimensiones) - 1

    mejor_personal = posiciones.copy()
    puntajes_personales = funcion(posiciones)
    mejor_global = mejor_personal[cp.argmin(puntajes_personales)]
    puntaje_global = puntajes_personales.min()

    for _ in range(max_iteraciones):
        r1 = cp.random.rand(num_particulas, dimensiones)
        r2 = cp.random.rand(num_particulas, dimensiones)

        velocidades = (w * velocidades
                       + c1 * r1 * (mejor_personal - posiciones)
                       + c2 * r2 * (mejor_global - posiciones))

        posiciones += velocidades
        for d in range(dimensiones):
            posiciones[:, d] = cp.clip(posiciones[:, d], limites_inf[d], limites_sup[d])

        nuevos_puntajes = funcion(posiciones)

        mejora = nuevos_puntajes < puntajes_personales
        mejora = mejora[:, cp.newaxis]

        mejor_personal = cp.where(mejora, posiciones, mejor_personal)
        puntajes_personales = cp.where(mejora[:, 0], nuevos_puntajes, puntajes_personales)

        if nuevos_puntajes.min() < puntaje_global:
            idx = nuevos_puntajes.argmin()
            mejor_global = posiciones[idx]
            puntaje_global = nuevos_puntajes.min()

    return puntaje_global.item(), cp.asnumpy(mejor_global)

# --- Programa principal
if __name__ == "__main__":
    dimensiones = 3
    limites = [(0, 10)] * dimensiones
    limites_inf = cp.array([lim[0] for lim in limites])
    limites_sup = cp.array([lim[1] for lim in limites])

    # Espacio de búsqueda aleatoria
    espacio_parametros = {
        'num_particulas': [10, 20, 30, 40, 50],
        'w': [0.1, 0.3, 0.5, 0.7, 0.9],
        'c1': [0.5, 1.0, 1.5, 2.0, 2.5],
        'c2': [0.5, 1.0, 1.5, 2.0, 2.5]
    }

    combinaciones = random.sample(
        list(itertools.product(
            espacio_parametros['num_particulas'],
            espacio_parametros['w'],
            espacio_parametros['c1'],
            espacio_parametros['c2']
        )),
        250
    )

    mejor_puntaje = float('inf')
    mejores_parametros = None
    mejor_solucion = None

    print("Iniciando búsqueda aleatoria de hiperparámetros con PSO en GPU...\n")
    inicio = time.time()

    for params in combinaciones:
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
    for i, param in enumerate(mejores_parametros):
        print(f"  Parámetro {i+1}: {param}")
    print("Variables óptimas encontradas:")
    for i, val in enumerate(mejor_solucion):
        print(f"  x{i+1} = {val}")

    # Guardar resultados en CSV
    nombre_csv = "resultados_pso_randomsearch_gpu.csv"
    existe = os.path.exists(nombre_csv)

    with open(nombre_csv, mode='a', newline='') as archivo:
        writer = csv.writer(archivo)
        if not existe:
            writer.writerow(["num_particulas", "w", "c1", "c2", "tiempo", "puntaje", "x1", "x2", "x3"])
        writer.writerow([*mejores_parametros, round(duracion, 4), mejor_puntaje, *mejor_solucion])
    print(f"\nResultado agregado a: {nombre_csv}")
