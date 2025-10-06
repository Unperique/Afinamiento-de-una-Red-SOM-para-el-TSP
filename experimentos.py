import time
import matplotlib.pyplot as plt
from SD import load_problem, tour_length, gap
from SD import plot_tour  # si definiste esta función ahí
from SOM_TSP import som_tsp  # si decides poner el SOM en otro archivo

# === 1. Cargar el problema ===
cities, opt_tour = load_problem("att48.tsp")
opt_length = 33524  # valor conocido del óptimo en TSPLIB

# === 2. Experimento 1 ===
print("Experimento 1: parámetros base")
start = time.time()
tour = som_tsp(cities, n_nodes=200, epochs=4000)
elapsed = time.time() - start
length = tour_length(tour)
print(f"Longitud: {length:.0f}, Gap: {gap(opt_length, length):.2f}%, Tiempo: {elapsed:.2f}s")
plot_tour(tour, title="Resultado Experimento 1")

# === 3. Experimento 2 (variando parámetros) ===
print("\nExperimento 2: más neuronas (300)")
start = time.time()
tour = som_tsp(cities, n_nodes=300, epochs=4000)
elapsed = time.time() - start
length = tour_length(tour)
print(f"Longitud: {length:.0f}, Gap: {gap(opt_length, length):.2f}%, Tiempo: {elapsed:.2f}s")
plot_tour(tour, title="Resultado Experimento 2")
