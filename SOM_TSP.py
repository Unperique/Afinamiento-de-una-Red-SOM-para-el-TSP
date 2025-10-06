# SOM_TSP.py
import numpy as np
import random
from math import exp

def som_tsp(cities, n_nodes=200, epochs=4000, learning_rate=0.8, radius=None):
    """
    Implementación básica del Self-Organizing Map (SOM) para el TSP.

    cities: lista de coordenadas complejas (x + yj)
    n_nodes: cantidad de neuronas en el anillo SOM
    epochs: número de iteraciones de entrenamiento
    learning_rate: tasa de aprendizaje inicial
    radius: radio inicial de vecindad (por defecto n_nodes / 10)
    """
    if radius is None:
        radius = n_nodes / 10

    # Aceptar varios tipos de colección para `cities` (lista, set, frozenset, numpy array)
    # Convertimos a lista para preservar indexación por entero y mantener el orden
    # si viene como iterable sin orden (como set), la conversión a lista será en
    # el orden iterado por Python. Para una TSP mejor definida, preferible usar
    # una lista ordenada desde `load_problem`.
    if not isinstance(cities, (list, tuple, np.ndarray)):
        try:
            cities = list(cities)
        except TypeError:
            raise TypeError("`cities` debe ser un iterable de coordenadas (x + yj)")

    # Convertir las ciudades a un arreglo de coordenadas
    city_coords = np.array([[c.real, c.imag] for c in cities])
    n_cities = len(city_coords)

    # Inicializar nodos SOM en un círculo unitario
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    nodes = np.array([[np.cos(a), np.sin(a)] for a in angles])

    # Entrenamiento SOM
    for epoch in range(epochs):
        # Seleccionar una ciudad aleatoria
        city = city_coords[random.randint(0, n_cities - 1)]

        # Calcular distancias a todas las neuronas
        distances = np.linalg.norm(nodes - city, axis=1)
        winner_idx = np.argmin(distances)

        # Calcular tasa de aprendizaje y radio que decaen con el tiempo
        lr = learning_rate * exp(-epoch / epochs)
        rad = radius * exp(-epoch / (epochs / np.log(radius)))

        # Actualizar posiciones de las neuronas vecinas
        for i in range(n_nodes):
            # Distancia circular entre índices (porque es un anillo)
            dist_to_winner = min(abs(i - winner_idx), n_nodes - abs(i - winner_idx))
            if dist_to_winner < rad:
                influence = exp(-dist_to_winner ** 2 / (2 * (rad ** 2)))
                nodes[i] += influence * lr * (city - nodes[i])

    # Asignar cada ciudad a su neurona más cercana
    city_to_node = {}
    for i, city in enumerate(city_coords):
        distances = np.linalg.norm(nodes - city, axis=1)
        nearest_node = np.argmin(distances)
        city_to_node[i] = nearest_node

    # Ordenar las ciudades según la secuencia de neuronas
    sorted_cities = [cities[i] for i in np.argsort([city_to_node[i] for i in range(n_cities)])]

    return sorted_cities
