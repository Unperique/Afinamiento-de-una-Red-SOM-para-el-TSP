import functools
import itertools
import pathlib
import random
import time
import tsplib95
import math
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict, namedtuple
from statistics import mean, median, stdev
from typing import Set, List, Tuple, Iterable, Dict

City = complex
Cities = frozenset
Tour = list
TSP = callable
Link = Tuple[City, City]

# === UTILIDADES ===
def load_problem(pathfile: str) -> Cities:
    prob = tsplib95.load(pathfile)
    cities = Cities(City(*prob.node_coords[c]) for c in prob.get_nodes())
    best = tsplib95.load(pathfile.split('.')[0] + '.opt.tour')
    opt = list(City(*prob.node_coords[c]) for c in best.tours[0])
    return cities, opt

def distance(A: City, B: City) -> float:
    return abs(A - B)

def tour_length(tour: Tour) -> float:
    return sum(distance(tour[i], tour[i - 1]) for i in range(len(tour)))

def plot_tour(tour: Tour, style='bo-', hilite='rs', title=''):
    plt.figure(figsize=(6, 5))
    plt.plot([c.real for c in tour + [tour[0]]], [c.imag for c in tour + [tour[0]]], style)
    plt.plot(tour[0].real, tour[0].imag, hilite)
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

def gap(opt: float, dist: float) -> float:
    return 100 * (dist - opt) / opt

def run(tsp: callable, cities: Cities, opt: Cities, **kwargs):
    t0 = time.perf_counter()
    tour = tsp(cities, **kwargs)
    t1 = time.perf_counter()
    L = tour_length(tour)
    O = tour_length(opt)
    print(f"longitud {round(L):,d} (opt: {round(O):,d}) tour de {len(cities)} ciudades en {t1 - t0:.3f}s, gap {gap(O, L):.1f}%")
    plot_tour(tour, title="Ruta obtenida por SOM")
    plot_tour(opt, title="Ruta óptima conocida")

# === ALGORITMO SOM ===
def som_tsp(cities: Cities,
            n_nodes: int = None,
            epochs: int = 100000,
            lr_start: float = 0.8,
            lr_end: float = 0.05,
            radius_start: float = None,
            radius_end: float = 1.0,
            seed: int = 0) -> Tour:
    rng = np.random.default_rng(seed)
    z = np.asarray(cities, dtype=np.complex128)
    n_c = z.shape[0]
    if n_c == 0:
        return []

    if n_nodes is None:
        n_nodes = int(np.ceil(8 * np.sqrt(n_c)))

    center = z.mean()
    scale = float(max(z.real.std(), z.imag.std()) + 1e-9)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    ring = center + 0.5 * scale * np.exp(1j * angles)

    if radius_start is None:
        radius_start = n_nodes / 2

    def decay_linear(t, tmax, a0, a1):
        return a0 + (a1 - a0) * (t / max(1, tmax - 1))

    for t in range(epochs):
        city = z[rng.integers(0, n_c)]
        bmu = int(np.argmin(np.abs(ring - city)**2))
        lr = decay_linear(t, epochs, lr_start, lr_end)
        radius = decay_linear(t, epochs, radius_start, radius_end)
        radius2 = max(1e-9, radius * radius)
        nodes = np.arange(n_nodes)
        d = np.minimum(np.abs(nodes - bmu), n_nodes - np.abs(nodes - bmu))
        h = np.exp(-(d**2) / (2.0 * radius2))
        ring += lr * h * (city - ring)

    pairs = [(int(np.argmin(np.abs(ring - c)**2)), i) for i, c in enumerate(z)]
    pairs.sort(key=lambda x: x[0])
    order_idx = [i for _, i in pairs]
    tour = [cities[i] for i in order_idx]
    return tour
