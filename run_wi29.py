import time
from math import sqrt
from SD import load_problem, tour_length, gap
from SD import plot_tour
from SOM_TSP import som_tsp

# Reuse simple two_opt implementation
from run_berlin52 import two_opt


def run():
    cities, opt = load_problem('wi29.tsp')
    opt_len = tour_length(opt)
    n = len(cities)
    print(f"Loaded wi29: n={n}, opt_len={opt_len:.2f}")

    n_nodes = int(8 * sqrt(n))
    epochs = 4000

    t0 = time.time()
    tour = som_tsp(cities, n_nodes=n_nodes, epochs=epochs)
    t1 = time.time()
    L = tour_length(tour)
    gap_pct = gap(opt_len, L)
    print(f"SOM: n_nodes={n_nodes}, epochs={epochs}, Longitud={L:.2f}, Gap={gap_pct:.2f}%, Time={t1-t0:.2f}s")

    t2 = time.time()
    tour_2opt = two_opt(tour)
    t3 = time.time()
    L2 = tour_length(tour_2opt)
    gap2 = gap(opt_len, L2)
    print(f"After 2-opt: Longitud={L2:.2f}, Gap={gap2:.2f}%, Time={t3-t2:.2f}s")

    try:
        plot_tour(tour, title=f"wi29 SOM n_nodes={n_nodes}")
        plot_tour(tour_2opt, title=f"wi29 SOM+2opt")
    except Exception as e:
        print(f"Plot failed: {e}")

    print('\n---RESULTS FOR MD---')
    print(f'SOM_length: {L:.2f}')
    print(f'SOM_gap: {gap_pct:.4f}%')
    print(f'2opt_length: {L2:.2f}')
    print(f'2opt_gap: {gap2:.4f}%')

if __name__ == "__main__":
    run()
