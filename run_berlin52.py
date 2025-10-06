import time
from math import sqrt
from SD import load_problem, tour_length, gap
from SD import plot_tour
from SOM_TSP import som_tsp

# simple 2-opt implementation with break-on-no-improvement
def two_opt(tour, max_no_improve=5):
    best = tour[:]
    best_len = tour_length(best)
    no_improve = 0
    n = len(best)
    while no_improve < max_no_improve:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_tour = best[:i] + best[i:j][::-1] + best[j:]
                new_len = tour_length(new_tour)
                if new_len < best_len - 1e-9:
                    best = new_tour
                    best_len = new_len
                    improved = True
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    return best


def run():
    cities, opt = load_problem('berlin52.tsp')
    opt_len = tour_length(opt)
    n = len(cities)
    print(f"Loaded berlin52: n={n}, opt_len={opt_len:.2f}")

    n_nodes = int(8 * sqrt(n))
    epochs = 4000

    t0 = time.time()
    tour = som_tsp(cities, n_nodes=n_nodes, epochs=epochs)
    t1 = time.time()
    L = tour_length(tour)
    gap_pct = gap(opt_len, L)
    print(f"SOM: n_nodes={n_nodes}, epochs={epochs}, Longitud={L:.2f}, Gap={gap_pct:.2f}%, Time={t1-t0:.2f}s")

    # apply 2-opt
    t2 = time.time()
    tour_2opt = two_opt(tour)
    t3 = time.time()
    L2 = tour_length(tour_2opt)
    gap2 = gap(opt_len, L2)
    print(f"After 2-opt: Longitud={L2:.2f}, Gap={gap2:.2f}%, Time={t3-t2:.2f}s")

    # Optionally save plots (commented out by default)
    try:
        plot_tour(tour, title=f"berlin52 SOM n_nodes={n_nodes}")
        plot_tour(tour_2opt, title=f"berlin52 SOM+2opt")
    except Exception as e:
        print(f"Plot failed: {e}")

    # Print tour lengths for MD copy
    print('\n---RESULTS FOR MD---')
    print(f'SOM_length: {L:.2f}')
    print(f'SOM_gap: {gap_pct:.4f}%')
    print(f'2opt_length: {L2:.2f}')
    print(f'2opt_gap: {gap2:.4f}%')

if __name__ == "__main__":
    run()
