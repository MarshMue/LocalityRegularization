import numpy as np
from triangulation import customCVX, cvxLP, exactLP
import time
from tqdm import tqdm
from utilities import simplexSample
import matplotlib.pyplot as plt
from example import delaunay_simplex
import datetime
import pickle



if __name__ == "__main__":
    # generate points


    # n = 150

    ns = [2**x for x in range(10, 17)]
    # ns = [512, 512*4, 512*16]
    # ns = [10**x for x in range(4, 7)]

    num_new_points = 1
    n_samples = 50
    total_points = n_samples
    rho = 1e-7

    ds = [2**x for x in range(1, 10)]

    # arrays to save timing data
    solve_timesQP = np.zeros((len(ns), len(ds), total_points))
    solve_timesLP = np.zeros((len(ns), len(ds), total_points))
    solve_timesExactLP = np.zeros((len(ns), len(ds), total_points))
    dSparse_solve_times = np.zeros((len(ns), len(ds), total_points))

    for i, d in enumerate(tqdm(ds)):
        # make n larger than d by some factor
        for n_idx, n in enumerate(tqdm(ns)):
            for _ in range(num_new_points):
                # sample new points
                X = np.random.rand(n, d)

                idxes = np.arange(n)

                # iterate over simplices and generate points to then solve QP to identify vertices and compute runtime
                for simplex_i in range(n_samples):

                    # sample a point in the convex hull to then query the triangle.
                    # We do not care about how well the triangle is identified, only the runtime here
                    weights = simplexSample(n, 1).reshape(1, -1)

                    p = weights @ X


                    array_idx = simplex_i


                    # delaunay sparse comparison
                    Xcopy = X.copy()
                    pcopy = p.copy()
                    start = time.time()
                    dSparse_idx, _ = delaunay_simplex(Xcopy, pcopy, parallel=True)
                    dSparse_solve_times[n_idx, i, array_idx] = time.time() - start

                    # QP solve
                    start = time.time()
                    idx, weights = customCVX(X, p, rho)
                    solve_timesQP[n_idx, i, array_idx] = time.time() - start

                    # LP solve
                    start = time.time()
                    LPidx = cvxLP(X, p)
                    solve_timesLP[n_idx, i, array_idx] = time.time() - start

                    # exact
                    start = time.time()
                    LPidx = exactLP(X, p)
                    solve_timesExactLP[n_idx, i, array_idx] = time.time() - start


    # these plots are used for a quick interpretation of the data, the figures in the paper use the saved data but
    # made more aesthetically pleasing
    fig, ax = plt.subplots(figsize=(8, 6))

    for n_idx, n in enumerate(ns):
        # fit line of best fit
        log_y = np.log(solve_timesQP[n_idx].mean(axis=1))
        log_x = np.log(ds)
        coefficients = np.polyfit(log_x, log_y, 1)
        a = coefficients[0]
        b = np.exp(coefficients[1])

        # plot cvx times
        l, = ax.loglog(ds, solve_timesQP[n_idx].mean(axis=1), label=fr'cvxQP (n={n}), ${b:.3f}x^{{{a:.3f}}}$')
        ax.fill_between(ds, solve_timesQP[n_idx].mean(axis=1) + solve_timesQP[n_idx].std(axis=1), solve_timesQP[n_idx].mean(axis=1) - solve_timesQP[n_idx].std(axis=1), alpha=0.3)
        c = l.get_color()

        # plot dsparse times
        log_y = np.log(dSparse_solve_times[n_idx].mean(axis=1))
        coefficients = np.polyfit(log_x, log_y, 1)
        a = coefficients[0]
        b = np.exp(coefficients[1])

        ax.loglog(ds, dSparse_solve_times[n_idx].mean(axis=1), linestyle='--', color=c, label=fr'delaunaySparse (n={n}), ${b:.3f}x^{{{a:.3f}}}$')
        ax.fill_between(ds, dSparse_solve_times[n_idx].mean(axis=1) + dSparse_solve_times[n_idx].std(axis=1), dSparse_solve_times[n_idx].mean(axis=1) - dSparse_solve_times[n_idx].std(axis=1), color=c, alpha=0.3)

        # LP times
        # fit line of best fit
        log_y = np.log(solve_timesLP[n_idx].mean(axis=1))
        coefficients = np.polyfit(log_x, log_y, 1)
        a = coefficients[0]
        b = np.exp(coefficients[1])

        # plot cvx times
        ax.loglog(ds, solve_timesLP[n_idx].mean(axis=1), label=fr'cvxLP (n={n}), ${b:.3f}x^{{{a:.3f}}}$', color=c, linestyle=":")
        ax.fill_between(ds, solve_timesLP[n_idx].mean(axis=1) + solve_timesLP[n_idx].std(axis=1),
                        solve_timesLP[n_idx].mean(axis=1) - solve_timesLP[n_idx].std(axis=1), color=c, alpha=0.3)

        # Exact LP times
        # fit line of best fit
        log_y = np.log(solve_timesExactLP[n_idx].mean(axis=1))
        coefficients = np.polyfit(log_x, log_y, 1)
        a = coefficients[0]
        b = np.exp(coefficients[1])

        # plot cvx times
        ax.loglog(ds, solve_timesExactLP[n_idx].mean(axis=1), label=fr'ExactLP (n={n}), ${b:.3f}x^{{{a:.3f}}}$', color=c,
                  linestyle="dashdot")
        ax.fill_between(ds, solve_timesExactLP[n_idx].mean(axis=1) + solve_timesExactLP[n_idx].std(axis=1),
                        solve_timesExactLP[n_idx].mean(axis=1) - solve_timesExactLP[n_idx].std(axis=1), color=c, alpha=0.3)

    # timestamp files
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    save_times_dict = {"ns": ns, "ds" : ds, "QP": solve_timesQP, "LP": solve_timesLP, "ExactLP": solve_timesExactLP, "DelaunaySparse" : dSparse_solve_times, "rho": rho}

    with open("data/" + dtstr + "_runtime_data.pkl", 'wb') as f:
        pickle.dump(save_times_dict, f)

    ax.set_xlabel("d")
    ax.set_ylabel("time (s)")

    ax.legend()
    fig.suptitle(f"Runtime vs Dimension")
    fig.tight_layout()


    fig.savefig("images/" + dtstr + f"scaling_all_script.pdf")


    # plt.show()












