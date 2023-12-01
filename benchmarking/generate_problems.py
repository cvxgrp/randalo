import numpy as np

min_n_or_ss = [100, 1_000, 10_000, 100_000]
ratios_for_n_over_s = [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2]
distributions = ["gauss_iid", "t5_iid", "varying_norm", "equicorrelated", "sparse"]
sparsity_solution = [True, False]


for sparse in sparsity_solution:
    for min_n_or_s in min_n_or_ss:
        for ratio in ratios_for_n_over_s:
            if ratio <= 1:
                n = min_n_or_s
                s = int(n / ratio)
                p = 5 * s if sparse else s
            else:
                s = min_n_or_s
                n = ratio * s
                p = 5 * s if sparse else s
            beta = np.zeros(p)
            beta[:s] = 1 / np.sqrt(s)
            for i, distribution in enumerate(distributions):
                rng = np.random.default_rng(
                    seed=int(str(p) + str(n) + str(s) + str(i) + "1" if sparse else "0")
                )
                match distribution:
                    case "gauss_iid":
                        X = rng.normal(size=(n, p))
                    case "t5_iid":
                        X = rng.standard_t(5, size=(n, p))
                    case "varying_norm":
                        X = ...
                        continue
                    case "equicorrelated":
                        x0 = rng.normal(size=n)
                        a = np.sqrt(0.5)
                        b = np.sqrt(1 - a**2)
                        X = a * rng.normal(size=(n, p)) + b * x0[:, None]
                    case "sparse":
                        X = ...
                        continue

                y = X @ beta + rng.normal(size=n)
                np.savez(
                    f'{min_n_or_s=}-{ratio=}-{distribution}-{"sparse_beta" if sparse else "dense_beta"}',
                    X=X,
                    beta=beta,
                    y=y,
                    n=n,
                    s=s,
                    p=p,
                )
