import np

min_n_or_ps = [100, 1_000, 10_000, 100_000]
ratios_for_n_over_s = [5, 4, 3, 2, 1, 0.8, 0.6, 0.4, 0.2]
distributions = ["gauss_iid", "t5_iid", "varying_norm", "equicorrelated", "sparse"]
sparsity_solution = [True, False]


for sparse in sparsity_solution:
    for min_n_or_p in min_n_or_ps:
        for ratio in ratios_for_n_over_s:
            if ratio <= 1:
                n = min_n_or_p
                s = ratio * n
                p = 5 * s if sparse else s
            else:
                p = min_n_or_p
                n = ratio * p
                s = p // 5 if sparse else p
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
                        X = ...
                        continue
                    case "sparse":
                        X = ...
                        continue

                y = X @ beta + rng.normal(size=n)
                np.savez(
                    f'{min_n_or_p=}-{ratio=}-{distribution}-{"sparse_beta" if sparse else "dense_beta"}',
                    X=X,
                    beta=beta,
                    y=y,
                    n=n,
                    s=s,
                    p=p,
                )
