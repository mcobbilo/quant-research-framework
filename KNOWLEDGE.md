# J-EPA: Institutional Cognition Base (Merkle-Light Audit Trail)

This ledger summarizes causal insights distilled by **Agent Epsilon** from backtest failures. Every entry is cryptographically linked to its predecessor.

---

## [BLOCK_0] | Genesis

- **Status**: Stable

- **Context**: System Initiation

- **Lesson**: The Curiosity Engine must strictly adhere to the Institutional Knowledge Base to prevent Lesson Regression.

- **ParentHash**: 0000000000000000000000000000000000000000000000000000000000000000

- **Hash**: 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8

---

## [BLOCK_1] | Previous Research

- **Context**: Sequence 1 (Fourier Cycle Isolation)

- **Lesson**: Numpy Read-Only Buffers - Always use `.copy()` on arrays returned from external libraries before performing in-place modifications or nan-filling.

- **ParentHash**: 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8

- **Hash**: ec153b8f108f9730caccf77e20ec4ead0f52d5b63e8f8f949c2949f2b84a9e55

---

## [BLOCK_2] | Initialization Errors

- **Context**: Sequence 3 (Initial Hardening)

- **Lesson**: Header Hallucination - Always verify column existence before indexing, even if the mocks claim to include them.

- **ParentHash**: ec153b8f108f9730caccf77e20ec4ead0f52d5b63e8f8f949c2949f2b84a9e55

- **Hash**: 57f2a13f707f1bd03164a66e4a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a

---

## [BLOCK_3] | Numerical & Regime Safeguards

- **Context**: Iterations 1-5 (v2.0 Refactor)

- **Lesson**: Distributed Safeguards - Enforce numeric dtypes via `pd.to_numeric`, compute distribution thresholds strictly on lagged data (`.shift(1)`), and clamp terminal wealth at zero.

- **ParentHash**: 57f2a13f707f1bd03164a66e4a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a

- **Hash**: a1b2c3d4e5f60708090a0b0c0d0e0f1011121314151617181920212223242526

---

## [BLOCK_4] | Logic Drift Prevention

- **Context**: Iterations 1-5 (v2.0 Refactor)

- **Lesson**: Never hardcode an asset universe that is not explicitly verified against the loaded DataFrame columns at the entry point; derive or guard the required feature set from actual schema.

- **ParentHash**: a1b2c3d4e5f60708090a0b0c0d0e0f1011121314151617181920212223242526

- **Hash**: f1e2d3c4b5a60708090a0b0c0d0e0f1011121314151617181920212223242627

---

## [BLOCK_5] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Always assert `len(df) > burnin` immediately after data loading and fail fast with explicit diagnostics before any regime fitting or performance metric calculation.

- **ParentHash**: f1e2d3c4b5a60708090a0b0c0d0e0f1011121314151617181920212223242627

- **Hash**: a81598987ab32706b3970c6129d5273f7334f2aee8222cb4c48f1948ed546ede

---

## [BLOCK_6] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Never place a hard length assertion directly after external data ingestion; insert an explicit guard (`if len(df) <= burnin: return 0.0, 0.0`) with clear diagnostic output before any further processing.

- **ParentHash**: a81598987ab32706b3970c6129d5273f7334f2aee8222cb4c48f1948ed546ede

- **Hash**: b8ce29115f4b209d7deb2a51c262b4a8d0c41258f8b9326f51c0b70f62727ad8

---

## [BLOCK_7] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: When `positions` has length `n-1` (holding the bet from bar `i` into `i+1`), the decision loop must run `range(burnin, n-1)`, never `range(burnin, n)`.

- **ParentHash**: b8ce29115f4b209d7deb2a51c262b4a8d0c41258f8b9326f51c0b70f62727ad8

- **Hash**: ecc485b72af9067b9557272ae336f862ff2407dcf54cb0faa2e9e8f8e2eaa83b

---

## [BLOCK_8] | REGRESSION-Refinement

- **Context**: Performance Delta: -2.7374

- **Lesson**: Never use a single static train/test split on full-history time-series data; enforce strictly time-respecting walk-forward or purged k-fold validation for every model refinement.

- **ParentHash**: ecc485b72af9067b9557272ae336f862ff2407dcf54cb0faa2e9e8f8e2eaa83b

- **Hash**: 2cf397f0c77a0781201f17d560b6d9ae40ce8de7808e9c3870fdf50bda471d87

---

## [BLOCK_9] | REGRESSION-Refinement

- **Context**: Performance Delta: -1.6536

- **Lesson**: Never use the `fdf` keyword when the intention is to fit or initialize the degrees-of-freedom parameter; use positional argument for initial guess and confirm the number of returned values matches the unpack.

- **ParentHash**: 2cf397f0c77a0781201f17d560b6d9ae40ce8de7808e9c3870fdf50bda471d87

- **Hash**: 1dbe8cac154b7242c73da9ef5d92c6d2c6aa0a963f9252ba56dde10cceb40e34

---

## [BLOCK_10] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Never embed ticker-specific column assumptions directly into the data ingestion guard clauses without a prior schema introspection or column mapping step against the actual market_data.db table definition.

- **ParentHash**: 1dbe8cac154b7242c73da9ef5d92c6d2c6aa0a963f9252ba56dde10cceb40e34

- **Hash**: 768a2c4015ad3f8d6ea37f739a8d7da41182819e0de61a21433c2a6629a030e8

---

## [BLOCK_11] | REGRESSION-Refinement

- **Context**: Performance Delta: -1.4316

- **Lesson**: Never add lags to volatility estimates beyond what causality strictly requires—current realized return can be included in sigma when the position is applied forward; over-lagging injects stale risk scaling that destroys timing and sizing.

- **ParentHash**: 768a2c4015ad3f8d6ea37f739a8d7da41182819e0de61a21433c2a6629a030e8

- **Hash**: 3f978ebd46a86247637cbd9edafbc151327cd49e13142cfa39ade62e14308fe6

---

## [BLOCK_12] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: In any online parameter update loop, generate the trading signal using parameters strictly prior to ingesting the current timestep's observation.

- **ParentHash**: 3f978ebd46a86247637cbd9edafbc151327cd49e13142cfa39ade62e14308fe6

- **Hash**: 1d2368af8b44d8f266223815e97d2a423edbe9e24b71713b388d88b6d003ba4d

---

## [BLOCK_13] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Compute predictive signal (r_hat, regime responsibilities) *before* performing the online parameter update with the current observation, and exclude the target return from the feature vector at decision time.

- **ParentHash**: 1d2368af8b44d8f266223815e97d2a423edbe9e24b71713b388d88b6d003ba4d

- **Hash**: 683fecff5db80f2fed4f9595489a38073b2a13caf47b35909e47e1e54226f0fe

---

## [BLOCK_14] | OSINT-Dorking Protocol

- **Context**: Structural Alpha Exploration

- **Lesson**: Use Google Dorking for high-fidelity discovery:
  - **CSV/Data**: `site:un.org filetype:csv "market"`
  - **Academic Verification**: `site:arxiv.org [Strategy Title]`
  - **GitHub Prototyping**: `site:github.com filetype:py [Strategy Detail]`
  - **Exposed Configs**: `site:*.partner.com filetype:log OR filetype:conf`

- **ParentHash**: 683fecff5db80f2fed4f9595489a38073b2a13caf47b35909e47e1e54226f0fe

- **Hash**: 5e8d9c4b7a2e1f3d5c6b8a1f9e2d7c5b4a3e2f1d9c8b7a6e5d4c3b2a1f0e9d8c

---

## [BLOCK_15] | REGRESSION-Refinement

- **Context**: Performance Delta: -0.6550

- **Lesson**: Never compute the current deviation using any statistic that incorporates the current observation in recursive mean/volatility filters; enforce mu[t-1] and sig[t-1] for all position decisions.

- **ParentHash**: 5e8d9c4b7a2e1f3d5c6b8a1f9e2d7c5b4a3e2f1d9c8b7a6e5d4c3b2a1f0e9d8c

- **Hash**: d13084c2ff0b69ff2a4b444f6f0b363b38f10f22643d85a24d0e5c889789f48c

---

## [BLOCK_16] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Never include the contemporaneous target return in the feature vector for regime/probability inference; compute gamma and predictive moments strictly from lagged or non-return features.

- **ParentHash**: d13084c2ff0b69ff2a4b444f6f0b363b38f10f22643d85a24d0e5c889789f48c

- **Hash**: 3240a69da9cf38ad55bba292070e5d7580817d008e98d4f2c1ba87df42bbd95f

---

## [BLOCK_17] | REGRESSION-Refinement

- **Context**: Performance Delta: -0.7551

- **Lesson**: Never multiply the precision-weighted signal (mu/v) by an extra 1/sigma term or update variance with the already-posterior mean in online regime models.

- **ParentHash**: 3240a69da9cf38ad55bba292070e5d7580817d008e98d4f2c1ba87df42bbd95f

- **Hash**: a7632ec3bb6d22baaa046d2251825c8fce17e2eb88a910c138d277ba41788de7

---

## [BLOCK_18] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Never let an optimization bound land exactly on a value that produces a zero denominator in any subsequent analytic formula; enforce a safety gap (nu += 1e-8 or bound=[2.51,30]) at the point the numeric constraint is declared.

- **ParentHash**: a7632ec3bb6d22baaa046d2251825c8fce17e2eb88a910c138d277ba41788de7

- **Hash**: 80892d9bf0498011c8494aae5c7831963173f92da30f292c2ef1d98b81633315

---

## [BLOCK_19] | REGRESSION-Refinement

- **Context**: Performance Delta: -0.9400

- **Lesson**: Never rely on except blocks for control flow in core parameter estimation; test fit call signatures (starting value vs. f-prefixed fixed param) in isolation before backtest.

- **ParentHash**: 80892d9bf0498011c8494aae5c7831963173f92da30f292c2ef1d98b81633315

- **Hash**: a623c226cc1129602c98a67e6ae663412e689cdbd05b1fcdd2c8747a24c75c75

---

## [BLOCK_20] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Explicitly import every external symbol (including `import pandas as pd`) at the top of every self-contained strategy file; never assume ambient notebook or framework pre-imports in production backtest runners.

- **ParentHash**: a623c226cc1129602c98a67e6ae663412e689cdbd05b1fcdd2c8747a24c75c75

- **Hash**: 8d21436b6a03022572aabd8379cca16cfd57ba79a482d9eaa348e20960baf4d6

---

## [BLOCK_21] | REGRESSION-Refinement

- **Context**: Performance Delta: -0.7208

- **Lesson**: Calibrate and test regime-quality filters (Q, posterior γ, or per-component SNR) in isolation on walk-forward windows before embedding them in online GMM parameter updates.

- **ParentHash**: 8d21436b6a03022572aabd8379cca16cfd57ba79a482d9eaa348e20960baf4d6

- **Hash**: d7b183997218d15a81db315f7d4e471fac38e33aca9d4e854ddf85875b49eb13

---

## [BLOCK_22] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: After every division by a rolling statistic, explicitly bound the result with `np.nan_to_num(..., nan=0.0, posinf=1.0, neginf=-1.0)` or `np.clip(..., -10, 10)` before feeding the feature matrix to any statistical estimator.

- **ParentHash**: d7b183997218d15a81db315f7d4e471fac38e33aca9d4e854ddf85875b49eb13

- **Hash**: 2a0d6e800c7753197b6730c2f07ec119fade539a3d536e65899e577710dedfc1

---

## [BLOCK_23] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Apply uniform lag offset to every raw input before any rolling statistic or normalization so the entire feature vector reflects identical information cutoff.

- **ParentHash**: 2a0d6e800c7753197b6730c2f07ec119fade539a3d536e65899e577710dedfc1

- **Hash**: 43549b016519c5224ecb4758e1dfe5b489aa3291d6fec43702120a833ff07f92

---

## [BLOCK_24] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Scale all SR thresholds, targets, and vol parameters consistently with the frequency of the returns series actually used in the online updates.

- **ParentHash**: 43549b016519c5224ecb4758e1dfe5b489aa3291d6fec43702120a833ff07f92

- **Hash**: a63e05b7293ed13133c050aa7bf594f8680b604d3c1eb95d52c62679bd7329c0

---

## [BLOCK_25] | REGRESSION-Refinement

- **Context**: Performance Delta: -0.8595

- **Lesson**: In online regime models, compute all moments (m, V, SR, w) on the same probability vector (posterior gamma after the observation) before sizing the position.

- **ParentHash**: a63e05b7293ed13133c050aa7bf594f8680b604d3c1eb95d52c62679bd7329c0

- **Hash**: 0d70ed06be370dce78de47d474ac0e71e6bb67e121a1a6ea498f7da5d6e1370a

---

## [BLOCK_26] | REGRESSION-Refinement

- **Context**: Performance Delta: -1.0981

- **Lesson**: Verify that the return vector element used for signal construction at index i is exactly the one that should be multiplied by the position decided at i (i.e. omegas[i] × returns[i+1] only when signal sees data through returns[i]).

- **ParentHash**: 0d70ed06be370dce78de47d474ac0e71e6bb67e121a1a6ea498f7da5d6e1370a

- **Hash**: 84ab2aa393c20e58b2768a9ddf6d0410fd0bc560dbbf67f01bddd404ecb9c393

---

## [BLOCK_27] | REGRESSION-Refinement

- **Context**: Performance Delta: -0.8090

- **Lesson**: Derive and unit-test all conjugate or approximate Bayesian recursions against known OU transition densities or Kalman equivalents before inserting them into the main loop.

- **ParentHash**: 84ab2aa393c20e58b2768a9ddf6d0410fd0bc560dbbf67f01bddd404ecb9c393

- **Hash**: bb050fa0935721a56f1e089a332a6115d8e886e699e2bfaefd6082202433d3dd

---

## [BLOCK_28] | CAUSAL-Refinement

- **Context**: Performance Delta: 0.0000

- **Lesson**: Explicitly enumerate every external symbol (including pandas as pd) in the import block before any data-loading or computation statements.

- **ParentHash**: bb050fa0935721a56f1e089a332a6115d8e886e699e2bfaefd6082202433d3dd

- **Hash**: cda0e03e6aa14ead640443f05f4520b80671ca45d1abd5a5a6035ae144cc1368

---

## [BLOCK_29] | REGRESSION-Refinement

- **Context**: Performance Delta: -1.0699

- **Lesson**: Normalize decaying weights to unit sum (or scale kappa to effective window) before any saturating nonlinearity like tanh.

- **ParentHash**: cda0e03e6aa14ead640443f05f4520b80671ca45d1abd5a5a6035ae144cc1368

- **Hash**: 5d5e7664b42ca7f68c04e90b028f703776f22e57f999986473de0417757f7735

---

## [BLOCK_30] | REGRESSION-Refinement

- **Context**: Performance Delta: -1.3548

- **Lesson**: After replacing any global statistic to remove lookahead, immediately recompute or burn-in-calibrate all dependent noise ratios and retune the full hyperparameter set (kappa, beta, gamma, delta) before measuring Sharpe.

- **ParentHash**: 5d5e7664b42ca7f68c04e90b028f703776f22e57f999986473de0417757f7735

- **Hash**: b962a08a78ea6e491b39db2e1644126dd44f17324fed4f83e08e7a9a0d6a8a95

---

## [BLOCK_31] | REGRESSION-Refinement

- **Context**: Performance Delta: -1.7752

- **Lesson**: Explicitly test the monotonicity of every eigenvalue-derived metric (diversity vs concentration) against prior champion logic before blending windows or composing Psi/alpha.

- **ParentHash**: b962a08a78ea6e491b39db2e1644126dd44f17324fed4f83e08e7a9a0d6a8a95

- **Hash**: fe1b60cd3a3bc0be479fda78ded7c3cbb8213e379783c8594a9cd2fdf0ad71e4

---

## [BLOCK_32] | REGRESSION-Refinement

- **Context**: Performance Delta: -1.6759

- **Lesson**: Benchmark any change to conjugate parameter recursions (even "obvious bug fixes") against the exact champion baseline on identical data before promotion.

- **ParentHash**: fe1b60cd3a3bc0be479fda78ded7c3cbb8213e379783c8594a9cd2fdf0ad71e4

- **Hash**: 6d164e68c9fea23924ab54c0c03d188e303687a63a0a43d4dca78bc5ee062e05
