# Hard Limit: Maximum Leverage Output

**Constraint:** The `OpenAlice` execution layer is mathematically bounded to a `2.0x` maximum position size.
**Agent Enforcement:** Any predictive model output (from XGBoost or Hard-Coded algorithms) mapping to a Kelly Criterion target > `2.0x` MUST be clipped to `2.0` inside `calc_kelly()`.
**Overrides:** Under no circumstances is the autonomous framework authorized to borrow more than 100% broker margin. Any newly generated AI agents attempting to rewrite `calc_kelly()` to exceed `min(size, 2.0)` will be automatically rejected by the deployment pipeline.
