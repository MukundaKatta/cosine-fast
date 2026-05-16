# cosine-fast

[![crates.io](https://img.shields.io/crates/v/cosine-fast.svg)](https://crates.io/crates/cosine-fast)

Hot-loop cosine similarity for f32 slices. Scalar core that the
compiler auto-vectorizes well; `precompute_norm` for cheap repeated
queries.

```rust
use cosine_fast::{cosine, batch_cosine};
let a = vec![1.0f32, 2.0, 3.0];
let b = vec![2.0, 4.0, 6.0];
assert!((cosine(&a, &b) - 1.0).abs() < 1e-5);
```

Zero deps. MIT or Apache-2.0.

## Repository Health

This repository includes a dependency-free health check for core documentation, metadata, and CI wiring. Run it locally before publishing changes:

```sh
python3 scripts/check_repository_health.py
```

The same check runs in GitHub Actions on pushes and pull requests.
