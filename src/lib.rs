//! # cosine-fast
//!
//! Hot-loop cosine similarity for f32 slices. Scalar core that the
//! compiler auto-vectorizes well on AArch64 NEON and x86 AVX2; an
//! optional `precompute_norm` lets you skip the per-call sqrt when the
//! same query is compared against many candidates.
//!
//! ## Example
//!
//! ```
//! use cosine_fast::{cosine, batch_cosine};
//! let a = vec![1.0f32, 0.0, 0.0];
//! let b = vec![0.0f32, 1.0, 0.0];
//! assert!((cosine(&a, &b) - 0.0).abs() < 1e-6);
//!
//! let q = vec![1.0f32, 2.0, 3.0];
//! let cands = vec![
//!     vec![1.0, 2.0, 3.0], // self
//!     vec![0.0, 0.0, 1.0],
//! ];
//! let out = batch_cosine(&q, cands.iter().map(|v| v.as_slice()));
//! assert!((out[0] - 1.0).abs() < 1e-6);
//! ```

#![deny(missing_docs)]

/// Cosine similarity between two equal-length f32 slices.
///
/// Returns 0.0 when either input has zero norm.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        let ai = a[i];
        let bi = b[i];
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute the L2 norm of `v`. Useful as `precompute_norm` for hot-path
/// queries.
pub fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity when you already know one side's norm.
/// `b_norm` should equal `norm(b)`.
pub fn cosine_with_norm(a: &[f32], b: &[f32], b_norm: f32) -> f32 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    for i in 0..a.len() {
        let ai = a[i];
        let bi = b[i];
        dot += ai * bi;
        na += ai * ai;
    }
    let denom = na.sqrt() * b_norm;
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute cosine similarity between `q` and every candidate.
pub fn batch_cosine<'a, I>(q: &[f32], candidates: I) -> Vec<f32>
where
    I: IntoIterator<Item = &'a [f32]>,
{
    let q_norm = norm(q);
    if q_norm == 0.0 {
        return candidates.into_iter().map(|_| 0.0).collect();
    }
    let mut out = Vec::new();
    for c in candidates {
        assert_eq!(c.len(), q.len(), "vector length mismatch");
        let mut dot = 0.0_f32;
        let mut nc = 0.0_f32;
        for i in 0..q.len() {
            dot += q[i] * c[i];
            nc += c[i] * c[i];
        }
        let denom = q_norm * nc.sqrt();
        out.push(if denom == 0.0 { 0.0 } else { dot / denom });
    }
    out
}
