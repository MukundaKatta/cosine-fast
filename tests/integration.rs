use cosine_fast::{batch_cosine, cosine, cosine_with_norm, norm};

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn parallel_vectors_are_one() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![2.0, 4.0, 6.0];
    assert!(approx(cosine(&a, &b), 1.0));
}

#[test]
fn orthogonal_is_zero() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert!(approx(cosine(&a, &b), 0.0));
}

#[test]
fn antiparallel_is_neg_one() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    assert!(approx(cosine(&a, &b), -1.0));
}

#[test]
fn zero_vector_returns_zero() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 2.0, 3.0];
    assert_eq!(cosine(&a, &b), 0.0);
}

#[test]
fn cosine_with_norm_matches_plain() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let nb = norm(&b);
    assert!(approx(cosine(&a, &b), cosine_with_norm(&a, &b, nb)));
}

#[test]
fn batch_returns_one_score_per_candidate() {
    let q = vec![1.0, 0.0];
    let cands = [vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
    let out = batch_cosine(&q, cands.iter().map(|v| v.as_slice()));
    assert_eq!(out.len(), 3);
    assert!(approx(out[0], 1.0));
    assert!(approx(out[1], 0.0));
    assert!(approx(out[2], -1.0));
}

#[test]
#[should_panic(expected = "vector length mismatch")]
fn length_mismatch_panics() {
    cosine(&[1.0, 2.0], &[1.0]);
}

#[test]
fn norm_is_euclidean_length() {
    assert!(approx(norm(&[3.0, 4.0]), 5.0));
    assert_eq!(norm(&[0.0, 0.0]), 0.0);
}

#[test]
fn cosine_with_zero_norm_returns_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![0.0, 0.0, 0.0];
    assert_eq!(cosine_with_norm(&a, &b, 0.0), 0.0);
}

#[test]
fn batch_zero_query_is_all_zero() {
    let q = vec![0.0, 0.0];
    let cands = [vec![1.0, 0.0], vec![0.0, 1.0]];
    let out = batch_cosine(&q, cands.iter().map(|v| v.as_slice()));
    assert_eq!(out, vec![0.0, 0.0]);
}

#[test]
fn batch_empty_candidates_is_empty() {
    let q = vec![1.0, 2.0];
    let cands: Vec<&[f32]> = Vec::new();
    let out = batch_cosine(&q, cands);
    assert!(out.is_empty());
}
