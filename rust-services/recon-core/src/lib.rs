use ndarray::{Array1, Array2, Axis};

/// Perform one MART iteration over all rays.
///
/// projections:  length M (measured y)
/// system_matrix: shape (M, N) (A)
/// volume:      length N (x)
/// relaxation:  relaxation parameter (lambda)
pub fn mart_step(
    projections: &Array1<f32>,
    system_matrix: &Array2<f32>,
    volume: &mut Array1<f32>,
    relaxation: f32,
) {
    let (m, n) = system_matrix.dim();
    assert_eq!(projections.len(), m);
    assert_eq!(volume.len(), n);

    for i in 0..m {
        let row = system_matrix.index_axis(Axis(0), i); // A_i*

        // estimated projection: y_hat_i = sum_j A_ij * x_j
        let mut y_hat = 0.0f32;
        for j in 0..n {
            y_hat += row[j] * volume[j];
        }

        if y_hat <= 0.0 {
            // avoid division by zero / nonsense updates
            continue;
        }

        let ratio = projections[i] / y_hat;
        let factor = ratio.powf(relaxation);

        for j in 0..n {
            let a_ij = row[j];
            if a_ij > 0.0 {
                volume[j] *= factor;
            }
        }
    }
}

/// Simple MART reconstruction loop.
///
/// - projections: length M
/// - system_matrix: shape (M, N)
/// - n_iters: number of MART passes over all rays
/// - relaxation: relaxation parameter
///
/// Returns reconstructed volume (length N).
pub fn mart_reconstruct(
    projections: &Array1<f32>,
    system_matrix: &Array2<f32>,
    n_iters: usize,
    relaxation: f32,
) -> Array1<f32> {
    let n = system_matrix.dim().1;
    let mut volume = Array1::<f32>::from_elem(n, 1.0); // uniform initial guess

    for _ in 0..n_iters {
        mart_step(projections, system_matrix, &mut volume, relaxation);
    }

    volume
}
