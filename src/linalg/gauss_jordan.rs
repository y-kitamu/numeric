use std::mem;

use anyhow::{Context, Result};
use num::ToPrimitive;

use crate::{linalg::LinAlgError, Matrix};

/// Linear equation (Ax = b) solution by Gauss-Jordan elimination.
/// On output, `a` is replaced by its matrix inverse, and
/// `b` is replaced by the corresponding set of solution vectors.
pub fn gauss_jordan<T>(mut a: Matrix<T>, mut b: Matrix<T>) -> Result<(Matrix<T>, Matrix<T>)>
where
    T: Clone + ToPrimitive,
{
    let n = a.rows();
    let m = b.cols();

    let mut indxc = Vec::with_capacity(n);
    let mut indxr = Vec::with_capacity(n);
    let mut ipiv = vec![0; n];

    let mut irow = 0;
    let mut icol = 0;
    for i in 0..n {
        let mut big = 0.0;
        for j in 0..n {
            if ipiv[j] != 1 {
                for k in 0..n {
                    if ipiv[k] == 0 {
                        if a[j][k].to_f32().unwrap().abs() >= big {
                            big = a[j][k].to_f32().unwrap().abs();
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }
        ipiv[icol] += 1;

        if irow != icol {
            a.swap_rows(irow, icol);
            b.swap_rows(irow, icol);
        }
        indxr.push(irow);
        indxc.push(icol);

        if a[icol][icol].to_f32().unwrap().abs() < 1e-7 {
            return Err(LinAlgError::SingularMatrix("gauss_jordan".to_string()))?;
        }
    }
    Ok((a, b))
}
