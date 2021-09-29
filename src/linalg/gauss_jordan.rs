use anyhow::Result;

use crate::{linalg::LinAlgError, MatLinAlgBound, Matrix};

/// Linear equation (Ax = b) solution by Gauss-Jordan elimination.
/// On output, `a` is replaced by its matrix inverse, and
/// `b` is replaced by the corresponding set of solution vectors.
pub fn gauss_jordan<T>(mut a: Matrix<T>, mut b: Matrix<T>) -> Result<(Matrix<T>, Matrix<T>)>
where
    T: MatLinAlgBound,
{
    let n = a.rows();
    let m = b.cols();

    let mut indxc = Vec::with_capacity(n);
    let mut indxr = Vec::with_capacity(n);
    let mut ipiv = vec![0; n];

    let mut irow = 0;
    let mut icol = 0;
    for _ in 0..n {
        let mut big: T = 0.0.into();
        for j in 0..n {
            if ipiv[j] != 1 {
                for k in 0..n {
                    if ipiv[k] == 0 {
                        if a[j][k] >= big {
                            big = a[j][k];
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

        let pivinv = 1.0 / a[icol][icol].to_f32().unwrap();
        for l in 0..n {
            let val = a[icol][l].to_f32().unwrap() * pivinv;
            a[icol][l] = val.into();
        }
        for l in 0..m {
            let val = b[icol][l].to_f32().unwrap() * pivinv;
            b[icol][l] *= val.into();
        }
        for l in 0..n {
            if l != icol {
                let scale: f32 = a[l][icol].to_f32().unwrap();
                a[l][icol] = 0f32.into();
                for k in 0..n {
                    let val = a[l][k].to_f32().unwrap() - scale * a[icol][k].to_f32().unwrap();
                    a[l][k] = val.into();
                }
                for k in 0..m {
                    let val = b[l][k].to_f32().unwrap() - scale * b[icol][k].to_f32().unwrap();
                    b[l][k] = val.into();
                }
            }
        }
    }

    for i in n - 1..=0 {
        if indxr[i] != indxc[i] {
            a.swap_rows(indxr[i], indxc[i]);
            b.swap_rows(indxr[i], indxc[i]);
        }
    }

    Ok((a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_jordan() {
        let rows = 2;
        let cols = 2;
        #[rustfmt::skip]
        let da: Vec<f32> = vec![
            1.0, 0.0,
            0.0, 1.0
        ];
        let db: Vec<f32> = vec![4.0, -3.0];
        let a = Matrix::new(rows, cols, da);
        let b = Matrix::new(rows, 1, db);
        let (a, b) = gauss_jordan(a, b).unwrap();
        assert_eq!(a.rows(), rows);
        assert_eq!(a.cols(), cols);
        println!("{:?}", a);
        println!("{:?}", b);
        assert!(a[0][0].abs() < 1e-5, "a[0][0] = {}", a[0][0]);
        assert!(a[1][1].abs() < 1e-5);
        assert!((a[1][0] - 0.5).abs() < 1e-5);
        assert!((a[0][1] - 1.0 / 3.0).abs() < 1e-5);
        assert!((b[0][0] + 1.0).abs() < 1e-5);
        assert!((b[1][0] - 2.0).abs() < 1e-5);
    }
}
