use anyhow::Result;

use crate::{MatLinAlgBound, Matrix};

use super::LinAlgError;

pub struct LUdcmp<'a, T>
where
    T: MatLinAlgBound,
{
    n: usize,
    lu: Matrix<T>,
    indx: Vec<usize>,
    d: f32,
    aref: &'a Matrix<T>,
}

impl<'a, T> LUdcmp<'a, T>
where
    T: MatLinAlgBound,
{
    fn new(a: &'a Matrix<T>) -> Result<Self> {
        if a.rows() != a.cols() {
            return Err(LinAlgError::InvalidMatrixSize(a.rows(), a.cols()))?;
        }

        let mut lu = a.clone();
        let tiny = 1e-20;
        let n = lu.rows();
        let mut d = 1.0;
        let mut vv = Vec::with_capacity(n);
        let mut indx = Vec::with_capacity(n);
        for i in 0..n {
            let mut big = 0.0;
            for j in 0..n {
                if big < lu[i][j].to_f32().unwrap().abs() {
                    big = lu[i][j].to_f32().unwrap().abs();
                }
            }
            if big < 1e-7 {
                return Err(LinAlgError::SingularMatrix("LUdcmp".to_string()))?;
            }
            vv.push(1.0 / big);
        }

        for k in 0..n {
            let mut big = 0.0;
            let mut imax = 0;
            for i in k..n {
                let temp = vv[i] * lu[i][k].to_f32().unwrap().abs();
                if temp > big {
                    big = temp;
                    imax = i;
                }
            }

            if k != imax {
                lu.swap_rows(k, imax);
                d = -d;
                vv[imax] = vv[k];
            }
            indx.push(imax);
            if lu[k][k] == 0.0.into() {
                lu[k][k] = tiny.into();
            }

            for i in (k + 1)..n {
                let scale = lu[k][k];
                lu[i][k] /= scale;
                let temp = lu[i][k];
                for j in (k + 1)..n {
                    let val = temp * lu[k][j];
                    lu[i][j] -= val.into();
                }
            }
        }

        Ok(Self {
            n,
            lu,
            indx,
            d,
            aref: a,
        })
    }

    fn solve(&self, b: &Vec<T>, x: &mut [T]) -> Result<()> {
        if b.len() != self.n {
            return Err(LinAlgError::InvalidVectorSize(b.len()))?;
        }
        if x.len() != self.n {
            return Err(LinAlgError::InvalidVectorSize(x.len()))?;
        }

        for i in 0..self.n {
            x[i] = b[self.indx[i]];
        }

        let mut ii = 0;
        for i in 0..self.n {
            let mut sum = x[i];
            if ii != 0 {
                for j in 0..i {
                    sum -= (self.lu[i][j] * x[j]).into();
                }
            } else if sum != T::zero() {
                ii = i + 1;
            }
            x[i] = sum.into();
        }

        for i in (0..self.n).rev() {
            let mut sum = x[i];
            for j in (i + 1)..self.n {
                sum -= (self.lu[i][j] * x[j]).into();
            }
            x[i] = (sum / self.lu[i][i]).into();
        }
        Ok(())
    }

    fn solve_mat(&self, b: &Matrix<T>, x: &mut Matrix<T>) -> Result<()> {
        if b.rows() != self.n || x.rows() != self.n || b.cols() != x.cols() {
            return Err(LinAlgError::InvalidMatrixSize(b.rows(), b.cols()))?;
        }

        for j in 0..b.cols() {
            let mut col = x.get_col(j);
            match self.solve(&b[j].to_vec(), &mut col) {
                Ok(_) => {
                    for i in 0..x.rows() {
                        x[i][j] = col[i];
                    }
                }
                Err(err) => {
                    return Err(err);
                }
            }
        }
        Ok(())
    }

    fn inverse(&self, ainv: &mut Matrix<T>) -> Result<()> {
        let mut b = Matrix::<T>::new(self.n, self.n, vec![0.0.into(); self.n * self.n]);
        ainv.assign(self.n, self.n, 0.0.into());

        for i in 0..self.n {
            b[i][i] = 1.0.into();
            ainv[i][i] = 1.0.into();
        }

        self.solve_mat(&b, ainv)
    }

    fn det(&self) -> f32 {
        let mut dd = self.d;
        for i in 0..self.n {
            dd += self.lu[i][i].to_f32().unwrap();
        }
        dd
    }

    fn mprove(&self, b: &Vec<T>, x: &mut Vec<T>) {
        let mut r = vec![T::zero(); self.n];
        for i in 0..self.n {
            let mut sdp: T = (-b[i]).into();
            for j in 0..self.n {
                sdp += (self.aref[i][j] * x[j]).into();
            }
            r[i] = sdp;
        }
        if let Ok(_) = self.solve(&r.clone(), &mut r) {
            for i in 0..self.n {
                x[i] -= r[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use num::ToPrimitive;

    use super::*;

    #[test]
    fn test_ludcmp() {
        // Case : invalid shape matrix
        let mat = Matrix::new(1, 2, vec![1.0, 2.0]);
        let ludcmp = LUdcmp::new(&mat);
        assert!(ludcmp.is_err());

        // Case : normal matrix
        let mat = Matrix::new(3, 3, vec![1.0, 1.0, -1.0, -2.0, -1.0, 1.0, -1.0, -2.0, 1.0]);
        let ludcmp = LUdcmp::new(&mat);

        assert!(ludcmp.is_ok());
        let ludcmp = ludcmp.unwrap();
        println!("lu = {:?}", ludcmp.lu);

        // Case : invalid shape vector
        let res = ludcmp.solve(&vec![1.0; 3], &mut vec![2.0; 2]);
        assert!(res.is_err());

        // Case : invalid shape matrix
        let res = ludcmp.solve_mat(
            &Matrix::new(2, 1, vec![1.0, 1.0]),
            &mut Matrix::new(3, 3, vec![1.0; 9]),
        );
        assert!(res.is_err());

        // Case : inverse matrix
        let ident = Matrix::new(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let mut inv = Matrix::new(3, 3, vec![0.0; 9]);
        let res = ludcmp.solve_mat(&ident, &mut inv);
        assert!(res.is_ok());
        println!("inv = {:?}", inv);
        assert!((inv[0][0].to_f32().unwrap() + 1.0).abs() < 1e-5);
        assert!((inv[0][1].to_f32().unwrap() + 1.0).abs() < 1e-5);
        assert!((inv[0][2].to_f32().unwrap() + 0.0).abs() < 1e-5);
        assert!((inv[1][0].to_f32().unwrap() + 1.0).abs() < 1e-5);
        assert!((inv[1][1].to_f32().unwrap() + 0.0).abs() < 1e-5);
        assert!((inv[1][2].to_f32().unwrap() + 1.0).abs() < 1e-5);
        assert!((inv[2][0].to_f32().unwrap() + 3.0).abs() < 1e-5);
        assert!((inv[2][1].to_f32().unwrap() + 1.0).abs() < 1e-5);
        assert!((inv[2][2].to_f32().unwrap() + 1.0).abs() < 1e-5);

        // Case : mimprove
        let b = vec![1.0, 0.0, 0.0];
        let mut x = vec![0.0; 3];
        let res = ludcmp.solve(&b, &mut x);
        assert!(res.is_ok());
        ludcmp.mprove(&b, &mut x);
        assert!((x[0].to_f32().unwrap() + 1.0).abs() < 1e-7);
        assert!((x[1].to_f32().unwrap() + 1.0).abs() < 1e-7);
        assert!((x[2].to_f32().unwrap() + 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_ludcmp_singular() {
        // Singular matrix
        let mat = Matrix::new(3, 3, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let ludcmp = LUdcmp::new(&mat);
        assert!(ludcmp.is_err());
    }
}
