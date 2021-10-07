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
        let mut lu = a.clone();
        let tiny = 1e-20;
        let n = lu.rows();
        let mut d = 1.0;
        let mut vv = Vec::with_capacity(n);
        let mut indx = Vec::with_capacity(n);
        for i in 0..n {
            let mut big = 0.0;
            for j in 0..n {
                if big < lu[i][j].to_f32().unwrap() {
                    big = lu[i][j].to_f32().unwrap();
                }
            }
            if big < 1e-7 {
                return Err(LinAlgError::SingularMatrix("LUdcmp".to_string()))?;
            }
            vv.push(big);
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
                lu[i][k] /= lu[k][k];
                let temp = lu[i][k];
                for j in (k + 1)..n {
                    lu[i][j] -= (temp * lu[k][j]).into();
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

    fn solve(&self, b: &Vec<T>, x: &mut Vec<T>) -> Result<()> {
        if b.len() != self.n {
            return Err(LinAlgError::InvalidVectorSize(b.len()))?;
        }
        if x.len() != self.n {
            return Err(LinAlgError::InvalidVectorSize(x.len()))?;
        }

        for i in 0..self.n {
            x[i] = b[i];
        }

        let mut ii = 0;
        for i in 0..self.n {
            let p = self.indx[i];
            let mut sum = x[p].to_f32().unwrap();
            x[p] = x[i];
            if ii != 0 {
                for j in 0..(i - 1) {
                    sum -= self.lu[i][j].to_f32().unwrap() * x[j].to_f32().unwrap();
                }
            } else if sum != 0.0 {
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
            self.solve(&b[j].to_vec(), &mut x[j].to_vec());
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

    fn mprove(&self, b: &Vec<T>, x: &Vec<T>) {}
}
