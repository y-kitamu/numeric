use anyhow::Result;

use crate::{MatLinAlgBound, Matrix};

use super::LinAlgError;

struct SVD<T>
where
    T: MatLinAlgBound,
{
    m: usize,
    n: usize,
    u: Matrix<T>,
    v: Matrix<T>,
    w: Vec<T>,
    eps: f32,
    tsh: f32,
}

impl<T> SVD<T>
where
    T: MatLinAlgBound,
{
    pub fn new(a: Matrix<T>) -> Self {
        let m = a.rows();
        let n = a.cols();
        let u = a;
        let v = Matrix::new(n, n, vec![T::zero(); n * n]);
        let w = vec![T::zero(); n];
        let eps = f32::EPSILON;
        let tsh = 0.0;
        let mut svd = SVD {
            m,
            n,
            u,
            v,
            w,
            eps,
            tsh,
        };
        svd.decompose();
        svd.reorder();
        svd.tsh = 0.5 * (m as f32 + n as f32 + 1.0f32).sqrt() * svd.w[0].to_f32().unwrap() * eps;
        svd
    }

    fn decompose(&mut self) {}

    fn reorder(&mut self) {}

    pub fn solve(&self, b: &[T], x: &mut [T]) -> Result<()> {
        if b.len() != self.m || x.len() != self.n {
            return Err(LinAlgError::InvalidSize())?;
        }

        let mut tmp = vec![T::zero(); self.n];
        for j in 0..self.n {
            let mut s = T::zero();
            if self.w[j] > T::zero() {
                for i in 0..self.m {
                    s += (self.u[i][j] * b[i]).into();
                }
                s /= self.w[j];
            }
            tmp[j] = s;
        }
        for j in 0..self.n {
            let mut s = T::zero();
            for i in 0..self.n {
                s += (self.v[j][i] * tmp[i]).into();
            }
            x[j] = s;
        }
        Ok(())
    }

    pub fn solve_mat(&self, b: &Matrix<T>, x: &mut Matrix<T>) -> Result<()> {
        if b.rows() != self.n || x.rows() != self.m || b.cols() != x.cols() {
            return Err(LinAlgError::InvalidSize())?;
        }
        let mut tmp = vec![T::zero(); self.n];
        for j in 0..self.m {
            self.solve(&b.get_col(j), &mut tmp);
            for i in 0..self.n {
                x[i][j] = tmp[i];
            }
        }
        Ok(())
    }

    pub fn rank(&self) -> usize {
        let mut cnt = 0;
        for j in 0..self.n {
            if self.w[j].to_f32().unwrap() > self.tsh {
                cnt += 1;
            }
        }
        cnt
    }

    pub fn nullity(&self) -> usize {
        let mut cnt = 0;
        for j in 0..self.n {
            if self.w[j].to_f32().unwrap() <= self.tsh {
                cnt += 1;
            }
        }
        cnt
    }

    pub fn range(&self) -> Matrix<T> {
        let mut cnt = 0;
        let rnk = self.rank();
        let mut range_mat = Matrix::new(self.m, rnk, vec![T::zero(); self.m * rnk]);
        for j in 0..self.n {
            if self.w[j].to_f32().unwrap() > self.tsh {
                for i in 0..self.m {
                    range_mat[i][cnt] = self.u[i][j];
                }
                cnt += 1;
            }
        }
        range_mat
    }

    pub fn nullspace(&self) -> Matrix<T> {
        let mut cnt = 0;
        let nul = self.nullity();
        let mut null_mat = Matrix::new(self.m, nul, vec![T::zero(); self.m * nul]);
        for j in 0..self.n {
            if self.w[j].to_f32().unwrap() <= self.tsh {
                for i in 0..self.n {
                    null_mat[i][cnt] = self.v[i][j];
                }
                cnt += 1;
            }
        }
        null_mat
    }
}
