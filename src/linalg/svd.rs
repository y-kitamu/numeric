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
        let u = Matrix::identity(m);
        let v = Matrix::identity(n);
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
        svd.decompose(a);
        svd.reorder();
        svd.tsh = 0.5 * (m as f32 + n as f32 + 1.0f32).sqrt() * svd.w[0].to_f32().unwrap() * eps;
        svd
    }

    /// https://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.PDF
    fn decompose(&mut self, mut a: Matrix<T>) {
        // householder reduction
        for k in 0..self.n {
            // householder reduction for U
            let mut sum = T::zero();
            for i in k..self.m {
                sum += (a[i][k] * a[i][k]).into();
            }
            let s = sum.sqrt();
            let t: T = (a[k][k] - s.copysign(a[k][k])).into();
            sum += (T::from(t * t) - T::from(a[k][k] * a[k][k])).into();
            let norm = sum.sqrt();
            a[k][k] = s;
            for j in k..self.m {
                let mut sum = T::zero();
                for i in k..self.m {
                    sum += (self.u[j][i] * a[i][k]).into();
                }
                let tau: T = (T::from(sum / norm) * 2.0.into()).into();
                for i in k..self.m {
                    self.u[j][i] -= (tau * a[i][k]).into();
                }
            }
            // householder reduction for V
            if k < self.n - 1 {
                let mut sum = T::zero();
                for i in (k + 1)..self.n {
                    sum += (a[k][i] * a[k][i]).into();
                }
                let s = sum.sqrt();
                let t: T = (a[k][k + 1] - s.copysign(a[k][k + 1])).into();
                sum += (T::from(t * t) - T::from(a[k][k + 1] * a[k][k + 1])).into();
                let norm = sum.sqrt();
                a[k][k + 1] = s;
                for j in (k + 1)..self.n {
                    let mut sum = T::zero();
                    for i in (k + 1)..self.n {
                        sum += (self.v[i][j] * a[k][i]).into();
                    }
                    let tau: T = (T::from(sum / norm) * 2.0.into()).into();
                    for i in (k + 1)..self.n {
                        self.v[i][j] -= (tau * a[k][i]).into();
                    }
                }
            }
        }
        for i in 0..self.m {
            for j in 0..self.n {
                if i == j || (i + 1) == j {
                    continue;
                }
                a[i][j] = T::zero();
            }
        }
        // Golub-Reinsch SVD
        let eps: T = 1e-7.into();
        let mut p = self.n - 1;
        let mut q = 0;
        loop {
            for i in 0..(self.n - 1) {
                if a[i][i + 1].abs() < (T::from(a[i][i].abs() + a[i + 1][i + 1]).abs() * eps).into()
                {
                    a[i][i + 1] = T::zero();
                }
            }
            for i in (1..self.n).rev() {
                if a[i - 1][i].abs() > eps {
                    q = self.n - 1 - i;
                    break;
                }
            }
            if q == self.n - 1 {
                break;
            }
            for i in 0..(self.n - q) {
                if a[i][i + 1].abs() < eps {
                    p = i + 1;
                }
            }
            let mut flag = false;
            for i in p..(self.n - q) {
                if a[i][i].abs() < eps {
                    // Givens rotation
                    let norm = (a[i][i + 1].square() + a[i + 1][i + 1].square()).into();
                    let nsin: T = (a[i][i + 1] / norm).into();
                    let cos: T = (a[i + 1][i + 1] / norm).into();
                    a[i + 1][i + 1] =
                        (T::from(nsin * a[i][i + 1]) + T::from(cos * a[i + 1][i + 1])).into();
                    a[i][i + 1] = T::zero();
                    flag = true;
                }
            }
            if flag {
                continue;
            }
            // Golub–Kahan SVD step
            // Set C = lower, right 2 × 2 sub matrix of B(2, 2)T x B(2, 2)
            let idx = self.n - q - 2;
            let c00 = a[idx][idx].square();
            let c10: T = (a[idx][idx] * a[idx][idx + 1]).into();
            let c11: T = (a[idx][idx + 1].square() + a[idx + 1][idx + 1].square()).into();
            let nume0 = T::from(c10 * 2.0.into()).square();
            let nume1: T = (T::from(c00 * c11) - c10.square()).into();
            let nume: T = (nume0 - T::from(nume1 * 4.0.into())).into();
            let l1: T = (((c00 + c11) + nume) * 0.5.into()).into();
            let l2: T = (T::from((c00 + c11) - nume) * 0.5.into()).into();
            let mu = if T::from(l1 - a[idx + 1][idx + 1]).abs()
                < T::from(l2 - a[idx + 1][idx + 1]).abs()
            {
                l1
            } else {
                l2
            };
            let givens_right = |cos: T, sin: T, mat: &mut Matrix<T>, idx: usize| {
                let ak0k0: T =
                    (T::from(cos * mat[idx][idx]) - T::from(sin * mat[idx][idx + 1])).into();
                let ak0k1: T =
                    (T::from(sin * mat[idx][idx]) + T::from(cos * mat[idx][idx + 1])).into();
                let ak1k0: T = (T::from(cos * mat[idx + 1][idx])
                    - T::from(sin * mat[idx + 1][idx + 1]))
                .into();
                let ak1k1: T = (T::from(sin * mat[idx][idx + 1])
                    - T::from(cos * mat[idx + 1][idx + 1]))
                .into();
                mat[idx][idx] = ak0k0;
                mat[idx][idx + 1] = ak0k1;
                mat[idx + 1][idx] = ak1k0;
                mat[idx + 1][idx + 1] = ak1k1;
            };
            let mut alpha: T = (a[p][p].square() - mu).into();
            let mut beta: T = (a[p][p] * a[p][p + 1]).into();
            for k in p..(self.n - q - 1) {
                let norm = alpha.norm(beta);
                let cos: T = (alpha / norm).into();
                let sin: T = (T::from(-beta) / norm).into();
                givens_right(cos, sin, &mut a, k);
                givens_right(cos, sin, &mut self.v, k);
                alpha = a[k][k];
                beta = a[k + 1][k];
                let norm = alpha.norm(beta);
                let cos: T = (alpha / norm).into();
                let sin: T = (T::from(-beta) / norm).into();
                givens_right(cos, sin, &mut a, k);
                givens_right(cos, sin, &mut self.u, k);
                if k < self.n - q - 2 {
                    alpha = a[k][k + 1];
                    beta = a[k][k + 2];
                }
            }
        }
        for i in 0..self.n {
            self.w[i] = a[i][i];
        }
    }

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
        if b.rows() != self.n || x.rows() != self.n || b.cols() != x.cols() {
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
