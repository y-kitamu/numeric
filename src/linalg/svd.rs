use crate::{MatLinAlgBound, Matrix};

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
        svd.tsh = 0.5 * (m as f32 + n as f32 + 1.0f32).sqrt() * w[0].to_f32().unwrap() * eps;
    }

    fn decompose(&mut self) {}

    fn reorder(&mut self) {}

    pub fn solve(&mut self) {}

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
