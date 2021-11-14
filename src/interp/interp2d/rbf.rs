use crate::{linalg::lu_decomposition::LUdcmp, MatLinAlgBound, Matrix};

pub trait RBFFunc {
    fn rbf(&self, r: f64) -> f64;
}

pub struct MultiQuadric {
    r02: f64,
}

impl MultiQuadric {
    pub fn new(scale: f64) -> Self {
        Self { r02: scale * scale }
    }
}

impl RBFFunc for MultiQuadric {
    fn rbf(&self, r: f64) -> f64 {
        (r * r + self.r02).sqrt()
    }
}

struct RBF<'a, T>
where
    T: RBFFunc,
{
    dim: usize,
    n: usize,
    w: Vec<f64>,
    pts: &'a Matrix<f64>,
    vals: &'a Vec<f64>,
    rbf_fn: T,
    norm: bool,
}

impl<'a, T> RBF<'a, T>
where
    T: RBFFunc,
{
    pub fn new(pts: &'a Matrix<f64>, vals: &'a Vec<f64>, rbf_fn: T, norm: bool) -> Self {
        let dim = pts.cols();
        let n = pts.rows();
        let mut w = vec![0.0; n];

        let mut mat = Matrix::new(n, n, vec![0.0; n * n]);
        for i in 0..n {
            for j in 0..n {
                let mut dist = 0.0;
                for k in 0..dim {
                    dist += (pts[i][k] - pts[j][k]).square();
                }
                dist = dist.sqrt();
                mat[i][j] = rbf_fn.rbf(dist);
                if norm && dist > 1e-8 {
                    mat[i][j] /= dist;
                }
            }
        }
        LUdcmp::new(&mat).unwrap().solve(&vals, &mut w).unwrap();

        Self {
            dim: pts.cols(),
            n,
            w,
            pts,
            vals,
            norm,
            rbf_fn,
        }
    }

    pub fn interp(&self, pt: &Vec<f64>) -> f64 {
        let mut ans = 0.0;
        let mut den = 0.0;
        for i in 0..self.n {
            let mut dist = 0.0;
            for k in 0..self.dim {
                dist += (self.pts[i][k] - pt[k]).square();
            }
            dist = dist.sqrt();
            ans += self.w[i] * self.rbf_fn.rbf(dist);
            den += dist;
        }
        if self.norm {
            ans /= den;
        }
        ans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf() {}
}
