use std::mem::swap;

use num::ToPrimitive;

use crate::{MatLinAlgBound, Matrix};

/// Matrix multipy b = Ax, where A is band-diagnoal
/// with m1 rows below the diagnoal and m2 rows above.
/// A is stored in a compact form.
pub fn banmul<T>(a: &Matrix<T>, m1: usize, m2: usize, x: &Vec<T>, b: &mut Vec<T>)
where
    T: MatLinAlgBound,
{
    let n = b.len();
    let im1 = m1.to_isize().unwrap();
    for i in 0..n {
        let k = i.to_isize().unwrap() - im1;
        let max = std::cmp::min(m1 + m2 + 1, (n.to_isize().unwrap() - k).to_usize().unwrap());
        b[i] = T::zero();
        for j in std::cmp::max(0, -k).to_usize().unwrap()..max {
            b[i] += (a[i][j] * x[(j.to_isize().unwrap() + k).to_usize().unwrap()]).into();
        }
    }
}

pub struct Bandec<T>
where
    T: MatLinAlgBound,
{
    n: usize,
    m1: usize,
    m2: usize,
    au: Matrix<T>,
    al: Matrix<T>,
    indx: Vec<usize>,
    d: f32,
}

impl<T> Bandec<T>
where
    T: MatLinAlgBound,
{
    pub fn new(a: Matrix<T>, m1: usize, m2: usize) -> Self {
        let TINY = 1.0e-10;
        let n = a.rows();
        let mut au = a;
        let mut al = Matrix::new(au.rows(), au.cols(), vec![T::zero(); au.rows() * au.cols()]);
        let mut indx = Vec::with_capacity(n);
        let mut d = 1.0;

        let mm = m1 + m2 + 1;
        let mut l = m1;
        for i in 0..m1 {
            for j in (m1 - i)..mm {
                au[i][j - l] = au[i][j];
            }
            l -= 1;
            for j in (mm - l - 1)..mm {
                au[i][j] = T::zero();
            }
        }

        l = m1;
        for k in 0..n {
            let mut dum = au[k][0].to_f32().unwrap();
            let mut i = k;
            if l < n {
                l += 1;
            }
            for j in (k + 1)..l {
                if au[j][0].to_f32().unwrap().abs() > dum.abs() {
                    dum = au[j][0].to_f32().unwrap();
                    i = j;
                }
            }
            indx.push(i + 1);
            if dum.abs() < TINY {
                dum = TINY;
            }
            if i != k {
                d = -d;
                au.swap_rows(k, i);
            }
            for i in (k + 1)..l {
                dum = au[i][0].to_f32().unwrap() / au[k][0].to_f32().unwrap();
                al[k][i - k - 1] = dum.into();
                for j in 1..mm {
                    au[i][j - 1] =
                        (au[i][j].to_f32().unwrap() - dum * au[k][j].to_f32().unwrap()).into();
                }
                au[i][mm - 1] = 0.0.into();
            }
        }

        Bandec {
            n,
            m1,
            m2,
            au,
            al,
            indx,
            d,
        }
    }

    pub fn solve(&self, b: &Vec<T>, x: &mut Vec<T>) {
        let mm = self.m1 + self.m2 + 1;
        let mut l = self.m1;
        for k in 0..self.n {
            x[k] = b[k];
        }
        for k in 0..self.n {
            let j = self.indx[k] - 1;
            if j != k {
                x.swap(j, k);
            }
            if l < self.n {
                l += 1;
            }
            for j in (k + 1)..l {
                let temp = self.al[k][j - k - 1] * x[k];
                x[j] -= temp.into();
            }
        }
        l = 1;
        for i in (0..self.n).rev() {
            let mut dum = x[i];
            for k in 1..l {
                dum -= (self.au[i][k] * x[k + i]).into();
            }
            x[i] = (dum / self.au[i][0]).into();
            if l < mm {
                l += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandec() {
        let a = Matrix::new(
            4,
            4,
            vec![
                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            ],
        );
        let bandec = Bandec::new(a, 2, 1);

        let b = vec![3.0, 6.0, 10.0, 9.0];
        let mut x = vec![1.0f32; 4];
        bandec.solve(&b, &mut x);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
        assert!((x[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_banmul() {
        #[rustfmt::skip]
        let a = Matrix::new(4, 4, vec![
            0.0, 0.0, 1.0, 1.0,
            0.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.0,
        ]);
        let m1 = 2;
        let m2 = 1;
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut b = vec![0.0f32; 4];
        banmul(&a, m1, m2, &x, &mut b);

        assert!((b[0] - 3.0).abs() < 1e-10);
        assert!((b[1] - 6.0).abs() < 1e-10);
        assert!((b[2] - 10.0).abs() < 1e-10);
        assert!((b[3] - 9.0).abs() < 1e-10);
    }
}
