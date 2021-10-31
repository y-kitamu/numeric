use anyhow::Result;

use crate::{MatLinAlgBound, Matrix};

use super::LinAlgError;

pub struct QRdcmp<T>
where
    T: MatLinAlgBound,
{
    n: usize,
    qt: Matrix<T>,
    r: Matrix<T>,
    singular: bool,
}

impl<T> QRdcmp<T>
where
    T: MatLinAlgBound,
{
    pub fn new(a: &Matrix<T>) -> Self {
        let n = a.rows();
        let mut qt = Matrix::new(n, n, vec![T::zero(); n * n]);
        let mut r = a.clone();
        let mut singular = false;

        let mut c = vec![T::zero(); n];
        let mut d = vec![T::zero(); n];

        for k in 0..(n - 1) {
            let mut scale = 0.0f32;
            for i in k..n {
                scale = scale.max(r[i][k].to_f32().unwrap().abs());
            }
            if scale < 1e-5 {
                singular = true;
            } else {
                for i in k..n {
                    r[i][k] /= scale.into();
                }
                let mut sum = T::zero();
                for i in k..n {
                    sum += (r[i][k] * r[i][k]).into();
                }
                let sigma: T = sum
                    .to_f32()
                    .unwrap()
                    .sqrt()
                    .copysign(r[k][k].to_f32().unwrap())
                    .into();
                r[k][k] += sigma;
                c[k] = (sigma * r[k][k]).into();
                d[k] = (T::from(-scale) * sigma).into();
                for j in (k + 1)..n {
                    let mut sum = T::zero();
                    for i in k..n {
                        sum += (r[i][k] * r[i][j]).into();
                    }
                    let tau: T = (sum / c[k]).into();
                    for i in k..n {
                        let tmp = tau * r[i][k];
                        r[i][j] -= tmp.into();
                    }
                }
            }
        }
        d[n - 1] = r[n - 1][n - 1];
        println!("d[n - 1] = {}", d[n - 1].to_f32().unwrap());
        if d[n - 1].to_f32().unwrap().abs() < 1e-5 {
            singular = true;
        }
        for i in 0..n {
            qt[i][i] = 1.0.into();
        }
        for k in 0..(n - 1) {
            if c[k].to_f32().unwrap().abs() > 0.0 {
                for j in 0..n {
                    let mut sum = T::zero();
                    for i in k..n {
                        sum += (r[i][k] * qt[i][j]).into();
                    }
                    sum /= c[k];
                    for i in k..n {
                        qt[i][j] -= (sum * r[i][k]).into();
                    }
                }
            }
        }
        for i in 0..n {
            r[i][i] = d[i];
            for j in 0..i {
                r[i][j] = T::zero();
            }
        }
        Self { n, qt, r, singular }
    }

    pub fn solve(&self, b: &Vec<T>, x: &mut Vec<T>) -> Result<()> {
        self.qtmult(b, x);
        self.rsolve(&x.clone(), x)
    }

    pub fn qtmult(&self, b: &Vec<T>, x: &mut Vec<T>) {
        for i in 0..self.n {
            let mut sum = T::zero();
            for j in 0..self.n {
                sum += (self.qt[i][j] * b[j]).into();
            }
            x[i] = sum;
        }
    }

    pub fn rsolve(&self, b: &Vec<T>, x: &mut Vec<T>) -> Result<()> {
        if self.singular {
            return Err(LinAlgError::SingularMatrix("qt".to_string()))?;
        }
        for i in (0..self.n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..self.n {
                sum -= (self.r[i][j] * x[j]).into();
            }
            x[i] = (sum / self.r[i][i]).into();
        }
        Ok(())
    }

    pub fn update(&mut self, u: &Vec<T>, v: &Vec<T>) {
        let mut w = u.clone();
        let k = {
            let mut m = 0;
            for k in (0..self.n).rev() {
                if w[k] != T::zero() {
                    m = k;
                }
            }
            m
        };
        for i in (0..k).rev() {
            self.rotate(i, w[i], (-w[i + 1]).into());
            let awi = w[i].to_f32().unwrap().abs();
            let awwi = w[i + 1].to_f32().unwrap().abs();
            if awi < 1e-5 {
                w[i] = awwi.into();
            } else if awi > awwi {
                w[i] = (awi * (1.0 + awwi * awwi / (awi * awi)).sqrt()).into();
            } else {
                w[i] = (awwi * (1.0 + awi * awi / (awwi * awwi)).sqrt()).into();
            }
        }
        for i in 0..self.n {
            self.r[0][i] += (w[0] * v[i]).into();
        }
        for i in 0..k {
            self.rotate(i, self.r[i][i], (-self.r[i + 1][i]).into());
        }
        for i in 0..self.n {
            if self.r[i][i].to_f32().unwrap().abs() < 1e-5 {
                self.singular = true;
                break;
            }
        }
    }

    pub fn rotate(&mut self, i: usize, a: T, b: T) {
        let absa = a.to_f32().unwrap().abs();
        let absb = b.to_f32().unwrap().abs();
        let (c, s) = if absa < 1e-5 {
            (0.0, if b >= T::zero() { 1.0 } else { -1.0 })
        } else if absa > absb {
            let fact = absb / absa;
            let tmp = (1.0 / (1.0 + fact * fact).sqrt()).copysign(a.to_f32().unwrap());
            (tmp, fact * tmp)
        } else {
            let fact = absa / absb;
            let tmp = (1.0 / (1.0 + (fact * fact)).sqrt()).copysign(b.to_f32().unwrap());
            (fact * tmp, tmp)
        };
        for j in i..self.n {
            let y = self.r[i][j].to_f32().unwrap();
            let w = self.r[i + 1][j].to_f32().unwrap();
            self.r[i][j] = (c * y - s * w).into();
            self.r[i + 1][j] = (s * y + c * w).into();
        }
        for j in 0..self.n {
            let y = self.qt[i][j].to_f32().unwrap();
            let w = self.qt[i + 1][j].to_f32().unwrap();
            self.qt[i][j] = (c * y - s * w).into();
            self.qt[i + 1][j] = (s * y + c * w).into();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qrdcmp() {
        let mat = Matrix::new(3, 3, vec![1.0, 1.0, -1.0, -2.0, -1.0, 1.0, -1.0, -2.0, 1.0]);
        let qrd = QRdcmp::new(&mat);
        let b = vec![0.0, -1.0, -2.0];
        let mut x = vec![0.0f32; 3];
        let res = qrd.solve(&b, &mut x);
        println!("qrd.qt = {:?}", qrd.qt);
        println!("qrd.r = {:?}", qrd.r);
        assert!(res.is_ok());
        println!("x = {:?}", x);
        println!(
            "qt * b = {:?}",
            qrd.qt.clone() * Matrix::new(3, 1, b.clone())
        );
        println!("r * x = {:?}", qrd.r.clone() * Matrix::new(3, 1, x.clone()));
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[1] - 2.0).abs() < 1e-5);
        assert!((x[2] - 3.0).abs() < 1e-5);
    }
}
