use anyhow::Result;

use crate::{MatLinAlgBound, Matrix};

use super::LinAlgError;

pub struct Cholesky<T>
where
    T: MatLinAlgBound,
{
    n: usize,
    el: Matrix<T>,
}

impl<T> Cholesky<T>
where
    T: MatLinAlgBound,
{
    pub fn new(a: &Matrix<T>) -> Result<Self> {
        let n = a.rows();
        let mut el = a.clone();

        for i in 0..n {
            for j in i..n {
                let mut sum = el[i][j];
                for k in 0..i {
                    sum -= (el[i][k] * el[j][k]).into();
                }
                if i == j {
                    if sum <= T::zero() {
                        return Err(LinAlgError::NegativeValueNotAllowed())?;
                    }
                    el[i][i] = sum.to_f32().unwrap().sqrt().into();
                } else {
                    el[j][i] = (sum / el[i][i]).into();
                }
            }
        }
        for i in 0..n {
            for j in 0..i {
                el[j][i] = T::zero();
            }
        }
        Ok(Cholesky { n, el })
    }

    pub fn solve(&self, b: &Vec<T>, x: &mut Vec<T>) -> Result<()> {
        if b.len() != self.n || x.len() != self.n {
            return Err(LinAlgError::InvalidSize())?;
        }
        for i in 0..self.n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= (x[j] * self.el[i][j]).into();
            }
            x[i] = (sum / self.el[i][i]).into();
        }
        for i in (0..self.n).rev() {
            let mut sum = x[i];
            for j in (i + 1)..self.n {
                sum -= (x[j] * self.el[j][i]).into();
            }
            x[i] = (sum / self.el[i][i]).into();
        }
        Ok(())
    }

    /// Multiply Ly = b.
    pub fn elmult(&self, y: &Vec<T>, b: &mut Vec<T>) -> Result<()> {
        if b.len() != self.n || y.len() != self.n {
            return Err(LinAlgError::InvalidSize())?;
        }
        for i in 0..self.n {
            b[i] = T::zero();
            for j in 0..=i {
                b[i] += (self.el[i][j] * y[j]).into();
            }
        }
        Ok(())
    }

    /// Solve Ly = b.
    pub fn elsolve(&self, b: &Vec<T>, y: &mut Vec<T>) -> Result<()> {
        if b.len() != self.n || y.len() != self.n {
            return Err(LinAlgError::InvalidSize())?;
        }
        for i in 0..self.n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= (self.el[i][j] * y[j]).into();
            }
            y[i] = (sum / self.el[i][i]).into();
        }
        Ok(())
    }

    pub fn inverse(&self, ainv: &mut Matrix<T>) {
        ainv.resize(self.n, self.n);
        for i in 0..self.n {
            for j in 0..=i {
                let mut sum = if i == j { 1.0.into() } else { T::zero() };
                for k in (j..i).rev() {
                    sum -= (self.el[i][k] * ainv[j][k]).into();
                }
                ainv[j][i] = (sum / self.el[i][i]).into();
            }
        }
        for i in (0..self.n).rev() {
            for j in 0..=i {
                let mut sum = if i < j { T::zero() } else { ainv[j][i] };
                for k in (i + 1)..self.n {
                    sum -= (self.el[k][i] * ainv[j][k]).into();
                }
                let tmp = (sum / self.el[i][i]).into();
                ainv[j][i] = tmp;
                ainv[i][j] = tmp;
            }
        }
    }

    pub fn logdet(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.n {
            sum += self.el[i][i].to_f64().unwrap().ln();
        }
        return 2.0 * sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cholesky() {
        let mat = Matrix::new(
            3,
            3,
            vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0f64],
        );
        let res = Cholesky::new(&mat);
        assert!(res.is_ok());
        let ch = res.unwrap();
        println!("ch.el = {:?}", ch.el);
        let mut y = vec![0.0; 3];
        let b = vec![-4.0, 2.0, 4.0];
        let res = ch.solve(&b, &mut y);
        assert!(res.is_ok());
        println!("y = {:?}", y);
        assert!((y[0] + 1.0).abs() < 1e-5);
        assert!((y[1] - 2.0).abs() < 1e-5);
        assert!((y[2] - 3.0).abs() < 1e-5);

        let mut bb = vec![0.0; 3];
        let res = ch.elmult(&y, &mut bb);
        assert!(res.is_ok());
        println!("bb = {:?}", bb);
        assert!((bb[0] + 1.4142135381).abs() < 1e-5);
        assert!((bb[1] - 3.1565966252126034).abs() < 1e-5);
        assert!((bb[2] - 1.8311084505353095).abs() < 1e-5);

        let mut ainv = Matrix::new(3, 3, vec![0.0; 9]);
        ch.inverse(&mut ainv);
        println!("ainv = {:?}", ainv);
        assert!((ainv[0][0] - 0.75).abs() < 1e-5);
        assert!((ainv[0][1] - 0.5).abs() < 1e-5);
        assert!((ainv[0][2] - 0.25).abs() < 1e-5);
        assert!((ainv[1][0] - 0.5).abs() < 1e-5);
        assert!((ainv[1][1] - 1.0).abs() < 1e-5);
        assert!((ainv[1][2] - 0.5).abs() < 1e-5);
        assert!((ainv[2][0] - 0.25).abs() < 1e-5);
        assert!((ainv[2][1] - 0.5).abs() < 1e-5);
        assert!((ainv[2][2] - 0.75).abs() < 1e-5);
    }
}
