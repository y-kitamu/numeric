use crate::accessor_impl;

use super::Interp;
use anyhow::Result;

pub struct Spline1D<'a> {
    n: usize,
    mm: usize,
    jsav: usize,
    dj: usize,
    cor: usize,
    xx: &'a Vec<f64>,
    yy: &'a [f64],
    y2: Vec<f64>,
}

impl<'a> Spline1D<'a> {
    pub fn new(xx: &'a Vec<f64>, yy: &'a [f64]) -> Result<Self> {
        let mut obj = Self {
            n: xx.len(),
            mm: 2,
            jsav: 0,
            dj: 0,
            cor: 0,
            xx,
            yy,
            y2: vec![0.0; xx.len()],
        };
        match obj.calcy2() {
            Ok(_) => Ok(obj),
            Err(err) => Err(err),
        }
    }

    pub fn calcy2(&mut self) -> Result<()> {
        let mut l = vec![0.0; self.n];
        let mut u = vec![0.0; self.n];
        let mut d = vec![0.0; self.n];
        let x = self.xx;
        let y = self.yy;
        let mut lhs1 = (y[1] - y[0]) / (x[1] - x[0]);

        for i in 1..(self.n - 1) {
            let lhs0 = lhs1;
            lhs1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
            if i > 1 {
                let a = (x[i] - x[i - 1]) / 6.0;
                l[i] = a / d[i - 1];
            }
            let b = (x[i + 1] - x[i - 1]) / 3.0;
            d[i] = b - l[i] * u[i - 1];
            if i < self.n - 2 {
                u[i] = (x[i + 1] - x[i]) / 6.0;
            }
            self.y2[i] = lhs1 - lhs0 - self.y2[i - 1] * l[i];
        }
        for i in (1..(self.n - 1)).rev() {
            let tmp = (self.y2[i] - u[i] * self.y2[i + 1]) / d[i];
            self.y2[i] = tmp;
        }
        Ok(())
    }
}

impl<'a> Interp for Spline1D<'a> {
    accessor_impl!((get = n, set = set_n): usize);
    accessor_impl!((get = mm, set = set_mm): usize);
    accessor_impl!((get = jsav, set = set_jsav): usize);
    accessor_impl!((get = dj, set = set_dj): usize);
    accessor_impl!((get = cor, set = set_cor): usize);

    fn xx(&self) -> &Vec<f64> {
        self.xx
    }

    fn rawinterp(&self, j: usize, x: f64) -> anyhow::Result<f64> {
        if self.xx[j] == self.xx[j + 1] {
            return Ok(self.yy[j]);
        }
        let a = (self.xx[j + 1] - x) / (self.xx[j + 1] - self.xx[j]);
        let b = 1.0 - a;
        let c = (a * a * a - a) * (self.xx[j + 1] - self.xx[j]).powi(2) / 6.0;
        let d = (b * b * b - b) * (self.xx[j + 1] - self.xx[j]).powi(2) / 6.0;
        let y = a * self.yy[j] + b * self.yy[j + 1] + c * self.y2[j] + d * self.y2[j + 1];
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spline1d() {
        let x = vec![0.0, 2.0, 3.0, 5.0, 7.0, 8.0];
        let y = vec![4.0, 2.0, 6.0, 8.0, 6.0, 8.0];
        let sp1d = Spline1D::new(&x, &y);
        assert!(sp1d.is_ok());
        let mut sp1d = sp1d.unwrap();
        let y = sp1d.interp(6.0).unwrap();
        assert!((y - 6.50358166).abs() < 1e-5, "y = {}", y);
    }
}
