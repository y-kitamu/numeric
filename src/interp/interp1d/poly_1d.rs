use crate::{accessor_impl, interp::InterpError};
use anyhow::Result;

use super::Interp;

pub struct Poly1D<'a> {
    n: usize,
    mm: usize,
    jsav: usize,
    dj: usize,
    cor: usize,
    dy: f64,
    xx: &'a Vec<f64>,
    yy: &'a [f64],
}

impl<'a> Poly1D<'a> {
    pub fn new(xx: &'a Vec<f64>, yy: &'a Vec<f64>, mm: usize) -> Self {
        Self {
            n: xx.len(),
            mm,
            jsav: 0,
            dj: 0,
            cor: 0,
            dy: 0.0,
            xx,
            yy,
        }
    }

    pub fn set_yy(&mut self, yy: &'a [f64]) {
        self.yy = yy;
    }
}

impl<'a> Interp for Poly1D<'a> {
    accessor_impl!((get = n, set = set_n): usize);
    accessor_impl!((get = mm, set = set_mm): usize);
    accessor_impl!((get = jsav, set = set_jsav): usize);
    accessor_impl!((get = dj, set = set_dj): usize);
    accessor_impl!((get = cor, set = set_cor): usize);

    fn xx(&self) -> &Vec<f64> {
        self.xx
    }

    fn rawinterp(&self, j: usize, x: f64) -> Result<f64> {
        let xx = self.xx();
        let yy = self.yy;
        let mut c = vec![0.0; self.mm()];
        let mut d = vec![0.0; self.mm()];
        let mut ns: isize = 0;
        let mut dif = x - xx[j];
        for i in 0..self.mm() {
            let tmp = (x - xx[j + i]).abs();
            if tmp < dif {
                ns = i as isize;
                dif = tmp;
            }
            c[i] = yy[j + i];
            d[i] = yy[j + i];
        }
        let mut y = yy[j + ns as usize];
        ns -= 1;
        for m in 1..self.mm() {
            for i in 0..(self.mm() - m) {
                let ho = xx[j + i] - x;
                let hp = xx[j + i + m] - x;
                let w = c[i + 1] - d[i];
                let mut dn = ho - hp;
                if dn.abs() < 1e-5 {
                    return Err(InterpError::IdenticalX())?;
                }
                dn = w / dn;
                d[i] = hp * dn;
                c[i] = ho * dn;
            }
            let dy = if (2 * (ns + 1) as usize) < self.mm() - m {
                c[(ns + 1) as usize]
            } else {
                ns -= 1;
                d[(ns + 1) as usize]
            };
            y += dy;
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_1d() {
        let xx = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0];
        let yy = vec![1.0, 4.0, 9.0, 16.0, 25.0, 100.0, 400.0];
        let mut p1d = Poly1D::new(&xx, &yy, 3);
        let res = p1d.interp(1.5).unwrap();
        assert!((res - 2.25).abs() < 1e-5, "res = {}", res);
        let res = p1d.interp(9.0).unwrap();
        assert!((res - 81.0).abs() < 1e-5, "res = {}", res);
    }

    #[test]
    fn test_poly_1d_linear() {
        let xx = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0];
        let yy = vec![10.0, 15.0, 30.0, 10.0, 0.0, 5.0, -5.0];
        let mut lint = Poly1D::new(&xx, &yy, 2);
        let res = lint.interp(1.0).unwrap();
        assert!((res - 10.0).abs() < 1e-5);
        assert!((lint.interp(1.5).unwrap() - 12.5).abs() < 1e-5);
        let res = lint.interp(20.0).unwrap();
        assert!((res + 5.0).abs() < 1e-5, "res = {}", res);
        let res = lint.interp(15.0).unwrap();
        assert!((res - 0.0).abs() < 1e-5, "res = {}", res);
        let res = lint.interp(6.0).unwrap();
        assert!((res - 1.0).abs() < 1e-5, "res = {}", res);
    }
}
