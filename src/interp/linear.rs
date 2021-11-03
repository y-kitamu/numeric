use crate::accessor_impl;

use super::Interp;

pub struct LinearInterp<'a> {
    n: usize,
    mm: usize,
    jsav: usize,
    dj: usize,
    cor: usize,
    xx: &'a Vec<f64>,
    yy: &'a Vec<f64>,
}

impl<'a> LinearInterp<'a> {
    pub fn new(xx: &'a Vec<f64>, yy: &'a Vec<f64>) -> Self {
        Self {
            n: xx.len(),
            mm: 2,
            jsav: 0,
            cor: 0,
            dj: 0,
            xx,
            yy,
        }
    }
}

impl<'a> Interp for LinearInterp<'a> {
    accessor_impl!((get = n, set = set_n): usize);
    accessor_impl!((get = mm, set = set_mm): usize);
    accessor_impl!((get = jsav, set = set_jsav): usize);
    accessor_impl!((get = dj, set = set_dj): usize);
    accessor_impl!((get = cor, set = set_cor): usize);

    fn xx(&self) -> &Vec<f64> {
        self.xx
    }

    fn rawinterp(&self, j: usize, x: f64) -> f64 {
        println!("j = {}, x = {}", j, x);
        if self.xx[j] == self.xx[j + 1] {
            return self.yy[j];
        }
        self.yy[j]
            + ((x - self.xx[j]) / (self.xx[j + 1] - self.xx[j])) * (self.yy[j + 1] - self.yy[j])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interp() {
        let xx = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0];
        let yy = vec![10.0, 15.0, 30.0, 10.0, 0.0, 5.0, -5.0];
        let mut lint = LinearInterp::new(&xx, &yy);
        let res = lint.interp(1.0);
        assert!((res - 10.0).abs() < 1e-5);
        assert!((lint.interp(1.5) - 12.5).abs() < 1e-5);
        let res = lint.interp(20.0);
        assert!((res + 5.0).abs() < 1e-5, "res = {}", res);
        let res = lint.interp(15.0);
        assert!((res - 0.0).abs() < 1e-5, "res = {}", res);
        let res = lint.interp(6.0);
        assert!((res - 1.0).abs() < 1e-5, "res = {}", res);
    }
}
