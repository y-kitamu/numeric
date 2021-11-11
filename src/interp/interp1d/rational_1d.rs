use crate::accessor_impl;

use super::Interp;

pub struct Rational1D<'a> {
    n: usize,
    mm: usize,
    jsav: usize,
    dj: usize,
    cor: usize,
    xx: &'a Vec<f64>,
    yy: &'a Vec<f64>,
}

impl<'a> Rational1D<'a> {
    pub fn new(xx: &'a Vec<f64>, yy: &'a Vec<f64>, mm: usize) -> Self {
        Self {
            n: xx.len(),
            mm,
            jsav: 0,
            dj: 0,
            cor: 0,
            xx,
            yy,
        }
    }
}

impl<'a> Interp for Rational1D<'a> {
    accessor_impl!((get = n, set = set_n): usize);
    accessor_impl!((get = mm, set = set_mm): usize);
    accessor_impl!((get = jsav, set = set_jsav): usize);
    accessor_impl!((get = dj, set = set_dj): usize);
    accessor_impl!((get = cor, set = set_cor): usize);

    fn xx(&self) -> &Vec<f64> {
        self.xx
    }

    fn rawinterp(&self, j: usize, x: f64) -> anyhow::Result<f64> {
        let xx = self.xx;
        let yy = self.yy;
        let mut c = vec![0.0; self.mm()];
        let mut d = vec![0.0; self.mm()];
        for i in 0..self.mm() {
            if (x - xx[j + i]).abs() < 1e-7 {
                return Ok(xx[j + i]);
            }
            c[i] = yy[j + i];
            d[i] = yy[j + i];
        }
        let mut y = c[0];
        for m in 0..(self.mm() - 1) {
            for i in 0..(self.mm() - m - 1) {
                let xval = (x - xx[j + i]) / (x - xx[j + i + m + 1]);
                let dnume = c[i + 1] * (c[i + 1] - d[i]);
                let cnume = xval * d[i] * (c[i + 1] - d[i]);
                let deno = xval * d[i] - c[i + 1];
                d[i] = dnume / deno;
                c[i] = cnume / deno;
            }
            y += c[0];
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational1d() {
        let func = |x: f64| (x * x + 2.0 * x + 1.0) / (x + 2.0);
        let xx = vec![1.0, 2.0, 3.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let yy = vec![
            func(1.0),
            func(2.0),
            func(3.0),
            func(6.0),
            func(8.0),
            func(9.0),
            func(10.0),
            func(11.0),
            func(12.0),
        ];
        let mut rat1d = Rational1D::new(&xx, &yy, 4);
        println!("yy = {:?}", yy);
        let res = rat1d.interp(4.0).unwrap();
        let tval = func(4.0);
        assert!((res - tval).abs() < 1e-2, "y = {}, true = {}", res, tval);

        let res = rat1d.interp(5.0).unwrap();
        let tval = func(5.0);
        assert!((res - tval).abs() < 1e-2, "y = {}, true = {}", res, tval);
    }
}
