use crate::accessor_impl;

use super::Interp;

pub struct BaryRat1D<'a> {
    n: usize,
    mm: usize,
    jsav: usize,
    dj: usize,
    cor: usize,
    xx: &'a Vec<f64>,
    yy: &'a Vec<f64>,
    w: Vec<f64>,
}

impl<'a> BaryRat1D<'a> {
    pub fn new(xx: &'a Vec<f64>, yy: &'a Vec<f64>, mm: usize, calc: usize) -> Self {
        let n = xx.len();
        let w = match calc {
            1 => BaryRat1D::calcW(xx, mm),
            _ => BaryRat1D::calcW2(xx, mm),
        };
        Self {
            n,
            mm,
            jsav: 0,
            dj: 0,
            cor: 0,
            xx,
            yy,
            w,
        }
    }

    fn calcW(xx: &Vec<f64>, mm: usize) -> Vec<f64> {
        let n = xx.len();
        let mut w = vec![0.0; n];
        for k in 0..n {
            let mini = if k < mm { 0 } else { k - mm };
            let maxi = std::cmp::min(n - mm, k);
            // println!("k = {}, mini = {}, maxi = {}", k, mini, maxi);
            let mut sum = 0.0;
            let mut temp = if (mini & 1) == 1 { -1.0 } else { 1.0 };
            for i in mini..maxi {
                let maxj = std::cmp::min(i + mm, n - 1);
                let mut term = 1.0;
                for j in i..=maxj {
                    if j != k {
                        term *= xx[k] - xx[j];
                    }
                }
                sum += temp / term;
                temp = -temp;
            }
            w[k] = sum;
        }
        w
    }

    fn calcW2(xx: &Vec<f64>, mm: usize) -> Vec<f64> {
        let n = xx.len();
        let mut w = vec![0.0; n];
        for i in 0..n {
            let mut term = 1.0;
            for j in 0..n {
                if i != j {
                    term *= xx[i] - xx[j];
                }
            }
            w[i] = 1.0 / term;
        }
        w
    }
}

impl<'a> Interp for BaryRat1D<'a> {
    accessor_impl!((get = n, set = set_n): usize);
    accessor_impl!((get = mm, set = set_mm): usize);
    accessor_impl!((get = jsav, set = set_jsav): usize);
    accessor_impl!((get = dj, set = set_dj): usize);
    accessor_impl!((get = cor, set = set_cor): usize);

    fn xx(&self) -> &Vec<f64> {
        self.xx
    }

    fn rawinterp(&self, jlo: usize, x: f64) -> anyhow::Result<f64> {
        let mut nume = 0.0;
        let mut deno = 0.0;
        for i in 0..self.n {
            if (x - self.xx[i]).abs() < 1e-7 {
                return Ok(self.yy[i]);
            }
            let tmp = self.w[i] / (x - self.xx[i]);
            nume += tmp * self.yy[i];
            deno += tmp;
        }
        Ok(nume / deno)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interp() {
        let mut xx = vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
        let mut yy = vec![1.0, 4.0, 9.0, 25.0, 49.0, 100.0];
        let mut by1d = BaryRat1D::new(&xx, &yy, 2, 1);
        let res = by1d.interp(1.5);
        assert!(res.is_ok());
        // assert!((res - 2.5).abs() < 1e-5, "res = {}", res);

        let mut by1d = BaryRat1D::new(&xx, &yy, 2, 2);
        let res = by1d.interp(1.5);
        assert!(res.is_ok());
    }
}
