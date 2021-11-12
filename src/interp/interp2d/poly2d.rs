use crate::{
    interp::{poly_1d::Poly1D, Interp},
    Matrix,
};

pub struct Poly2D<'a> {
    m: usize,
    n: usize,
    mm: usize,
    nn: usize,
    y: &'a Matrix<f64>,
    yv: Vec<f64>,
    x1v: &'a Vec<f64>,
    x2interp: Poly1D<'a>,
}

impl<'a> Poly2D<'a> {
    pub fn new(
        x1v: &'a Vec<f64>,
        x2v: &'a Vec<f64>,
        ym: &'a Matrix<f64>,
        mp: usize,
        np: usize,
    ) -> Self {
        let yv = vec![0.0; x1v.len()];
        let x2interp = Poly1D::new(x2v, x2v, np);
        Self {
            m: x1v.len(),
            n: x2v.len(),
            mm: mp,
            nn: np,
            y: ym,
            yv,
            x1v,
            x2interp,
        }
    }

    pub fn interp(&mut self, x1p: f64, x2p: f64) -> f64 {
        let mut x1interp = Poly1D::new(self.x1v, self.x1v, self.mm);
        let i = if x1interp.cor() == 1 {
            x1interp.hunt(x1p)
        } else {
            x1interp.locate(x1p)
        };
        let j = if self.x2interp.cor() == 1 {
            self.x2interp.hunt(x2p)
        } else {
            self.x2interp.locate(x2p)
        };
        for i in 0..self.m {
            self.x2interp.set_yy(&self.y[i]);
            self.yv[i] = self.x2interp.rawinterp(j, x2p).unwrap();
        }
        x1interp.set_yy(&self.yv);
        x1interp.rawinterp(i, x1p).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly2d() {
        let x1 = vec![0.0, 1.0, 2.0, 3.0, 5.0];
        let x2 = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        #[rustfmt::skip]
        let y = Matrix::new(5, 5, vec![
            1.0, 2.0, 3.0, 4.0, 5.0,
            2.0, 3.0, 4.0, 5.0, 6.0,
            3.0, 4.0, 5.0, 6.0, 7.0,
            4.0, 5.0, 6.0, 7.0, 8.0,
            5.0, 6.0, 7.0, 8.0, 9.0,
        ]);
        let mut interp = Poly2D::new(&x1, &x2, &y, 2, 2);

        let res = interp.interp(0.0, 0.0);
        assert!((res - 1.0).abs() < 1e-5, "res = {}", res);

        let res = interp.interp(0.5, 0.5);
        assert!((res - 2.0).abs() < 1e-5);

        let res = interp.interp(2.5, 3.0);
        assert!((res - 6.0).abs() < 1e-5);

        let res = interp.interp(3.5, 5.0);
        assert!((res - 7.5).abs() < 1e-5);
    }
}
