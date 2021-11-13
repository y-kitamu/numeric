use crate::{
    interp::{spline1d::Spline1D, Interp},
    Matrix,
};

pub struct Spline2D<'a> {
    m: usize,
    n: usize,
    y: &'a Matrix<f64>,
    x1: &'a Vec<f64>,
    srp: Vec<Spline1D<'a>>,
}

impl<'a> Spline2D<'a> {
    pub fn new(x1v: &'a Vec<f64>, x2v: &'a Vec<f64>, ym: &'a Matrix<f64>) -> Self {
        let m = x1v.len();
        let srp = (0..m)
            .map(|i| Spline1D::new(x2v, &ym[i]).unwrap())
            .collect();
        Self {
            m,
            n: x2v.len(),
            y: ym,
            x1: x1v,
            srp,
        }
    }

    pub fn interp(&mut self, x1p: f64, x2p: f64) -> f64 {
        let yv: Vec<f64> = self
            .srp
            .iter_mut()
            .map(|srp| srp.interp(x2p).unwrap())
            .collect();
        // println!("len srp = {}, yv = {:?}", self.srp.len(), yv);
        let mut scol = Spline1D::new(self.x1, &yv).unwrap();
        scol.interp(x1p).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spline() {
        let x1 = vec![0.0, 2.0, 3.0, 5.0, 7.0, 8.0];
        let x2 = vec![0.0, 2.0, 3.0, 5.0, 7.0, 8.0];
        #[rustfmt::skip]
        let y = Matrix::new(6, 6, vec![
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0
        ]);
        let mut sp2d = Spline2D::new(&x1, &x2, &y);
        let y = sp2d.interp(6.0, 6.0);
        assert!((y - 6.50358166).abs() < 1e-5, "y = {}", y);

        let x1 = vec![0.0, 2.0, 3.0, 5.0, 7.0];
        let x2 = vec![0.0, 2.0, 3.0, 5.0, 7.0, 8.0];
        #[rustfmt::skip]
        let y = Matrix::new(6, 6, vec![
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
            4.0, 2.0, 6.0, 8.0, 6.0, 8.0,
        ]);
        let mut sp2d = Spline2D::new(&x1, &x2, &y);
        let y = sp2d.interp(4.0, 6.0);
        assert!((y - 6.50358166).abs() < 1e-5, "y = {}", y);
    }
}
