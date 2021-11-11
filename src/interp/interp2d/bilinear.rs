use crate::{
    interp::{linear::LinearInterp, Interp},
    Matrix,
};

pub struct Bilinear<'a> {
    m: usize,
    n: usize,
    y: &'a Matrix<f64>,
    x1interp: LinearInterp<'a>,
    x2interp: LinearInterp<'a>,
}

impl<'a> Bilinear<'a> {
    fn new(x1v: &'a Vec<f64>, x2v: &'a Vec<f64>, y: &'a Matrix<f64>) -> Self {
        Bilinear {
            m: x1v.len(),
            n: x2v.len(),
            y,
            x1interp: LinearInterp::new(x1v, x1v),
            x2interp: LinearInterp::new(x2v, x2v),
        }
    }

    fn interp(&mut self, x1p: f64, x2p: f64) -> f64 {
        let i = if self.x1interp.cor() == 1 {
            self.x1interp.hunt(x1p)
        } else {
            self.x1interp.locate(x1p)
        };
        let j = if self.x2interp.cor() == 1 {
            self.x2interp.hunt(x2p)
        } else {
            self.x2interp.locate(x2p)
        };
        let t = (x1p - self.x1interp.xx()[i]) / (self.x1interp.xx()[i + 1] - self.x1interp.xx()[i]);
        let u = (x2p - self.x2interp.xx()[j]) / (self.x2interp.xx()[j + 1] - self.x2interp.xx()[j]);
        // println!("i = {}, j = {}, t = {}, u = {}", i, j, t, u);
        let yy = (1.0 - t) * (1.0 - u) * self.y[i][j]
            + t * (1.0 - u) * self.y[i + 1][j]
            + (1.0 - t) * u * self.y[i][j + 1]
            + t * u * self.y[i + 1][j + 1];
        yy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear() {
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
        let mut interp = Bilinear::new(&x1, &x2, &y);

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
