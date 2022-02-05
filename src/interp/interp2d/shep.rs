/// Implementation of Shepard Interpolation
/// (Section 3.7.3 of Numerical Recipes in C)
use crate::{interp::InterpError, Matrix};
use anyhow::Result;

/// Struct for Shep interpolation
pub struct Shep<'a> {
    /// Given scattered points (2d-array). \[n_point, n_dimension\]
    pts: &'a Matrix<f64>,
    /// values of the points corresponding to `pts`. length = n_point
    vals: &'a [f64],
    /// desired exponent
    power: f64,
}

impl<'a> Shep<'a> {
    pub fn new(pts: &'a Matrix<f64>, vals: &'a [f64], power: f64) -> Self {
        Self { pts, vals, power }
    }

    pub fn interp(&self, pt: &[f64]) -> Result<f64> {
        if pt.len() != self.pts.cols() {
            return Err(InterpError::SizeNotMatch())?;
        }
        let dists: Vec<f64> = (0..self.pts.rows())
            .map(|idx| {
                let square_dist: f64 = self.pts[idx]
                    .iter()
                    .zip(pt.iter())
                    .map(|(x1, x2)| (x2 - x1) * (x2 - x1))
                    .sum();
                square_dist.sqrt()
            })
            .collect();
        for (dist, val) in dists.iter().zip(self.vals.iter()) {
            if *dist < 1e-8 {
                return Ok(*val);
            }
        }
        let powers: Vec<f64> = dists.iter().map(|d| d.powf(-self.power)).collect();
        let deno: f64 = powers.iter().sum();
        let nume: f64 = powers
            .iter()
            .zip(self.vals.iter())
            .map(|(d, v)| d * v)
            .sum();
        Ok(nume / deno)
    }
}

#[cfg(test)]
mod tests {
    use num::ToPrimitive;

    use super::*;

    #[test]
    fn test_shep() {
        // 4 points (0, 0), (1, 0), (1, 1), (0, 1)
        let pts = Matrix::new(4, 2, vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let vals = vec![0.0, 1.0, 2.0, 1.0];

        (1..3).for_each(|pow| {
            let shep = Shep::new(&pts, &vals, pow.to_f64().unwrap());
            let res = shep.interp(&[0.5, 0.5]).unwrap();
            assert!((res - 1.0).abs() < 1e-5, "{}", res);
            let res = shep.interp(&[0.0, 1.0 + 1e-7]).unwrap();
            assert!((res - 1.0).abs() < 1e-5, "{}", res);
            let res = shep.interp(&[0.0, 0.0 + 1e-7]).unwrap();
            assert!((res - 0.0).abs() < 1e-5, "{}", res);
            let res = shep.interp(&[1.0, 1.0 + 1e-7]).unwrap();
            assert!((res - 2.0).abs() < 1e-5, "{}", res);
        })
    }
}
