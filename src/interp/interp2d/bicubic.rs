use anyhow::Result;

use crate::{interp::InterpError, Matrix};

fn bcucof(
    y: &Vec<f64>,
    y1: &Vec<f64>,
    y2: &Vec<f64>,
    y12: &Vec<f64>,
    d1: f64,
    d2: f64,
    c: &mut Matrix<f64>,
) {
    #[rustfmt::skip]
    let wt_d: Vec<isize> = vec![
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0,
        2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1,
        0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1,
        -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
        9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2,
        -6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2,
        2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0,
        -6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1,
        4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1,
    ];
    let d1d2 = d1 * d2;
    let mut cl: Vec<f64> = vec![0.0; 16];
    let mut x: Vec<f64> = vec![0.0; 16];
    let wt = Matrix::new(16, 16, wt_d);
    for i in 0..4 {
        x[i] = y[i];
        x[i + 4] = y1[i] * d1;
        x[i + 8] = y2[i] * d2;
        x[i + 12] = y12[i] * d1d2;
    }
    for i in 0..16 {
        let mut xx = 0.0;
        for k in 0..16 {
            xx += wt[i][k] as f64 * x[k];
        }
        cl[i] = xx;
    }
    for i in 0..4 {
        for j in 0..4 {
            c[i][j] = cl[i * 4 + j];
        }
    }
}

pub fn bcuint(
    y: &Vec<f64>,
    y1: &Vec<f64>,
    y2: &Vec<f64>,
    y12: &Vec<f64>,
    x1l: f64,
    x1u: f64,
    x2l: f64,
    x2u: f64,
    x1: f64,
    x2: f64,
) -> Result<(f64, f64, f64)> {
    let d1 = x1u - x1l;
    let d2 = x2u - x2l;
    let mut c = Matrix::new(4, 4, vec![0.0; 16]);
    bcucof(y, y1, y2, y12, d1, d2, &mut c);
    if x1u == x1l || x2u == x2l {
        return Err(InterpError::IdenticalX())?;
    }
    let t = (x1 - x1l) / d1;
    let u = (x2 - x2l) / d2;
    let mut y = 0.0;
    let mut y1 = 0.0;
    let mut y2 = 0.0;
    for i in (0..3).rev() {
        y = t * y + ((c[i][3] * u + c[i][2]) * u + c[i][i]) * u + c[i][0];
        y2 = t * y2 + (3.0 * c[i][3] * u + 2.0 * c[i][2]) * u + c[i][1];
        y1 = t * y1 + (3.0 * c[3][i] * t + 2.0 * c[2][i]) * t + c[1][i];
    }
    y1 /= d1;
    y2 /= d2;
    Ok((y, y1, y2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bicubic() {
        let y: Vec<f64> = vec![0.0, 1.0, 5.0, 4.0];
        let y1: Vec<f64> = vec![0.0, 2.0, 2.0, 0.0];
        let y2: Vec<f64> = vec![0.0, 0.0, 4.0, 4.0];
        let y12: Vec<f64> = vec![0.0, 2.0, 6.0, 4.0];
        let res = bcuint(&y, &y1, &y2, &y12, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5);
        assert!(res.is_ok());
        // println!("ay = {}, ay1 = {}, ay2 = {}", ay, ay1, ay2);
        // assert!((ay - 0.75).abs() < 1e-5);
        // assert!((ay1 - 1.0).abs() < 1e-5);
        // assert!((ay2 - 2.0).abs() < 1e-5);
    }
}
