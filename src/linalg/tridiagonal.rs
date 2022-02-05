use anyhow::Result;

use crate::MatLinAlgBound;

use super::LinAlgError;

/// Solves for a vector u[0..n-1] the tridiagonal linear set given by the equation:
pub fn tridiag<T>(a: &Vec<T>, b: &Vec<T>, c: &Vec<T>, r: &Vec<T>, u: &mut Vec<f32>) -> Result<()>
where
    T: MatLinAlgBound,
{
    if b[0] == T::zero() {
        return Err(LinAlgError::ZeroDiagonalElemement(0))?;
    }
    let n = a.len();
    let mut bet = b[0].to_f32().unwrap();
    let mut gam = vec![0.0; n];
    u[0] = (r[0].to_f32().unwrap() / bet).into();

    for j in 1..n {
        gam[j] = c[j - 1].to_f32().unwrap() / bet;
        bet = b[j].to_f32().unwrap() - a[j].to_f32().unwrap() * gam[j];
        if bet.abs() < 1e-10 {
            return Err(LinAlgError::ZeroDivision())?;
        }
        let temp = a[j].to_f32().unwrap() * u[j - 1];
        u[j] = (r[j].to_f32().unwrap() - temp) / bet;
    }

    for j in (0..n - 1).rev() {
        let temp = gam[j + 1] * u[j + 1];
        u[j] -= temp;
    }

    Ok(())
}

pub fn cyclic<T>(
    a: &Vec<T>,
    b: &Vec<T>,
    c: &Vec<T>,
    alpha: T,
    beta: T,
    r: &Vec<T>,
    x: &mut Vec<f32>,
) -> Result<()>
where
    T: MatLinAlgBound,
{
    let n = a.len();
    if n <= 2 {
        return Err(LinAlgError::InvalidVectorSize(n))?;
    }

    let mut bb = vec![T::zero(); n];
    let gamma: T = (-b[0]).into();
    bb[0] = (b[0] - gamma).into();
    bb[n - 1] = (b[n - 1] - (T::from(alpha * beta) / gamma).into()).into();

    for i in 1..(n - 1) {
        bb[i] = b[i];
    }
    if let Err(err) = tridiag(a, &bb, c, r, x) {
        return Err(err);
    }
    let mut u = vec![T::zero(); n];
    u[0] = gamma;
    u[n - 1] = alpha;
    let mut z = vec![0.0f32; n];
    if let Err(err) = tridiag(a, &bb, c, &u, &mut z) {
        return Err(err);
    }
    let fact = (x[0] + beta.to_f32().unwrap() * x[n - 1] / gamma.to_f32().unwrap())
        / (1.0 + z[0] + beta.to_f32().unwrap() * z[n - 1] / gamma.to_f32().unwrap());
    for i in 0..n {
        x[i] -= fact * z[i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tridiag() {
        // Case : normal
        let a = vec![0.0, 1.0, 1.0, 1.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0, 1.0, 0.0];
        let r = vec![4.0, 8.0, 12.0, 11.0];
        let mut u = vec![0.0f32; 4];

        let res = tridiag(&a, &b, &c, &r, &mut u);
        assert!(res.is_ok(), "{:?}", res);

        println!("u = {:?}", u);
        assert!((u[0] - 1.0).abs() < 1e-5, "u[0] = {}", u[0]);
        assert!((u[1] - 2.0).abs() < 1e-5, "u[0] = {}", u[1]);
        assert!((u[2] - 3.0).abs() < 1e-5, "u[0] = {}", u[2]);
        assert!((u[3] - 4.0).abs() < 1e-5, "u[0] = {}", u[3]);

        // Case : error
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let res = tridiag(&a, &b, &c, &r, &mut u);
        assert!(res.is_err());

        // Case : error
        let b = vec![0.0, 1.0, 1.0, 1.0];
        let res = tridiag(&a, &b, &c, &r, &mut u);
        assert!(res.is_err());
    }

    #[test]
    fn test_cyclic() {
        let a = vec![0.0, 1.0, 1.0, 1.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0, 1.0, 0.0];
        let r = vec![8.0, 8.0, 12.0, 13.0];
        let alpha = 2.0;
        let beta = 1.0;
        let mut x = vec![0.0f32; 4];
        let res = cyclic(&a, &b, &c, alpha, beta, &r, &mut x);
        assert!(res.is_ok());

        println!("x = {:?}", x);
        assert!((x[0] - 1.0).abs() < 1e-5, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-5, "x[0] = {}", x[1]);
        assert!((x[2] - 3.0).abs() < 1e-5, "x[0] = {}", x[2]);
        assert!((x[3] - 4.0).abs() < 1e-5, "x[0] = {}", x[3]);
    }
}
