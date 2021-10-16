use anyhow::Result;

use crate::MatLinAlgBound;

use super::LinAlgError;

/// Solves for a vector u[0..n-1] the tridiagonal linear set given by the equation:
fn tridiag<T>(a: &Vec<T>, b: &Vec<T>, c: &Vec<T>, r: &Vec<T>, u: &mut Vec<f32>) -> Result<()>
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
}
