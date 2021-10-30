use anyhow::Result;

use crate::MatLinAlgBound;

use super::LinAlgError;

fn toeplz<T>(r: &Vec<T>, x: &mut Vec<T>, y: &Vec<T>) -> Result<()>
where
    T: MatLinAlgBound,
{
    let n = y.len();
    let n1 = n - 1;
    if r[n1].to_f32().unwrap().abs() < 1e-7 {
        return Err(LinAlgError::ZeroDiagonalElemement(0))?;
    }

    x[0] = (y[0] / r[n1]).into();
    if n1 == 0 {
        return Ok(());
    }
    let mut g = vec![T::zero(); n1];
    let mut h = vec![T::zero(); n1];
    g[0] = (r[n1 - 1] / r[n1]).into();
    h[0] = (r[n1 + 1] / r[n1]).into();
    for m in 0..n {
        let m1 = m + 1;
        let mut sxn: T = (-y[m1]).into();
        let mut sd: T = (-r[n1]).into();
        for j in 0..(m + 1) {
            sxn += (r[n1 + m1 - j] * x[j]).into();
            sd += (r[n1 + m1 - j] * g[m - j]).into();
        }
        if sd.to_f32().unwrap().abs() < 1e-7 {
            return Err(LinAlgError::SingularPrincipleMinor())?;
        }
        x[m1] = (sxn / sd).into();
        for j in 0..(m + 1) {
            let tmp = x[m1] * g[m - j];
            x[j] -= tmp.into();
        }
        if m1 == n1 {
            return Ok(());
        }
        let mut sgn: T = (-r[n1 - m1 - 1]).into();
        let mut shn: T = (-r[n1 + m1 + 1]).into();
        let mut sgd: T = (-r[n1]).into();
        for j in 0..(m + 1) {
            sgn += (r[n1 + j - m1] * g[j]).into();
            shn += (r[n1 + m1 - j] * h[j]).into();
            sgd += (r[n1 + j - m1] * h[m - j]).into();
        }
        if sgd.to_f32().unwrap().abs() < 1e-7 {
            return Err(LinAlgError::SingularPrincipleMinor())?;
        }
        g[m1] = (sgn / sgd).into();
        h[m1] = (shn / sd).into();

        let mut k = m;
        let m2 = (m + 2) >> 1;
        let pp = g[m1];
        let qq = h[m1];
        for j in 0..m2 {
            let pt1 = g[j];
            let pt2 = g[k];
            let qt1 = h[j];
            let qt2 = h[k];
            g[j] = (pt1 - T::from(pp * qt2)).into();
            g[k] = (pt2 - T::from(pp * qt1)).into();
            h[j] = (qt1 - T::from(qq * qt2)).into();
            h[k] = (qt2 - T::from(qq * qt1)).into();
            if k > 0 {
                k -= 1;
            }
        }
    }
    return Err(LinAlgError::ShouldNotArriveHere())?;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toeplz() {
        let r = vec![1.0, 2.0, 3.0, 4.0, 0.0];
        let mut x = vec![0.0; 3];
        let y = vec![10.0, 16.0, 17.0f64];
        let res = toeplz(&r, &mut x, &y);
        assert!(res.is_ok());
        println!("{:?}", x);
        assert!((x[0] - 1.0).abs() < 1e-7);
        assert!((x[1] - 2.0).abs() < 1e-7);
        assert!((x[2] - 3.0).abs() < 1e-7);
    }
}
