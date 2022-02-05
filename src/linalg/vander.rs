use crate::MatLinAlgBound;

pub fn vander<T>(x: &Vec<T>, w: &mut Vec<T>, q: &Vec<T>)
where
    T: MatLinAlgBound,
{
    let n = q.len();
    let mut c = vec![T::zero(); n];
    for i in 0..n {
        let xx: T = (-x[i]).into();
        for j in (n - i - 1)..(n - 1) {
            let val = c[j + 1];
            c[j] += (val * xx).into();
        }
        c[n - 1] += xx;
    }
    for i in 0..n {
        let xx = x[i];
        let mut b: T = (1.0).into();
        let mut s = q[n - 1];
        let mut t: T = (1.0).into();
        for j in (1..n).rev() {
            b = (c[j] + T::from(xx * b)).into();
            s += (q[j - 1] * b).into();
            t = (T::from(xx * t) + b).into();
        }
        w[i] = (s / t).into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vander() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut w = vec![0.0, 0.0, 0.0];
        let q = vec![9.0, 20.0, 50.0];
        // let q = vec![3.0, 6.0, 14.0];
        vander(&x, &mut w, &q);
        println!("{:?}", w);
        assert!((w[0] - 2.0).abs() < 1e-5);
        assert!((w[1] - 3.0).abs() < 1e-5);
        assert!((w[2] - 4.0).abs() < 1e-5);
    }
}
