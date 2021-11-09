pub fn poly_coeff(x: &Vec<f64>, y: &Vec<f64>, coeff: &mut Vec<f64>) {
    let n = x.len();
    let mut factor = vec![0.0; n + 1];
    let mut deno = vec![1.0; n];

    factor[n] = 1.0;
    for i in 0..n {
        coeff[i] = 0.0;
        for j in (n - i)..n {
            let tmp = factor[j + 1] - factor[j] * x[i];
            factor[j] = tmp;
        }
        factor[n] *= -x[i];
        factor[n - i - 1] = 1.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            deno[i] *= x[i] - x[j];
        }
    }

    for i in 0..n {
        let mut nume = 1.0;
        for j in 0..n {
            coeff[n - j - 1] += y[i] * nume / deno[i];
            nume = factor[j + 1] + nume * x[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_coeff() {
        let xx = vec![2.0, 3.0, 4.0];
        let yy = vec![17.0, 34.0, 57.0];
        let mut coeff = vec![0.0; 3];

        poly_coeff(&xx, &yy, &mut coeff);
        println!("coeff = {:?}", coeff);
        assert!((coeff[0] - 1.0).abs() < 1e-5);
        assert!((coeff[1] - 2.0).abs() < 1e-5);
        assert!((coeff[2] - 3.0).abs() < 1e-5);
    }
}
