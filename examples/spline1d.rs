use numerics::interp::{spline1d::Spline1D, Interp};

fn main() {
    let x = vec![0.0, 2.0, 3.0, 5.0, 7.0, 8.0];
    let y = vec![4.0, 2.0, 6.0, 8.0, 6.0, 8.0];
    let mut sp1d = Spline1D::new(&x, &y).unwrap();

    for i in 0..100 {
        let xx = 8.0 * (i as f64) / 100.0;
        let yy = sp1d.interp(xx).unwrap();
        println!("[{}, {}],", xx, yy);
    }
}
