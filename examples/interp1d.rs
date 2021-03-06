/// Interporation samples
use std::{env, path::PathBuf};

use numerics::interp::{
    barycentric_1d::BaryRat1D, poly_1d::Poly1D, rational_1d::Rational1D, spline1d::Spline1D, Interp,
};

fn interp_poly(mm: usize) -> Vec<(f64, f64)> {
    let func = |x: f64| (x * x + 2.0 * x + 1.0) / (x + 2.0);
    let xx = vec![1.0, 2.0, 3.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let yy = vec![
        func(1.0),
        func(2.0),
        func(3.0),
        func(6.0),
        func(8.0),
        func(9.0),
        func(10.0),
        func(11.0),
        func(12.0),
    ];
    let mut interp = Poly1D::new(&xx, &yy, mm);
    let xy: Vec<(f64, f64)> = (0..120)
        .map(|i| {
            let x = i as f64 / 10.0;
            (x, interp.interp(x).unwrap())
        })
        .collect();
    xy
}

fn interp_rati() -> Vec<(f64, f64)> {
    let func = |x: f64| (x * x + 2.0 * x + 1.0) / (x + 2.0);
    let xx = vec![1.0, 2.0, 3.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let yy = vec![
        func(1.0),
        func(2.0),
        func(3.0),
        func(6.0),
        func(8.0),
        func(9.0),
        func(10.0),
        func(11.0),
        func(12.0),
    ];
    let mut ra1d = Rational1D::new(&xx, &yy, 4);
    let xy: Vec<(f64, f64)> = (0..120)
        .map(|i| {
            let x = i as f64 / 10.0;
            (x, ra1d.interp(x).unwrap())
        })
        .collect();
    xy
}

fn interp_barycentric(mm: usize, calc: usize) -> Vec<(f64, f64)> {
    let func = |x: f64| (x * x + 2.0 * x + 1.0) / (x + 2.0);
    let xx = vec![1.0, 2.0, 3.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let yy = vec![
        func(1.0),
        func(2.0),
        func(3.0),
        func(6.0),
        func(8.0),
        func(9.0),
        func(10.0),
        func(11.0),
        func(12.0),
    ];
    let mut inter = BaryRat1D::new(&xx, &yy, mm, calc);
    let xy: Vec<(f64, f64)> = (0..120)
        .map(|i| {
            let x = i as f64 / 10.0;
            (x, inter.interp(x).unwrap())
        })
        .collect();
    xy
}

fn interp_spline() -> Vec<(f64, f64)> {
    let func = |x: f64| (x * x + 2.0 * x + 1.0) / (x + 2.0);
    let xx = vec![1.0, 2.0, 3.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let yy = vec![
        func(1.0),
        func(2.0),
        func(3.0),
        func(6.0),
        func(8.0),
        func(9.0),
        func(10.0),
        func(11.0),
        func(12.0),
    ];
    let mut inter = Spline1D::new(&xx, &yy).unwrap();
    let xy: Vec<(f64, f64)> = (0..120)
        .map(|i| {
            let x = i as f64 / 10.0;
            (x, inter.interp(x).unwrap())
        })
        .collect();
    xy
}

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("args = {:?}", args);
    let (xy, filename) = match args.get(1).unwrap_or(&"".to_string()).as_str() {
        "barycentric" => {
            let mm = args
                .get(2)
                .unwrap_or(&"4".to_string())
                .parse::<usize>()
                .unwrap();
            let calc = args
                .get(3)
                .unwrap_or(&"1".to_string())
                .parse::<usize>()
                .unwrap();
            let xy = interp_barycentric(mm, calc);
            let filename = format!("barycentric{}_{}.csv", mm, calc);
            (xy, filename)
        }
        "rational" => {
            let xy = interp_rati();
            let filename = format!("rational1d");
            (xy, filename)
        }
        "poly1d" => {
            let mm = args
                .get(2)
                .unwrap_or(&"2".to_string())
                .parse::<usize>()
                .unwrap();
            let xy = interp_poly(mm);
            let filename = format!("poly1d{}.csv", mm);
            (xy, filename)
        }
        "spline" => {
            let xy = interp_spline();
            let filename = format!("spline1d.csv");
            (xy, filename)
        }
        _ => {
            let xy = interp_rati();
            let filename = format!("rational1d");
            (xy, filename)
        }
    };

    let mut out_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out_file.push(format!("output/interp/{}", filename));
    let out_dir = out_file.parent().unwrap();
    if !out_dir.exists() {
        if let Err(_) = std::fs::create_dir_all(out_dir.clone()) {
            println!("Failed to create directory : {:?}", out_dir.to_str());
            return;
        }
    }
    let mut wtr = csv::Writer::from_path(out_file.clone()).unwrap();
    for (x, y) in xy {
        wtr.write_record(&[x.to_string(), y.to_string()]).unwrap();
    }
    wtr.flush().unwrap();
    println!("Save to {}", out_file.to_str().unwrap());
}
