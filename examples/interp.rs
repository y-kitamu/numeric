/// Interporation samples
use std::{io, path::PathBuf};

use numerics::interp::{rational_1d::Rational1D, Interp};

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

fn main() {
    let xy = interp_rati();

    let filename = "rational1d.csv";
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
