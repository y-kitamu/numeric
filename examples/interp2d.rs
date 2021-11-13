use std::path::PathBuf;

use clap::Parser;
use numerics::{interp::bicubic_spline::Spline2D, Matrix};

#[derive(Parser)]
#[clap(version = "1.0")]
struct Opts {
    #[clap(short, long, default_value = "default")]
    interp: String,
}

fn spline2d() -> Vec<(f64, f64, f64)> {
    let x1 = vec![0.0, 1.0, 2.0, 4.0, 6.0, 10.0];
    let x2 = vec![0.0, 2.0, 3.0, 5.0, 7.0, 10.0];
    let y: Vec<f64> = x1
        .iter()
        .flat_map(|v1| x2.iter().map(|v2| v1 * v1 + v2 * v2).collect::<Vec<f64>>())
        .collect();
    let y = Matrix::new(x1.len(), x2.len(), y);

    let mut interp = Spline2D::new(&x1, &x2, &y);
    (0..100)
        .flat_map(|i| {
            (0..100)
                .map(|j| {
                    let yy = i as f64 * 0.1;
                    let xx = j as f64 * 0.1;
                    let zz = interp.interp(yy, xx);
                    (xx, yy, zz)
                })
                .collect::<Vec<(f64, f64, f64)>>()
        })
        .collect()
}

fn main() {
    let opts: Opts = Opts::parse();

    let (xyz, filename) = match opts.interp.as_str() {
        "spline" => {
            let filename = "spline2d.csv";
            let xyz = spline2d();
            (xyz, filename)
        }
        _ => {
            let filename = "spline2d.csv";
            let xyz = spline2d();
            (xyz, filename)
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
    for (x, y, z) in xyz {
        wtr.write_record(&[x.to_string(), y.to_string(), z.to_string()])
            .unwrap();
    }
    wtr.flush().unwrap();
    println!("Save to {}", out_file.to_str().unwrap());
}
