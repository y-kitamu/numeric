use crate::{MatLinAlgBound, Matrix};

pub struct LUdcmp<'a, T>
where
    T: MatLinAlgBound,
{
    n: usize,
    lu: Matrix<T>,
    indx: Vec<usize>,
    d: f32,
    aref: &'a Matrix<T>,
}

impl<'a, T> LUdcmp<'a, T>
where
    T: MatLinAlgBound,
{
    fn new(a: &Matrix<T>) -> Self {
        let tiny = 1e-20;

        let n = a.rows();
        let d = 1.0;
        let mut vv = Vec::with_capacity(n);
        for i in 0..n {
            let mut big = 0.0;
            for j in 0..n {
                if big < a[i][j].to_f32().unwrap() {
                    big = a[i][j].to_f32().unwrap();
                }
            }
            vv.push(big);
        }

        for k in 0..n {}
    }

    fn solve(b: &Vec<T>, x: &mut Vec<T>) {}

    fn solve_mat(b: &Matrix<T>, x: &mut Matrix<T>) {}

    fn inverse(ainv: &Matrix<T>) {}

    fn det() -> T {}

    fn mprove(b: &Vec<T>, x: &Vec<T>) {}
}
