use std::{
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    ptr,
};

use anyhow::Result;
use num::{ToPrimitive, Zero};
use thiserror::Error;

pub mod interp;
pub mod linalg;
mod macros;

pub trait MatLinAlgBound:
    Copy
    + Clone
    + ToPrimitive
    + Zero
    + Add
    + AddAssign
    + Mul
    + MulAssign
    + Sub
    + SubAssign
    + Div
    + DivAssign
    + PartialOrd
    + Neg
    + From<f32>
    + From<<Self as Neg>::Output>
    + From<<Self as Add>::Output>
    + From<<Self as Mul>::Output>
    + From<<Self as Sub>::Output>
    + From<<Self as Div>::Output>
{
    fn abs(self) -> Self {
        let val = self.clone();
        if val < Self::zero() {
            return (-val).into();
        }
        self.clone()
    }

    fn sqrt(self) -> Self {
        let val = self.clone().to_f32().unwrap();
        val.sqrt().into()
    }

    fn square(self) -> Self {
        (self * self).into()
    }

    fn copysign(self, sign: Self) -> Self {
        if sign < Self::zero() {
            if self < Self::zero() {
                self
            } else {
                (-self).into()
            }
        } else {
            if self < Self::zero() {
                (-self).into()
            } else {
                self
            }
        }
    }

    fn norm(self, rhs: Self) -> Self {
        Self::from(Self::from(self * self) + Self::from(rhs * rhs)).sqrt()
    }
}

impl MatLinAlgBound for f32 {}
impl MatLinAlgBound for f64 {}

#[derive(Error, Debug)]
pub enum MatrixOpsError {
    #[error("Invalid matrix size of `{0}`. (row, col) = ({1}, {2}).")]
    InvalidMatrixSize(String, usize, usize),
}

#[derive(Debug, Clone)]
pub struct Matrix<T>
where
    T: Clone,
{
    nrows: usize,
    ncols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Clone,
{
    pub fn new(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        Matrix { nrows, ncols, data }
    }

    pub fn rows(&self) -> usize {
        self.nrows
    }

    pub fn cols(&self) -> usize {
        self.ncols
    }

    pub fn get_col(&self, idx: usize) -> Vec<T> {
        (0..self.nrows).map(|i| self[i][idx].clone()).collect()
    }

    pub fn resize(&mut self, new_rows: usize, new_cols: usize) {
        self.nrows = new_rows;
        self.ncols = new_cols;
    }

    pub fn assign(&mut self, new_rows: usize, new_cols: usize, val: T) {
        self.nrows = new_rows;
        self.ncols = new_cols;
        self.data = vec![val; self.nrows * self.ncols];
    }

    pub fn swap_rows(&mut self, rhs: usize, lhs: usize) {
        unsafe {
            ptr::swap_nonoverlapping(
                self[lhs].as_mut_ptr(),
                self[rhs].as_mut_ptr(),
                self[rhs].len(),
            );
        }
    }

    pub fn swap_cols(&mut self, rhs: usize, lhs: usize) {
        if lhs >= self.ncols || rhs >= self.ncols {
            return;
        }
        unsafe {
            for i in 0..self.nrows {
                ptr::swap(self[i].as_mut_ptr().add(rhs), self[i].as_mut_ptr().add(lhs));
            }
        }
    }
}

impl<T> Matrix<T>
where
    T: Clone + Zero + From<f32>,
{
    pub fn identity(n: usize) -> Self {
        let mut data = vec![T::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = 1.0.into();
        }
        Matrix {
            nrows: n,
            ncols: n,
            data,
        }
    }

    pub fn pseudo_identity(nrows: usize, ncols: usize) -> Self {
        let mut data = vec![T::zero(); nrows * ncols];
        for i in 0..(nrows.min(ncols)) {
            data[i * ncols + i] = 1.0.into();
        }
        Matrix { nrows, ncols, data }
    }
}

impl<T> Index<usize> for Matrix<T>
where
    T: Clone,
{
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.ncols..(index + 1) * self.ncols]
    }
}

impl<T> IndexMut<usize> for Matrix<T>
where
    T: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index * self.ncols..(index + 1) * self.ncols]
    }
}

impl<T> Add for Matrix<T>
where
    T: Copy + Clone + AddAssign,
{
    type Output = Result<Self>;

    fn add(mut self, rhs: Self) -> Self::Output {
        if self.nrows != rhs.rows() || self.ncols != rhs.cols() {
            return Err(MatrixOpsError::InvalidMatrixSize(
                "rhs".to_string(),
                rhs.rows(),
                rhs.cols(),
            ))?;
        }
        self += rhs;
        Ok(self)
    }
}

impl<T> AddAssign for Matrix<T>
where
    T: Copy + Clone + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        for y in 0..self.nrows {
            let rcol = &rhs[y];
            for x in 0..self.ncols {
                self[y][x] += rcol[x];
            }
        }
    }
}

impl<T> Mul for Matrix<T>
where
    T: Copy + Clone + Mul + From<<T as Mul>::Output> + AddAssign + Zero,
{
    type Output = Result<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.ncols != rhs.rows() {
            return Err(MatrixOpsError::InvalidMatrixSize(
                "rhs".to_string(),
                rhs.rows(),
                rhs.cols(),
            ))?;
        }
        let mut out_vec = vec![T::zero(); self.nrows * rhs.cols()];
        for y in 0..self.nrows {
            for x in 0..rhs.cols() {
                let idx = y * rhs.cols() + x;
                for k in 0..self.ncols {
                    out_vec[idx] += (self[y][k] * rhs[k][x]).into();
                }
            }
        }
        Ok(Matrix::new(self.nrows, rhs.cols(), out_vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_identity() {
        let n = 3;
        let mat: Matrix<f64> = Matrix::identity(n);
        assert_eq!(mat.rows(), n);
        assert_eq!(mat.cols(), n);
        assert!((mat[0][0] - 1.0).abs() < 1e-5);
        assert!((mat[1][1] - 1.0).abs() < 1e-5);
        assert!((mat[2][2] - 1.0).abs() < 1e-5);
        assert!(mat[0][1].abs() < 1e-5);
        assert!(mat[0][2].abs() < 1e-5);
        assert!(mat[1][0].abs() < 1e-5);
        assert!(mat[1][2].abs() < 1e-5);
        assert!(mat[2][0].abs() < 1e-5);
        assert!(mat[2][1].abs() < 1e-5);
    }

    #[test]
    fn test_matrix_mul() {
        let rows = 3;
        let cols = 4;
        let lhs = Matrix::new(
            rows,
            cols,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 3.0, 4.0],
        );

        // Case : Assertion Failure
        let out = lhs.clone() * Matrix::new(1, 1, vec![1.0]);
        assert!(out.is_err());

        // Case : Normal operation in float
        let out = (lhs.clone() * Matrix::new(4, 1, vec![1.0, 2.0, 3.0, 4.0])).unwrap();
        assert!((out[0][0].to_f32().unwrap() - 10.0).abs() < 1e-5);
        assert!((out[1][0].to_f32().unwrap() - 17.0).abs() < 1e-5);
        assert!((out[2][0].to_f32().unwrap() - 30.0).abs() < 1e-5);

        // Case : Normal operation in integer
        let lhs = Matrix::new(rows, cols, vec![1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 3, 4]);
        let out = (lhs.clone() * Matrix::new(4, 1, vec![1, 2, 3, 4])).unwrap();
        assert_eq!(out[0][0].to_isize().unwrap(), 10);
        assert_eq!(out[1][0].to_isize().unwrap(), 17);
        assert_eq!(out[2][0].to_isize().unwrap(), 30);
    }

    #[test]
    fn test_matrix_add() {
        let rows = 3;
        let cols = 4;
        #[rustfmt::skip]
        let lhs = Matrix::new(rows, cols, vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12
        ]);
        let rhs = Matrix::new(1, 1, vec![1]);
        let out = lhs.clone() + rhs;
        assert!(out.is_err());

        let out = lhs.clone() + lhs.clone();
        assert!(out.is_ok());
        let out = out.unwrap();
        assert_eq!(out[0][0], 2);
        assert_eq!(out[0][1], 4);
        assert_eq!(out[0][2], 6);
        assert_eq!(out[0][3], 8);
        assert_eq!(out[1][0], 10);
        assert_eq!(out[1][1], 12);
        assert_eq!(out[1][2], 14);
        assert_eq!(out[1][3], 16);
        assert_eq!(out[2][0], 18);
        assert_eq!(out[2][1], 20);
        assert_eq!(out[2][2], 22);
        assert_eq!(out[2][3], 24);
    }

    #[test]
    fn test_matrix() {
        let rows = 3;
        let cols = 4;
        #[rustfmt::skip]
        let data = vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
        ];
        let mut mat = Matrix::new(rows, cols, data);
        assert_eq!(mat.rows(), rows);
        assert_eq!(mat.cols(), cols);
        assert_eq!(mat[0][0], 1);
        assert_eq!(mat[1].len(), cols);
        assert_eq!(mat[2][3], 12);

        let col = mat.get_col(1);
        assert_eq!(col.len(), 3);
        assert_eq!(col[0], 2);
        assert_eq!(col[1], 6);
        assert_eq!(col[2], 10);

        mat.swap_rows(0, 1);
        assert_eq!(mat[0][0], 5);
        assert_eq!(mat[1][2], 3);

        mat.swap_cols(1, 0);
        assert_eq!(mat[0][0], 6);
        assert_eq!(mat[1][1], 1);

        mat[2][3] = 8;
        assert_eq!(mat[2][3], 8);
        assert_eq!(mat[2][2], 11);

        mat.resize(10, 11);
        assert_eq!(mat.rows(), 10);
        assert_eq!(mat.cols(), 11);

        mat.assign(5, 4, 1);
        assert_eq!(mat.rows(), 5);
        assert_eq!(mat.cols(), 4);
        assert_eq!(mat[0][0], 1);
        assert_eq!(mat[4][3], 1);
    }
}
