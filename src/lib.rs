use std::{
    ops::{Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    ptr,
};

use num::ToPrimitive;

pub mod linalg;

pub trait MatLinAlgBound:
    Copy
    + Clone
    + ToPrimitive
    + MulAssign
    + Mul
    + Sub
    + SubAssign
    + PartialOrd
    + From<f32>
    + From<<Self as std::ops::Mul>::Output>
    + From<<Self as std::ops::Sub>::Output>
{
}

impl MatLinAlgBound for f32 {}
impl MatLinAlgBound for f64 {}

#[derive(Debug)]
pub struct Matrix<T>
where
    T: Clone,
{
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Clone,
{
    fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Matrix { rows, cols, data }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn resize(&mut self, new_rows: usize, new_cols: usize) {
        self.rows = new_rows;
        self.cols = new_cols;
    }

    pub fn assign(&mut self, new_rows: usize, new_cols: usize, val: T) {
        self.rows = new_rows;
        self.cols = new_cols;
        self.data = vec![val; self.rows * self.cols];
    }

    pub fn swap_rows(&mut self, rhs: usize, lhs: usize) {
        unsafe {
            ptr::swap_nonoverlapping(
                self[rhs].as_mut_ptr(),
                self[lhs].as_mut_ptr(),
                self[rhs].len(),
            );
        }
    }
}

impl<T> Index<usize> for Matrix<T>
where
    T: Clone,
{
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.cols..(index + 1) * self.cols]
    }
}

impl<T> IndexMut<usize> for Matrix<T>
where
    T: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index * self.cols..(index + 1) * self.cols]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        mat.swap_rows(0, 1);
        assert_eq!(mat[0][0], 5);
        assert_eq!(mat[1][2], 3);

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
