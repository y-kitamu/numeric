use std::{
    ops::{Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    ptr,
};

use num::ToPrimitive;

pub mod linalg;

pub trait MatLinAlgBound:
    Copy
    + Clone
    + ToPrimitive
    + Mul
    + MulAssign
    + Sub
    + SubAssign
    + Div
    + DivAssign
    + PartialOrd
    + From<f32>
    + From<<Self as std::ops::Mul>::Output>
    + From<<Self as std::ops::Sub>::Output>
    + From<<Self as std::ops::Div>::Output>
{
}

impl MatLinAlgBound for f32 {}
impl MatLinAlgBound for f64 {}

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
    fn new(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        Matrix { nrows, ncols, data }
    }

    pub fn rows(&self) -> usize {
        self.nrows
    }

    pub fn cols(&self) -> usize {
        self.ncols
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
