use crate::MatLinAlgBound;

/// Sparse vector data structure
pub struct NRsparseCol<T>
where
    T: MatLinAlgBound,
{
    nrows: usize,
    nvals: usize,        // maximum number of non-zeros
    row_ind: Vec<usize>, // row indices of non-zeros
    val: Vec<T>,         // Array of non-zero values.
}

impl<T> NRsparseCol<T>
where
    T: MatLinAlgBound,
{
    pub fn new(nrows: usize, nvals: usize) -> Self {
        NRsparseCol {
            nrows,
            nvals,
            row_ind: vec![0; nvals],
            val: vec![T::zero(); nvals],
        }
    }

    pub fn resize(&mut self, nrows: usize, nvals: usize) {
        self.nrows = nrows;
        self.nvals = nvals;
        self.row_ind = vec![0; nvals];
        self.val = vec![T::zero(); nvals];
    }
}

/// Sparse matrix data structure
pub struct NRsparseMat<T>
where
    T: MatLinAlgBound,
{
    nrows: usize,
    ncols: usize,
    nvals: usize,
    col_ptr: Vec<usize>,
    row_ind: Vec<usize>,
    val: Vec<T>,
}

impl<T> NRsparseMat<T>
where
    T: MatLinAlgBound,
{
    pub fn new(nrows: usize, ncols: usize, nvals: usize) -> Self {
        NRsparseMat {
            nrows,
            ncols,
            nvals,
            col_ptr: vec![0; ncols + 1],
            row_ind: vec![0; nvals],
            val: vec![T::zero(); nvals],
        }
    }

    /// Multiply A by a vector x.
    pub fn ax(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.nrows];
        for j in 0..self.ncols {
            for i in self.col_ptr[j]..self.col_ptr[j + 1] {
                y[self.row_ind[i]] += (x[j] * self.val[i]).into();
            }
        }
        y
    }

    /// Mutiply transpose of A by a vector x.
    pub fn atx(&self, x: &[T]) -> Vec<T> {
        let mut y = vec![T::zero(); self.ncols];
        for j in 0..self.ncols {
            for i in self.col_ptr[j]..self.col_ptr[j + 1] {
                y[j] += (x[self.row_ind[i]] * self.val[i]).into();
            }
        }
        y
    }

    pub fn transpose(&self) -> Self {
        let m = self.nrows;
        let n = self.ncols;
        let mut at = NRsparseMat::new(n, m, self.nvals);
        let mut count = vec![0; m];
        for i in 0..n {
            for j in self.col_ptr[i]..self.col_ptr[i + 1] {
                count[self.row_ind[j]] += 1;
            }
        }
        for j in 0..m {
            at.col_ptr[j + 1] = at.col_ptr[j] + count[j];
            count[j] = 0;
        }
        for i in 0..n {
            for j in self.col_ptr[i]..self.col_ptr[i + 1] {
                let k = self.row_ind[j];
                let index = at.col_ptr[k] + count[k];
                at.row_ind[index] = i;
                at.val[index] = self.val[j];
                count[k] += 1;
            }
        }
        at
    }
}
