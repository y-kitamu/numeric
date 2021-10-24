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

pub struct ADAT<'a, T>
where
    T: MatLinAlgBound,
{
    a: &'a NRsparseMat<T>,
    at: &'a NRsparseMat<T>,
    adat: NRsparseMat<T>,
}

impl<'a, T> ADAT<'a, T>
where
    T: MatLinAlgBound,
{
    pub fn new(a: &'a NRsparseMat<T>, at: &'a NRsparseMat<T>) -> Self {
        let m = at.ncols;
        let mut done = vec![usize::max_value(); m];
        let mut nvals = 0;
        for j in 0..m {
            for i in at.col_ptr[j]..at.col_ptr[j + 1] {
                let k = at.row_ind[i];
                for l in a.col_ptr[k]..a.col_ptr[k + 1] {
                    let h = a.row_ind[l];
                    if done[h] != j {
                        done[h] = j;
                        nvals += 1;
                    }
                }
            }
        }
        let mut adat = NRsparseMat::new(m, m, nvals);
        let mut done = vec![usize::max_value(); m];
        let mut nvals = 0;
        for j in 0..m {
            adat.col_ptr[j] = nvals;
            for i in at.col_ptr[j]..at.col_ptr[j + 1] {
                let k = at.row_ind[i];
                for l in a.col_ptr[k]..a.col_ptr[k + 1] {
                    let h = a.row_ind[l];
                    if done[h] != j {
                        adat.row_ind[nvals] = h;
                        done[h] = j;
                        nvals += 1;
                    }
                }
            }
        }
        adat.col_ptr[m] = nvals;
        for j in 0..m {
            let i = adat.col_ptr[j];
            let size = adat.col_ptr[j + 1] - i;
            if size > 1 {
                adat.row_ind[i..(i + size)].sort();
            }
        }
        ADAT { a, at, adat }
    }

    pub fn updateD(&mut self, D: &Vec<T>) {
        let m = self.a.nrows;
        let n = self.a.ncols;
        let mut temp = vec![T::zero(); n];
        let mut temp2 = vec![T::zero(); n];
        for i in 0..m {
            for j in self.at.col_ptr[i]..self.at.col_ptr[i + 1] {
                let k = self.at.row_ind[j];
                temp[k] = (self.at.val[j] * D[k]).into();
            }
            for j in self.at.col_ptr[i]..self.at.col_ptr[i + 1] {
                let k = self.at.row_ind[j];
                for l in self.a.col_ptr[k]..self.a.col_ptr[k + 1] {
                    let h = self.a.row_ind[l];
                    temp2[h] += (temp[k] * self.a.val[l]).into();
                }
            }
            for j in self.adat.col_ptr[i]..self.adat.col_ptr[i + 1] {
                let k = self.adat.row_ind[j];
                self.adat.val[j] = temp2[k];
                temp2[k] = T::zero();
            }
        }
    }

    pub fn reference(&'a self) -> &'a NRsparseMat<T> {
        &self.adat
    }
}
