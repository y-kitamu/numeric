use crate::MatLinAlgBound;

use super::sparse::NRsparseMat;

/// Trait for solving sparce linear equations by the preconditioned conjugate gradient method
pub trait Linbcg<T>
where
    T: MatLinAlgBound,
{
    fn asolve(&self, b: &Vec<T>, x: &mut Vec<T>, itrnsp: usize);
    fn atimes(&self, x: &Vec<T>, r: &mut Vec<T>, itrnsp: usize);

    fn solve(
        &self,
        b: &Vec<T>,
        x: &mut Vec<T>,
        itol: usize,
        tol: T,
        itmax: usize,
        iter: usize,
        err: T,
    ) {
        let bkden = 1.0;
        let eps = 1e-14;
        let n = b.len();
        let mut p = vec![T::zero(); n];
        let mut pp = vec![T::zero(); n];
        let mut r = vec![T::zero(); n];
        let mut rr = vec![T::zero(); n];
        let mut z = vec![T::zero(); n];
        let mut zz = vec![T::zero(); n];
        let mut iter = 0;
        let mut znrm = T::zero();
        self.atimes(x, &mut r, 0);
        for j in 0..n {
            r[j] = (b[j] - r[j]).into();
            rr[j] = r[j];
        }
        // self.atimes(r, rr, 0);
        let bnrm = match itol {
            1 => {
                self.asolve(&r, &mut z, 0);
                self.snrm(&b, itol)
            }
            2 => {
                self.asolve(&b, &mut z, 0);
                let tmp = self.snrm(&z, itol);
                self.asolve(&r, &mut z, 0);
                tmp
            }
            3 | 4 => {
                self.asolve(&b, &mut z, 0);
                let tmp = self.snrm(&z, itol);
                self.asolve(&r, &mut z, 0);
                znrm = self.snrm(&z, itol);
                tmp
            }
            _ => T::zero(),
        };

        let mut err = T::zero();
        let mut akden = T::zero();
        let mut bknum = T::zero();
        let mut bkden: T = 1.0.into();
        while iter < itmax {
            iter += 1;
            self.asolve(&rr, &mut zz, 1);
            bknum = z.iter().zip(&rr).fold(T::zero(), |acc, (zval, rval)| {
                acc + (zval.clone() * rval.clone()).into()
            });
            if iter == 1 {
                for j in 0..n {
                    p[j] = z[j];
                    pp[j] = zz[j];
                }
            } else {
                let bk: T = (bknum / bkden).into();
                for j in 0..n {
                    p[j] = T::from(bk * p[j]) + z[j];
                    pp[j] = T::from(bk * pp[j]) + zz[j];
                }
            }
            bkden = bknum;
            self.atimes(&p, &mut z, 0);
            akden = z.iter().zip(&pp).fold(T::zero(), |acc, (zval, pval)| {
                acc + (zval.clone() * pval.clone()).into()
            });
            let ak: T = (bknum / akden).into();
            self.atimes(&pp, &mut zz, 1);
            for j in 0..n {
                x[j] += (ak * p[j]).into();
                r[j] -= (ak * z[j]).into();
                rr[j] -= (ak * zz[j]).into();
            }
            self.asolve(&r, &mut z, 0);
            if itol == 1 {
                err = (self.snrm(&r, itol) / bnrm).into();
            } else if itol == 2 {
                err = (self.snrm(&z, itol) / bnrm).into();
            } else if itol == 3 || itol == 4 {
                let zminrm = znrm;
                znrm = self.snrm(&z, itol);
                if T::from(zminrm - znrm).to_f64().unwrap() > znrm.to_f64().unwrap() * eps {
                    let dxnrm: T =
                        (T::from(ak.to_f32().unwrap().abs()) * self.snrm(&p, itol)).into();
                    err = (T::from(znrm / T::from(zminrm - znrm)) * dxnrm).into();
                } else {
                    err = (znrm / bnrm).into();
                    continue;
                }
                let xnrm = self.snrm(&x, itol);
                if err <= T::from(T::from(0.5f32) * xnrm) {
                    err /= xnrm;
                } else {
                    err = (znrm / bnrm).into();
                    continue;
                }
            }
            if err < tol {
                break;
            }
        }
    }

    fn snrm(&self, sx: &Vec<T>, itol: usize) -> T {
        let n = sx.len();
        match itol {
            3 => {
                let mut ans = T::zero();
                for i in 0..n {
                    ans += (sx[i] * sx[i]).into();
                }
                ans.to_f32().unwrap().sqrt().into()
            }
            _ => {
                let mut isamax = 0;
                for i in 0..n {
                    if sx[i].to_f32().unwrap().abs() > sx[isamax].to_f32().unwrap().abs() {
                        isamax = i;
                    }
                }
                sx[isamax].to_f32().unwrap().abs().into()
            }
        }
    }
}

pub struct NRsparseLingcb<'a, T>
where
    T: MatLinAlgBound,
{
    mat: &'a NRsparseMat<T>,
    n: usize,
}

impl<'a, T> NRsparseLingcb<'a, T>
where
    T: MatLinAlgBound,
{
    fn new(mat: &'a NRsparseMat<T>) -> Self {
        NRsparseLingcb { mat, n: mat.nrows }
    }
}

impl<'a, T> Linbcg<T> for NRsparseLingcb<'a, T>
where
    T: MatLinAlgBound,
{
    fn asolve(&self, b: &Vec<T>, x: &mut Vec<T>, itrnsp: usize) {
        for i in 0..self.n {
            let mut diag = 1.0f32;
            for j in self.mat.col_ptr[i]..self.mat.col_ptr[i + 1] {
                if self.mat.row_ind[j] == i {
                    diag = self.mat.val[j].to_f32().unwrap();
                    break;
                }
            }
            x[i] = (b[i].to_f32().unwrap() / diag).into();
        }
    }

    fn atimes(&self, x: &Vec<T>, r: &mut Vec<T>, itrnsp: usize) {
        if itrnsp > 0 {
            *r = self.mat.atx(x);
        } else {
            *r = self.mat.ax(x);
        }
    }
}
