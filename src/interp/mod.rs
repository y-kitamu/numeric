use crate::accessor;
use anyhow::Result;
use thiserror::Error;

pub mod linear;
pub mod poly_1d;
pub mod rational_1d;
pub mod spline1d;

#[derive(Error, Debug)]
pub enum InterpError {
    #[error("IdenticalX")]
    IdenticalX(),

    #[error("ZeroDiagonal")]
    ZeroDiagonal(),
}

pub trait Interp {
    accessor!((get = n, set = set_n): usize);
    accessor!((get = mm, set = set_mm): usize);
    accessor!((get = jsav, set = set_jsav): usize);
    accessor!((get = dj, set = set_dj): usize);
    accessor!((get = cor, set = set_cor): usize);
    fn xx(&self) -> &Vec<f64>;

    fn interp(&mut self, x: f64) -> Result<f64> {
        let jlo = if self.cor() > 0 {
            self.hunt(x)
        } else {
            self.locate(x)
        };
        self.rawinterp(jlo, x)
    }

    fn locate(&mut self, x: f64) -> usize {
        let mut min = 0;
        let mut max = self.n();
        let xx = self.xx();
        let ascend = xx[max - 1] >= xx[0];
        while max - min > 1 {
            let mid = (min + max) >> 1;
            if (xx[mid] < x) == ascend {
                min = mid;
            } else {
                max = mid;
            }
        }
        let cor = if min > self.jsav() {
            (min - self.jsav() > self.dj()) as usize
        } else {
            (self.jsav() - min > self.dj()) as usize
        };
        self.set_cor(cor);
        self.set_jsav(min);
        if min < (self.mm() - 2) >> 1 || self.n() < self.mm() {
            0
        } else {
            std::cmp::min(self.n() - self.mm(), min - ((self.mm() - 2) >> 1))
        }
    }

    fn hunt(&mut self, x: f64) -> usize {
        let mut jl = self.jsav();
        let mut inc = 1;
        let mut ju = jl;
        let n = self.n();
        if n < 2 || self.mm() < 2 || self.mm() > n {
            return self.locate(x);
        }
        let xx = self.xx();
        let ascend = xx[n - 1] >= xx[0];
        if (x >= xx[jl]) == ascend {
            loop {
                ju = jl + inc;
                if ju > n - 1 {
                    ju = n - 1;
                    break;
                } else if (x < xx[ju]) == ascend {
                    break;
                } else {
                    jl = ju;
                    inc += inc;
                }
            }
        } else {
            ju = jl;
            loop {
                jl = jl - inc;
                if jl <= 0 {
                    jl = 0;
                    break;
                } else if (x >= xx[jl]) == ascend {
                    break;
                } else {
                    ju = jl;
                    inc += inc;
                }
            }
        }
        while ju - jl > 1 {
            let jm = (ju + jl) >> 1;
            if (x >= xx[jm]) == ascend {
                jl = jm;
            } else {
                ju = jm;
            }
        }
        let cor = if jl > self.jsav() {
            (jl - self.jsav() > self.dj()) as usize
        } else {
            (self.jsav() - jl > self.dj()) as usize
        };
        self.set_cor(cor);
        self.set_jsav(jl);
        if jl < ((self.mm() - 2) >> 1) || self.n() < self.mm() {
            0
        } else {
            std::cmp::min(self.n() - self.mm(), jl - ((self.mm() - 2) >> 1))
        }
    }

    fn rawinterp(&self, jlo: usize, x: f64) -> Result<f64>;
}
