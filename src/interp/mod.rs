use thiserror::Error;

pub mod interp1d;
pub mod interp2d;

pub use interp1d::*;
pub use interp2d::*;

#[derive(Error, Debug)]
pub enum InterpError {
    #[error("IdenticalX")]
    IdenticalX(),

    #[error("ZeroDiagonal")]
    ZeroDiagonal(),
}
