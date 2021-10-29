use thiserror::Error;

#[derive(Error, Debug)]
pub enum LinAlgError {
    #[error("Invalid vector or matrix size")]
    InvalidSize(),

    #[error("Invalid vector size : {0}")]
    InvalidVectorSize(usize),

    #[error("Invalid matrix size : (row, col) = ({0}, {1})")]
    InvalidMatrixSize(usize, usize),

    #[error("Matrix is singular : {0}")]
    SingularMatrix(String),

    #[error("Diagonal element of the matrix is zero at row = col = {0}")]
    ZeroDiagonalElemement(usize),

    #[error("Try to divide by zero.")]
    ZeroDivision(),

    #[error("Singular principle minor")]
    SingularPrincipleMinor(),

    #[error("Should not arrive here")]
    ShouldNotArriveHere(),
}

pub mod banddiagonal;
pub mod gauss_jordan;
pub mod lingcd;
pub mod lu_decomposition;
pub mod sparse;
pub mod svd;
pub mod toeplz;
pub mod tridiagonal;
pub mod vander;
