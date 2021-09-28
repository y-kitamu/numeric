use thiserror::Error;

#[derive(Error, Debug)]
pub enum LinAlgError {
    #[error("Matrix is singular : {0}")]
    SingularMatrix(String),
}

pub mod gauss_jordan;
