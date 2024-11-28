use ndarray::{linalg, prelude::*, LinalgScalar};

pub type Mat<'a, T> = ArrayView2<'a, T>;
pub type MatMut<'a, T> = ArrayViewMut2<'a, T>;

pub trait MatMul {
    fn matmul<T: LinalgScalar>(&self, lhs: Mat<T>, rhs: Mat<T>, out: MatMut<T>);
}

pub struct NdArray;

impl MatMul for NdArray {
    fn matmul<T: LinalgScalar>(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<T>) {
        linalg::general_mat_mul(T::one(), &lhs, &rhs, T::zero(), &mut out);
    }
}

#[cfg(all(test, feature = "matmul"))]
mod tests {
    use std::{fmt::Display, time::Instant};

    use super::*;
    use crate::rand::*;

    const N: usize = 3;

    fn test<T: LinalgScalar + Display>(
        lhs: ArrayView2<T>,
        rhs: ArrayView2<T>,
        matmul: impl MatMul,
    ) {
        let mut out = matrix_of::<_, N, N>(zero);

        let before = Instant::now();
        matmul.matmul(lhs, rhs, out.view_mut());
        let elapsed = before.elapsed();

        println!("{lhs}\n");
        println!("{rhs}\n");
        println!("{out}\n");
        println!("elapsed: {elapsed:?}\n");
    }

    #[test]
    fn ndarray() {
        test(
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            NdArray,
        );
        test(
            matrix_of::<_, N, N>(float).view(),
            matrix_of::<_, N, N>(float).view(),
            NdArray,
        );
    }
}
