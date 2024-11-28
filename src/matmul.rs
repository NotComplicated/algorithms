use ndarray::{linalg, prelude::*, LinalgScalar};
use std::{error::Error, fmt, mem::MaybeUninit};

pub type Mat<'a, T> = ArrayView2<'a, T>;
pub type MatMut<'a, T> = ArrayViewMut2<'a, T>;

pub trait MatMul {
    unsafe fn matmul_unchecked<T: LinalgScalar>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        out: MatMut<MaybeUninit<T>>,
    );
    fn matmul<'out, T: LinalgScalar>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        mut out: MatMut<'out, MaybeUninit<T>>,
    ) -> Result<MatMut<'out, T>, MatMulError> {
        let ((a, b), (c, d), (e, f)) = (lhs.dim(), rhs.dim(), out.dim());
        if b != c || a != e || d != f {
            Err(MatMulError((a, b), (c, d)))
        } else {
            let out = unsafe {
                self.matmul_unchecked(lhs, rhs, out.view_mut());
                out.assume_init()
            };
            Ok(out)
        }
    }
}

pub struct NdArray;

impl MatMul for NdArray {
    unsafe fn matmul_unchecked<T: LinalgScalar>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        mut out: MatMut<MaybeUninit<T>>,
    ) {
        out.map_inplace(|elem| {
            elem.write(T::zero());
        });
        let mut out = out.assume_init();
        linalg::general_mat_mul(T::one(), &lhs, &rhs, T::zero(), &mut out);
    }
}

pub struct Naive;

impl MatMul for Naive {
    unsafe fn matmul_unchecked<T: LinalgScalar>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        out: MatMut<MaybeUninit<T>>,
    ) {
        todo!()
    }
}

pub struct MatMulError((usize, usize), (usize, usize));

impl fmt::Debug for MatMulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self((a, b), (c, d)) = self;
        write!(f, "Cannot multiply {a}x{b} matrix with {c}x{d} matrix",)
    }
}

impl fmt::Display for MatMulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for MatMulError {}

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
        let mut out = matrix_of::<_, N, N>(MaybeUninit::uninit);

        let before = Instant::now();
        let out = matmul.matmul(lhs, rhs, out.view_mut()).unwrap();
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

    #[test]
    fn naive() {
        test(
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            Naive,
        );
        test(
            matrix_of::<_, N, N>(float).view(),
            matrix_of::<_, N, N>(float).view(),
            Naive,
        );
    }
}
