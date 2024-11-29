use ndarray::{linalg, prelude::*, LinalgScalar, Zip};
use std::{error::Error, fmt, mem::MaybeUninit};

pub type Mat<'a, T> = ArrayView2<'a, T>;
pub type MatMut<'a, T> = ArrayViewMut2<'a, T>;

pub trait Elem: LinalgScalar + Send + Sync {}

impl<T: LinalgScalar + Send + Sync> Elem for T {}

pub trait MatMul {
    fn matmul_impl<T: Elem>(&self, lhs: Mat<T>, rhs: Mat<T>, out: MatMut<MaybeUninit<T>>);
    fn matmul<'out, T: Elem>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        mut out: MatMut<'out, MaybeUninit<T>>,
    ) -> Result<MatMut<'out, T>, MatMulError> {
        let ((a, b), (c, d), (e, f)) = (lhs.dim(), rhs.dim(), out.dim());
        if b != c || a != e || d != f {
            Err(MatMulError((a, b), (c, d)))
        } else {
            self.matmul_impl(lhs, rhs, out.view_mut());
            let out = unsafe { out.assume_init() };
            Ok(out)
        }
    }
}

pub struct NdArray;

impl MatMul for NdArray {
    fn matmul_impl<T: Elem>(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<MaybeUninit<T>>) {
        out.map_inplace(|elem| {
            elem.write(T::zero());
        });
        let mut out = unsafe { out.assume_init() };
        linalg::general_mat_mul(T::one(), &lhs, &rhs, T::zero(), &mut out);
    }
}

pub struct Naive<const PAR: bool = false>;

impl<const PAR: bool> MatMul for Naive<PAR> {
    fn matmul_impl<T: Elem>(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<MaybeUninit<T>>) {
        let indices = Zip::indexed(&mut out);
        let per_elem = |(i, j), elem: &mut MaybeUninit<T>| {
            let (lhs_row, rhs_col) = (lhs.row(i), rhs.column(j));
            let zip = Zip::from(&lhs_row).and(&rhs_col);
            elem.write(zip.fold(T::zero(), |acc, &l, &r| acc + l * r));
        };
        if PAR {
            indices.par_for_each(per_elem);
        } else {
            indices.for_each(per_elem);
        }
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

    const N: usize = 500;

    fn test<T: Elem + Display>(lhs: ArrayView2<T>, rhs: ArrayView2<T>, matmul: impl MatMul) {
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
    fn naive_single_thread() {
        test(
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            Naive::<false>,
        );
        test(
            matrix_of::<_, N, N>(float).view(),
            matrix_of::<_, N, N>(float).view(),
            Naive::<false>,
        );
    }

    #[test]
    fn naive_multi_thread() {
        test(
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            Naive::<true>,
        );
        test(
            matrix_of::<_, N, N>(float).view(),
            matrix_of::<_, N, N>(float).view(),
            Naive::<true>,
        );
    }
}
