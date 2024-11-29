use ndarray::{linalg, prelude::*, LinalgScalar, Zip};
use std::{error::Error, fmt, mem::MaybeUninit};

pub type Mat<'a, T> = ArrayView2<'a, T>;
pub type MatMut<'a, T> = ArrayViewMut2<'a, T>;

pub trait Elem: LinalgScalar + Send + Sync {}

impl<T: LinalgScalar + Send + Sync> Elem for T {}

pub trait MatMul {
    unsafe fn matmul_unchecked<T: Elem>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        out: MatMut<MaybeUninit<T>>,
    );
    fn matmul<'out, T: Elem>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        mut out: MatMut<'out, MaybeUninit<T>>,
    ) -> Result<MatMut<'out, T>, MatMulError> {
        let ((a, b), (c, d), (e, f)) = (lhs.dim(), rhs.dim(), out.dim());
        if b != c || a != e || d != f {
            Err(MatMulError::InvalidShape((a, b), (c, d)))
        } else {
            Ok(unsafe {
                self.matmul_unchecked(lhs, rhs, out.view_mut());
                out.assume_init()
            })
        }
    }
}

pub struct NdArray;

impl MatMul for NdArray {
    unsafe fn matmul_unchecked<T: Elem>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        mut out: MatMut<MaybeUninit<T>>,
    ) {
        out.map_inplace(|elem| {
            elem.write(T::zero());
        });
        let mut out = unsafe { out.assume_init() };
        linalg::general_mat_mul(T::one(), &lhs, &rhs, T::zero(), &mut out);
    }
}

pub struct Naive<const PAR: bool>;

impl<const PAR: bool> MatMul for Naive<PAR> {
    unsafe fn matmul_unchecked<T: Elem>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        mut out: MatMut<MaybeUninit<T>>,
    ) {
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

pub struct DivideAndConquer;

impl MatMul for DivideAndConquer {
    unsafe fn matmul_unchecked<T: Elem>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        mut out: MatMut<MaybeUninit<T>>,
    ) {
        match lhs.shape() {
            [1, 1] => unsafe {
                let out_elem = out.uget_mut((0, 0)).assume_init_mut();
                *out_elem = *out_elem + *lhs.uget((0, 0)) * *rhs.uget((0, 0));
            },

            [rows, cols] if rows == cols => {
                let mid = rows / 2;
                let (lhs_1, lhs_2) = lhs.split_at(Axis(0), mid);
                let (lhs_1_1, lhs_1_2) = lhs_1.split_at(Axis(1), mid);
                let (lhs_2_1, lhs_2_2) = lhs_2.split_at(Axis(1), mid);

                let (rhs_1, rhs_2) = rhs.split_at(Axis(0), mid);
                let (rhs_1_1, rhs_1_2) = rhs_1.split_at(Axis(1), mid);
                let (rhs_2_1, rhs_2_2) = rhs_2.split_at(Axis(1), mid);

                let (out_1, out_2) = out.split_at(Axis(0), mid);
                let (mut out_1_1, mut out_1_2) = out_1.split_at(Axis(1), mid);
                let (mut out_2_1, mut out_2_2) = out_2.split_at(Axis(1), mid);

                self.matmul_unchecked(lhs_1_1, rhs_1_1, out_1_1.view_mut());
                self.matmul_unchecked(lhs_1_2, rhs_2_1, out_1_1.view_mut());
                self.matmul_unchecked(lhs_1_1, rhs_1_2, out_1_2.view_mut());
                self.matmul_unchecked(lhs_1_2, rhs_2_2, out_1_2.view_mut());
                self.matmul_unchecked(lhs_2_1, rhs_1_1, out_2_1.view_mut());
                self.matmul_unchecked(lhs_2_2, rhs_2_1, out_2_1.view_mut());
                self.matmul_unchecked(lhs_2_1, rhs_1_2, out_2_2.view_mut());
                self.matmul_unchecked(lhs_2_2, rhs_2_2, out_2_2.view_mut());
            }

            _ => unreachable!(),
        }
    }

    fn matmul<'out, T: Elem>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        mut out: MatMut<'out, MaybeUninit<T>>,
    ) -> Result<MatMut<'out, T>, MatMulError> {
        let ((a, b), (c, d), (e, f)) = (lhs.dim(), rhs.dim(), out.dim());
        if a.count_ones() != 1 || [b, c, d, e, f].iter().any(|&x| x != a) {
            Err(MatMulError::Other(concat!(
                "Divide and conquer algorithm",
                " requires two NxN input matrices",
                " and one NxN output matrix,",
                " where N is a power of 2.",
            )))
        } else {
            out.map_inplace(|elem| {
                elem.write(T::zero());
            });
            Ok(unsafe {
                self.matmul_unchecked(lhs, rhs, out.view_mut());
                out.assume_init()
            })
        }
    }
}

pub enum MatMulError {
    InvalidShape((usize, usize), (usize, usize)),
    Other(&'static str),
}

impl fmt::Debug for MatMulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatMulError::InvalidShape((a, b), (c, d)) => {
                write!(f, "Cannot multiply {a}x{b} matrix with {c}x{d} matrix")
            }
            MatMulError::Other(s) => f.write_str(s),
        }
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

    const N: usize = 256;

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

    #[test]
    fn divide_and_conquer() {
        test(
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            DivideAndConquer,
        );
        test(
            matrix_of::<_, N, N>(float).view(),
            matrix_of::<_, N, N>(float).view(),
            DivideAndConquer,
        );
    }

    #[test]
    fn rectangles() {
        test(
            matrix_of::<_, N, 10>(float).view(),
            matrix_of::<_, 10, N>(float).view(),
            NdArray,
        );
        test(
            matrix_of::<_, N, 10>(float).view(),
            matrix_of::<_, 10, N>(float).view(),
            Naive::<false>,
        );
    }
}
