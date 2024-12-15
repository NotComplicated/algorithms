use ndarray::{linalg, par_azip, prelude::*, LinalgScalar};
use std::{array, error::Error, fmt, mem::MaybeUninit};

pub type Mat<'a, T> = ArrayView2<'a, T>;
pub type MatMut<'a, T> = ArrayViewMut2<'a, T>;

pub trait MatMul<T: LinalgScalar> {
    unsafe fn matmul_unchecked(&self, lhs: Mat<T>, rhs: Mat<T>, out: MatMut<MaybeUninit<T>>);
    fn matmul<'out>(
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

impl<T: LinalgScalar> MatMul<T> for NdArray {
    unsafe fn matmul_unchecked(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<MaybeUninit<T>>) {
        out.map_inplace(|elem| {
            elem.write(T::zero());
        });
        let mut out = unsafe { out.assume_init() };
        linalg::general_mat_mul(T::one(), &lhs, &rhs, T::zero(), &mut out);
    }
}

pub struct Naive<const PAR: bool>;

impl<T: LinalgScalar> MatMul<T> for Naive<false> {
    unsafe fn matmul_unchecked(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<MaybeUninit<T>>) {
        azip! {
            (index (i, j), out in &mut out) {
                let out = out.write(T::zero());
                azip! {
                    (&lhs in &lhs.row(i), &rhs in &rhs.column(j)) {
                        *out = *out + lhs * rhs;
                    }
                }
            }
        }
    }
}

impl<T: LinalgScalar + Send + Sync> MatMul<T> for Naive<true> {
    unsafe fn matmul_unchecked(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<MaybeUninit<T>>) {
        par_azip! {
            (index (i, j), out in &mut out) {
                let out = out.write(T::zero());
                azip! {
                    (&lhs in &lhs.row(i), &rhs in &rhs.column(j)) {
                        *out = *out + lhs * rhs;
                    }
                }
            }
        }
    }
}

macro_rules! get_square_partition {
    ($mat:ident, $mid:ident) => {{
        let (mat1, mat2) = $mat.split_at(Axis(0), $mid);
        let (mat11, mat12) = mat1.split_at(Axis(1), $mid);
        let (mat21, mat22) = mat2.split_at(Axis(1), $mid);
        [[mat11, mat12], [mat21, mat22]]
    }};
}

fn square_partition<T>(mat: Mat<T>, mid: usize) -> [[Mat<T>; 2]; 2] {
    get_square_partition!(mat, mid)
}

fn square_partition_mut<T>(mat: MatMut<T>, mid: usize) -> [[MatMut<T>; 2]; 2] {
    get_square_partition!(mat, mid)
}

pub struct DivideAndConquer;

impl<T: LinalgScalar> MatMul<T> for DivideAndConquer {
    unsafe fn matmul_unchecked(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<MaybeUninit<T>>) {
        match lhs.shape() {
            [1, 1] => unsafe {
                let out_elem = out.uget_mut((0, 0)).assume_init_mut();
                *out_elem = *out_elem + *lhs.uget((0, 0)) * *rhs.uget((0, 0));
            },

            [n, _] => {
                let mid = n / 2;
                let lhs_parts = square_partition(lhs, mid);
                let rhs_parts = square_partition(rhs, mid);
                let mut out_parts = square_partition_mut(out, mid);

                self.matmul_unchecked(lhs_parts[0][0], rhs_parts[0][0], out_parts[0][0].view_mut());
                self.matmul_unchecked(lhs_parts[0][1], rhs_parts[1][0], out_parts[0][0].view_mut());
                self.matmul_unchecked(lhs_parts[0][0], rhs_parts[0][1], out_parts[0][1].view_mut());
                self.matmul_unchecked(lhs_parts[0][1], rhs_parts[1][1], out_parts[0][1].view_mut());
                self.matmul_unchecked(lhs_parts[1][0], rhs_parts[0][0], out_parts[1][0].view_mut());
                self.matmul_unchecked(lhs_parts[1][1], rhs_parts[1][0], out_parts[1][0].view_mut());
                self.matmul_unchecked(lhs_parts[1][0], rhs_parts[0][1], out_parts[1][1].view_mut());
                self.matmul_unchecked(lhs_parts[1][1], rhs_parts[1][1], out_parts[1][1].view_mut());
            }

            _ => unreachable!(),
        }
    }

    fn matmul<'out>(
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

pub struct Strassen;

impl<T: LinalgScalar> MatMul<T> for Strassen {
    unsafe fn matmul_unchecked(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<MaybeUninit<T>>) {
        match lhs.shape() {
            [1, 1] => unsafe {
                let out_elem = out.uget_mut((0, 0)).assume_init_mut();
                *out_elem = *out_elem + *lhs.uget((0, 0)) * *rhs.uget((0, 0));
            },

            &[n, _] => {
                let mid = n / 2;
                let lhs_parts = square_partition(lhs, mid);
                let rhs_parts = square_partition(rhs, mid);
                let mut out_parts = square_partition_mut(out, mid);

                let s1 = &rhs_parts[0][1] - &rhs_parts[1][1];
                let s2 = &lhs_parts[0][0] + &lhs_parts[0][1];
                let s3 = &lhs_parts[1][0] + &lhs_parts[1][1];
                let s4 = &rhs_parts[1][0] - &rhs_parts[0][0];
                let s5 = &lhs_parts[0][0] + &lhs_parts[1][1];
                let s6 = &rhs_parts[0][0] + &rhs_parts[1][1];
                let s7 = &lhs_parts[0][1] - &lhs_parts[1][1];
                let s8 = &rhs_parts[1][0] + &rhs_parts[1][1];
                let s9 = &lhs_parts[0][0] - &lhs_parts[1][0];
                let s10 = &rhs_parts[0][0] + &rhs_parts[0][1];

                let mut p_mats: [_; 7] = array::from_fn(|_| {
                    let mut p_mat = Array2::uninit((mid, mid));
                    p_mat.map_inplace(|elem| {
                        elem.write(T::zero());
                    });
                    p_mat
                });

                unsafe {
                    self.matmul_unchecked(lhs_parts[0][0].view(), s1.view(), p_mats[0].view_mut());
                    self.matmul_unchecked(s2.view(), rhs_parts[1][1].view(), p_mats[1].view_mut());
                    self.matmul_unchecked(s3.view(), rhs_parts[0][0].view(), p_mats[2].view_mut());
                    self.matmul_unchecked(lhs_parts[1][1].view(), s4.view(), p_mats[3].view_mut());
                    self.matmul_unchecked(s5.view(), s6.view(), p_mats[4].view_mut());
                    self.matmul_unchecked(s7.view(), s8.view(), p_mats[5].view_mut());
                    self.matmul_unchecked(s9.view(), s10.view(), p_mats[6].view_mut());

                    azip! {
                        (out in &mut out_parts[0][0], p4 in &p_mats[4], p3 in &p_mats[3], p1 in &p_mats[1], p5 in &p_mats[5]) {
                            out.write(p4.assume_init() + p3.assume_init() - p1.assume_init() + p5.assume_init());
                        }
                    }
                    azip! {
                        (out in &mut out_parts[0][1], p0 in &p_mats[0], p1 in &p_mats[1]) {
                            out.write(p0.assume_init() + p1.assume_init());
                        }
                    }
                    azip! {
                        (out in &mut out_parts[1][0], p2 in &p_mats[2], p3 in &p_mats[3]) {
                            out.write(p2.assume_init() + p3.assume_init());
                        }
                    }
                    azip! {
                        (out in &mut out_parts[1][1], p4 in &p_mats[4], p0 in &p_mats[0], p2 in &p_mats[2], p6 in &p_mats[6]) {
                            out.write(p4.assume_init() + p0.assume_init() - p2.assume_init() - p6.assume_init());
                        }
                    }
                }
            }

            _ => unreachable!(),
        }
    }

    fn matmul<'out>(
        &self,
        lhs: Mat<T>,
        rhs: Mat<T>,
        mut out: MatMut<'out, MaybeUninit<T>>,
    ) -> Result<MatMut<'out, T>, MatMulError> {
        let ((a, b), (c, d), (e, f)) = (lhs.dim(), rhs.dim(), out.dim());
        if a.count_ones() != 1 || [b, c, d, e, f].iter().any(|&x| x != a) {
            Err(MatMulError::Other(concat!(
                "Strassen algorithm requires",
                " two NxN input matrices",
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

    const N: usize = 64;

    fn test<T: LinalgScalar + Display>(
        lhs: ArrayView2<T>,
        rhs: ArrayView2<T>,
        matmul: impl MatMul<T>,
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
    fn strassen() {
        test(
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            matrix_of::<_, N, N>(|| range(0..10)).view(),
            Strassen,
        );
        test(
            matrix_of::<_, N, N>(float).view(),
            matrix_of::<_, N, N>(float).view(),
            Strassen,
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
