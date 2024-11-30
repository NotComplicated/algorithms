use ndarray::{iter::Indices, linalg, prelude::*, LinalgScalar, Zip};
use std::{array, cell::UnsafeCell, error::Error, fmt, mem::MaybeUninit};

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

impl<const PAR: bool> Naive<PAR> {
    fn zip_per_elem_impl<'a, T: LinalgScalar>(
        &self,
        lhs: Mat<'a, T>,
        rhs: Mat<'a, T>,
        out: MatMut<'a, MaybeUninit<T>>,
    ) -> (
        Zip<(Indices<Ix2>, ArrayViewMut2<'a, MaybeUninit<T>>), Ix2>,
        impl Fn((usize, usize), &'a mut MaybeUninit<T>) + use<'a, T, PAR>,
    ) {
        let indices = Zip::indexed(out);
        let per_elem = move |(i, j), elem: &mut MaybeUninit<T>| {
            let (lhs_row, rhs_col) = (lhs.row(i), rhs.column(j));
            let zip = Zip::from(&lhs_row).and(&rhs_col);
            elem.write(zip.fold(T::zero(), |acc, &l, &r| acc + l * r));
        };
        (indices, per_elem)
    }
}

impl<T: LinalgScalar> MatMul<T> for Naive<false> {
    unsafe fn matmul_unchecked(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<MaybeUninit<T>>) {
        let (indices, per_elem) = self.zip_per_elem_impl(lhs.view(), rhs.view(), out.view_mut());
        indices.for_each(per_elem);
    }
}

impl<T: LinalgScalar + Send + Sync> MatMul<T> for Naive<true> {
    unsafe fn matmul_unchecked(&self, lhs: Mat<T>, rhs: Mat<T>, mut out: MatMut<MaybeUninit<T>>) {
        let (indices, per_elem) = self.zip_per_elem_impl(lhs.view(), rhs.view(), out.view_mut());
        indices.par_for_each(per_elem);
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

pub struct Strassen<T> {
    s_mats: UnsafeCell<[Array2<T>; 10]>,
    p_mats: UnsafeCell<[Array2<MaybeUninit<T>>; 7]>,
}

impl<T> Default for Strassen<T> {
    fn default() -> Self {
        Self {
            s_mats: UnsafeCell::new(array::from_fn(|_| {
                Array2::from_shape_fn((0, 0), |_| unreachable!())
            })),
            p_mats: UnsafeCell::new(array::from_fn(|_| Array2::uninit((0, 0)))),
        }
    }
}

impl<T: LinalgScalar> MatMul<T> for Strassen<T> {
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

                macro_rules! s {
                    ($s_i:tt: $lhs:ident[$lhs_i:tt][$lhs_j:tt] $op:tt $rhs:ident[$rhs_i:tt][$rhs_j:tt]) => {
                        let s = unsafe { &mut (*self.s_mats.get())[$s_i] };
                        azip! {
                            (s in s, &lhs in &$lhs[$lhs_i][$lhs_j], &rhs in &$rhs[$rhs_i][$rhs_j]) {
                                *s = lhs $op rhs;
                            }
                        }
                    };
                }

                s!(0: rhs_parts[0][1] - rhs_parts[1][1]);
                s!(1: lhs_parts[0][0] + lhs_parts[0][1]);
                s!(2: lhs_parts[1][0] + lhs_parts[1][1]);
                s!(3: rhs_parts[1][0] - rhs_parts[0][0]);
                s!(4: lhs_parts[0][0] + lhs_parts[1][1]);
                s!(5: rhs_parts[0][0] + rhs_parts[1][1]);
                s!(6: lhs_parts[0][1] - lhs_parts[1][1]);
                s!(7: rhs_parts[1][0] + rhs_parts[1][1]);
                s!(8: lhs_parts[0][0] - lhs_parts[1][0]);
                s!(9: rhs_parts[0][0] + rhs_parts[0][1]);

                unsafe {
                    let get_s = |i: usize| (*self.s_mats.get()).get_unchecked(i).view();
                    let get_p = |i: usize| (*self.p_mats.get()).get_unchecked_mut(i).view_mut();

                    self.matmul_unchecked(lhs_parts[0][0].view(), get_s(0), get_p(0));
                    self.matmul_unchecked(get_s(1), rhs_parts[1][1].view(), get_p(1));
                    self.matmul_unchecked(get_s(2), rhs_parts[0][0].view(), get_p(2));
                    self.matmul_unchecked(lhs_parts[1][1].view(), get_s(3), get_p(3));
                    self.matmul_unchecked(get_s(4), get_s(5), get_p(4));
                    self.matmul_unchecked(get_s(6), get_s(7), get_p(5));
                    self.matmul_unchecked(get_s(8), get_s(9), get_p(6));

                    azip! {
                        (out in &mut out_parts[0][0], p4 in get_p(4), p3 in get_p(3), p1 in get_p(1), p5 in get_p(5)) {
                            out.write(p4.assume_init() + p3.assume_init() - p1.assume_init() + p5.assume_init());
                        }
                    }
                    azip! {
                        (out in &mut out_parts[0][1], p0 in get_p(0), p1 in get_p(1)) {
                            out.write(p0.assume_init() + p1.assume_init());
                        }
                    }
                    azip! {
                        (out in &mut out_parts[1][0], p2 in get_p(2), p3 in get_p(3)) {
                            out.write(p2.assume_init() + p3.assume_init());
                        }
                    }
                    azip! {
                        (out in &mut out_parts[1][1], p4 in get_p(4), p0 in get_p(0), p2 in get_p(2), p6 in get_p(6)) {
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
            let n = a / 2;
            Ok(unsafe {
                *self.s_mats.get() = array::from_fn(|_| Array2::<T>::zeros((n, n)));
                *self.p_mats.get() = array::from_fn(|_| Array2::<T>::uninit((n, n)));
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
            Strassen::default(),
        );
        test(
            matrix_of::<_, N, N>(float).view(),
            matrix_of::<_, N, N>(float).view(),
            Strassen::default(),
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
