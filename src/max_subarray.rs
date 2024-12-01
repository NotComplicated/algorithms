use std::ops::{Add, Sub};

pub trait Elem:
    Default
    + Clone
    + PartialOrd
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
{
}

impl<T> Elem for T where
    for<'a> T: Default + Clone + PartialOrd + Add<&'a T, Output = T> + Sub<&'a T, Output = T>
{
}

pub trait MaxSubarray {
    fn max_subarray<'a, T: Elem>(&self, arr: &'a [T]) -> (&'a [T], Option<T>);
}

pub struct BruteForce;

impl MaxSubarray for BruteForce {
    fn max_subarray<'a, T: Elem>(&self, arr: &'a [T]) -> (&'a [T], Option<T>) {
        let subarrs = (0..arr.len()).flat_map(|i| (i + 1..=arr.len()).map(move |j| &arr[i..j]));
        subarrs.fold((&[], None), |(max_subarr, max), subarr| {
            let sum = subarr.iter().fold(T::default(), |sum, elem| sum + elem);
            if max.as_ref().is_some_and(|max| max > &sum) {
                (max_subarr, max)
            } else {
                (subarr, Some(sum))
            }
        })
    }
}

pub struct DivideAndConquer;

impl MaxSubarray for DivideAndConquer {
    fn max_subarray<'a, T: Elem>(&self, arr: &'a [T]) -> (&'a [T], Option<T>) {
        if arr.is_empty() {
            (&[], None)
        } else if let [elem] = arr {
            (arr, Some(elem.clone()))
        } else {
            let mid = arr.len() / 2;
            let (l_subarr, l_sum) = self.max_subarray(&arr[..mid]);
            let (r_subarr, r_sum) = self.max_subarray(&arr[mid..]);
            let (x_subarr, x_sum) = {
                let (mut sum, mut l_sum, mut r_sum, mut l, mut r) =
                    (T::default(), None, None, mid, mid);
                for i in (0..mid).rev() {
                    sum = sum + &arr[i];
                    if l_sum.as_ref().is_none_or(|l_sum| l_sum < &sum) {
                        l_sum = Some(sum.clone());
                        l = i;
                    }
                }
                sum = T::default();
                for (i, item) in arr.iter().enumerate().skip(mid) {
                    sum = sum + item;
                    if r_sum.as_ref().is_none_or(|r_sum| r_sum < &sum) {
                        r_sum = Some(sum.clone());
                        r = i;
                    }
                }
                let x_sum = l_sum.and_then(|l_sum| r_sum.map(|r_sum| l_sum + &r_sum));
                (&arr[l..=r], x_sum)
            };
            if l_sum >= r_sum && l_sum >= x_sum {
                (l_subarr, l_sum)
            } else if r_sum >= l_sum && r_sum >= x_sum {
                (r_subarr, r_sum)
            } else {
                (x_subarr, x_sum)
            }
        }
    }
}

pub struct OnePass;

impl MaxSubarray for OnePass {
    fn max_subarray<'a, T: Elem>(&self, arr: &'a [T]) -> (&'a [T], Option<T>) {
        let (mut val, mut min, mut min_i, mut sum, mut range) =
            (T::default(), T::default(), 0, None, 0..0);
        for (elem, i) in arr.iter().zip(1..) {
            val = val + elem;
            let curr_sum = val.clone() - &min;
            if sum.as_ref().is_none_or(|sum| sum < &curr_sum) {
                sum = Some(curr_sum);
                range = min_i..i;
            }
            if min >= val {
                min = val.clone();
                min_i = i;
            }
        }
        (&arr[range], sum)
    }
}

#[cfg(all(test, feature = "max_subarray"))]
mod tests {
    use super::*;
    use std::{hint, time::Instant};

    fn test(max_subarray: impl MaxSubarray) {
        assert_eq!((&[] as &[i32], None), max_subarray.max_subarray(&[]));
        assert_eq!((&[13] as _, Some(13)), max_subarray.max_subarray(&[13]));
        assert_eq!((&[-13] as _, Some(-13)), max_subarray.max_subarray(&[-13]));
        assert_eq!(
            (&[3] as _, Some(3)),
            max_subarray.max_subarray(&[1, -4, 3, -4])
        );
        assert_eq!(
            (&[18.0, 20.0, -7.0, 12.0] as _, Some(43.0)),
            max_subarray.max_subarray(hint::black_box(&[
                18.0, 20.0, -7.0, 12.0, -5.0, -22.0, 15.0, -4.0, 7.0,
            ]))
        );
        let now = Instant::now();
        assert_eq!(
            (&[18, 20, -7, 12] as _, Some(43)),
            max_subarray.max_subarray(hint::black_box(&[
                13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7,
            ]))
        );
        println!("{:?}", now.elapsed());
    }

    #[test]
    fn brute_force() {
        test(BruteForce);
    }

    #[test]
    fn divide_and_conquer() {
        test(DivideAndConquer);
    }

    #[test]
    fn one_pass() {
        test(OnePass);
    }
}
