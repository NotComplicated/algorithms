use std::iter::Sum;

pub trait MaxSubarray {
    fn max_subarray<'a, T: Sum<&'a T> + PartialOrd>(&self, arr: &'a [T]) -> (&'a [T], Option<T>);
}

pub struct BruteForce;

impl MaxSubarray for BruteForce {
    fn max_subarray<'a, T: Sum<&'a T> + PartialOrd>(&self, arr: &'a [T]) -> (&'a [T], Option<T>) {
        let subarrs = (0..arr.len() - 1).flat_map(|i| (i + 1..arr.len()).map(move |j| &arr[i..j]));
        subarrs.fold((&[], None), |(max_subarr, max), subarr| {
            let sum = subarr.iter().sum();
            if max.as_ref().is_some_and(|max| max > &sum) {
                (max_subarr, max)
            } else {
                (subarr, Some(sum))
            }
        })
    }
}

#[cfg(all(test, feature = "max_subarray"))]
mod tests {
    use std::{hint, time::Instant};

    use super::*;

    #[test]
    fn brute_force() {
        let now = Instant::now();
        assert_eq!(
            BruteForce.max_subarray(hint::black_box(&[
                13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7
            ])),
            (&[18, 20, -7, 12] as &[i32], Some(43))
        );
        println!("{:?}", now.elapsed());
    }
}
