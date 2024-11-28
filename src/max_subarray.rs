use std::{
    iter::Sum,
    ops::{AddAssign, Sub},
};

pub trait MaxSubarray<T> {
    type Spread;

    fn max_subarray<'a>(&self, arr: &'a [T]) -> (&'a [T], Option<Self::Spread>);
}

pub struct BruteForce;

impl<T> MaxSubarray<T> for BruteForce
where
    T: PartialOrd + for<'a> Sum<&'a T>,
{
    type Spread = T;

    fn max_subarray<'a>(&self, arr: &'a [T]) -> (&'a [T], Option<Self::Spread>) {
        let subarrs = (0..arr.len()).flat_map(|i| (i + 1..=arr.len()).map(move |j| &arr[i..j]));
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

pub struct DivideAndConquer;

impl<T> MaxSubarray<T> for DivideAndConquer {
    type Spread = T;

    fn max_subarray<'a>(&self, arr: &'a [T]) -> (&'a [T], Option<Self::Spread>) {
        todo!()
    }
}

pub struct OnePass;

impl<T, S> MaxSubarray<T> for OnePass
where
    T: Clone + PartialOrd + for<'a> AddAssign<&'a T>,
    S: PartialOrd,
    for<'a, 'b> &'a T: Sub<&'b T, Output = S>,
{
    type Spread = S;

    fn max_subarray<'a>(&self, arr: &'a [T]) -> (&'a [T], Option<Self::Spread>) {
        if let [first, rest @ ..] = arr {
            let (mut val, mut min, mut out, mut from, mut to) =
                (first.clone(), first.clone(), first - first, 0, 1);
            for (item, i) in rest.iter().zip(1..) {
                val += item;
                if min >= val {
                    min = val.clone();
                    from = i + 1;
                } else {
                    let spread = &val - &min;
                    if &out < &spread {
                        out = spread;
                        to = i;
                    }
                }
            }
            (&arr[from..=to], Some(out))
        } else {
            (&[], None)
        }
    }
}

#[cfg(all(test, feature = "max_subarray"))]
mod tests {
    use super::*;
    use std::time::Instant;

    type Int = i32;
    const ARR: &[Int] = &[
        13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7,
    ];
    const EXPECT: (&[Int], Option<Int>) = (&[18, 20, -7, 12], Some(43));

    fn test(max_subarray: impl MaxSubarray<Int, Spread = Int>) {
        let now = Instant::now();
        assert_eq!(EXPECT, max_subarray.max_subarray(ARR));
        println!("{:?}", now.elapsed());
    }

    #[test]
    fn brute_force() {
        test(BruteForce);
    }

    #[test]
    fn one_pass() {
        test(OnePass);
    }
}
