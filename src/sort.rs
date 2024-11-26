pub fn is_sorted(items: &[impl Ord]) -> bool {
    !items.windows(2).any(|window| window[0] > window[1])
}

pub trait Sort {
    fn sort<T: Ord>(self, items: &mut [T]);
}

pub struct Stdlib;

impl Sort for Stdlib {
    fn sort<T: Ord>(self, items: &mut [T]) {
        items.sort();
    }
}

pub struct Insertion;

impl Sort for Insertion {
    fn sort<T: Ord>(self, items: &mut [T]) {
        for j in 1..items.len() {
            for i in (1..=j).rev() {
                if items[i - 1] > items[i] {
                    items.swap(i - 1, i);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rand::*;
    use std::{any, time::Instant};

    #[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
    struct Custom {
        int: u8,
        string: String,
    }

    impl Custom {
        fn random() -> Self {
            Self {
                int: range(1..5),
                string: string(),
            }
        }
    }

    fn name<T>(t: &T) -> &'static str {
        any::type_name_of_val(t).rsplit("::").next().unwrap()
    }

    fn test<const N: usize>(sorter: impl Sort, mut items: [impl Ord; N]) {
        println!("Sorter: {}", name(&sorter));
        println!("Sorting: {}", name(&items[0]));
        println!("N: {N}");

        assert!(!is_sorted(&items));
        let before = Instant::now();
        sorter.sort(&mut items);
        println!("Elapsed: {}µs", before.elapsed().as_micros());

        assert!(is_sorted(&items));
        println!();
    }

    #[test]
    fn stdlib_sort_int() {
        test(Stdlib, ten_of(int));
        test(Stdlib, hundred_of(int));
        test(Stdlib, thousand_of(int));
    }

    #[test]
    fn stdlib_sort_string() {
        test(Stdlib, ten_of(string));
        test(Stdlib, hundred_of(string));
        test(Stdlib, thousand_of(string));
    }

    #[test]
    fn stdlib_sort_custom() {
        test(Stdlib, ten_of(Custom::random));
        test(Stdlib, hundred_of(Custom::random));
        test(Stdlib, thousand_of(Custom::random));
    }

    #[test]
    fn insertion_sort_int() {
        test(Insertion, ten_of(int));
        test(Insertion, hundred_of(int));
        test(Insertion, thousand_of(int));
    }

    #[test]
    fn insertion_sort_string() {
        test(Insertion, ten_of(string));
        test(Insertion, hundred_of(string));
        test(Insertion, thousand_of(string));
    }

    #[test]
    fn insertion_sort_custom() {
        test(Insertion, ten_of(Custom::random));
        test(Insertion, hundred_of(Custom::random));
        test(Insertion, thousand_of(Custom::random));
    }
}
