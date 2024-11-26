use std::{
    cmp::Ordering,
    ops::{Deref, DerefMut},
};

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
        for i in 1..items.len() {
            let j = items[..i]
                .iter()
                .rev()
                .position(|item| item < &items[i])
                .map_or(0, |k| i - k);
            if j != i {
                unsafe {
                    let tmp = (&raw const items[i]).read();
                    (&raw const items[j]).copy_to(&raw mut items[j + 1], i - j);
                    (&raw mut items[j]).write(tmp);
                }
            }
        }
    }
}

pub struct Selection;

impl Sort for Selection {
    fn sort<T: Ord>(self, items: &mut [T]) {
        for i in 0..items.len() - 1 {
            let min = items[i..].iter().min().unwrap();
            let j = unsafe { (min as *const T).offset_from(&items[0]) } as _;
            items.swap(i, j);
        }
    }
}

macro_rules! ordfloat {
    ($ord:ident($float:ty)) => {
        #[derive(PartialOrd, Copy, Clone, Default, Debug)]
        pub struct $ord(pub $float);

        impl Deref for $ord {
            type Target = $float;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $ord {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl PartialEq for $ord {
            fn eq(&self, other: &Self) -> bool {
                self.0.to_bits() == other.0.to_bits()
            }
        }

        impl Eq for $ord {}

        impl Ord for $ord {
            fn cmp(&self, other: &Self) -> Ordering {
                self.total_cmp(other)
            }
        }
    };
}
ordfloat!(OrdF32(f32));
ordfloat!(OrdF64(f64));

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

    #[test]
    fn selection_sort_int() {
        test(Selection, ten_of(int));
        test(Selection, hundred_of(int));
        test(Selection, thousand_of(int));
    }

    #[test]
    fn selection_sort_string() {
        test(Selection, ten_of(string));
        test(Selection, hundred_of(string));
        test(Selection, thousand_of(string));
    }

    #[test]
    fn selection_sort_custom() {
        test(Selection, ten_of(Custom::random));
        test(Selection, hundred_of(Custom::random));
        test(Selection, thousand_of(Custom::random));
    }

    #[test]
    fn ord_float() {
        test(Stdlib, hundred_of(float).map(OrdF32));
        test(Insertion, hundred_of(float).map(OrdF32));
        test(Selection, hundred_of(float).map(OrdF32));
    }
}
