use std::{
    cmp::Ordering,
    ops::{Deref, DerefMut},
    ptr,
};

pub fn is_sorted(items: &[impl Ord]) -> bool {
    !items.windows(2).any(|window| window[0] > window[1])
}

// SAFETY - from and to are within 0..items.len(), from >= to
unsafe fn shift_back<T>(items: &mut [T], from: usize, to: usize) {
    let (from, to) = (items.as_mut_ptr().add(from), items.as_mut_ptr().add(to));
    unsafe {
        let tmp = from.read();
        to.copy_to(to.add(1), from.offset_from(to) as _);
        to.write(tmp);
    }
}

// SAFETY - i and j are within 0..items.len()
unsafe fn swap<T>(items: &mut [T], i: usize, j: usize) {
    unsafe { items.as_mut_ptr().add(i).swap(items.as_mut_ptr().add(j)) };
}

pub trait Sort {
    fn sort(&self, items: &mut [impl Ord]);
}

pub struct Stdlib;

impl Sort for Stdlib {
    fn sort(&self, items: &mut [impl Ord]) {
        items.sort();
    }
}

pub struct Insertion;

impl Sort for Insertion {
    fn sort(&self, items: &mut [impl Ord]) {
        for i in 1..items.len() {
            let (Ok(j) | Err(j)) = items[..i].binary_search(&items[i]);
            unsafe { shift_back(items, i, j) };
        }
    }
}

pub struct Selection;

impl Sort for Selection {
    fn sort(&self, items: &mut [impl Ord]) {
        if let Some(last) = items.len().checked_sub(1) {
            for i in 0..last {
                unsafe {
                    let min = items[i..].iter().min().unwrap_unchecked();
                    let j = ptr::from_ref(min).offset_from(items.as_ptr()) as _;
                    swap(items, i, j);
                }
            }
        }
    }
}

pub struct Merge;

impl Sort for Merge {
    fn sort(&self, items: &mut [impl Ord]) {
        let len = items.len();
        if len > 1 {
            let mid = len / 2;
            self.sort(&mut items[..mid]);
            self.sort(&mut items[mid..]);
            let (mut i, mut j) = (0, mid);
            while i < j && j < len {
                unsafe {
                    if items.get_unchecked(i) > items.get_unchecked(j) {
                        shift_back(items, j, i);
                        j = j.unchecked_add(1);
                    }
                    i = i.unchecked_add(1);
                }
            }
        }
    }
}

pub struct Quick;

impl Sort for Quick {
    fn sort(&self, items: &mut [impl Ord]) {
        if items.len() > 1 {
            let mut pivot = 0;
            for i in 1..items.len() {
                unsafe {
                    if items.get_unchecked(i) < items.get_unchecked(pivot) {
                        shift_back(items, i, pivot);
                        pivot = pivot.unchecked_add(1);
                    }
                }
            }
            if pivot == 0 {
                pivot = 1;
            }
            self.sort(&mut items[..pivot]);
            self.sort(&mut items[pivot..]);
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

#[cfg(all(test, feature = "sort"))]
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
        println!("Elapsed: {:?}", before.elapsed());

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
    fn merge_sort_int() {
        test(Merge, ten_of(int));
        test(Merge, hundred_of(int));
        test(Merge, thousand_of(int));
    }

    #[test]
    fn merge_sort_string() {
        test(Merge, ten_of(string));
        test(Merge, hundred_of(string));
        test(Merge, thousand_of(string));
    }

    #[test]
    fn merge_sort_custom() {
        test(Merge, ten_of(Custom::random));
        test(Merge, hundred_of(Custom::random));
        test(Merge, thousand_of(Custom::random));
    }

    #[test]
    fn quick_sort_int() {
        test(Quick, ten_of(int));
        test(Quick, hundred_of(int));
        test(Quick, thousand_of(int));
    }

    #[test]
    fn quick_sort_string() {
        test(Quick, ten_of(string));
        test(Quick, hundred_of(string));
        test(Quick, thousand_of(string));
    }

    #[test]
    fn quick_sort_custom() {
        test(Quick, ten_of(Custom::random));
        test(Quick, hundred_of(Custom::random));
        test(Quick, thousand_of(Custom::random));
    }

    #[test]
    fn ord_float() {
        test(Stdlib, hundred_of(float).map(OrdF32));
        test(Insertion, hundred_of(float).map(OrdF32));
        test(Selection, hundred_of(float).map(OrdF32));
        test(Merge, hundred_of(float).map(OrdF32));
        test(Quick, hundred_of(float).map(OrdF32));
    }

    #[test]
    #[ignore]
    fn fifty_thousand() {
        test(Stdlib, fifty_thousand_of(int));
        test(Insertion, fifty_thousand_of(int));
        test(Selection, fifty_thousand_of(int));
        test(Merge, fifty_thousand_of(int));
        test(Quick, fifty_thousand_of(int));
    }
}
