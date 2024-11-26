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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rand::*;

    #[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
    struct Custom {
        int: u8,
        string: String,
    }

    fn test<const N: usize>(sorter: impl Sort, mut items: [impl Ord; N]) {
        assert!(!is_sorted(&items));
        sorter.sort(&mut items);
        assert!(is_sorted(&items));
    }

    #[test]
    fn stdlib_sort_ints() {
        test(Stdlib, array_of(int));
    }

    #[test]
    fn stdlib_sort_strings() {
        test(Stdlib, array_of(string));
    }

    #[test]
    fn stdlib_sort_custom() {
        test(
            Stdlib,
            array_of(|| Custom {
                int: range(1..5),
                string: string(),
            }),
        );
    }
}
