use rand::{
    distributions::{uniform::SampleUniform, Alphanumeric, DistString},
    prelude::*,
};
use std::{
    array,
    cell::RefCell,
    ops::{Range, RangeInclusive},
};

const ARRAY_SIZE: usize = 20;
const STRING_SIZE: RangeInclusive<usize> = 1..=50;

pub fn array_of<T>(mut randomizer: impl FnMut() -> T) -> [T; ARRAY_SIZE] {
    array::from_fn(|_| randomizer())
}

thread_local! {
    static RNG: RefCell<ThreadRng> = RefCell::new(thread_rng());
}

pub fn int() -> i32 {
    RNG.with_borrow_mut(|rng| rng.gen())
}

pub fn string() -> String {
    RNG.with_borrow_mut(|rng| {
        let len = rng.gen_range(STRING_SIZE);
        Alphanumeric.sample_string(rng, len)
    })
}

pub fn range<T: SampleUniform + PartialOrd>(range: Range<T>) -> T {
    RNG.with_borrow_mut(|rng| rng.gen_range(range))
}
