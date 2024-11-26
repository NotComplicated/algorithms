use rand::{
    distributions::{uniform::SampleUniform, Alphanumeric, DistString, Standard},
    prelude::*,
};
use std::{array, cell::RefCell, ops::Range};

fn array_of<T, const N: usize>(mut randomizer: impl FnMut() -> T) -> [T; N] {
    array::from_fn(|_| randomizer())
}

pub fn ten_of<T>(randomizer: impl FnMut() -> T) -> [T; 10] {
    array_of(randomizer)
}

pub fn hundred_of<T>(randomizer: impl FnMut() -> T) -> [T; 100] {
    array_of(randomizer)
}

pub fn thousand_of<T>(randomizer: impl FnMut() -> T) -> [T; 1_000] {
    array_of(randomizer)
}

pub fn fifty_thousand_of<T>(randomizer: impl FnMut() -> T) -> [T; 50_000] {
    array_of(randomizer)
}

thread_local! {
    static RNG: RefCell<ThreadRng> = RefCell::new(thread_rng());
}

fn gen<T>() -> T
where
    Standard: Distribution<T>,
{
    RNG.with_borrow_mut(|rng| rng.gen())
}

pub fn int() -> i32 {
    gen()
}

pub fn float() -> f32 {
    gen()
}

pub fn string() -> String {
    RNG.with_borrow_mut(|rng| {
        let len = rng.gen_range(1..=50);
        Alphanumeric.sample_string(rng, len)
    })
}

pub fn range<T: SampleUniform + PartialOrd>(range: Range<T>) -> T {
    RNG.with_borrow_mut(|rng| rng.gen_range(range))
}
