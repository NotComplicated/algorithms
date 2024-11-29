use ndarray::prelude::*;
use num_traits::Zero;
use rand::{
    distributions::{uniform::SampleUniform, Alphanumeric, DistString, Standard},
    prelude::*,
};
use std::{array, cell::RefCell, mem::MaybeUninit, ops::Range};

fn array_of<T, const N: usize>(mut f: impl FnMut() -> T) -> [T; N] {
    array::from_fn(|_| f())
}

pub fn ten_of<T>(f: impl FnMut() -> T) -> [T; 10] {
    array_of(f)
}

pub fn hundred_of<T>(f: impl FnMut() -> T) -> [T; 100] {
    array_of(f)
}

pub fn thousand_of<T>(f: impl FnMut() -> T) -> [T; 1_000] {
    array_of(f)
}

pub fn fifty_thousand_of<T>(f: impl FnMut() -> T) -> [T; 50_000] {
    array_of(f)
}

pub fn matrix_of<T: Clone, const M: usize, const N: usize>(mut f: impl FnMut() -> T) -> Array2<T> {
    arr2(&array_of::<_, M>(|| array_of::<_, N>(&mut f)))
}

pub fn fill_with<T>(mut f: impl FnMut() -> T) -> impl FnMut(&mut [MaybeUninit<T>]) -> &mut [T] {
    move |items| {
        for item in items.iter_mut() {
            item.write(f());
        }
        unsafe { &mut *(items as *mut _ as *mut _) }
    }
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

pub fn zero<Z: Zero>() -> Z {
    Z::zero()
}
