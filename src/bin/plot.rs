use std::{any, array, mem::MaybeUninit, time::Instant};

use algorithms::*;
use gnuplot::{
    AlignType, AutoOption, Axes2D, AxesCommon, Coordinate, Figure, LegendOption, PlotOption, Tick,
};
use sort::Sort;

const N: usize = 18;
const XS: [usize; N] = const {
    let mut xs = [0; N];
    let (mut i, mut x) = (0, 1);
    while i < N {
        xs[i] = x;
        x *= 2;
        i += 1;
    }
    xs
};
const ITEMS_LEN: usize = XS[N - 1];

fn plot(sorter: impl Sort, axes: &mut Axes2D) {
    let mut items: [_; ITEMS_LEN] = array::from_fn(|_| MaybeUninit::uninit());
    let ys = XS.map(|x| {
        let items = rand::fill_with(rand::string)(&mut items[..x]);
        let before = Instant::now();
        sorter.sort(items);
        before.elapsed().as_nanos() as u64
    });
    axes.lines_points(
        XS,
        ys,
        &[
            PlotOption::LineWidth(2.0),
            PlotOption::PointSize(2.0),
            PlotOption::Caption(any::type_name_of_val(&sorter).split("::").last().unwrap()),
        ],
    );
}

fn main() {
    let mut figure = Figure::new();
    figure.set_title("Sorting Algorithms");

    let x_ticks = (1..)
        .map(|x| Tick::Major(2f32.powi(x), AutoOption::<f32>::Auto))
        .take(N);
    let y_ticks = (1..)
        .zip([
            "10ns", "100ns", "1µs", "10µs", "100µs", "1ms", "10ms", "100ms", "1s", "10s", "100s",
        ])
        .map(|(y, label)| Tick::Major(10f32.powi(y), AutoOption::Fix(label)));

    let mut axes = figure
        .axes2d()
        .set_legend(
            Coordinate::Graph(0.05),
            Coordinate::Graph(0.95),
            &[LegendOption::Placement(
                AlignType::AlignLeft,
                AlignType::AlignTop,
            )],
            &[],
        )
        .set_x_grid(true)
        .set_x_label("Array length", &[])
        .set_x_ticks_custom(x_ticks, &[], &[])
        .set_x_log(Some(2.0))
        .set_y_grid(true)
        .set_y_label("Time (ns)", &[])
        .set_y_ticks_custom(y_ticks, &[], &[])
        .set_y_log(Some(10.0));

    plot(sort::Insertion, &mut axes);
    plot(sort::Selection, &mut axes);
    plot(sort::Merge, &mut axes);
    plot(sort::Quick, &mut axes);
    plot(sort::Stdlib, &mut axes);

    figure.show().unwrap();
}
