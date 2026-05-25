use criterion::{criterion_group, criterion_main, Criterion};
use glam::Vec2;
use inox2d::math::{
	interp::{bi_interpolate_vec2s_additive, InterpRange, InterpolateMode},
	types::Vec2x4,
};
use rand::{random, random_range};
use simd_aligned::VecSimd;

fn random_vec2_data(size: usize) -> VecSimd<Vec2x4> {
	let mut out = VecSimd::with(Vec2::ZERO, size);
	for value in out.flat_mut() {
		*value = Vec2::new(random(), random());
	}

	out
}

fn bi_interpolate_vec2s_additive_bench(criterion: &mut Criterion) {
	criterion.bench_function("bi_interpolate_vec2s_additive_nearest", |b| {
		b.iter_batched(
			|| {
				let in_top_left = random_vec2_data(1_000_000);
				let in_top_right = random_vec2_data(1_000_000);
				let in_bottom_left = random_vec2_data(1_000_000);
				let in_bottom_right = random_vec2_data(1_000_000);
				let out = VecSimd::with(Vec2::ZERO, 1_000_000);

				let in_top = InterpRange::new(in_top_left, in_top_right);
				let in_bottom = InterpRange::new(in_bottom_left, in_bottom_right);

				let range_in = InterpRange::new(
					Vec2::new(random_range(-1.0..0.0), random_range(-1.0..0.0)),
					Vec2::new(random_range(0.0..1.0), random_range(0.0..1.0)),
				);

				let t = Vec2::new(
					random_range(range_in.beg.x..range_in.end.x),
					random_range(range_in.beg.y..range_in.end.y),
				);

				(t, range_in, in_top, in_bottom, out)
			},
			|(t, range_in, in_top, in_bottom, mut out)| {
				let in_top_slice = InterpRange::new(&in_top.beg, &in_top.end);
				let in_bottom_slice = InterpRange::new(&in_bottom.beg, &in_bottom.end);

				bi_interpolate_vec2s_additive(
					t,
					range_in,
					in_top_slice,
					in_bottom_slice,
					InterpolateMode::Nearest,
					&mut out,
				);
			},
			criterion::BatchSize::SmallInput,
		);
	});

	criterion.bench_function("bi_interpolate_vec2s_additive_linear", |b| {
		b.iter_batched(
			|| {
				let in_top_left = random_vec2_data(1_000_000);
				let in_top_right = random_vec2_data(1_000_000);
				let in_bottom_left = random_vec2_data(1_000_000);
				let in_bottom_right = random_vec2_data(1_000_000);
				let out = VecSimd::with(Vec2::ZERO, 1_000_000);

				let in_top = InterpRange::new(in_top_left, in_top_right);
				let in_bottom = InterpRange::new(in_bottom_left, in_bottom_right);

				let range_in = InterpRange::new(
					Vec2::new(random_range(-1.0..0.0), random_range(-1.0..0.0)),
					Vec2::new(random_range(0.0..1.0), random_range(0.0..1.0)),
				);

				let t = Vec2::new(
					random_range(range_in.beg.x..range_in.end.x),
					random_range(range_in.beg.y..range_in.end.y),
				);

				(t, range_in, in_top, in_bottom, out)
			},
			|(t, range_in, in_top, in_bottom, mut out)| {
				let in_top_slice = InterpRange::new(&in_top.beg, &in_top.end);
				let in_bottom_slice = InterpRange::new(&in_bottom.beg, &in_bottom.end);

				bi_interpolate_vec2s_additive(
					t,
					range_in,
					in_top_slice,
					in_bottom_slice,
					InterpolateMode::Linear,
					&mut out,
				);
			},
			criterion::BatchSize::SmallInput,
		);
	});
}

criterion_main!(benches);
criterion_group!(benches, bi_interpolate_vec2s_additive_bench);
