use glam::Vec2;
use simd_aligned::arch::f32x8;
use simd_aligned::VecSimd;

use crate::math::types::Vec2x4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolateMode {
	/// Round to nearest
	Nearest,
	/// Linear interpolation
	Linear,
	// there's more but I'm not adding them for now.
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InterpRange<T> {
	pub beg: T,
	pub end: T,
}

impl<T> InterpRange<T> {
	#[inline]
	pub fn new(beg: T, end: T) -> Self {
		Self { beg, end }
	}
}

impl InterpRange<Vec2> {
	#[inline]
	pub fn to_x(self) -> InterpRange<f32> {
		InterpRange {
			beg: self.beg.x,
			end: self.end.x,
		}
	}

	#[inline]
	pub fn to_y(self) -> InterpRange<f32> {
		InterpRange {
			beg: self.beg.y,
			end: self.end.y,
		}
	}
}

#[inline]
fn interpolate_nearest(t: f32, range_in: InterpRange<f32>, range_out: InterpRange<f32>) -> f32 {
	debug_assert!(
		range_in.beg <= t && t <= range_in.end,
		"{} <= {} <= {}",
		range_in.beg,
		t,
		range_in.end
	);

	if (range_in.end - t) < (t - range_in.beg) {
		range_out.end
	} else {
		range_out.beg
	}
}

#[inline]
fn interpolate_linear(t: f32, range_in: InterpRange<f32>, range_out: InterpRange<f32>) -> f32 {
	debug_assert!(
		range_in.beg <= t && t <= range_in.end,
		"{} is out of input range [{}, {}]",
		t,
		range_in.beg,
		range_in.end,
	);

	(t - range_in.beg) * (range_out.end - range_out.beg) / (range_in.end - range_in.beg) + range_out.beg
}

#[inline]
pub fn interpolate_f32(t: f32, range_in: InterpRange<f32>, range_out: InterpRange<f32>, mode: InterpolateMode) -> f32 {
	match mode {
		InterpolateMode::Nearest => interpolate_nearest(t, range_in, range_out),
		InterpolateMode::Linear => interpolate_linear(t, range_in, range_out),
	}
}

#[inline]
pub fn interpolate_vec2(
	t: f32,
	range_in: InterpRange<f32>,
	range_out: InterpRange<Vec2>,
	mode: InterpolateMode,
) -> Vec2 {
	let x = interpolate_f32(t, range_in, range_out.to_x(), mode);
	let y = interpolate_f32(t, range_in, range_out.to_y(), mode);
	Vec2 { x, y }
}

pub fn interpolate_f32s_additive(
	t: f32,
	range_in: InterpRange<f32>,
	range_out: InterpRange<&VecSimd<f32x8>>,
	mode: InterpolateMode,
	out: &mut VecSimd<f32x8>,
) {
	for ((&ob, &oe), o) in range_out
		.beg
		.flat()
		.iter()
		.zip(range_out.end.flat())
		.zip(out.flat_mut())
	{
		*o += interpolate_f32(t, range_in, InterpRange::new(ob, oe), mode);
	}
}

pub fn interpolate_vec2s_additive(
	t: f32,
	range_in: InterpRange<f32>,
	range_out: InterpRange<&VecSimd<Vec2x4>>,
	mode: InterpolateMode,
	out: &mut VecSimd<Vec2x4>,
) {
	for ((&ob, &oe), o) in range_out
		.beg
		.flat()
		.iter()
		.zip(range_out.end.flat())
		.zip(out.flat_mut())
	{
		*o += interpolate_vec2(t, range_in, InterpRange::new(ob, oe), mode);
	}
}

#[inline]
pub fn bi_interpolate_f32(
	t: Vec2,
	range_in: InterpRange<Vec2>,
	out_top: InterpRange<f32>,
	out_bottom: InterpRange<f32>,
	mode: InterpolateMode,
) -> f32 {
	let beg = interpolate_f32(t.x, range_in.to_x(), out_top, mode);
	let end = interpolate_f32(t.x, range_in.to_x(), out_bottom, mode);
	interpolate_f32(t.y, range_in.to_y(), InterpRange::new(beg, end), mode)
}

#[inline]
pub fn bi_interpolate_vec2(
	t: Vec2,
	range_in: InterpRange<Vec2>,
	out_top: InterpRange<Vec2>,
	out_bottom: InterpRange<Vec2>,
	mode: InterpolateMode,
) -> Vec2 {
	let beg = interpolate_vec2(t.x, range_in.to_x(), out_top, mode);
	let end = interpolate_vec2(t.x, range_in.to_x(), out_bottom, mode);
	interpolate_vec2(t.y, range_in.to_y(), InterpRange::new(beg, end), mode)
}

pub fn bi_interpolate_f32s_additive(
	t: Vec2,
	range_in: InterpRange<Vec2>,
	out_top: InterpRange<&VecSimd<f32x8>>,
	out_bottom: InterpRange<&VecSimd<f32x8>>,
	mode: InterpolateMode,
	out: &mut VecSimd<f32x8>,
) {
	for (((&otb, &ote), (&obb, &obe)), o) in (out_top.beg.flat().iter().zip(out_top.end.flat()))
		.zip(out_bottom.beg.flat().iter().zip(out_bottom.end.flat()))
		.zip(out.flat_mut())
	{
		*o += bi_interpolate_f32(
			t,
			range_in,
			InterpRange::new(otb, ote),
			InterpRange::new(obb, obe),
			mode,
		)
	}
}

pub fn bi_interpolate_vec2s_additive(
	t: Vec2,
	range_in: InterpRange<Vec2>,
	out_top: InterpRange<&VecSimd<Vec2x4>>,
	out_bottom: InterpRange<&VecSimd<Vec2x4>>,
	mode: InterpolateMode,
	out: &mut VecSimd<Vec2x4>,
) {
	#[cfg(all(any(target_arch = "x86_64", target_arch = "x86")))]
	if is_x86_feature_detected!("avx2") {
		// SAFETY: You must check CPU features before calling a function
		// compiled using uplevel features.
		return unsafe { x86_64_avx2::bi_interpolate_vec2s_additive_avx2(t, range_in, out_top, out_bottom, mode, out) };
	}

	for (((&otb, &ote), (&obb, &obe)), o) in (out_top.beg.flat().iter().zip(out_top.end.flat()))
		.zip(out_bottom.beg.flat().iter().zip(out_bottom.end.flat()))
		.zip(out.flat_mut())
	{
		*o += bi_interpolate_vec2(
			t,
			range_in,
			InterpRange::new(otb, ote),
			InterpRange::new(obb, obe),
			mode,
		)
	}
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "x86")))]
mod x86_64_avx2;

#[cfg(test)]
mod tests {
	use super::*;
	use simd_aligned::VecSimd;

	/// Do the interpolation without an aligned array.
	///
	/// This only exists because simd_aligned doesn't provide a convenience
	/// method for initializing aligned arrays. DO NOT actually use this in
	/// production code as it is slow.
	fn bi_interpolate_vec2s_additive_unaligned(
		t: Vec2,
		range_in: InterpRange<Vec2>,
		out_top: InterpRange<&[Vec2]>,
		out_bottom: InterpRange<&[Vec2]>,
		mode: InterpolateMode,
		out: &mut [Vec2],
	) {
		let mut aligned_out_top_beg = VecSimd::with(Vec2::ZERO, out_top.beg.len());
		let mut aligned_out_top_end = VecSimd::with(Vec2::ZERO, out_top.end.len());
		let mut aligned_out_bottom_beg = VecSimd::with(Vec2::ZERO, out_bottom.beg.len());
		let mut aligned_out_bottom_end = VecSimd::with(Vec2::ZERO, out_bottom.end.len());
		let mut aligned_out = VecSimd::with(Vec2::ZERO, out.len());

		aligned_out_top_beg.flat_mut().copy_from_slice(out_top.beg);
		aligned_out_top_end.flat_mut().copy_from_slice(out_top.end);
		aligned_out_bottom_beg.flat_mut().copy_from_slice(out_bottom.beg);
		aligned_out_bottom_end.flat_mut().copy_from_slice(out_bottom.end);
		aligned_out.flat_mut().copy_from_slice(out);

		bi_interpolate_vec2s_additive(
			t,
			range_in,
			InterpRange::new(&aligned_out_top_beg, &aligned_out_top_end),
			InterpRange::new(&aligned_out_bottom_beg, &aligned_out_bottom_end),
			mode,
			&mut aligned_out,
		);

		out.copy_from_slice(aligned_out.flat());
	}

	#[test]
	fn test_linear_interpolation() {
		assert_eq!(
			interpolate_linear(0.0, InterpRange::new(0.0, 1.0), InterpRange::new(-5.0, 5.0)),
			-5.0
		);
		assert_eq!(
			interpolate_linear(1.0, InterpRange::new(0.0, 1.0), InterpRange::new(-5.0, 5.0)),
			5.0
		);
		assert_eq!(
			interpolate_linear(0.5, InterpRange::new(0.0, 1.0), InterpRange::new(-5.0, 5.0)),
			0.0
		);
		assert_eq!(
			interpolate_linear(0.0, InterpRange::new(-0.5, 0.0), InterpRange::new(-5.0, 5.0)),
			5.0
		);
	}

	#[test]
	fn test_bi_interpolate_vec2s_additive_linear() {
		// Test vectors randomly generated by printing out vectors from the
		// test bench.
		let t = Vec2::new(-0.09409535, 0.42562026);
		let range_in = InterpRange::new(Vec2::new(-0.64288485, -0.28533518), Vec2::new(0.424587, 0.45015025));
		let out_top = InterpRange::new(
			vec![
				Vec2::new(0.8977584, 0.80095005),
				Vec2::new(0.56985533, 0.072586596),
				Vec2::new(0.16844785, 0.7011595),
				Vec2::new(0.875262, 0.47915757),
				Vec2::new(0.85566926, 0.27344626),
				Vec2::new(0.92075604, 0.5903343),
				Vec2::new(0.9817382, 0.08148116),
				Vec2::new(0.09791291, 0.8297892),
				Vec2::new(0.559472, 0.31804848),
				Vec2::new(0.9877534, 0.75839597),
				Vec2::new(0.24279994, 0.86313826),
			],
			vec![
				Vec2::new(0.067943215, 0.116503954),
				Vec2::new(0.48291522, 0.65917104),
				Vec2::new(0.3740812, 0.7955075),
				Vec2::new(0.72866577, 0.67857397),
				Vec2::new(0.81998867, 0.75347126),
				Vec2::new(0.17495865, 0.6438539),
				Vec2::new(0.44213963, 0.8384317),
				Vec2::new(0.049688518, 0.82894814),
				Vec2::new(0.25530463, 0.15305531),
				Vec2::new(0.11776304, 0.6246206),
				Vec2::new(0.19543058, 0.4828195),
			],
		);

		let out_bottom = InterpRange::new(
			vec![
				Vec2::new(0.5181033, 0.27322984),
				Vec2::new(0.7298904, 0.99417347),
				Vec2::new(0.17261189, 0.6518882),
				Vec2::new(0.30163157, 0.20842135),
				Vec2::new(0.63835543, 0.59212905),
				Vec2::new(0.5530297, 0.34918338),
				Vec2::new(0.40671325, 0.04971701),
				Vec2::new(0.7481877, 0.50074065),
				Vec2::new(0.39253706, 0.78923327),
				Vec2::new(0.22278339, 0.6697099),
				Vec2::new(0.5387332, 0.25920153),
			],
			vec![
				Vec2::new(0.3036, 0.42818177),
				Vec2::new(0.79054284, 0.69804317),
				Vec2::new(0.40255207, 0.3045717),
				Vec2::new(0.8957452, 0.34809977),
				Vec2::new(0.20738769, 0.526328),
				Vec2::new(0.954661, 0.0610494),
				Vec2::new(0.9445757, 0.9847535),
				Vec2::new(0.53999746, 0.60769874),
				Vec2::new(0.03272271, 0.108495116),
				Vec2::new(0.7805427, 0.21582133),
				Vec2::new(0.24908477, 0.63060415),
			],
		);
		let mut out = vec![Vec2::ZERO; 11];

		bi_interpolate_vec2s_additive_unaligned(
			t,
			range_in,
			InterpRange::new(out_top.beg.as_slice(), out_top.end.as_slice()),
			InterpRange::new(out_bottom.beg.as_slice(), out_bottom.end.as_slice()),
			InterpolateMode::Linear,
			&mut out,
		);

		assert_eq!(
			out,
			vec![
				Vec2::new(0.40993863, 0.3560989),
				Vec2::new(0.75320375, 0.8263308),
				Vec2::new(0.29026896, 0.4825483),
				Vec2::new(0.6134979, 0.29028425),
				Vec2::new(0.43081963, 0.5570308),
				Vec2::new(0.7520994, 0.21495411),
				Vec2::new(0.6839332, 0.528427),
				Vec2::new(0.62221146, 0.56485415),
				Vec2::new(0.21407755, 0.43239254),
				Vec2::new(0.5105612, 0.44481146),
				Vec2::new(0.38410854, 0.45739365),
			]
		);
	}

	#[test]
	fn test_bi_interpolate_vec2s_additive_nearest() {
		let t = Vec2::new(-0.09409535, 0.42562026);
		let range_in = InterpRange::new(Vec2::new(-0.64288485, -0.28533518), Vec2::new(0.424587, 0.45015025));
		let out_top = InterpRange::new(
			vec![
				Vec2::new(0.8977584, 0.80095005),
				Vec2::new(0.56985533, 0.072586596),
				Vec2::new(0.16844785, 0.7011595),
				Vec2::new(0.875262, 0.47915757),
				Vec2::new(0.85566926, 0.27344626),
				Vec2::new(0.92075604, 0.5903343),
				Vec2::new(0.9817382, 0.08148116),
				Vec2::new(0.09791291, 0.8297892),
				Vec2::new(0.559472, 0.31804848),
				Vec2::new(0.9877534, 0.75839597),
				Vec2::new(0.24279994, 0.86313826),
			],
			vec![
				Vec2::new(0.067943215, 0.116503954),
				Vec2::new(0.48291522, 0.65917104),
				Vec2::new(0.3740812, 0.7955075),
				Vec2::new(0.72866577, 0.67857397),
				Vec2::new(0.81998867, 0.75347126),
				Vec2::new(0.17495865, 0.6438539),
				Vec2::new(0.44213963, 0.8384317),
				Vec2::new(0.049688518, 0.82894814),
				Vec2::new(0.25530463, 0.15305531),
				Vec2::new(0.11776304, 0.6246206),
				Vec2::new(0.19543058, 0.4828195),
			],
		);

		let out_bottom = InterpRange::new(
			vec![
				Vec2::new(0.5181033, 0.27322984),
				Vec2::new(0.7298904, 0.99417347),
				Vec2::new(0.17261189, 0.6518882),
				Vec2::new(0.30163157, 0.20842135),
				Vec2::new(0.63835543, 0.59212905),
				Vec2::new(0.5530297, 0.34918338),
				Vec2::new(0.40671325, 0.04971701),
				Vec2::new(0.7481877, 0.50074065),
				Vec2::new(0.39253706, 0.78923327),
				Vec2::new(0.22278339, 0.6697099),
				Vec2::new(0.5387332, 0.25920153),
			],
			vec![
				Vec2::new(0.3036, 0.42818177),
				Vec2::new(0.79054284, 0.69804317),
				Vec2::new(0.40255207, 0.3045717),
				Vec2::new(0.8957452, 0.34809977),
				Vec2::new(0.20738769, 0.526328),
				Vec2::new(0.954661, 0.0610494),
				Vec2::new(0.9445757, 0.9847535),
				Vec2::new(0.53999746, 0.60769874),
				Vec2::new(0.03272271, 0.108495116),
				Vec2::new(0.7805427, 0.21582133),
				Vec2::new(0.24908477, 0.63060415),
			],
		);
		let mut out = vec![Vec2::ZERO; 11];

		bi_interpolate_vec2s_additive_unaligned(
			t,
			range_in,
			InterpRange::new(out_top.beg.as_slice(), out_top.end.as_slice()),
			InterpRange::new(out_bottom.beg.as_slice(), out_bottom.end.as_slice()),
			InterpolateMode::Nearest,
			&mut out,
		);

		assert_eq!(
			out,
			vec![
				Vec2::new(0.3036, 0.42818177),
				Vec2::new(0.79054284, 0.69804317),
				Vec2::new(0.40255207, 0.3045717),
				Vec2::new(0.8957452, 0.34809977),
				Vec2::new(0.20738769, 0.526328),
				Vec2::new(0.954661, 0.0610494),
				Vec2::new(0.9445757, 0.9847535),
				Vec2::new(0.53999746, 0.60769874),
				Vec2::new(0.03272271, 0.108495116),
				Vec2::new(0.7805427, 0.21582133),
				Vec2::new(0.24908477, 0.63060415)
			]
		);

		let t = Vec2::new(-0.4, 0.0);
		let mut out = vec![Vec2::ZERO; 11];

		bi_interpolate_vec2s_additive_unaligned(
			t,
			range_in,
			InterpRange::new(out_top.beg.as_slice(), out_top.end.as_slice()),
			InterpRange::new(out_bottom.beg.as_slice(), out_bottom.end.as_slice()),
			InterpolateMode::Nearest,
			&mut out,
		);

		assert_eq!(
			out,
			vec![
				Vec2::new(0.8977584, 0.80095005),
				Vec2::new(0.56985533, 0.072586596),
				Vec2::new(0.16844785, 0.7011595),
				Vec2::new(0.875262, 0.47915757),
				Vec2::new(0.85566926, 0.27344626),
				Vec2::new(0.92075604, 0.5903343),
				Vec2::new(0.9817382, 0.08148116),
				Vec2::new(0.09791291, 0.8297892),
				Vec2::new(0.559472, 0.31804848),
				Vec2::new(0.9877534, 0.75839597),
				Vec2::new(0.24279994, 0.86313826)
			]
		);

		let t = Vec2::new(-0.4, 0.4);
		let mut out = vec![Vec2::ZERO; 11];

		bi_interpolate_vec2s_additive_unaligned(
			t,
			range_in,
			InterpRange::new(out_top.beg.as_slice(), out_top.end.as_slice()),
			InterpRange::new(out_bottom.beg.as_slice(), out_bottom.end.as_slice()),
			InterpolateMode::Nearest,
			&mut out,
		);

		assert_eq!(
			out,
			vec![
				Vec2::new(0.5181033, 0.27322984),
				Vec2::new(0.7298904, 0.99417347),
				Vec2::new(0.17261189, 0.6518882),
				Vec2::new(0.30163157, 0.20842135),
				Vec2::new(0.63835543, 0.59212905),
				Vec2::new(0.5530297, 0.34918338),
				Vec2::new(0.40671325, 0.04971701),
				Vec2::new(0.7481877, 0.50074065),
				Vec2::new(0.39253706, 0.78923327),
				Vec2::new(0.22278339, 0.6697099),
				Vec2::new(0.5387332, 0.25920153)
			]
		);

		let t = Vec2::new(0.4, 0.0);
		let mut out = vec![Vec2::ZERO; 11];

		bi_interpolate_vec2s_additive_unaligned(
			t,
			range_in,
			InterpRange::new(out_top.beg.as_slice(), out_top.end.as_slice()),
			InterpRange::new(out_bottom.beg.as_slice(), out_bottom.end.as_slice()),
			InterpolateMode::Nearest,
			&mut out,
		);

		assert_eq!(
			out,
			vec![
				Vec2::new(0.067943215, 0.116503954),
				Vec2::new(0.48291522, 0.65917104),
				Vec2::new(0.3740812, 0.7955075),
				Vec2::new(0.72866577, 0.67857397),
				Vec2::new(0.81998867, 0.75347126),
				Vec2::new(0.17495865, 0.6438539),
				Vec2::new(0.44213963, 0.8384317),
				Vec2::new(0.049688518, 0.82894814),
				Vec2::new(0.25530463, 0.15305531),
				Vec2::new(0.11776304, 0.6246206),
				Vec2::new(0.19543058, 0.4828195)
			]
		);
	}
}
