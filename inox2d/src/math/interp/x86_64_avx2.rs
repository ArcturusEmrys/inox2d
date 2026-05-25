use std::arch::x86_64;
use std::arch::x86_64::__m256;

use super::*;
use glam::Vec2;
use simd_aligned::traits::Simd;

trait AsAvx256Register {
	fn as_avx(&self) -> &[__m256];

	fn as_avx_mut(&mut self) -> &mut [__m256];
}

impl AsAvx256Register for [Vec2x4] {
	fn as_avx(&self) -> &[__m256] {
		//SAFETY: Vec2x4 is a repr(transparent) wrapper around f32x8, which is
		//itself repr(C) and has the same alignment and size requirements as
		//__m256. In fact, it will actually become __m256 (making this a no-op)
		//if the binary is set to be compiled using AVX2.
		unsafe { std::mem::transmute(self) }
	}

	fn as_avx_mut(&mut self) -> &mut [__m256] {
		//SAFETY: See above.
		unsafe { std::mem::transmute(self) }
	}
}

const _: () = assert!(align_of::<__m256>() % align_of::<Vec2>() == 0);

#[target_feature(enable = "avx2")]
pub unsafe fn bi_interpolate_vec2s_additive_avx2(
	t: Vec2,
	range_in: InterpRange<Vec2>,
	out_top: InterpRange<&VecSimd<Vec2x4>>,
	out_bottom: InterpRange<&VecSimd<Vec2x4>>,
	mode: InterpolateMode,
	out: &mut VecSimd<Vec2x4>,
) {
	if mode == InterpolateMode::Nearest {
		let nearest = if (range_in.end.x - t.x) < (t.x - range_in.beg.x) {
			if (range_in.end.y - t.y) < (t.y - range_in.beg.y) {
				out_bottom.end
			} else {
				out_top.end
			}
		} else {
			if (range_in.end.y - t.y) < (t.y - range_in.beg.y) {
				out_bottom.beg
			} else {
				out_top.beg
			}
		};

		for (out, nearest) in (&mut *out).iter_mut().zip((&*nearest).iter()) {
			*out += *nearest;
		}
	} else {
		for (((&otb, &ote), (&obb, &obe)), o) in ((&*out_top.beg).as_avx().iter().zip((&*out_top.end).as_avx().iter()))
			.zip(
				(&*out_bottom.beg)
					.as_avx()
					.iter()
					.zip((&*out_bottom.end).as_avx().iter()),
			)
			.zip((&mut *out).as_avx_mut().iter_mut())
		{
			/*
			let beg_lerp = (t.x - range_in.beg.x) * (ote - otb) / (range_in.end.x - range_in.beg.x) + otb;
			let end_lerp = (t.x - range_in.beg.x) * (obe - obb) / (range_in.end.x - range_in.beg.x) + obb;
			let out = (t.y - range_in.beg.y) * (end_lerp - beg_lerp) / (range_in.end.y - range_in.beg.y) + beg_lerp; */

			let t_x_delta = x86_64::_mm256_set1_ps((t.x - range_in.beg.x) / (range_in.end.x - range_in.beg.x));
			let t_y_delta = x86_64::_mm256_set1_ps((t.y - range_in.beg.y) / (range_in.end.y - range_in.beg.y));

			let beg_lerp =
				x86_64::_mm256_add_ps(x86_64::_mm256_mul_ps(x86_64::_mm256_sub_ps(ote, otb), t_x_delta), otb);
			let end_lerp =
				x86_64::_mm256_add_ps(x86_64::_mm256_mul_ps(x86_64::_mm256_sub_ps(obe, obb), t_x_delta), obb);
			let out_lerp = x86_64::_mm256_add_ps(
				x86_64::_mm256_mul_ps(x86_64::_mm256_sub_ps(end_lerp, beg_lerp), t_y_delta),
				beg_lerp,
			);

			*o = x86_64::_mm256_add_ps(*o, out_lerp);
		}
	}
}
