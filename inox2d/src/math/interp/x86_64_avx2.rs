use std::arch::x86_64;
use std::arch::x86_64::__m256;

use super::*;
use glam::Vec2;
use simd_aligned::traits::Simd;

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
		let t_x_delta = (t.x - range_in.beg.x) / (range_in.end.x - range_in.beg.x);
		let t_y_delta = (t.y - range_in.beg.y) / (range_in.end.y - range_in.beg.y);

		if t_x_delta.is_nan() || t_y_delta.is_nan() {
			for (((&otb, &ote), (&obb, &obe)), o) in (out_top.beg.iter().zip(out_top.end.iter()))
				.zip(out_bottom.beg.iter().zip(out_bottom.end.iter()))
				.zip(out.iter_mut())
			{
				let beg_lerp = (t_x_delta * (ote - otb) + otb).erase_nans(InterpRange::new(otb, ote));
				let end_lerp = (t_x_delta * (obe - obb) + obb).erase_nans(InterpRange::new(obb, obe));
				*o += (t_y_delta * (end_lerp - beg_lerp) + beg_lerp).erase_nans(InterpRange::new(beg_lerp, end_lerp));
			}
		} else {
			for (((&otb, &ote), (&obb, &obe)), o) in (out_top.beg.iter().zip(out_top.end.iter()))
				.zip(out_bottom.beg.iter().zip(out_bottom.end.iter()))
				.zip(out.iter_mut())
			{
				let beg_lerp = t_x_delta * (ote - otb) + otb;
				let end_lerp = t_x_delta * (obe - obb) + obb;
				*o += t_y_delta * (end_lerp - beg_lerp) + beg_lerp;
			}
		}
	}
}
