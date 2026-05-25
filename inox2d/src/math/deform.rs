use glam::Vec2;
use simd_aligned::VecSimd;

use crate::math::types::Vec2x4;

/// Different kinds of deform.
// TODO: Meshgroup.
pub(crate) enum Deform {
	/// Specifying a displacement for every vertex.
	Direct(VecSimd<Vec2x4>),
}

/// Element-wise add direct deforms up and write result.
pub(crate) fn linear_combine<'deforms>(
	direct_deforms: impl Iterator<Item = &'deforms VecSimd<Vec2x4>>,
	result: &mut VecSimd<Vec2x4>,
) {
	result.flat_mut().iter_mut().for_each(|deform| *deform = Vec2::ZERO);

	for direct_deform in direct_deforms {
		if direct_deform.len() != result.len() {
			panic!("Trying to combine direct deformations with wrong dimensions.");
		}

		result
			.iter_mut()
			.zip(direct_deform.iter())
			.for_each(|(sum, addition)| *sum += *addition);
	}
}
