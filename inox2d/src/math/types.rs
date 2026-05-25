use std::ops::{Add, AddAssign};

use glam::Vec2;
use simd_aligned::{arch::f32x8, traits::Simd};

/// Wrapper type to allow using Vec2 with simd_aligned.
#[repr(transparent)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct Vec2x4(pub f32x8);

impl Simd for Vec2x4 {
	type Element = Vec2;
	type LanesType = [Vec2; Self::LANES];

	const LANES: usize = 4;

	fn splat(t: Self::Element) -> Self {
		Self(f32x8::new([t.x, t.y, t.x, t.y, t.x, t.y, t.x, t.y]))
	}

	fn as_array(&self) -> &[Self::Element] {
		// SAFETY: Four Vec2s are made up of eight f32s.
		// Both f32x8 and Vec2 are repr(C) and we are transparent.
		// Vec2's alignment will always be lower than f32x8.
		let self_array = unsafe { std::mem::transmute::<_, &Self::LanesType>(self) };
		self_array.as_ref()
	}

	fn sum(&self) -> Self::Element {
		self.as_array().iter().sum()
	}
}

impl Add for Vec2x4 {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		Self(self.0 + rhs.0)
	}
}

impl AddAssign for Vec2x4 {
	fn add_assign(&mut self, rhs: Self) {
		self.0 += rhs.0
	}
}
