use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

use glam::Vec2;
use simd_aligned::arch::{f32x4, f32x8};
use simd_aligned::traits::Simd;

/// Wrapper type to allow using Vec2 with simd_aligned.
#[repr(transparent)]
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct Vec2x4(pub f32x8);

impl Vec2x4 {
	/// Retrieve all X components of the Vec2.
	pub fn x(&self) -> f32x4 {
		let arr = self.0.as_array_ref();
		f32x4::new([arr[0], arr[2], arr[4], arr[6]])
	}

	/// Construct a new Vec2 from its X components only.
	pub fn from_x(x: f32x4) -> Self {
		let arr = x.as_array_ref();

		Vec2x4(f32x8::new([arr[0], 0.0, arr[1], 0.0, arr[2], 0.0, arr[3], 0.0]))
	}

	/// Retrieve all Y components of the Vec2.
	pub fn y(&self) -> f32x4 {
		let arr = self.0.as_array_ref();
		f32x4::new([arr[1], arr[3], arr[5], arr[7]])
	}

	/// Construct a new Vec2 from its Y components only.
	pub fn from_y(y: f32x4) -> Self {
		let arr = y.as_array_ref();

		Vec2x4(f32x8::new([0.0, arr[0], 0.0, arr[1], 0.0, arr[2], 0.0, arr[3]]))
	}

	/// Construct a new Vec2 from its X and Y components.
	pub fn from_x_and_y(x: f32x4, y: f32x4) -> Self {
		Self::from_x(x) + Self::from_y(y)
	}
}

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

impl Sub for Vec2x4 {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output {
		Self(self.0 - rhs.0)
	}
}

impl SubAssign for Vec2x4 {
	fn sub_assign(&mut self, rhs: Self) {
		self.0 -= rhs.0
	}
}

impl Mul<f32> for Vec2x4 {
	type Output = Self;

	fn mul(self, rhs: f32) -> Self::Output {
		Self(self.0 * rhs)
	}
}

impl Mul<Vec2x4> for f32 {
	type Output = Vec2x4;

	fn mul(self, rhs: Vec2x4) -> Self::Output {
		Vec2x4(self * rhs.0)
	}
}

impl Div<f32> for Vec2x4 {
	type Output = Self;

	fn div(self, rhs: f32) -> Self::Output {
		Self(self.0 / rhs)
	}
}

impl Div<Vec2x4> for f32 {
	type Output = Vec2x4;

	fn div(self, rhs: Vec2x4) -> Self::Output {
		Vec2x4(self / rhs.0)
	}
}
