#![feature(portable_simd)]

extern crate nalgebra as na;

use cblas::sdot;
use na::DVector;

use std::arch::aarch64::{float32x4_t, vaddvq_f32, vmlaq_f32};
use std::mem::transmute;
use std::simd::f32x4;

pub fn dot_stdsimd4(xs: &[f32], ys: &[f32]) -> f32 {
    let mut simd_result = f32x4::splat(0.0);
    let mut scalar_result = 0.0;
    let stride = 4;

    let split_idx = std::cmp::min(xs.len(), ys.len()) / stride * stride;
    let (simd_xs, scalar_xs) = xs.split_at(split_idx);
    let (simd_ys, scalar_ys) = ys.split_at(split_idx);

    for (x, y) in simd_xs.chunks(stride).zip(simd_ys.chunks(stride)) {
        simd_result = simd_result + f32x4::from_slice(x) * f32x4::from_slice(y);
    }

    for (x_scalar, y_scalar) in scalar_xs.iter().zip(scalar_ys.iter()) {
        scalar_result += x_scalar * y_scalar;
    }

    scalar_result + simd_result.as_array().iter().sum::<f32>()
}

pub fn dot_stdsimd8(xs: &[f32], ys: &[f32]) -> f32 {
    let mut simd_result = std::simd::f32x8::splat(0.0);
    let mut scalar_result = 0.0;
    let stride = 8;

    let split_idx = std::cmp::min(xs.len(), ys.len()) / stride * stride;
    let (simd_xs, scalar_xs) = xs.split_at(split_idx);
    let (simd_ys, scalar_ys) = ys.split_at(split_idx);

    for (x, y) in simd_xs.chunks(stride).zip(simd_ys.chunks(stride)) {
        simd_result =
            simd_result + std::simd::f32x8::from_slice(x) * std::simd::f32x8::from_slice(y);
    }

    for (x_scalar, y_scalar) in scalar_xs.iter().zip(scalar_ys.iter()) {
        scalar_result += x_scalar * y_scalar;
    }

    scalar_result + simd_result.as_array().iter().sum::<f32>()
}

#[cfg(all(target_arch = "aarch64"))]
pub fn dot_neon(xs: &[f32], ys: &[f32]) -> f32 {
    // To do this, we want to call vmlaq_f32 for the 4xf32 multiply accumulated.
    let mut simd_result: float32x4_t = unsafe { transmute(f32x4::splat(0.0)) };
    let stride = 4;

    let split_idx = std::cmp::min(xs.len(), ys.len()) / stride * stride;
    let (simd_xs, scalar_xs) = xs.split_at(split_idx);
    let (simd_ys, scalar_ys) = ys.split_at(split_idx);

    for (x, y) in simd_xs.chunks(stride).zip(simd_ys.chunks(stride)) {
        unsafe {
            let simdx: float32x4_t = transmute(f32x4::from_slice(x));
            let simdy: float32x4_t = transmute(f32x4::from_slice(y));
            simd_result = vmlaq_f32(simd_result, simdx, simdy);
        }
    }

    let mut scalar_result = 0.0;
    for (x_scalar, y_scalar) in scalar_xs.iter().zip(scalar_ys.iter()) {
        scalar_result += x_scalar * y_scalar;
    }

    scalar_result + unsafe { vaddvq_f32(simd_result) }
}

pub fn dot_cblas(x: &[f32], y: &[f32]) -> f32 {
    unsafe { sdot(x.len() as i32, x, 1, y, 1) }
}

pub fn dot_nalgebra(x: &DVector<f32>, y: &DVector<f32>) -> f32 {
    x.dot(&y)
}

pub fn dot_unrolled4(x: &[f32]) -> f32 {
    let (mut txx1, mut txx2, mut txx3, mut txx4) = (0f32, 0f32, 0f32, 0f32);
    for i in (0..x.len()).step_by(4) {
        txx1 += x[i] * x[i];
        txx2 += x[i + 1] * x[i + 1];
        txx3 += x[i + 2] * x[i + 2];
        txx4 += x[i + 3] * x[i + 3];
    }
    // should also add up any overflow...
    txx1 + txx2 + txx3 + txx4
}

pub fn dot_unrolled8(x: &[f32]) -> f32 {
    let (mut txx1, mut txx2, mut txx3, mut txx4, mut txx5, mut txx6, mut txx7, mut txx8) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in (0..x.len()).step_by(8) {
        txx1 += x[i] * x[i];
        txx2 += x[i + 1] * x[i + 1];
        txx3 += x[i + 2] * x[i + 2];
        txx4 += x[i + 3] * x[i + 3];
        txx5 += x[i + 4] * x[i + 4];
        txx6 += x[i + 5] * x[i + 5];
        txx7 += x[i + 6] * x[i + 6];
        txx8 += x[i + 7] * x[i + 7];
    }
    // should also add up any overflow...
    txx1 + txx2 + txx3 + txx4 + txx5 + txx6 + txx7 + txx8
}

pub fn dot(x: &[f32]) -> f32 {
    let mut tx = 0f32;
    for i in 0..x.len() {
        tx += x[i] * x[i];
    }
    tx
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simple_dot() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(dot(&x), 140.0);
    }

    #[test]
    fn test_unrolled_dot() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(dot(&x), dot_unrolled4(&x));
        assert_eq!(dot(&x), dot_unrolled8(&x));
    }

    #[test]
    fn test_stdsimd_dot() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(dot(&x), dot_stdsimd4(&x, &x));
        assert_eq!(dot(&x), dot_stdsimd8(&x, &x));
    }

    #[test]
    fn test_neon_dot() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(dot(&x), dot_neon(&x, &x));
    }

    // #[test]
    fn test_cblas_dot() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(dot(&x), dot_cblas(&x, &x));
    }
}
