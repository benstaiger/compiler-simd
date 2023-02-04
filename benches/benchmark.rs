use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DVector;
// test ndarray as well?
use simd::*;


pub fn dot_benchmark(c: &mut Criterion) {
    let x = vec![0.0; 1024*1024];
    c.bench_function("for loop (2^20 f32)", |b| b.iter(|| dot(black_box(&x))));
    c.bench_function("unrolled(4) for loop (2^20 f32)", |b| b.iter(|| dot_unrolled4(black_box(&x))));
    c.bench_function("unrolled(8) for loop (2^20 f32)", |b| b.iter(|| dot_unrolled8(black_box(&x))));
    c.bench_function("std::simd(4) (2^20 f32)", |b| b.iter(|| dot_stdsimd4(black_box(&x), black_box(&x))));
    c.bench_function("std::simd(8) (2^20 f32)", |b| b.iter(|| dot_stdsimd8(black_box(&x), black_box(&x))));
    let x_nalgebra = DVector::from_element(1024*1024, 0.0);
    c.bench_function("nalgebra", |b| b.iter(|| dot_nalgebra(black_box(&x_nalgebra), black_box(&x_nalgebra))));
}

criterion_group!(benches, dot_benchmark);
criterion_main!(benches);