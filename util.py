import cutlass

cutlass_dtype_lut = {
  'e4m3': cutlass.Float8E4M3FN,
  'e2m1': cutlass.Float4E2M1FN,
  'e8m0': cutlass.Float8E8M0FNU,
  'fp8': cutlass.Float8E4M3FN,
  'fp16': cutlass.Float16,
  'fp32': cutlass.Float32
}

dtype_bytes = {
  'mxfp8': 1,
  'mxfp4': 0.5,
  'nvfp4': 0.5,
  'fp4': 0.5,
  'e2m1': 0.5,
  'e4m3': 1,
  'e8m0': 1,
  'fp8': 1,
  'fp16': 2,
  'fp32': 4,
}
