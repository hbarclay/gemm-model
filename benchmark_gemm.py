import argparse
import csv
import time

import sys
sys.path.append('./cutlass/CuTeDSL/blackwell/')
from dense_gemm_persistent import run as run_gemm
from dense_blockscaled_gemm_persistent import run as run_bsgemm
import cutlass

from util import *

CTA_SHAPES =  [
  (64,64),
  (64,128),
  (64,192),
  (64,256),
  (128,64),
  (128,128),
  (128,192),
  (128,256),
]

CTA_SHAPES_BS = [
  (128,64),
  (128,128),
  (128,192),
  (128,256),
]

CGA_SHAPES =  [
  (2,1),
  (2,2),
]

def parse_args():
  parser = argparse.ArgumentParser(description='Profile CuTeDSL GEMMs')
  parser.add_argument('--input_csv', type=str, default="testlists/dsv3.csv", help='Path to input CSV with GEMM-mnk')
  parser.add_argument('--dtype', type=str, default="fp8", choices=['fp32', 'fp16', 'fp8', 'mxfp8', 'mxfp4', 'nvfp4'], help='Input/output datatype')
  parser.add_argument('--output_csv', type=str, required=False, help='Path to output CSV for results.')

  args = parser.parse_args()

  if args.output_csv is None:
    args.output_csv = f"results/dsv3_{args.dtype}.csv"

  return args

def pad_mnk(m, n, k, dtype):
  # Really, only the contiguous dimension must be 16-byte aligned, but just pad all dimensions here for simplicity
  elem_alignment = 16 // dtype_bytes[dtype]
  def align(x):
    return int(((x + elem_alignment - 1) // elem_alignment) * elem_alignment)
  return (align(m),align(n),align(k))

def main():
  args = parse_args()

  with open(args.input_csv, 'r') as f:
    reader = csv.reader(f)
    # assume column order is (m,n,k)
    header = next(reader)
    mnk_rows = [list(map(int, row)) for row in reader]

  acc_dtype = 'fp32'
  use_tma_store = True
  sf_dtype = ''
  sf_vec_size = 0

  is_blockscaled = args.dtype.startswith(('mx', 'nv'))
  cta_shapes = CTA_SHAPES_BS if is_blockscaled else CTA_SHAPES

  output_data = []
  for m,n,k in mnk_rows:
    for cta_m, cta_n in cta_shapes:
      for cluster_m, cluster_n in CGA_SHAPES:

        padded_m, padded_n, padded_k = pad_mnk(m,n,k,args.dtype)

        if is_blockscaled:
          if args.dtype == 'mxfp8':
            in_dtype = 'e4m3'
            sf_dtype = 'e8m0'
            sf_vec_size = 32
          elif args.dtype == 'mxfp4':
            in_dtype = 'e2m1'
            sf_dtype = 'e8m0'
            sf_vec_size = 32
          elif args.dtype == 'nvfp4':
            in_dtype = 'e2m1'
            sf_dtype = 'e8m0'
            sf_vec_size = 16
          else:
            raise TypeError(f'Unsupported datatype {args.dtype}')

          out_dtype = 'fp32'
          runtime = run_bsgemm(
              (padded_m,padded_n,padded_k,1),
              cutlass_dtype_lut[in_dtype],
              cutlass_dtype_lut[sf_dtype],
              sf_vec_size,
              cutlass_dtype_lut[out_dtype],
              'k',
              'k',
              'n',
              (cta_m * 2, cta_n),
              (cluster_m, cluster_n),
              0,
              2,
              5,
              True,
              True,
          )

        else:
          in_dtype = args.dtype
          out_dtype = args.dtype

          runtime = run_gemm(
              (padded_m,padded_n,padded_k,1),
              cutlass_dtype_lut[in_dtype],
              cutlass_dtype_lut[out_dtype],
              cutlass_dtype_lut[acc_dtype],
              'k',
              'k',
              'n',
              (cta_m * 2, cta_n),
              (cluster_m, cluster_n),
              True,
              use_tma_store,
              0,
              2,
              5,
              True,
              True,
          )


        output_data.append({
          'in_dtype': in_dtype,
          'acc_dtype': acc_dtype,
          'out_dtype': out_dtype,
          'sf_dtype': sf_dtype,
          'sf_vec_size': sf_vec_size,
          'a_major': 'k',
          'b_major': 'k',
          'c_major': 'n',
          'mma_m': cta_m * 2,
          'mma_n': cta_n,
          'cta_m': cta_m,
          'cta_n': cta_n,
          'cluster_m': cluster_m,
          'cluster_n': cluster_n,
          'm': m,
          'n': n,
          'k': k,
          'runtime_us': runtime
        })

  with open(args.output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
    writer.writeheader()
    writer.writerows(output_data)

if __name__ == '__main__':
  main()

