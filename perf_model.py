import math
from util import *

class PerfModel():
  def predict(self):
    pass

  def print_summary(self):
    pass

class SOLModel(PerfModel):
  def __init__(self, gpu, model_opts):
    self.gpu = gpu
    self.cfg = model_opts

  def _dram_sol(self):
    return (self.gpu.dram_bus_width * 2) * self.gpu.dram_clk_mhz * (10 ** 6) / 8

  def _mma_flops(self, dtype):
    if dtype in ['e2m1']:
      dtype = 'fp4'
    return self.gpu.mma_flops[dtype]

  def _math_sol(self, dtype):
    return self.gpu.sm_clk_mhz * self.gpu.num_sms * self._mma_flops(dtype) * (10 ** 6)

  def predict(self, m, n, k, in_dtype, out_dtype, sf_dtype, sf_vec_size, **kwargs):
    in_dtype = 'fp4' if in_dtype in ['e2m1'] else in_dtype
    in_dtype = 'fp8' if in_dtype in ['e3m2'] else in_dtype
    out_dtype = 'fp4' if out_dtype in ['e2m1'] else out_dtype
    out_dtype = 'fp8' if out_dtype in ['e3m2'] else out_dtype
    problem_flops = 2 * m * n * k
    problem_bytes = (m*k+k*n)*dtype_bytes[in_dtype] + (m*n)*dtype_bytes[out_dtype]
    if sf_vec_size:
      sf_load_bytes = (m*(k/sf_vec_size) + n*(k/sf_vec_size)) * dtype_bytes[sf_dtype]
      problem_bytes += sf_load_bytes

    math_us = (problem_flops / self._math_sol(in_dtype)) * (10 ** 6)
    dram_us = ((problem_bytes / self._dram_sol()) * 10 ** 6)

    return {
      'runtime': max(math_us, dram_us),
      'bound': "MATH" if math_us > dram_us else "DRAM"
    }

  def print_summary(self, result):
    print(result)


class WSPersistentGEMMModel(PerfModel):
  def __init__(self, gpu, model_opts):
    self.gpu = gpu
    self.cfg = model_opts

  def _fixed_overhead(self):
    if self.cfg.fixed_overhead_cycles:
      return (self.cfg.fixed_overhead_cycles / (self.gpu.sm_clk_mhz * (10**6)))

    return 0

  def _mma_flops(self, dtype):
    if dtype in ['e2m1']:
      dtype = 'fp4'
    return self.gpu.mma_flops[dtype]

  def _dram_sol(self):
    return (self.gpu.dram_bus_width * 2) * self.gpu.dram_clk_mhz * (10 ** 6) / 8

  def _math_sol(self, dtype, sms):
    return self.gpu.sm_clk_mhz * sms * self._mma_flops(dtype) * (10 ** 6)

  def _dma(self, sms, cta_m, cta_n, mma_k, cluster_m, cluster_n, dtype, sf_vec_size=0, sf_dtype=''):
    a_sf = 0
    b_sf = 0

    a_tile = cta_m * mma_k * dtype_bytes[dtype]
    if sf_vec_size:
      a_sf = cta_m * (mma_k / sf_vec_size) * dtype_bytes.get(sf_dtype, 0)

    b_tile = cta_n * mma_k * dtype_bytes[dtype]
    if sf_vec_size:
      b_sf = cta_n * (mma_k / sf_vec_size) * dtype_bytes.get(sf_dtype, 0)
    total_bytes = sms * (((a_tile+a_sf)/cluster_n) + ((b_tile+b_sf)/cluster_m))

    return total_bytes / self._dram_sol()

  def _ep_adjustment(self):
    if self.cfg.epilogue_min_latency:
      return (self.cfg.epilogue_min_latency / (self.gpu.sm_clk_mhz * (10**6)))

    return 0.0

  def _mainloop_one_wave(self, sms, cta_m, cta_n, k, cluster_m, cluster_n, in_dtype, out_dtype, sf_vec_size=0, sf_dtype=''):
    dma = self._dma(sms, cta_m, cta_n, k, cluster_m, cluster_n, in_dtype, sf_vec_size, sf_dtype)
    math_flops = 2 * cta_m * cta_n * k
    math = math_flops / self._math_sol(in_dtype, 1)
    epilogue_bytes = cta_m * cta_n * sms * dtype_bytes[out_dtype]
    epilogue = self._ep_adjustment() + (epilogue_bytes / self._dram_sol())

    return {"dma": dma, "math": math, "epilogue": epilogue}

  def predict(self, m, n, k, in_dtype, out_dtype, sf_dtype, sf_vec_size, mma_m, mma_n, cta_m, cta_n, cluster_m, cluster_n, **kwargs):
    mma_k = 32 / (dtype_bytes[in_dtype])

    result = {}

    fixed_overhead = self._fixed_overhead()
    result['fixed_overhead'] = fixed_overhead * 10**6

    # If the cluster size doesn't evenly divide the SM count, we assume fewer SMs are being used per wave
    # This still won't be accurate for larger clusters
    cga_m = cta_m * cluster_m
    cga_n = cta_n * cluster_n
    cga_tiles = math.ceil(m / cga_m) * math.ceil(n / cga_n)
    one_wave_clusters = self.gpu.num_sms // (cluster_m * cluster_n)
    full_wave_sms = one_wave_clusters * (cluster_m*cluster_n)
    tiles = cga_tiles * cluster_m * cluster_n

    ## First DMA
    first_wave_sms = tiles if not tiles > full_wave_sms else full_wave_sms
    first_dma = self._dma(first_wave_sms, cta_m, cta_n, mma_k, cluster_m, cluster_n, in_dtype, sf_vec_size, sf_dtype)
    result['first_dma'] = first_dma * 10**6

    ## Mainloop
    full_waves = tiles // full_wave_sms
    remainder = tiles % full_wave_sms
    result['mainloop'] = []
    mainloop = 0.0
    # For each wave
    for occupied_sms in [full_wave_sms] * full_waves + ([remainder] if remainder else []):
      mm = self._mainloop_one_wave(occupied_sms, cta_m, cta_n, k, cluster_m, cluster_n, in_dtype, out_dtype, sf_vec_size, sf_dtype)
      mainloop += max(mm['dma'], mm['math'], mm['epilogue'])
      result['mainloop'].append({k: v*10**6 for k,v in mm.items()})

    ## Last wave epilogue
    last_epilogue = mm['epilogue']
    result['last_epilogue'] = last_epilogue *10**6

    # constant + first dma + mainloop + last wave epilogue
    result['runtime'] = (fixed_overhead + first_dma + mainloop + last_epilogue) * (10 ** 6)

    return result


  def print_summary(self, result):
    print("========================================================================")
    print(f"MNK {result['m']} {result['n']} {result['k']}")
    print(f"CTA ({result['cta_m']} {result['cta_n']})")
    print(f"CLUSTER ({result['cluster_m']} {result['cluster_n']})")
    print(f"RATIO: {result['ratio']}")
    print(f"    predicted: {result['predicted_runtime']}")
    print(f"    actual: {result['actual_runtime']}")

    print(f"PROLOGUE")
    print(f"    Overhead: {result['fixed_overhead']}")
    print(f"    DMA: {result['first_dma']}")

    print(f"MAINLOOP")
    first_m = result['mainloop'][0]
    vals = [first_m['dma'], first_m['math'], first_m['epilogue']]
    sts = ['DMA', 'MATH', 'EPILOG']
    l = max(vals)
    limiter = sts[vals.index(l)]
    print(f"    Limiter: {limiter}")
    print(f"        DMA: {first_m['dma']}")
    print(f"        MATH: {first_m['math']}")
    print(f"        EPILOG: {first_m['epilogue']}")
    if len(result['mainloop']) > 1:
      last_m = result['mainloop'][-1]
      print(f"    Last Wave")
      print(f"        DMA: {last_m['dma']}")
      print(f"        MATH: {last_m['math']}")
      print(f"        EPILOG: {last_m['epilogue']}")

      

