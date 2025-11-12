import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf, DictConfig
from perf_model import *

@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
  assert cfg.input_csv is not None, "input_csv must be specified"
  assert cfg.output_dir is not None, "output_dir must be specified"

  if cfg.model == 'SOLModel':
    model = SOLModel(gpu=cfg.gpu, model_opts=cfg.model_opts)
  elif cfg.model == 'WSPersistentGEMMModel':
    model = WSPersistentGEMMModel(gpu=cfg.gpu, model_opts=cfg.model_opts)
  else:
    raise ValueError(f"Unknown model class: {cfg.model}")

  df = pd.read_csv(cfg.input_csv)
  required_columns = ['in_dtype', 'acc_dtype', 'out_dtype', 'sf_dtype', 'sf_vec_size', 'a_major', 'b_major', 'c_major', 'mma_m', 'mma_n', 'cta_m', 'cta_n', 'cluster_m', 'cluster_n', 'm', 'n', 'k', 'runtime_us']
  assert all(col in df.columns for col in required_columns), "Missing columns in CSV"

  results = []
  ratios = []
  for idx, row in df.iterrows():
    # assume runtime is the last column in the input csv
    params = {col: row[col] for col in required_columns[:-1]}
    pred_result = model.predict(**params)
    pred_runtime = pred_result['runtime']
    actual_runtime = row['runtime_us']
    ratio = pred_runtime / actual_runtime
    ratios.append(ratio)
    results.append({
      'row_idx': idx,
      **params,
      'actual_runtime': actual_runtime,
      'predicted_runtime': pred_runtime,
      'ratio': ratio,
      **pred_result
    })

  ratios.sort()
  sorted_results = sorted(results, key=lambda item: item['ratio'])
  os.makedirs(cfg.output_dir, exist_ok=True)

  print("========================================================================")
  print("5 highest ratio cases")
  print("========================================================================")
  for res in sorted_results[-5:]:
    model.print_summary(res)

  print("========================================================================")
  print("5 lowest ratio cases")
  print("========================================================================")
  for res in sorted_results[:5]:
    model.print_summary(res)

  output_csv_path = os.path.join(cfg.output_dir, f'predictions_{cfg.model}.csv')
  pd.DataFrame(results).to_csv(output_csv_path, index=False)

  plt.figure(figsize=(10, 6))
  plt.bar(range(len(ratios)), ratios, color='blue')
  plt.xlabel('GEMM Problem')
  plt.ylabel('Predicted Runtime / Runtime ')
  plt.title(f'Perf Ratio vs. {cfg.model}, {results[0]['in_dtype']}_{results[0]['out_dtype']}')
  plt.grid(True)
  plot_path = os.path.join(cfg.output_dir, f'scurve_{cfg.model}.png')
  plt.savefig(plot_path)
  plt.close()

if __name__ == '__main__':
  main()

