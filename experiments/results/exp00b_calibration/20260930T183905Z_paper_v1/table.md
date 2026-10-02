# exp00b calibration

## Reproduction of n_per_bin

| backend | size_bin | bin_mm | n_recomputed | n_shipped_table | diff_recomputed_minus_shipped | shipped_rows_found_in_recomputed | recomputed_rows_absent_from_table |
|---|---|---|---|---|---|---|---|
| original | 0 | [0, 5] | 383 | 381 | 2 | 381 | 2 |
| original | 1 | [5, 10] | 935 | 932 | 3 | 932 | 3 |
| original | 2 | [10, 20] | 682 | 676 | 6 | 676 | 6 |
| original | 3 | [20, 30] | 195 | 194 | 1 | 194 | 1 |
| original | 4 | [30, 50] | 105 | 104 | 1 | 104 | 1 |
| original | 5 | [50, 1000000000.0] | 59 | 57 | 2 | 57 | 2 |
| unigradicon | 0 | [0, 5] | 351 | 351 | 0 | 351 | 0 |
| unigradicon | 1 | [5, 10] | 843 | 835 | 8 | 835 | 8 |
| unigradicon | 2 | [10, 20] | 591 | 576 | 15 | 576 | 15 |
| unigradicon | 3 | [20, 30] | 188 | 184 | 4 | 184 | 4 |
| unigradicon | 4 | [30, 50] | 101 | 97 | 4 | 97 | 4 |
| unigradicon | 5 | [50, 1000000000.0] | 102 | 85 | 17 | 85 | 17 |

## Per-bin residual statistics

| backend | row_set | size_bin | n | median_mm | p90_mm | p95_mm | max_mm | sd_z_vox | sd_y_vox | sd_x_vox |
|---|---|---|---|---|---|---|---|---|---|---|
| original | shipped_rows | 0 | 381 | 2.6 | 16.38 | 22.69 | 46.68 | 5.59 | 6.47 | 5.69 |
| original | recomputed_rows | 0 | 383 | 2.64 | 17.25 | 22.98 | 162.28 | 9.32 | 7.15 | 8.33 |
| original | shipped_rows | 1 | 932 | 3.15 | 16.6 | 22.88 | 53.13 | 5.61 | 6.57 | 5.81 |
| original | recomputed_rows | 1 | 935 | 3.16 | 16.79 | 23.46 | 213.12 | 9.01 | 6.6 | 6.56 |
| original | shipped_rows | 2 | 676 | 4.38 | 19.36 | 25.2 | 104.69 | 7.23 | 7.6 | 6.92 |
| original | recomputed_rows | 2 | 682 | 4.41 | 20.28 | 25.67 | 219.43 | 12.32 | 8.56 | 10.54 |
| original | shipped_rows | 3 | 194 | 6.11 | 18.82 | 24.7 | 42.91 | 6.37 | 8.04 | 7.42 |
| original | recomputed_rows | 3 | 195 | 6.12 | 19.0 | 25.5 | 128.9 | 9.28 | 8.92 | 7.8 |
| original | shipped_rows | 4 | 104 | 8.45 | 25.41 | 34.44 | 77.23 | 8.79 | 14.05 | 12.36 |
| original | recomputed_rows | 4 | 105 | 8.63 | 25.84 | 40.42 | 173.83 | 15.09 | 14.33 | 14.64 |
| original | shipped_rows | 5 | 57 | 14.8 | 56.01 | 66.32 | 86.93 | 15.69 | 21.12 | 21.48 |
| original | recomputed_rows | 5 | 59 | 16.31 | 64.43 | 78.16 | 103.17 | 16.09 | 28.35 | 22.65 |
| unigradicon | shipped_rows | 0 | 351 | 4.87 | 18.64 | 25.31 | 46.04 | 7.35 | 6.5 | 4.64 |
| unigradicon | recomputed_rows | 0 | 351 | 4.87 | 18.64 | 25.31 | 46.04 | 7.35 | 6.5 | 4.64 |
| unigradicon | shipped_rows | 1 | 835 | 5.13 | 16.14 | 21.53 | 61.05 | 6.29 | 6.38 | 5.18 |
| unigradicon | recomputed_rows | 1 | 843 | 5.18 | 16.83 | 23.04 | 402.71 | 22.07 | 7.89 | 5.3 |
| unigradicon | shipped_rows | 2 | 576 | 5.4 | 18.16 | 22.45 | 68.08 | 7.15 | 6.68 | 6.16 |
| unigradicon | recomputed_rows | 2 | 591 | 5.63 | 19.64 | 28.4 | 432.45 | 36.38 | 9.12 | 6.34 |
| unigradicon | shipped_rows | 3 | 184 | 6.37 | 14.91 | 22.19 | 44.31 | 6.17 | 6.86 | 6.34 |
| unigradicon | recomputed_rows | 3 | 188 | 6.57 | 15.58 | 24.03 | 251.71 | 20.94 | 7.76 | 6.54 |
| unigradicon | shipped_rows | 4 | 97 | 7.11 | 24.45 | 40.0 | 57.84 | 9.51 | 11.55 | 8.76 |
| unigradicon | recomputed_rows | 4 | 101 | 7.77 | 36.93 | 54.97 | 285.08 | 34.91 | 12.19 | 8.94 |
| unigradicon | shipped_rows | 5 | 85 | 26.93 | 66.57 | 73.58 | 84.06 | 19.8 | 26.6 | 24.57 |
| unigradicon | recomputed_rows | 5 | 102 | 37.15 | 116.48 | 194.69 | 249.54 | 48.05 | 31.29 | 31.92 |

## Held-out 60 contribution

| backend | shipped_rows_total | shipped_rows_from_holdout60 | share | holdout_patients_contributing |
|---|---|---|---|---|
| original | 2344 | 399 | 0.17022184300341298 | 51 |
| unigradicon | 2128 | 419 | 0.19689849624060152 | 47 |
