# Longi inference smoke test — `11364907cf_00`

Test data lives outside the repo (NIfTI too large to vendor):

`/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/longi_infer_test_11364907cf_00/`

Built from `Longitudinal_CT_v2_val` via `nanounet_register_longi` (FU/BL co-registered, 24 clicks each).

```bash
cd "/Users/nielsrocholl/Documents/PhD DIAG - Local/Data/Datasets/longi_infer_test_11364907cf_00"
./run_longi_single.sh
```

Default model: `/Users/nielsrocholl/Downloads/longi-finetune` (`best-epoch=60-val_dice_macro=0.6602.ckpt`).

See that directory's `README.md` for single vs dataset-mode commands and null-baseline test.
