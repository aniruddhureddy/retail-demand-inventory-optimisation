# Source data

Original competition: https://github.com/Mcompetitions/M5-methods

Archive mirror: https://github.com/Nixtla/m5-forecasts

Download: https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip

Run python prepare_data.py from the repository root. The script downloads the archive and produces sales_subset.csv, calendar.csv, prices_subset.csv and selection_audit.csv locally. It uses only the validation-history sales file, through day 1913. Raw data is ignored by Git.

The cohort is fixed at CA_1, 20 food products per department, ranked using days 1241–1605 and requiring a sale by day 365. No evaluation-window outcomes determine product selection. The archive calendar's missing sequential day index is reconstructed after checking its consecutive dates.

provenance.json records the reference archive hash, selected-file hashes, selection cutoff and original source. Rerunning preparation refreshes that local record. Use it to identify source changes rather than assuming a mutable archive URL will always return identical bytes.

The archive is approximately 50 MB. If the mirror is unavailable, place the same original archive at data/m5.zip and rerun preparation. The committed report and notebook remain viewable without data downloads.

Dataset rights remain with the original providers; this repository does not grant a licence for their data. Downloaded data, source archives and row-level Parquet/feature files are not committed.
