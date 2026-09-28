# Submissions

| file | what | validation macro F0.5 |
|---|---|---|
| `v1_matching_results.zip` | stage-1 LightGBM, threshold 0.7 | 0.9606 |
| `v2_matching_results.zip` | + stage-2 context model | 0.9632 |
| `v3_matching_results.zip` | + normalisation/blocking fixes, more training data | 0.9673 |
| `v4_matching_results.zip` | + stage-2 group-support features (**best**) | 0.9679 |

Each zip contains `matching_results.tsv` for the leaderboard upload.

## Final package

`final_package/TEAM_submission.zip.part_*` is the final submission zip split into
parts under GitHub's 100 MB limit. Rejoin, then rename to `<team_name>_submission.zip`:

```bash
cat TEAM_submission.zip.part_* > TEAM_submission.zip          # macOS / Linux
copy /b TEAM_submission.zip.part_00+TEAM_submission.zip.part_01+TEAM_submission.zip.part_02 TEAM_submission.zip   # Windows
```

Contents: `output/` (both TSVs), `code/business_entity_resolution/` (src, README,
requirements, run_all.sh) and `Documentation_template.md`.
