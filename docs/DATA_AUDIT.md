# Dataset audit notes

Audit performed on the supplied `waterQuality1.csv` file.

| Check | Result |
|---|---:|
| Raw CSV rows | 7,999 |
| Input features | 20 |
| Target column | `is_safe` |
| Rows with invalid target tokens | 3 |
| Rows retained after target validation | 7,996 |
| Exact duplicate rows among retained records | 0 |
| Repeated 20-feature combinations | 0 |
| Class 0 (not safe) | 7,084 |
| Class 1 (safe) | 912 |
| Retained rows with negative ammonia | 10 |
| Minimum ammonia value observed | -0.08 |

The three invalid target rows were excluded; labels were not inferred or recreated. Ten negative ammonia values were retained and flagged as data-quality anomalies. These checks establish properties of the CSV copy used for the project; they do not independently verify the dataset's original laboratory provenance, sample locations, or label-generation process.
