# Data audit

All 21 JSONL files were checked offline. Existing data was not modified.

| Task | Train | Validation | Test | Balanced |
| --- | ---: | ---: | ---: | --- |
| AGNEWS | 8000 | 2000 | 2000 | Yes |
| QQP | 8000 | 1000 | 1000 | Yes |
| SST2 | 8000 | 2000 | 2000 | Yes |
| AMAZON | 8000 | 2000 | 2000 | Yes |
| YELP | 10000 | 2000 | 2000 | Yes |
| DBPEDIA | 14000 | 1750 | 1750 | Yes |
| YAHOO | 10000 | 2000 | 2000 | Yes |

All row counts match the current builder defaults. All required fields and labels are valid.
All real rows pass message construction, label mapping and template tokenization.
No conflicting-label groups were found. All available source idx columns are disjoint across splits.
AGNEWS has no idx column, so source-row disjointness cannot be checked there; exact and normalized text have no overlap.

## Findings

- SST2: the duplicate-text findings from this audit have been fixed by regenerating the data. Current verification: [sst2_dedup_check.json](sst2_dedup_check.json). Earlier SST2 line numbers in data_audit.json refer to the backup, not the regenerated files.
- DBPEDIA: regenerated with a 1536-token complete-chat filter; current verification: [dbpedia_length_filter_check.json](dbpedia_length_filter_check.json). Earlier outlier line numbers refer to the backup. All seven current tasks pass [length_check_1536.json](length_check_1536.json).
- YAHOO: regenerated with a 1536-token complete-chat length filter. Current results: [yahoo_length_filter_check.json](yahoo_length_filter_check.json). Earlier lengths and line numbers in data_audit.json refer to the previous exports.
- QQP: question pairs do not overlap, including when pair order is ignored. Individual questions are shared: train/val 128, train/test 130, val/test 28. This split is by pair, not by question identity.
- QQP: the earlier answer-parser issue has been fixed; the old parser probe in data_audit.json records its state at the time of that audit.
- YAHOO has empty question_content or best_answer fields, but no completely empty input: every example has a nonempty title. Empty optional fields do not currently break message construction.

Full details, counts and JSONL line numbers: [data_audit.json](data_audit.json).

Scope: local exported samples, code interfaces and token lengths. Source provenance and semantic near-duplicate detection were not verified.
