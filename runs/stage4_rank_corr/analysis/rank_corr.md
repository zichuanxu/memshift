Rank correlation across systems (exploratory: prereg v2 section 6, not a pre-registered test, no verdict)

| system | kind | Update S_src | Update S_def | Stability orphan | rank S_src | rank S_def | rank orphan |
|---|---|---|---|---|---|---|---|
| amem | native | 69.3 [63.7, 74.0] | 50.6 [42.9, 56.9] | 80.9 [75.6, 86.0] | 12 | 13 | 8.5 |
| cognee | native | 68.8 [64.4, 73.0] | 56.2 [47.3, 63.4] | 95.7 [92.7, 98.6] | 13 | 11 | 5 |
| hipporag | native | 74.3 [68.8, 79.2] | 58.3 [49.3, 65.3] | 82.7 [80.1, 85.2] | 7 | 8.5 | 7 |
| lightrag | native | 64.9 [59.7, 69.2] | 50.6 [41.4, 57.5] | 80.9 [76.4, 85.5] | 14 | 13 | 8.5 |
| mem0 | native | 70.6 [64.2, 76.2] | 50.6 [40.5, 58.6] | 70.0 [66.3, 73.9] | 10 | 13 | 13 |
| memos | native | 70.4 [65.0, 75.4] | 57.4 [49.8, 63.3] | 68.6 [63.8, 73.2] | 11 | 10 | 14 |
| oracle | baseline | 93.8 [90.9, 96.7] | 91.1 [85.7, 95.2] | 91.6 [88.7, 94.2] | 2 | 2 | 6 |
| rag | baseline | 73.4 [68.6, 78.0] | 59.1 [52.3, 66.2] | 67.6 [63.4, 71.5] | 8 | 7 | 15 |
| zero | baseline | 71.1 [65.9, 75.5] | 58.3 [51.5, 64.4] | 72.1 [67.6, 76.4] | 9 | 8.5 | 12 |
| holdmem_D0B0 | white-box | 57.6 [51.5, 63.4] | 47.2 [39.9, 52.8] | 74.1 [67.9, 80.0] | 15 | 15 | 11 |
| holdmem_D0B1_cited | white-box | 77.8 [72.7, 82.5] | 65.5 [58.4, 71.3] | 97.2 [94.5, 99.0] | 6 | 6 | 3 |
| holdmem_D0B1_source | white-box | 80.0 [75.1, 84.6] | 67.2 [59.1, 73.7] | 98.7 [96.8, 100.0] | 4 | 5 | 2 |
| holdmem_D1B0 | white-box | 79.4 [74.2, 84.7] | 81.7 [76.8, 87.0] | 77.2 [70.6, 83.6] | 5 | 4 | 10 |
| holdmem_D1B1_cited | white-box | 88.1 [83.3, 92.0] | 83.0 [77.7, 87.8] | 95.9 [93.9, 97.7] | 3 | 3 | 4 |
| holdmem_D1B1_source | white-box | 98.2 [96.6, 99.5] | 97.4 [94.8, 99.6] | 99.2 [98.0, 100.0] | 1 | 1 | 1 |

Ranks: 1 = highest, among the systems that have the metric.

Spearman rho(update_src, update_def) = 0.96 [0.85, 0.98] over 15 systems, 15 worlds
Spearman rho(update_src, stability_orphan) = 0.55 [0.43, 0.66] over 15 systems, 15 worlds
