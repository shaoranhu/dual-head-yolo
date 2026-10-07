# One-epoch fine-tuning comparison

Person/apparatus loss weights: 4:1; lr=1e-5; shared layers explicitly unfrozen.
Exploratory combined change, not an isolated loss-weight ablation.

|Task|Metric|Before|After|Change|
|---|---|---:|---:|---:|
|person|precision|0.7059|0.5881|-0.1178|
|person|recall|0.5449|0.5134|-0.0315|
|person|map50|0.6039|0.5110|-0.0929|
|person|map5095|0.3271|0.2549|-0.0722|
|apparatus|precision|0.5682|0.6099|+0.0417|
|apparatus|recall|0.3192|0.5455|+0.2263|
|apparatus|map50|0.4213|0.5717|+0.1504|
|apparatus|map5095|0.2524|0.3572|+0.1048|

Weighted validation loss sum: 19882.501955
All AP and PR values above use the existing custom evaluator; no claim of official COCO equivalence.