# ALMI evaluation results

Evaluation set: 747 held-out ALMI-X motions, 1000 steps per test (the paper uses 1122 CMU clips). Metric definitions: docs/EVALUATION.md. Paper values: Table 6 of arXiv 2504.14305v3. Eg and the action terms are defined differently from the paper, so compare trends, not absolute values.

| Level | Policy pair | Source | Evel | Eang | Ejpe | Ekpe | Eact_up | Eact_low | Eg | Survival |
|---|---|---|---|---|---|---|---|---|---|---|
| easy | lower-1 + upper-2 | ours | 0.1438 | 0.2742 | 0.0973 | 0.0739 | 0.0112 | 0.0407 | 0.0420 | 0.9746 |
| easy | lower-1 + upper-2 | paper | 0.1271 | 0.2738 | 0.1928 | 0.0526 | 0.0642 | 0.0171 | 0.7052 | 1.0000 |
| easy | lower-2 + upper-2 | ours | 0.1191 | 0.2182 | 0.0965 | 0.0734 | 0.0106 | 0.0328 | 0.0820 | 0.9993 |
| easy | lower-2 + upper-2 | paper | 0.1164 | 0.2669 | 0.1955 | 0.0452 | 0.0475 | 0.0171 | 0.7121 | 1.0000 |
| easy | lower-3 + upper-2 | ours | 0.1080 | 0.2283 | 0.0971 | 0.0746 | 0.0102 | 0.0300 | 0.0900 | 0.9719 |
| easy | lower-3 + upper-2 | paper | 0.1135 | 0.2647 | 0.1931 | 0.0460 | 0.0462 | 0.0170 | 0.6919 | 1.0000 |
| easy | lower-3 wide cmds long + upper-2 | ours | 0.1107 | 0.2295 | 0.0970 | 0.0748 | 0.0100 | 0.0298 | 0.0608 | 0.9980 |
| easy | lower-3 wide cmds + upper-2 | ours | 0.1828 | 0.2675 | 0.1015 | 0.0783 | 0.0126 | 0.0365 | 0.1010 | 0.8628 |
| medium | lower-1 + upper-2 | ours | 0.3430 | 0.3280 | 0.1012 | 0.0778 | 0.0149 | 0.0536 | 0.0537 | 0.7985 |
| medium | lower-1 + upper-2 | paper | 0.2262 | 0.3872 | 0.2173 | 0.0492 | 0.0604 | 0.0175 | 0.7730 | 0.9273 |
| medium | lower-2 + upper-2 | ours | 0.2449 | 0.4019 | 0.0992 | 0.0761 | 0.0128 | 0.0413 | 0.0894 | 0.8963 |
| medium | lower-2 + upper-2 | paper | 0.2213 | 0.3571 | 0.2032 | 0.0458 | 0.0607 | 0.0172 | 0.7748 | 0.9772 |
| medium | lower-3 + upper-2 | ours | 0.2235 | 0.4275 | 0.0993 | 0.0770 | 0.0127 | 0.0373 | 0.0868 | 0.9076 |
| medium | lower-3 + upper-2 | paper | 0.2192 | 0.3520 | 0.2007 | 0.0450 | 0.0598 | 0.0172 | 0.7604 | 0.9852 |
| medium | lower-3 wide cmds long + upper-2 | ours | 0.2267 | 0.4202 | 0.0989 | 0.0766 | 0.0120 | 0.0358 | 0.0610 | 0.9431 |
| medium | lower-3 wide cmds + upper-2 | ours | 0.2128 | 0.4628 | 0.1021 | 0.0791 | 0.0140 | 0.0421 | 0.0917 | 0.8701 |
| hard | lower-1 + upper-2 | ours | 0.6518 | 0.3598 | 0.1134 | 0.0873 | 0.0214 | 0.0730 | 0.0876 | 0.4578 |
| hard | lower-1 + upper-2 | paper | 0.2566 | 0.5172 | 0.2451 | 0.0537 | 0.0777 | 0.0179 | 0.9462 | 0.8743 |
| hard | lower-2 + upper-2 | ours | 0.4406 | 0.5824 | 0.1061 | 0.0827 | 0.0174 | 0.0539 | 0.1082 | 0.6807 |
| hard | lower-2 + upper-2 | paper | 0.2892 | 0.5395 | 0.2231 | 0.0482 | 0.0645 | 0.0178 | 0.9479 | 0.9233 |
| hard | lower-3 + upper-2 | ours | 0.4928 | 0.6162 | 0.1113 | 0.0873 | 0.0192 | 0.0570 | 0.1144 | 0.6124 |
| hard | lower-3 + upper-2 | paper | 0.2202 | 0.4812 | 0.2116 | 0.0458 | 0.0600 | 0.0175 | 0.8551 | 0.9723 |
| hard | lower-3 wide cmds long + upper-2 | ours | 0.3183 | 0.6117 | 0.1020 | 0.0801 | 0.0140 | 0.0423 | 0.0657 | 0.8648 |
| hard | lower-3 wide cmds + upper-2 | ours | 0.3664 | 0.6482 | 0.1035 | 0.0813 | 0.0161 | 0.0487 | 0.0929 | 0.8146 |
