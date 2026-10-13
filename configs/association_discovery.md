# Association Discovery

The final paper configuration discovers question-conditioned identity relations
from the estimation split only.  It stratifies all sub-samples and permutations
by country, so country is a control variable and never becomes retrieval
evidence.

| Symbol | CLI option | Paper configuration |
| --- | --- | --- |
| \(n_{\mathrm{cap}}\) | `--sample-per-country` | 100 |
| \(R\) | `--subsample-repeats` | 10 |
| \(B_{\mathrm{scr}}\) | `--screen-permutations` | 99 |
| \(B_{\mathrm{conf}}\) | `--permutations` | 999 |
| \(\tau_{\mathrm{scr}}\) | `--screen-quantile` | 0.95 |
| \(K_{\mathrm{scr}}\) | `--screen-max-candidates` | 25 |
| \(\alpha\) | `--alpha` | 0.05 |
| \(n_{Q,\min}\) | `--min-question-n` | 800 |
| \(\pi_{\max}\) | `--max-majority` | 0.90 |
| \(n_{\min}\) | `--min-country-group-n`, `--min-combo-group-n` | 30 |

The discovery program also requires at least 100 valid records per country.
All values can be deliberately overridden from the QIVER CLI by appending
the corresponding options after `--`.
