# Sources and licensing

Only runtime model import dependencies are copied, not upstream experiments or datasets:

| Model | Official source | Files | License |
|---|---|---:|---|
| iTransformer | https://github.com/thuml/iTransformer | 5 Python files | MIT, original LICENSE retained |
| TimesNet | https://github.com/thuml/Time-Series-Library | 3 Python files | MIT, original LICENSE retained |
| DLinear | https://github.com/cure-lab/LTSF-Linear | 1 Python file | Apache-2.0, original LICENSE retained |
| MoLE | https://github.com/RogerNi/MoLE | No upstream files redistributed | No explicit license found; obtain separately under applicable terms |

The nine bundled Python files are unchanged source snapshots. The parent research snapshots
had no Git history; their file hashes establish identity, not an inferred upstream commit.
MoLE is *Mixture-of-Linear-Experts for Long-term Time Series Forecasting* (AISTATS 2024).
Adapters, CLI, analysis and test code are project code; third-party licenses apply to the
corresponding vendor files only. Original copyright notices remain in each source/license.
Original project code is released under the MIT License (root LICENSE), as selected by the maintainer. Third-party files retain their own licenses.

## Snapshot SHA256

| File | SHA256 |
|---|---|
| `teprediction/vendor/LTSF-Linear/models/DLinear.py` | `190859fc67db235cdd9c27c7590776f402b59e7b67804b52ee2757715f7fc9e3` |
| `teprediction/vendor/Time-Series-Library/layers/Conv_Blocks.py` | `3ef9f4dfd5ef52ee42c72c907b270d3a842fe6809b0c7efeee013c52769bdafc` |
| `teprediction/vendor/Time-Series-Library/layers/Embed.py` | `0a29593e3796b5a3798bd3d37c1150fc2256c559abdee36e83eecd1a05787247` |
| `teprediction/vendor/Time-Series-Library/models/TimesNet.py` | `83a09e624cb76a3f36a5ca064e1f0a3c91fad5466df2985d802d949a260e98fe` |
| `teprediction/vendor/iTransformer/layers/Embed.py` | `cb1b8934d87659e656e6c92ba6290c4a76fbfb88389527ce43766c5b8cfd46ed` |
| `teprediction/vendor/iTransformer/layers/SelfAttention_Family.py` | `155f104d30a8576639a185efdb3b27a294d55147dfa17feaeaddbb965394c550` |
| `teprediction/vendor/iTransformer/layers/Transformer_EncDec.py` | `eed892a31468b1142777420f4b7ecb8cdbae020d835bbdd59a045c99bbe986df` |
| `teprediction/vendor/iTransformer/model/iTransformer.py` | `b89cebca695824ba1c074923527614f7fa5326d8dae884a3efa8f87aa9431be7` |
| `teprediction/vendor/iTransformer/utils/masking.py` | `c22de70ad076f4eeecc857aba984741da05da54375333399ff2742b041544986` |
