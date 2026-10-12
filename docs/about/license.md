# License

QueryGym is released under the **Apache License 2.0**.

## Apache License 2.0

Copyright 2024-2025 QueryGym Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See the full license text in the [LICENSE](https://github.com/ls3-lab/QueryGym/blob/main/LICENSE) file.

## Third-Party Licenses

querygym depends on several open-source packages. Please refer to their respective licenses:

- **typer**: MIT License
- **pyyaml**: MIT License
- **openai**: Apache License 2.0
- **tqdm**: MIT License / MPL-2.0

Optional dependencies:
- **datasets** (HuggingFace): Apache License 2.0
- **beir**: Apache License 2.0
- **pyserini**: Apache License 2.0

## Prompt Bank License

The prompts in `querygym/prompt_bank.yaml` are licensed under **CC-BY-4.0** (Creative Commons Attribution 4.0 International).

You are free to:
- Share — copy and redistribute the prompts
- Adapt — remix, transform, and build upon the prompts

Under the following terms:
- Attribution — You must give appropriate credit to the original authors

## Citation

If you use QueryGym in your research, please cite the toolkit paper:

```bibtex
@inproceedings{DBLP:conf/www/BigdeliRIACB26,
  author={Amin Bigdeli and Radin {Hamidi Rad} and Mert Incesu and Negar Arabzadeh and Charles L. A. Clarke and Ebrahim Bagheri},
  title={QueryGym: {A} Toolkit for Reproducible LLM-Based Query Reformulation},
  booktitle={Companion Proceedings of the ACM Web Conference 2026, WWW Companion 2026, Dubai, United Arab Emirates, 29 June 2026 - 3 July 2026},
  pages={196--199},
  publisher={ACM},
  year={2026},
  url={https://doi.org/10.1145/3774905.3793135},
  doi={10.1145/3774905.3793135},
}
```

If you refer to the SIGIR 2026 reproducibility results or the leaderboard, please also cite:

```bibtex
@inproceedings{DBLP:conf/sigir/BigdeliRLIACB26,
  author={Amin Bigdeli and Radin {Hamidi Rad} and Hai Son Le and Mert Incesu and Negar Arabzadeh and Charles L. A. Clarke and Ebrahim Bagheri},
  title={A Reproducibility Study of LLM-Based Query Reformulation},
  booktitle={Proceedings of the 49th International {ACM} {SIGIR} Conference on Research and Development in Information Retrieval, {SIGIR} 2026, Melbourne, VIC, Australia, July 20-24, 2026},
  pages={2824--2834},
  publisher={ACM},
  year={2026},
  url={https://doi.org/10.1145/3805712.3808560},
  doi={10.1145/3805712.3808560},
}
```
