# GCStack+GCScaler \[ISCA '25\]

This repository contains the source code for [GCStack+GCScaler \[ISCA '25\]](https://doi.org/10.1145/3695053.3731068), a fast and accurate GPU performance analysis mechanism leveraging fine-grained stall cycle accounting and interval analysis.
The repository also includes the source code for [GCStack \[IEEE CAL '24\]](https://doi.org/10.1109/LCA.2024.3476909), which is a fine-grained GPU stall cycle accounting method and part of the GCStack+GCScaler framework.
Please cite the following papers if you utilize GCStack+GCScaler and/or GCStack in your research.

```bibtex
@inproceedings{cha2025gcstack_gcscaler,
  author    = {Hanna Cha and Sungchul Lee and Jounghoo Lee and Yeonan Ha and Joonsung Kim and Youngsok Kim},
  title     = {{GCStack+GCScaler: Fast and Accurate GPU Performance Analyses Using Fine-Grained Stall Cycle Accounting and Interval Analysis}},
  booktitle = {Proc. 52nd IEEE/ACM International Symposium on Computer Architecture (ISCA)},
  year      = {2025},
}

@article{cha2024gcstack,
  author  = {Hanna Cha and Sungchul Lee and Yeonan Ha and Hanhwi Jang and Joonsung Kim and Youngsok Kim},
  title   = {{GCStack: A GPU Cycle Accounting Mechanism for Providing Accurate Insight Into GPU Performance}},
  journal = {IEEE Computer Architecture Letters (CAL)},
  year    = {2024},
}
```

## Directories
- `gcstack/`
    - The source code for GCStack, built on top of Accel-sim
    - See [gcstack/README.md](./gcstack/README.md)
- `gcscaler/`
    - The source code for GCScaler
    - See [gcscaler/README.md](./gcscaler/README.md)
