# Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis

This repository contains the code for benchmarking NVIDIA GPU performance. The relevant papers are as follows:

- Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Qiang Wang, and Xiaowen Chu. "[Benchmarking and Dissecting the Nvidia Hopper GPU Architecture.](https://ieeexplore.ieee.org/abstract/document/10579250?casa_token=tw5ix8vdZvsAAAAA:9K3snl2qTP8Pf_ZIN-3T9RCil_PniO2LVrRMPxP5gr8eUYdnag9L_YkhYsFdydmXtYQWfuz47ZKXQ4o)" In 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pp. 656-667. IEEE, 2024.
- Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Hongyuan Liu, Qiang Wang, and Xiaowen Chu. "[Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis.](https://arxiv.org/abs/2501.12084)" arXiv preprint arXiv:2501.12084 (2025).

If you find this work useful, please cite this project and our papers.

```
@inproceedings{luo2024benchmarking,
  title={Benchmarking and dissecting the nvidia hopper gpu architecture},
  author={Luo, Weile and Fan, Ruibo and Li, Zeyu and Du, Dayou and Wang, Qiang and Chu, Xiaowen},
  booktitle={2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages={656--667},
  year={2024},
  organization={IEEE}
}

@article{luo2025dissecting,
  title={Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis},
  author={Luo, Weile and Fan, Ruibo and Li, Zeyu and Du, Dayou and Liu, Hongyuan and Wang, Qiang and Chu, Xiaowen},
  journal={arXiv preprint arXiv:2501.12084},
  year={2025}
}
```

## Recommended environment

- CUDA 12.6 or above
- Ubuntu 20.04

## Build & Usage

In the folder, use `make` or `./compile.sh` to build, and use `./run.sh` or `./run_all.sh` to run.

## Acknowledgment

- https://github.com/shen203/GPU_Microbenchmark provides a reference for our regular unit tests.
- https://github.com/RRZE-HPC/gpu-benches provides a reference for our memory and TMA random access tests.
- We used the tools in https://github.com/blackjack2015/NV-DVFS-Benchmark to test the power consumption.