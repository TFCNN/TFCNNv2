# TFCNNv2.1
This version uses derivatives w.r.t **𝑥** rather than w.r.t **f(𝑥)** which is what [`TFCNNv2.h`](https://github.com/TFCNN/TFCNNv2) does. As a result this version has a slightly higher memory usage overhead.

**Note:** SELU weights should be normalised with `WEIGHT_INIT_NORMAL_LECUN` / `1.0/fan-in`
