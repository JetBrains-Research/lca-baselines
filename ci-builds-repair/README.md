# 🏟️ Long Code Arena Baselines
## CI builds repair

The `ci-builds-repair-benchmark` directory contains the benchmark code for running the evaulation of the 🤗 [CI builds repair](https://huggingface.co/datasets/JetBrains-Research/lca-ci-builds-repair) dataset.
The benchmark clones each repository to the local folder. The baseline model should fix the issue according to the provided logs and the local repository state, and then the benchmark pushes the repo to GitGub and requests the result of the GitHub CI.
Please refer to README inside `ci-builds-repair-benchmark` for specific instructions.
