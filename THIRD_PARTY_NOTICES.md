# Third-Party Notices

This repository includes or adapts code from third-party projects. Those third-party license terms apply to the relevant components.

## Qwen3.5

This repository includes Qwen3.5-related code and materials. See [LICENSE_QWEN3_5.txt](LICENSE_QWEN3_5.txt) for the upstream license terms.

The Qwen3.5 model implementation files under `src/g05/models/g05/qwen35/` include code ported from Hugging Face Transformers' Qwen3.5 implementation.

- Upstream project: https://github.com/huggingface/transformers
- Upstream file: `src/transformers/models/qwen3_5/modeling_qwen3_5.py`
- Copyright: Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
- License: Apache License, Version 2.0.

## Hugging Face LeRobot

Files under `src/g05/data/lerobot/` include code derived from Hugging Face LeRobot.

- Upstream project: https://github.com/huggingface/lerobot
- Copyright: Copyright 2024 The HuggingFace Inc. team.
- License: Apache License, Version 2.0.

The relevant source files retain their Apache License headers.

## PyTorch3D

The rotation conversion utilities in `src/g05/utils/data/rotation.py` and `src/g05/data_processor/transforms/rotation.py` include code adapted from PyTorch3D.

- Upstream project: https://github.com/facebookresearch/pytorch3d
- Copyright: Copyright (c) Meta Platforms, Inc. and affiliates.
- License: BSD-style license.

PyTorch3D BSD-style license notice:

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name Meta nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY.

## Google Research big_vision

`src/g05/models/g05/helpers/mask_helper.py` includes a block-wise causal mask routine adapted from Google Research big_vision.

- Upstream project: https://github.com/google-research/big_vision
- Copyright: Copyright 2024 Big Vision Authors.
- License: Apache License, Version 2.0.

## OpenPI

`src/g05/models/g05/model/modules.py` includes normalization and positional embedding components adapted from OpenPI's Big Vision-based Pi/Gemma implementation.

- Upstream project: https://github.com/Physical-Intelligence/openpi
- Copyright: Copyright 2024 Big Vision Authors.
- License: Apache License, Version 2.0.

## MolmoAct2 SO-101 Example

`experiments/so100/so100_policy_client.py` includes the follower-arm threading pattern ported from the MolmoAct2 SO-101 deployment example.

- Upstream project: https://github.com/irenegracekp/molmoact2-so101
- Copyright: Copyright (c) 2026 Irene Grace.
- License: MIT License.

MIT license notice:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
