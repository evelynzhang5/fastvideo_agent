name: fastvideo_setup

description:
Guide new contributors through setting up the FastVideo repository.

use_when:
- user wants to install FastVideo
- user is new to the repo
- environment setup fails

steps:

1. Explain repository structure

2. Setup environment

conda create -n fastvideo python=3.10
pip install -e .

3. Verify installation

python -m fastvideo.test

4. If errors occur
record them in session memory