#!/bin/bash
echo """This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf"""


echo """Start a Jupyter Notebook server with jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root

To access it, visit http://localhost:8888 on your host machine.
Ensure the following arguments were added to 'docker run' to expose the Jupyter Notebook server to your host machine:
      -p 8888:8888
"""

echo """Make local folders visible by bind mounting:
    -v <her2bdl local dev>:/her2bdl_dev (optional for development)
    -v <her2bdl local experiments>:/experiments
    -v <her2bdl local datasets>:/datasets
"""