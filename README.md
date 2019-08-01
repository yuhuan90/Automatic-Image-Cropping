Python version: 2.0

Requirements: Numpy, opencv, torch

Usage:
1. put the testing images into the folder "dataset/test". This code has only been tested on JPG images, images in other format should be okey. Any resolution is supported.
2. run demo_eval.py
3. The output crops are generated into the folder "dataset/testresult". For each testing image, we currently directly return the top4 crops without any post-processing.

Notification: the trained model naturally supports generating crops having fixed aspect ratios or resolutions. This function can be supported when necessary in the future.

