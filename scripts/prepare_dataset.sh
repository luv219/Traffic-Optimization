#!/bin/bash
# Prepares the YOLO dataset and downloads the sample traffic video
python "$(dirname "$0")/../detectors/prepare_data.py"
