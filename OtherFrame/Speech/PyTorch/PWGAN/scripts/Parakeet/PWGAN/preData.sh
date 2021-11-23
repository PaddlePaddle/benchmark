#!/usr/bin/env bash
pushd ParallelWaveGAN/egs/csmsc/voc1
bash run.sh --stage -1 --stop-stage 1
popd