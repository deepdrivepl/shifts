#!/usr/bin/env bash

./build.sh

docker save baseline | gzip -c > Submission.tar.gz
