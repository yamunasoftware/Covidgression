#!/bin/bash

echo "Started."
python -B src/data.py $1
echo "Duration: $SECONDS seconds"