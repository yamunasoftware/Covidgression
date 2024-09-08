#!/bin/bash

echo "Started."
python src/data.py $1
echo "Duration: $SECONDS seconds"