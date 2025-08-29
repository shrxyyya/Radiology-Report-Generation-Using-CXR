#!/bin/bash

HF_TOKEN="<HF_TOKEN>"

mkdir -p data


# Loop through part00 to part09
for i in $(seq -w 7 9); do
    PART="deid_png.part0$i"
    echo "Downloading $PART..."
    curl -L -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K/resolve/main/$PART" \
      -o "data/$PART"
done
