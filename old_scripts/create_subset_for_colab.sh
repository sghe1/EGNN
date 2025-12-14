#!/usr/bin/env bash

# Script to create a smaller dataset subset for Colab upload
# This extracts only the first N trajectories from train.tfrecord

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
DATA_DIR="${PROJECT_ROOT}/data/deforming_plate"
OUTPUT_DIR="${PROJECT_ROOT}/data/deforming_plate_colab"

# Number of trajectories to extract (adjust based on your needs)
NUM_TRAJECTORIES=${1:-100}  # Default: 100 trajectories (~900MB)

echo "Creating subset for Colab..."
echo "Extracting first ${NUM_TRAJECTORIES} trajectories from train.tfrecord"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Copy meta.json (needed for all methods)
cp "${DATA_DIR}/meta.json" "${OUTPUT_DIR}/meta.json"
echo "✓ Copied meta.json"

# Copy test and valid (smaller files)
cp "${DATA_DIR}/test.tfrecord" "${OUTPUT_DIR}/test.tfrecord"
cp "${DATA_DIR}/valid.tfrecord" "${OUTPUT_DIR}/valid.tfrecord"
echo "✓ Copied test.tfrecord and valid.tfrecord"

# Extract subset from train.tfrecord using TensorFlow
echo "Extracting subset from train.tfrecord..."
python3 "${SCRIPT_DIR}/extract_tfrecord_subset.py" \
    "${DATA_DIR}/train.tfrecord" \
    "${OUTPUT_DIR}/train.tfrecord" \
    "${NUM_TRAJECTORIES}"

if [ ! -f "${OUTPUT_DIR}/train.tfrecord" ]; then
    echo ""
    echo "⚠️  Extraction failed. Trying alternative method..."
    echo ""
    echo "If TensorFlow is not available, you can:"
    echo "1. Install TensorFlow: pip install tensorflow"
    echo "2. Or manually run the extraction script"
    echo ""
    echo "Manual command:"
    echo "  python3 ${SCRIPT_DIR}/extract_tfrecord_subset.py \\"
    echo "    ${DATA_DIR}/train.tfrecord \\"
    echo "    ${OUTPUT_DIR}/train.tfrecord \\"
    echo "    ${NUM_TRAJECTORIES}"
fi

# Check sizes
echo ""
echo "Original dataset sizes:"
du -sh "${DATA_DIR}"/*.tfrecord "${DATA_DIR}/meta.json" 2>/dev/null | sort -h

echo ""
echo "Subset dataset sizes:"
du -sh "${OUTPUT_DIR}"/*.tfrecord "${OUTPUT_DIR}/meta.json" 2>/dev/null | sort -h

echo ""
echo "✓ Subset created in: ${OUTPUT_DIR}"
echo ""
echo "To create zip for Colab:"
echo "  cd ${PROJECT_ROOT}"
echo "  zip -r deforming_plate_colab.zip data/deforming_plate_colab/"

