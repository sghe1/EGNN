#!/usr/bin/env python3
"""
Extract a subset of trajectories from a TFRecord file.
This creates a smaller dataset for Colab upload.
"""

import sys
import tensorflow as tf

def extract_subset(input_file, output_file, num_trajectories):
    """Extract first N trajectories from a TFRecord file."""
    print(f"Extracting first {num_trajectories} trajectories from {input_file}...")
    
    dataset = tf.data.TFRecordDataset([input_file])
    writer = tf.io.TFRecordWriter(output_file)
    
    count = 0
    for record in dataset:
        writer.write(record.numpy())
        count += 1
        if count >= num_trajectories:
            break
        if count % 10 == 0:
            print(f"  Extracted {count} trajectories...")
    
    writer.close()
    print(f"✓ Extracted {count} trajectories to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_tfrecord_subset.py <input_file> <output_file> <num_trajectories>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_trajectories = int(sys.argv[3])
    
    extract_subset(input_file, output_file, num_trajectories)

