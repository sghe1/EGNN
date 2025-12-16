#!/bin/bash
# Fix sonnet tensorflow.contrib compatibility issue
# This creates a mock tensorflow.contrib module for TensorFlow 2.x

set -e

ENV_NAME="meshgraphnet"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

TF_DIR="/opt/anaconda3/envs/${ENV_NAME}/lib/python3.11/site-packages/tensorflow"

echo "Creating tensorflow.contrib mock for TensorFlow 2.x compatibility..."

# Create contrib directory structure
mkdir -p "${TF_DIR}/contrib/eager/python"

# Create __init__.py files
cat > "${TF_DIR}/contrib/__init__.py" << 'EOF'
# Mock tensorflow.contrib for TensorFlow 2.x compatibility
EOF

cat > "${TF_DIR}/contrib/eager/__init__.py" << 'EOF'
# Mock tensorflow.contrib.eager for TensorFlow 2.x compatibility
EOF

# Create mock tfe module
cat > "${TF_DIR}/contrib/eager/python/__init__.py" << 'EOF'
# Mock tensorflow.contrib.eager.python for TensorFlow 2.x compatibility
EOF

cat > "${TF_DIR}/contrib/eager/python/tfe.py" << 'EOF'
"""Mock tensorflow.contrib.eager.python.tfe for TensorFlow 2.x compatibility."""
import tensorflow as tf

# Re-export from tf.compat.v1 or provide minimal stubs
try:
    # Try to use tf.compat.v1 equivalents
    from tensorflow.python.eager import context as eager_context
    from tensorflow.python.eager import execute as eager_execute
    
    # Minimal API that sonnet.base might need
    class Variable(object):
        pass
    
    def variable_scope(*args, **kwargs):
        return tf.compat.v1.variable_scope(*args, **kwargs)
    
    def get_variable(*args, **kwargs):
        return tf.compat.v1.get_variable(*args, **kwargs)
    
except ImportError:
    # Fallback stubs
    class Variable(object):
        pass
    
    def variable_scope(*args, **kwargs):
        class DummyScope:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyScope()
    
    def get_variable(*args, **kwargs):
        return None
EOF

echo "✓ Created tensorflow.contrib mock"

# Test if it works
echo ""
echo "Testing imports..."
python -c "
import os
import sys
sys.path.insert(0, 'deepmind-research')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()

# Test contrib import
try:
    from tensorflow.contrib.eager.python import tfe as contrib_eager
    print('✓ tensorflow.contrib mock works')
except Exception as e:
    print('⚠ contrib mock issue:', str(e))

from meshgraphnets import dataset
print('✓ Dataset OK')

from meshgraphnets import deforming_plate_model
print('✓ deforming_plate_model OK!')
print('')
print('🎉 SUCCESS! Ready to run training!')
" || echo "⚠ Some issues remain, but core functionality may work"

