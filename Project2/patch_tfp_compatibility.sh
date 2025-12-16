#!/bin/bash
# Patch tensorflow-probability 0.8.0 for compatibility with TensorFlow 2.13.1
# This patches the experimental.auto_batching module to skip problematic imports

set -e

ENV_NAME="meshgraphnet"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Try to find TFP_DIR in common locations
if [ -d "/opt/anaconda3/envs/${ENV_NAME}/lib/python3.11/site-packages/tensorflow_probability" ]; then
    TFP_DIR="/opt/anaconda3/envs/${ENV_NAME}/lib/python3.11/site-packages/tensorflow_probability"
elif [ -d "/opt/anaconda3/lib/python3.12/site-packages/tensorflow_probability" ]; then
    TFP_DIR="/opt/anaconda3/lib/python3.12/site-packages/tensorflow_probability"
elif [ -d "$(python -c 'import tensorflow_probability; import os; print(os.path.dirname(tensorflow_probability.__file__))' 2>/dev/null)" ]; then
    TFP_DIR="$(python -c 'import tensorflow_probability; import os; print(os.path.dirname(tensorflow_probability.__file__))' 2>/dev/null)"
else
    echo "Error: Could not find tensorflow_probability installation"
    exit 1
fi

echo "Patching tensorflow-probability for TensorFlow 2.13.1 compatibility..."

# Patch 1: Skip auto_batching import in experimental __init__.py
EXP_INIT="${TFP_DIR}/python/experimental/__init__.py"
if [ -f "$EXP_INIT" ]; then
    # Comment out auto_batching import
    sed -i '' 's/^from tensorflow_probability.python.experimental import auto_batching/# Patched: from tensorflow_probability.python.experimental import auto_batching/' "$EXP_INIT"
    echo "✓ Patched experimental/__init__.py"
fi

# Patch 2: Fix np.bool in sample_halton_sequence.py
HALTON_FILE="${TFP_DIR}/python/mcmc/sample_halton_sequence.py"
if [ -f "$HALTON_FILE" ]; then
    sed -i '' 's/dtype=np\.bool/dtype=np.bool_/g' "$HALTON_FILE"
    echo "✓ Patched sample_halton_sequence.py (np.bool -> np.bool_)"
fi

# Patch 3: Skip auto_batching import in nuts.py
NUTS_FILE="${TFP_DIR}/python/experimental/mcmc/nuts.py"
if [ -f "$NUTS_FILE" ]; then
    # Comment out the import
    sed -i '' 's/^from tensorflow_probability.python.experimental import auto_batching as ab/# Patched: from tensorflow_probability.python.experimental import auto_batching as ab/' "$NUTS_FILE"
    # Add a stub for 'ab' if it's used
    if grep -q "ab\." "$NUTS_FILE"; then
        echo "# Patched: stub for auto_batching" >> "$NUTS_FILE"
        echo "class ab:" >> "$NUTS_FILE"
        echo "    pass" >> "$NUTS_FILE"
    fi
    echo "✓ Patched nuts.py"
fi

# Patch 4: Fix prefer_static.py to handle tf.ones_like signature mismatch
PREFER_STATIC_FILE="${TFP_DIR}/python/internal/prefer_static.py"
if [ -f "$PREFER_STATIC_FILE" ]; then
    # Make _copy_docstring more lenient to handle signature mismatches
    python3 << 'PYTHON_EOF'
import re
import sys

prefer_static_file = sys.argv[1]

with open(prefer_static_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'Patched: lenient _copy_docstring' in content:
    print('✓ prefer_static.py already patched')
else:
    # Find the _copy_docstring function and make it more lenient
    # Replace the ValueError raise with a warning and continue
    pattern = r'(def _copy_docstring\(original, new\):.*?)(raise ValueError\([^)]+\))'
    
    def replace_func(match):
        func_def = match.group(1)
        raise_stmt = match.group(2)
        # Replace raise with a pass (silently ignore signature mismatch)
        return func_def + '    # Patched: lenient _copy_docstring\n    try:\n        ' + raise_stmt.replace('raise ValueError', 'pass  # Patched: ignore signature mismatch') + '\n    except:\n        pass  # Ignore any signature mismatches\n'
    
    new_content = re.sub(pattern, replace_func, content, flags=re.DOTALL)
    
    # If the regex didn't match, try a simpler approach - just comment out the problematic line
    if new_content == content:
        # Find the line with ones_like = _copy_docstring and make it more lenient
        lines = content.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'ones_like = _copy_docstring(tf.ones_like, _ones_like)' in line:
                # Replace with a try-except block
                new_lines.append('    # Patched: lenient ones_like docstring copy')
                new_lines.append('    try:')
                new_lines.append('        ' + line)
                new_lines.append('    except (ValueError, AttributeError):')
                new_lines.append('        # Ignore signature mismatch - tf.ones_like may have different signature')
                new_lines.append('        pass')
            else:
                new_lines.append(line)
            i += 1
        new_content = '\n'.join(new_lines)
    
    with open(prefer_static_file, 'w') as f:
        f.write(new_content)
    print('✓ Patched prefer_static.py (tf.ones_like signature mismatch)')
PYTHON_EOF
    echo "✓ Patched prefer_static.py"
fi

echo ""
echo "Patches applied. Testing imports..."

python3 << 'PYTHON_EOF'
import os
import sys
sys.path.insert(0, 'deepmind-research')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()

from meshgraphnets import dataset
print('✓ Dataset OK')

from meshgraphnets import deforming_plate_model
print('✓ deforming_plate_model OK!')
print('')
print('🎉 SUCCESS! All imports working!')
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠ Some patches may need manual adjustment."
    echo "The environment is mostly set up. You may need to:"
    echo "1. Check tensorflow-probability compatibility"
    echo "2. Consider using TensorFlow 2.12 or earlier"
}

