#!/usr/bin/env python3
"""
Fix tensorflow_probability prefer_static.py to handle tf.ones_like signature mismatch.
This patches the _copy_docstring function to be more lenient with signature mismatches.
"""

import sys
import os
import re

def find_tfp_dir():
    """Find tensorflow_probability installation directory."""
    try:
        import tensorflow_probability
        return os.path.dirname(tensorflow_probability.__file__)
    except ImportError:
        # Try common locations
        for base in ['/opt/anaconda3', os.path.expanduser('~/anaconda3'), os.path.expanduser('~/miniconda3')]:
            for python_version in ['3.12', '3.11', '3.10']:
                path = f"{base}/lib/python{python_version}/site-packages/tensorflow_probability"
                if os.path.exists(path):
                    return path
                # Also check base conda env
                path = f"{base}/lib/python{python_version}/site-packages/tensorflow_probability"
                if os.path.exists(path):
                    return path
    return None

def patch_prefer_static(tfp_dir):
    """Patch prefer_static.py to handle signature mismatches."""
    prefer_static_file = os.path.join(tfp_dir, 'python', 'internal', 'prefer_static.py')
    
    if not os.path.exists(prefer_static_file):
        print(f"Error: {prefer_static_file} not found")
        return False
    
    with open(prefer_static_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'Patched: lenient ones_like' in content:
        print("✓ prefer_static.py already patched")
        return True
    
    # Find the problematic line: ones_like = _copy_docstring(tf.ones_like, _ones_like)
    # Replace it with a try-except block
    pattern = r'(ones_like = _copy_docstring\(tf\.ones_like, _ones_like\))'
    
    def replace_func(match):
        original_line = match.group(1)
        return f'''    # Patched: lenient ones_like docstring copy
    try:
        {original_line}
    except (ValueError, AttributeError):
        # Ignore signature mismatch - tf.ones_like may have different signature (e.g., layout parameter)
        pass'''
    
    new_content = re.sub(pattern, replace_func, content)
    
    if new_content == content:
        # Try a different approach - find the line number and replace
        lines = content.split('\n')
        new_lines = []
        patched = False
        for i, line in enumerate(lines):
            if 'ones_like = _copy_docstring(tf.ones_like, _ones_like)' in line and not patched:
                # Add try-except around this line
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + '# Patched: lenient ones_like docstring copy')
                new_lines.append(' ' * indent + 'try:')
                new_lines.append(line)
                new_lines.append(' ' * indent + 'except (ValueError, AttributeError):')
                new_lines.append(' ' * indent + '    # Ignore signature mismatch - tf.ones_like may have different signature')
                new_lines.append(' ' * indent + '    pass')
                patched = True
            else:
                new_lines.append(line)
        
        if patched:
            new_content = '\n'.join(new_lines)
        else:
            print("Warning: Could not find the problematic line to patch")
            return False
    
    with open(prefer_static_file, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Patched {prefer_static_file}")
    return True

if __name__ == '__main__':
    tfp_dir = find_tfp_dir()
    if not tfp_dir:
        print("Error: Could not find tensorflow_probability installation")
        sys.exit(1)
    
    print(f"Found tensorflow_probability at: {tfp_dir}")
    if patch_prefer_static(tfp_dir):
        print("✓ Successfully patched prefer_static.py")
        sys.exit(0)
    else:
        print("✗ Failed to patch prefer_static.py")
        sys.exit(1)

