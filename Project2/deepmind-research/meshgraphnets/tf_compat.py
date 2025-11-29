"""
Compatibility shim for TensorFlow 1.15.

This module provides tensorflow.compat.v1 for TensorFlow 1.15,
allowing code written for TF 2.x to work with TF 1.15.

This is automatically imported by run_model.py if needed.
"""

import sys
import tensorflow as tf

# Check if we're using TF 1.x (which doesn't have compat.v1)
if not hasattr(tf, 'compat') or not hasattr(tf.compat, 'v1'):
    # Create compat.v1 module that mirrors tf for TF 1.15
    import types
    
    # Create a module-like object that forwards all attributes to tf
    compat_v1 = types.ModuleType('compat.v1')
    # Copy all public attributes from tf to compat_v1
    for attr_name in dir(tf):
        if not attr_name.startswith('_'):
            try:
                setattr(compat_v1, attr_name, getattr(tf, attr_name))
            except (AttributeError, TypeError):
                pass
    
    # Create compat module
    if not hasattr(tf, 'compat'):
        tf.compat = types.ModuleType('compat')
    tf.compat.v1 = compat_v1

# Ensure disable_v2_behavior exists (it doesn't in TF 1.15)
if not hasattr(tf, 'disable_v2_behavior'):
    def disable_v2_behavior():
        """No-op for TF 1.15 (v2 behavior doesn't exist)"""
        pass
    tf.disable_v2_behavior = disable_v2_behavior

# Ensure enable_resource_variables exists (it's the default in TF 1.15)
if not hasattr(tf, 'enable_resource_variables'):
    def enable_resource_variables():
        """No-op for TF 1.15 (resource variables are default)"""
        pass
    tf.enable_resource_variables = enable_resource_variables

