#!/usr/bin/env python3

# Read the new pyproject.toml content once
with open('/tmp/pyproject_toml_new_version.toml', 'rb') as f:
    NEW_PYPROJECT_CONTENT = f.read()

def filter_blob(blob, metadata):
    """Replace pyproject.toml content throughout git history"""
    filename = metadata.get('filename', b'')
    
    if filename == b'visionflow/pyproject.toml':
        blob.data = NEW_PYPROJECT_CONTENT
