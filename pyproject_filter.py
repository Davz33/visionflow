#!/usr/bin/env python3

import sys
import os

def filter_pyproject(blob, metadata):
    """Replace pyproject.toml content throughout git history"""
    
    # Check if this blob is the pyproject.toml file
    if metadata.get('filename') == b'visionflow/pyproject.toml':
        # Read the new version of pyproject.toml
        with open('/tmp/pyproject_toml_new_version.toml', 'rb') as f:
            blob.data = f.read()
    
    return blob

# This script will be used by git filter-repo
if __name__ == "__main__":
    # Import git_filter_repo module
    try:
        import git_filter_repo as fr
    except ImportError:
        print("git-filter-repo module not found")
        sys.exit(1)
    
    # Create blob callback
    def blob_callback(blob, metadata):
        if metadata.get('filename') == b'visionflow/pyproject.toml':
            with open('/tmp/pyproject_toml_new_version.toml', 'rb') as f:
                blob.data = f.read()
    
    # Set up filter
    args = fr.FilteringOptions.parse_args([
        '--blob-callback', 'pyproject_filter:blob_callback',
        '--force'
    ])
    
    filter = fr.RepoFilter(args)
    filter.run()
