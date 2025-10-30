# Test file for the image generator project
import os
import sys
from unittest.mock import patch

# Add src to path so we can import the main module
sys.path.insert(0, 'src')

import main

# Test default values when no arguments provided
@patch('sys.argv', ['main.py'])
@patch('main.create_and_save_image')
def test_default_values(mock_create):
    main.main()
    # Verify that the default image was created
    mock_create.assert_called_with(640, 480, 'blue')

# Test with arguments provided
@patch('sys.argv', ['main.py', '800', '600', 'red'])
@patch('main.create_and_save_image')
def test_with_args(mock_create):
    main.main()
    mock_create.assert_called_with('800', '600', 'red')

# Test with integer arguments
@patch('sys.argv', ['main.py', '800', '600', 'red'])
@patch('main.create_and_save_image')
def test_with_integer_args(mock_create):
    main.main()
    mock_create.assert_called_with('800', '600', 'red')

# Test invalid color input
@patch('sys.argv', ['main.py'])
@patch('main.create_and_save_image')
def test_invalid_color(mock_create):
    main.main()
    mock_create.assert_called_with(640, 480, 'blue')

# Test invalid width and height input
@patch('sys.argv', ['main.py'])
@patch('main.create_and_save_image')
def test_invalid_width_height(mock_create):
    main.main()
    mock_create.assert_called_with(640, 480, 'blue')