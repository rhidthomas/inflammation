"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    # NB: the comment 'yapf: disable' disables automatic formatting using
    # a tool called 'yapf' which we have used when creating this project
    test_array = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([0, 0]), daily_mean(test_array))


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_array = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([3, 4]), daily_mean(test_array))

def test_daily_max():
    """Test that max function works for an array of positive integers"""
    from inflammation.models import daily_max

    test_array = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([5, 6]), daily_max(test_array))    


def test_daily_min():
    """Test that max function works for an array of positive and negative integers"""
    from inflammation.models import daily_min

    test_array = np.array([[1, -2],
                           [3, 4],
                           [-5, 6]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([-5, -2]), daily_min(test_array))   

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])

# Parameterised versions of the above tests
@pytest.mark.parametrize(
    "test, expected",
    [
     ([[0, 0], [0, 0], [0, 0]], [0, 0]),
     ([[1, 2], [3, 4], [5, 6]], [3, 4])
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(np.array(expected), daily_mean(np.array(test)))


@pytest.mark.parametrize(
    "test, expected",
    [
     ([[0, 0], [0, 0], [0, 0]], [0, 0]),
     ([[1, 2], [3, -4], [-5, 1]], [3, 2]),
     ([[1, 2, 3], [3, 4, 7], [5, 0, -3]], [5, 4, 7])
    ])
def test_daily_max(test, expected):
    """Test max function works for array of zeroes and mixed positive/negative integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(np.array(expected), daily_max(np.array(test)))

@pytest.mark.parametrize(
    "test, expected",
    [
     ([[0, 0], [0, 0], [0, 0]], [0, 0]),
     ([[1, 2], [3, -4], [-5, 1]], [-5, -4]),
     ([[1, 2, 3], [3, 4, 7], [5, 0, -3]], [1, 0, -3])
    ])
def test_daily_min(test, expected):
    """Test min function works for array of zeroes and mixed positive/negative integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(np.array(expected), daily_min(np.array(test)))

@patch('inflammation.models.get_data_dir', return_value='/data_dir')
def test_load_csv(mock_get_data_dir):
    from inflammation.models import load_csv
    with patch('numpy.loadtxt') as mock_loadtxt:
        load_csv('test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[0]
        assert kwargs['fname'] == '/data_dir/test.csv'
        load_csv('/test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[1]
        assert kwargs['fname'] == '/test.csv'

# TODO(lesson-automatic) Implement tests for the other statistical functions
# TODO(lesson-mocking) Implement a unit test for the load_csv function
