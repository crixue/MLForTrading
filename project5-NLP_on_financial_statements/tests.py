import collections
from collections import OrderedDict
import copy
import pandas as pd
import numpy as np
from datetime import date, timedelta


def assert_structure(received_obj, expected_obj, obj_name):
    assert isinstance(received_obj, type(expected_obj)), \
        'Wrong type for output {}. Got {}, expected {}'.format(obj_name, type(received_obj), type(expected_obj))

    if hasattr(expected_obj, 'shape'):
        assert received_obj.shape == expected_obj.shape, \
            'Wrong shape for output {}. Got {}, expected {}'.format(obj_name, received_obj.shape, expected_obj.shape)
    elif hasattr(expected_obj, '__len__'):
        assert len(received_obj) == len(expected_obj), \
            'Wrong len for output {}. Got {}, expected {}'.format(obj_name, len(received_obj), len(expected_obj))

    if type(expected_obj) == pd.DataFrame:
        assert set(received_obj.columns) == set(expected_obj.columns), \
            'Incorrect columns for output {}\n' \
            'COLUMNS:          {}\n' \
            'EXPECTED COLUMNS: {}'.format(obj_name, sorted(received_obj.columns), sorted(expected_obj.columns))

        # This is to catch a case where __equal__ says it's equal between different types
        assert set([type(i) for i in received_obj.columns]) == set([type(i) for i in expected_obj.columns]), \
            'Incorrect types in columns for output {}\n' \
            'COLUMNS:          {}\n' \
            'EXPECTED COLUMNS: {}'.format(obj_name, sorted(received_obj.columns), sorted(expected_obj.columns))

        for column in expected_obj.columns:
            assert received_obj[column].dtype == expected_obj[column].dtype, \
                'Incorrect type for output {}, column {}\n' \
                'Type:          {}\n' \
                'EXPECTED Type: {}'.format(obj_name, column, received_obj[column].dtype, expected_obj[column].dtype)

    if type(expected_obj) in {pd.DataFrame, pd.Series}:
        assert set(received_obj.index) == set(expected_obj.index), \
            'Incorrect indices for output {}\n' \
            'INDICES:          {}\n' \
            'EXPECTED INDICES: {}'.format(obj_name, sorted(received_obj.index), sorted(expected_obj.index))

        # This is to catch a case where __equal__ says it's equal between different types
        assert set([type(i) for i in received_obj.index]) == set([type(i) for i in expected_obj.index]), \
            'Incorrect types in indices for output {}\n' \
            'INDICES:          {}\n' \
            'EXPECTED INDICES: {}'.format(obj_name, sorted(received_obj.index), sorted(expected_obj.index))


def does_data_match(obj_a, obj_b):
    if type(obj_a) == pd.DataFrame:
        # Sort Columns
        obj_b = obj_b.sort_index(1)
        obj_a = obj_a.sort_index(1)

    if type(obj_a) in {pd.DataFrame, pd.Series}:
        # Sort Indices
        obj_b = obj_b.sort_index()
        obj_a = obj_a.sort_index()
    try:
        data_is_close = np.isclose(obj_b, obj_a, equal_nan=True)
    except TypeError:
        data_is_close = obj_b == obj_a
    else:
        if isinstance(obj_a, collections.Iterable):
            data_is_close = data_is_close.all()

    return data_is_close


def assert_output(fn, fn_inputs, fn_expected_outputs, check_parameter_changes=True):
    assert type(fn_expected_outputs) == OrderedDict

    if check_parameter_changes:
        fn_inputs_passed_in = copy.deepcopy(fn_inputs)
    else:
        fn_inputs_passed_in = fn_inputs

    fn_raw_out = fn(**fn_inputs_passed_in)

    # Check if inputs have changed
    if check_parameter_changes:
        for input_name, input_value in fn_inputs.items():
            passed_in_unchanged = _is_equal(input_value, fn_inputs_passed_in[input_name])

            assert passed_in_unchanged, 'Input parameter "{}" has been modified inside the function. ' \
                                        'The function shouldn\'t modify the function parameters.'.format(input_name)

    fn_outputs = OrderedDict()
    if len(fn_expected_outputs) == 1:
        fn_outputs[list(fn_expected_outputs)[0]] = fn_raw_out
    elif len(fn_expected_outputs) > 1:
        assert type(fn_raw_out) == tuple,\
            'Expecting function to return tuple, got type {}'.format(type(fn_raw_out))
        assert len(fn_raw_out) == len(fn_expected_outputs),\
            'Expected {} outputs in tuple, only found {} outputs'.format(len(fn_expected_outputs), len(fn_raw_out))
        for key_i, output_key in enumerate(fn_expected_outputs.keys()):
            fn_outputs[output_key] = fn_raw_out[key_i]

    err_message = _generate_output_error_msg(
        fn.__name__,
        fn_inputs,
        fn_outputs,
        fn_expected_outputs)

    for fn_out, (out_name, expected_out) in zip(fn_outputs.values(), fn_expected_outputs.items()):
        assert_structure(fn_out, expected_out, out_name)
        correct_data = does_data_match(expected_out, fn_out)

        assert correct_data, err_message
