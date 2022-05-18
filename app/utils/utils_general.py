import json


def validate_and_convert_peaks(input_height, input_threshold, input_prominence, input_width,input_plateau_size):
    to_return= {
        "height" : convert_string_to_number_or_array(input_height),
        "threshold" : convert_string_to_number_or_array(input_threshold),
        "prominence" : convert_string_to_number_or_array(input_prominence),
        "width" : convert_string_to_number_or_array(input_width),
        "plateau_size" : convert_string_to_number_or_array(input_plateau_size),
    }

    return to_return

def validate_and_convert_power_spectral_density(input_verbose):
    if input_verbose:
        if isinstance(input_verbose, bool):
            return input_verbose
        elif input_verbose.isdigit():
            return int(input_verbose)
        elif isinstance(input_verbose, str):
            return input_verbose
    else:
        return None

def convert_string_to_number_or_array(input):
    to_return = None
    if input:
        if input.isdigit():
            to_return = float(input)
        else:
            to_return = json.loads(input)
    return to_return
