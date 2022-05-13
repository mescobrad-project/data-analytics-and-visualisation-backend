def validate_and_convert_peaks(input_height, input_threshold, input_prominence, input_width,input_plateau_size):
    to_return= {
        "height" : None,
        "threshold" : None,
        "prominence" : None,
        "width" : None,
        "plateau_size" : None,
    }
    if input_height:
        to_return["height"] = int(input_height) if input_height.isdigit() else None

    return to_return
