#!/bin/bash

#THIS FILE CAN BE USED AS A BASH SCRIPT IN NEURODESKTOP TO RUN A BATCH OF FREE
#SURFER COMMANDS FOR MULTIPLE FILES IF IT HASNT BEEN CHANGED ORIGINALLY IT WAS
#USED TO RUN THE mri_watershed command, if simple change just change the command
# THIS files needs to be situated in the parent folder of the target folder that
# contains input data. The ./mri_watershed "file" 
# Check if input folder is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_folder>"
    exit 1
fi

# Input folder containing the files
input_folder="$1"

# Iterate over each file in the input folder
for input_file in "$input_folder"/*; do
    # Check if file exists
    if [ -f "$input_file" ]; then
        # Extract file name without extension
        filename=$(basename -- "$input_file")
        filename_no_ext="${filename%.*}"

        # Generate output file name
        output_file="${filename_no_ext}_no_skull.nii"

        # Run the command
        mri_watershed "$input_file" "$output_file"

        echo "Processed: $input_file -> $output_file"
    fi
done
