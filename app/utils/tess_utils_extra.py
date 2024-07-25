import os
import shutil
import tarfile
import gzip

def untar_files(directory, output_directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("FLAIR.nii.gz"):
                file_path = os.path.join(root, file)
                untar_file(file_path, output_directory)
                # rename_files(file_path)
def rename_files(file_path):
    print(file_path)
    os.rename(file_path, file_path + ".nii")
def untar_file(file_path, output_directory):
    # output_file_path = os.path.join(output_directory, file_path)
    print(file_path)
    output_file_path = os.path.join(output_directory, os.path.basename(os.path.normpath(file_path)))
    output_file_path = output_file_path.replace(".gz", "")
    print(output_file_path)
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":
    # directory = input("Enter the directory path: ")
    # output_directory = input("Enter the output directory path: ")
    # untar_files(directory, output_directory)
    # with gzip.open("C:/sub-00001/anat/sub-00001_acq-T2sel_FLAIR.nii.gz", ) as f_in:
    #     with open("C:/Users/USER/Desktop/sub-00001-test/test1.nii", 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)
    #
    # with open("C:/Users/USER/EPU-NTUA/MES-CoBraD EPU - Working/06 Paradotea - Epistimoniki Tekmiriosi/WP6/MRIs_Epilepsy/sub-00001/anat/sub-00001_acq-T2sel_FLAIR.nii.gz"):
    #     print("Hello")
    untar_files(directory= "C:/Users/USER/EPU-NTUA/MES-CoBraD EPU - Working/06 Paradotea - Epistimoniki Tekmiriosi/WP6/MRIs_Epilepsy",
                output_directory="C:/Users/USER/EPU-NTUA/MES-CoBraD EPU - Working/06 Paradotea - Epistimoniki Tekmiriosi/WP6/MRIs_Epilepsy_Extracted")
