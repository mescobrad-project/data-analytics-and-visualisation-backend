import csv
import os, sys
from operator import index
from os.path import realpath, exists, join
from fastparquet import write as wr, ParquetFile

import glob, pandas
import pandas as pd
import pandas as pandas
from freesurfer_stats import CorticalParcellationStats
from fastapi import APIRouter, Query
from pip._internal.utils.misc import tabulate

router = APIRouter()

def load_whole_brain_measurements(stats_path) -> pandas.DataFrame:
    stats = CorticalParcellationStats.read(stats_path)
    stats.whole_brain_measurements['subject'] = stats.headers['subjectname']
    stats.whole_brain_measurements['source_basename'] = os.path.basename(stats_path)
    stats.whole_brain_measurements['hemisphere'] = stats.hemisphere
    return stats.whole_brain_measurements


def load_structural_measurements(stats_path) -> pandas.DataFrame:
    stats = CorticalParcellationStats.read(stats_path)
    stats.structural_measurements['subject'] = stats.headers['subjectname']
    stats.structural_measurements['source_basename'] = os.path.basename(stats_path)
    stats.structural_measurements['hemisphere'] = stats.hemisphere
    return stats.structural_measurements


@router.get("/list/datalakeendpoints", tags=["list_datalakeendpoints"])
async def list_datalakeendpoints() -> dict:
    endpoints = ["return_samseg_stats/whole_brain_measurements",
                 "return_samseg_stats/structural_measurements",
                "return_samseg_result", 4, ".."]
    # TODO:DataLake connection
    return {'endpoints': endpoints}

@router.get("/return_reconall_stats/whole_brain_measurements", tags=["return_reconall_stats"])
# Validation is done inline in the input of the function
async def return_samseg_stats(fs_dir: str = None, subject_id: str = None, hemisphere_requested: str = None):
    #TODO 1: check DataLake if report from FreeSurfer based on mriId exists
    #TODO 2: read the file "samseg.stats"
    # for testing I use the sample file from example_data folder
    # read data into pd

    whole_brain_measurements = pandas.concat(
        map(load_whole_brain_measurements, glob.glob('example_data/*h.aparc*.stats')),
        sort=False)
    whole_brain_measurements.reset_index(drop=True, inplace=True)
    whole_brain_measurements[['subject', 'source_basename', 'hemisphere']]
    if (hemisphere_requested!=None):
        whole_brain_measurements = whole_brain_measurements.loc[(whole_brain_measurements.hemisphere == hemisphere_requested)]
        print(hemisphere_requested)

    row_count = whole_brain_measurements.shape[0]
    column_count = whole_brain_measurements.shape[1]
    column_names = whole_brain_measurements.columns.tolist()
    final_row_data = []
    result_data = []
    for index, rows in whole_brain_measurements.iterrows():
        final_row_data.append(rows.to_dict())

    json_result = {'rows': row_count, 'cols': column_count, 'columns': column_names, 'rowData': final_row_data}
    result_data.append(json_result)

    return final_row_data #result_data


@router.get("/return_reconall_stats/structural_measurements", tags=["return_reconall_stats"])
# Validation is done inline in the input of the function
async def return_samseg_stats(fs_dir: str = None, subject_id: str = None) -> pandas.DataFrame:
    structural_measurements = pandas.concat(
        map(load_structural_measurements, glob.glob('example_data/*h.aparc*.stats')),
        sort=False)
    structural_measurements.reset_index(drop=True, inplace=True)
    results_array = []
    file_rows=0
    for file in structural_measurements['source_basename'].unique():
        def_only_col = structural_measurements.filter(items=['source_basename', 'structure_name', 'gray_matter_volume_mm^3'])
        pd = def_only_col.where(def_only_col['source_basename'] == file)
        pd = pd.dropna(how="all")
        for i, j in pd.iterrows():
            temp_to_append = {
                "id": i,
                "source_basename": j['source_basename'],
                "structure_name": j['structure_name'],
                "Volume": j['gray_matter_volume_mm^3']
            }
            results_array.append(temp_to_append)
    return results_array


@router.get("/return_samseg_result", tags=["return_samseg_stats"])
async def return_samseg_stats(fs_dir: str = None, subject_id: str = None) -> []:
    with open('example_data/samseg.stats', newline="") as csvfile:
        if not os.path.isfile('example_data/samseg.stats'):
            return []
        reader = csv.reader(csvfile, delimiter=',')
        results_array = []
        i = 0
        for row in reader:
            i += 1
            temp_to_append = {
                "id": i,
                "measure": row[0].strip("# Measure "),
                "value": row[1],
                "unit": row[2]
            }
            results_array.append(temp_to_append)
        return results_array

@router.get("/save_file_to_parquet", tags=["return_samseg_stats"])
async def save_file_to_parquet(fs_dir: str = None, subject_id: str = None) -> []:
    with open('example_data/samseg.stats', newline="") as csvfile:
        if not os.path.isfile('example_data/samseg.stats'):
            return []

        # If we use this method we will have to prepare the .stats file
        # First Remove empty rows and then rename the columns
        df = pd.read_csv(csvfile, delimiter=',', header=None, names=['measure', 'value', 'unit'])
        df = df.fillna('')
        df.to_parquet('samseg_sample_file.parquet', engine='fastparquet')
        return 'OK'

@router.get("/read_parquet_file", tags=["return_samseg_stats"])
async def save_file_to_parquet(fs_dir: str = None, subject_id: str = None) -> []:
    try:
        pf = ParquetFile('samseg_sample_file.parquet')
        df = pf.to_pandas()
        return df
    except Exception as exc:
        print(exc)
        print("error")
        return exc
