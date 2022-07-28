import csv
import os, sys
from operator import index
from os.path import realpath, exists, join

import glob, pandas
import pandas as pd
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

@router.get("/return_samseg_stats/whole_brain_measurements", tags=["return_samseg_stats/whole_brain_measurements"])
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


@router.get("/return_samseg_stats/structural_measurements", tags=["return_samseg_stats/structural_measurements"])
# Validation is done inline in the input of the function
async def return_samseg_stats(fs_dir: str = None, subject_id: str = None) -> pandas.DataFrame:
    structural_measurements = pandas.concat(
        map(load_structural_measurements, glob.glob('example_data/*h.aparc*.stats')),
        sort=False)
    structural_measurements.reset_index(drop=True, inplace=True)
    return tabulate(structural_measurements.values)


# @router.get("/return_samseg_result", tags=["return_samseg_result"])
# async def return_samseg_stats(fs_dir: str = None, subject_id: str = None) -> []:
#     stats = pd.read_csv('example_data/samseg.stats', header=None)
#     stats.set_axis (["measure", "value", "unit"], axis=1, inplace=True)
#     final_row_data = []
#     stats.insert(0, 'id', "")
#     for rows in stats.itertuples():
#         final_row_data.append(rows)
#     return final_row_data

@router.get("/return_samseg_result", tags=["return_samseg_result"])
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
