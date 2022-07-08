import os
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
                 "return_samseg_stats/structural_measurements", 3, 4, ".."]
    # TODO:DataLake connection
    return {'endpoints': endpoints}

@router.get("/return_samseg_stats/whole_brain_measurements", tags=["return_samseg_stats/whole_brain_measurements"])
# Validation is done inline in the input of the function
async def return_samseg_stats(fs_dir: str = None, subject_id: str = None):
    #TODO 1: check DataLake if report from FreeSurfer based on mriId exists
    #TODO 2: read the file "samseg.stats"
    # for testing I use the sample file from example_data folder
    # read data into pd

    whole_brain_measurements = pandas.concat(
        map(load_whole_brain_measurements, glob.glob('example_data/*h.aparc*.stats')),
        sort=False)
    whole_brain_measurements.reset_index(drop=True, inplace=True)
    whole_brain_measurements[['subject', 'source_basename', 'hemisphere']]

     # stats = CorticalParcellationStats.read('example_data/lh.aparc.a2009s.stats')
    # stats.headers['subjectname'] = 'fabian'
    # print(stats.structural_measurements[['structure_name', 'surface_area_mm^2', 'gray_matter_volume_mm^3']].head())
    # stats = np.load("example_data/samseg.stats", allow_pickle=True)
    # stats = np.loadtxt('example_data/samseg.stats', dtype="i1,i1,i4,f4,S32,f4,f4,f4,f4,f4")
    # for line in stats:
    #     print(line)
    # subcortical_data = np.array([seg[3] for seg in stats])
    # out_data = subcortical_data.flatten()
    # with pd.read_csv("example_data/samseg.stats", sep="\t", header=None) as X:
    #     for line in X:
    #         print(line)

    row_count = whole_brain_measurements.shape[0]
    column_count = whole_brain_measurements.shape[1]
    column_names = whole_brain_measurements.columns.tolist()
    final_row_data = []
    result_data = []
    for index, rows in whole_brain_measurements.iterrows():
        final_row_data.append(rows.to_dict())

    json_result = {'rows': row_count, 'cols': column_count, 'columns': column_names, 'rowData': final_row_data}
    result_data.append(json_result)

    return result_data


@router.get("/return_samseg_stats/structural_measurements", tags=["return_samseg_stats"])
# Validation is done inline in the input of the function
async def return_samseg_stats(fs_dir: str = None, subject_id: str = None) -> pandas.DataFrame:
    structural_measurements = pandas.concat(
        map(load_structural_measurements, glob.glob('example_data/*h.aparc*.stats')),
        sort=False)
    structural_measurements.reset_index(drop=True, inplace=True)
    return tabulate(structural_measurements.values)
