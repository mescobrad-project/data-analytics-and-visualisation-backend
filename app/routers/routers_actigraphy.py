import csv
from fastapi import Query, APIRouter
import pyActigraphy
import os
import pandas as pd
from pyActigraphy.analysis import Cosinor
import plotly.graph_objects as go
from lmfit import fit_report



# import plotly.graph_objs as go

router = APIRouter()
@router.get("/return_actigraphy_data", tags=["actigraphy_data"])
async def return_actigraphy_data():
    with open('example_data/actigraphy_relevant_dataset.csv', newline="") as csvfile:
        if not os.path.isfile('example_data/actigraphy_relevant_dataset.csv'):
            return []
        reader = csv.reader(csvfile, delimiter=',')
        results_array = []
        i = 0
        for row in reader:
            i += 1
            temp_to_append = {
                "id": i,
                "line": row[0],
                "date": row[1],
                "time": row[2],
                "activity": row[3],
                "marker": row[4],
                "whitelight": row[5],
                "sleep_wake": row[6],
                "interval_status": row[7]
            }
            results_array.append(temp_to_append)
            # print(results_array)
        return results_array

    return 1

@router.get("/return_actigraphy_general_data", tags=["actigraphy_general_data"])
async def return_actigraphy_general_data():
    with open('example_data/actigraphy_general_relevant_dataset.csv', newline="") as csvfile:
        if not os.path.isfile('example_data/actigraphy_relevant_dataset.csv'):
            return []
        reader = csv.reader(csvfile, delimiter=',')
        results_array = []
        i = 0
        for row in reader:
            i += 1
            temp_to_append = {
                "id": i,
                "interval_type": row[0],
                "interval": row[1],
                "date_start": row[2],
                "time_start": row[3],
                "date_stop": row[4],
                "time_stop": row[5],
                "duration": row[6],
                "invalid_sw": row[7],
                "efficiency": row[8],
                "wake_time": row[9],
                "sleep_time": row[10],
                "sleep": row[11],
                "exposure_white": row[12],
                "average_white": row[13],
                "max_white": row[14],
                "talt_white": row[15],
                "invalid_white": row[16]
            }
            results_array.append(temp_to_append)
            print(results_array)
        return results_array

    return 1

@router.get("/return_diary", tags=["actigraphy_analysis"])
async def return_diary():
    xx = 0
    try:
        # raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/raw_sample.csv', 'FR', True, None, None, float, float, ';')
        raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
        print(raw.start_time)
        print(raw.duration())
        xx = raw.IS()
        # raw = pyActigraphy.io.read_raw_rpx(fpath + 'raw_sample.csv')
    except:
        print("An exception occurred")

    # raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/Combined Export File.csv', 'ENG_US')

    # raw.start_time
    # raw.duration()
    # sleep_diary = raw.read_sleep_diary('example_data/actigraph/Neurophy_Actigraph.csv')
    # raw.sleep_diary.name
    # raw.sleep_diary.diary
    # raw.sleep_diary.summary()
    print(xx)
    return 1


def return_rawObject():
    xx=0
    try:
        # fpath = f"C:/Users/gdoukas/PycharmProjects/data-analytics-and-visualisation-backend/example_data/actigraph/example_data/actigraph/Neurophy_Actigraph.csv"
        raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/test_sample.csv','FR')
        print(type(raw))
        # raw = pyActigraphy.io.read_raw_rpx(fpath, 'FR')
        xx = raw.IS()
    except:
        print("An exception occurred")

    # raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/Combined Export File.csv', 'ENG_US')

    # raw.start_time
    # raw.duration()
    # sleep_diary = raw.read_sleep_diary('example_data/actigraph/Neurophy_Actigraph.csv')
    # raw.sleep_diary.name
    # raw.sleep_diary.diary
    # raw.sleep_diary.summary()
    return xx


ww = return_rawObject()
print(ww)


@router.get("/actigraphy_metrics")
async def actigraphymetrics(workflow_id: str,
                            step_id: str,
                            run_id: str,
                            i: int,
                            binarize: bool | None = Query(default=True),
                            threshold: int | None = Query(default=4),
                            metric: str | None = Query(default='IS', enum=['IS','ISm','IV','IVm', 'L5','M10','RA','L5p','ADAT','ADATp','M10p','RAp','kRA','kAR']),
                            prefix_offset: str | None = Query(default="Hour",
                                                              regex="^(Hour)$|^(Minute)$|^(Day)$")):

    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')
    if metric == 'IS':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        else:
            freq = f"{i}{pd.offsets.Minute._prefix}"
        xx = raw.IS(binarize=binarize, threshold=threshold, freq=freq)

        return {'IS': xx}
    elif metric == 'ISm':
        xx = raw.ISm(binarize=binarize, threshold=threshold)
        return {'ISm': xx}
    elif metric == 'IV':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        else:
            freq = f"{i}{pd.offsets.Minute._prefix}"
        xx = raw.IV(binarize=binarize, threshold=threshold, freq=freq)

        return {'IV': xx}
    elif metric == 'IVm':
        xx = raw.IVm(binarize=binarize, threshold=threshold)
        return {'IVm': xx}
    elif metric == 'L5':
        xx = raw.L5(binarize=binarize, threshold=threshold)
        return {'L5': xx}
    elif metric == 'M10':
        xx = raw.M10(binarize=binarize, threshold=threshold)
        return {'M10': xx}
    elif metric == 'RA':
        xx = raw.RA(binarize=binarize, threshold=threshold)
        return {'RA': xx}
    elif metric == 'L5p':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        elif prefix_offset == 'Minute':
            freq = f"{i}{pd.offsets.Minute._prefix}"
        else:
            freq = f"{i}{pd.offsets.Day._prefix}"
        xx = raw.L5p(binarize=binarize, threshold=threshold, period=freq)
        return {'L5p': xx}
    elif metric == 'ADAT':
        xx = raw.ADAT(binarize=binarize, threshold=threshold)
        return {'ADAT': xx}
    elif metric == 'ADATp':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        elif prefix_offset == 'Minute':
            freq = f"{i}{pd.offsets.Minute._prefix}"
        else:
            freq = f"{i}{pd.offsets.Day._prefix}"
        xx = raw.ADATp(binarize=binarize, threshold=threshold,period=freq)
        return {'ADATp': xx}
    elif metric == 'M10p':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        elif prefix_offset == 'Minute':
            freq = f"{i}{pd.offsets.Minute._prefix}"
        else:
            freq = f"{i}{pd.offsets.Day._prefix}"
        xx = raw.M10p(binarize=binarize, threshold=threshold,period=freq)
        return {'M10p': xx}
    elif metric == 'RAp':
        if prefix_offset == 'Hour':
            freq = f"{i}{pd.offsets.Hour._prefix}"
        elif prefix_offset == 'Minute':
            freq = f"{i}{pd.offsets.Minute._prefix}"
        else:
            freq = f"{i}{pd.offsets.Day._prefix}"
        xx = raw.RAp(binarize=binarize, threshold=threshold,period=freq)
        return {'RAp': xx}
    elif metric == 'kRA':
        xx = raw.kRA()
        return {'kRA': xx}
    elif metric == 'kAR':
        xx = raw.kAR()
        return {'kAR': xx}

@router.get("/cosinor_analysis_initial_values")
async def cosinoranalysisinitialvalues(workflow_id: str,
                                       step_id: str,
                                       run_id: str):

    cosinor = Cosinor()

    return {'Initial Values': cosinor.fit_initial_params.valuesdict()}

@router.get("/cosinor_analysis")
async def cosinoranalysis(workflow_id: str,
                          step_id: str,
                          run_id: str,
                          Period: int | None = Query(default=None),
                          set_new_values: bool | None = Query(default=False),
                          change_period: bool | None = Query(default=False)):

    raw = pyActigraphy.io.read_raw_rpx('example_data/actigraph/0345-024_18_07_2022_13_00_00_New_Analysis.csv')

    cosinor = Cosinor()

    if set_new_values == False:
        results = cosinor.fit(raw, verbose=True)
        return {'Result':results.params.valuesdict(), 'Akaike information criterium': results.aic, 'Reduced Chi^2': results.redchi, "report":fit_report(results)}
    else:
        if change_period == True:
            cosinor.fit_initial_params['Period'].value = Period
        results = cosinor.fit(raw, verbose=True)
        return {'Result': results.params.valuesdict(), 'Akaike information criterium': results.aic,
                'Reduced Chi^2': results.redchi, "report": fit_report(results)}









