import csv
from fastapi import Query, APIRouter
import pyActigraphy
import os

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
