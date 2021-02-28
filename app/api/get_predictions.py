#!/usr/bin/env python

"""
Copyright 2021, Benjamin L. Dowdell

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from flask import request, jsonify
from . import api
from app import pcr_model, knn_model, xgb_model
import pandas as pd


@api.route('/get_predictions', methods=['POST'])
def get_predictions():
    data = request.get_json()
    df_in = pd.read_json(data, orient='split')
    try:
        df_in = df_in[['CNC', 'GR', 'HRD', 'ZDEN']]  # ensure that the columns are in the correct order
        xgb_pred = xgb_model.predict(df_in)
        pcr_pred = pcr_model.predict(df_in)
        knn_pred = knn_model.predict(df_in)
        y_pred = (xgb_pred + pcr_pred + knn_pred) / 3
        df_out = pd.DataFrame(data=y_pred, columns=['pred_DTC', 'pred_DTS'])  # return the predictions as a pandas dataframe
        return df_out.to_json(orient='split')
    except KeyError:
        return 'Key Error, please check data frame column names', 500
