#!/usr/bin/env python

"""
Copyright 2021, Benjamin L Dowdell

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

import unittest
from app import create_app
import pandas as pd
import json


class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()
        self.headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        self.df_in = pd.DataFrame(
            data=[[0.3521, 55.1824, 0.8121, 2.3256]],
            columns=['CNC', 'GR', 'HRD', 'ZDEN']
            )
        self.df_in_misordered = pd.DataFrame(
            data=[[2.3256, 0.8121, 55.1824, 0.3521]],
            columns=['ZDEN', 'HRD', 'GR', 'CNC']
            )
        self.df_in_extra_cols = pd.DataFrame(
            data=[[8.5781, 0.3521, 55.1824, 0.8121, 0.78099, 6.8291, 2.3256]],
            columns=['CAL', 'CNC', 'GR', 'HRD', 'HRM', 'PE', 'ZDEN']
            )
        self.df_in_bad_col_names = pd.DataFrame(
            data=[[0.3521, 55.1824, 0.8121, 2.3256]],
            columns=['Phi', 'Gamma', 'RD', 'RHOB']
        )
        self.df_out = pd.DataFrame(
            data=[[102.225407, 196.408402]],
            columns=['pred_DTC', 'pred_DTS']
            )

    def tearDown(self):
        self.app_context.pop()

    def test_get_predictions(self):
        j_df = json.dumps(self.df_in.to_json(orient='split'))
        response = self.client.post('api/get_predictions', data=j_df, headers=self.headers)
        self.assertEqual(response.status_code, 200)

    def test_get_predictions_swapped_input_cols(self):
        j_df = json.dumps(self.df_in_misordered.to_json(orient='split'))
        response = self.client.post('api/get_predictions', data=j_df, headers=self.headers)
        df_pred = pd.read_json(response.data, orient='split')
        self.assertAlmostEqual(self.df_out.iloc[0, 0], round(df_pred.iloc[0, 0], 6))
        self.assertAlmostEqual(self.df_out.iloc[0, 1], round(df_pred.iloc[0, 1], 6))

    def test_get_predictions_extra_cols(self):
        j_df = json.dumps(self.df_in_extra_cols.to_json(orient='split'))
        response = self.client.post('api/get_predictions', data=j_df, headers=self.headers)
        df_pred = pd.read_json(response.data, orient='split')
        self.assertAlmostEqual(self.df_out.iloc[0, 0], round(df_pred.iloc[0, 0], 6))
        self.assertAlmostEqual(self.df_out.iloc[0, 1], round(df_pred.iloc[0, 1], 6))

    def test_get_predictions_bad_col_names(self):
        j_df = json.dumps(self.df_in_bad_col_names.to_json(orient='split'))
        response = self.client.post('api/get_predictions', data=j_df, headers=self.headers)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.data.decode('utf-8'), 'Key Error, please check data frame column names')

    def test_bad_url_404(self):
        j_df = json.dumps(self.df_in.to_json(orient='split'))
        response = self.client.post('bad/url', data=j_df, headers=self.headers)
        self.assertEqual(response.status_code, 404)
        