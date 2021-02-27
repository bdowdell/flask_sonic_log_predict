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

from flask import Flask
from flask_mail import Mail
from config import config
import joblib

mail = Mail()

# load the models using joblib and make them available to the app
xgb_model = joblib.load('app/api/static/models/xgbr_fitted.joblib')
pcr_model = joblib.load('app/api/static/models/pcr_fitted.joblib')
knn_model = joblib.load('app/api/static/models/knn_fitted.joblib')


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    mail.init_app(app)
    
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    return app
