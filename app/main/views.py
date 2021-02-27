#!/usr/bin/env python

"""
Copyright YYYY, FN MI LN

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

from flask import render_template, redirect, url_for, request, session, current_app
from . import main
from .. email import send_email
from . forms import ContactForm


@main.route('/')
@main.route('/index')
def index():
    return render_template('index.html')


@main.route('/about')
def about():
    return render_template('about.html')


@main.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        body = request.form.get('body')
        send_email(subject, email, name, current_app.config['ADMINS'], body, template='email/user_message')
        return redirect(url_for('.success'))
    return render_template('contact.html', form=form)


@main.route('/success')
def success():
    return render_template('success.html')
