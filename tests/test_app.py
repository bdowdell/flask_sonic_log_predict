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

import unittest
from app import create_app
from app.main.forms import ContactForm


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        self.client = self.app.test_client()

    def tearDown(self):
        self.app_context.pop()

    def test_index_page_get(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_about_page_get(self):
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)

    def test_contact_page_get(self):
        response = self.client.get('/contact')
        self.assertEqual(response.status_code, 200)

    def test_contact_page_post(self):
        form = ContactForm(
            name='Test Name',
            email='test@mail.com',
            subject='Hello World',
            body='Is there anybody out there?'
        )
        response = self.client.post('/contact', data=form.data)
        self.assertEqual(response.status_code, 302)  # successful validation should result in redirect
        response = self.client.post('/contact', data=form.data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(form.validate())

    def test_contact_page_bad_name_field(self):
        form = ContactForm(
            name=None,
            email='test@mail.com',
            subject='Hello World',
            body='This should not validate'
        )
        response = self.client.post('/contact', data=form.data)
        self.assertIsNot(form.validate(), True)  # name is required to validate
        self.assertEqual(response.status_code, 200)

    def test_contact_page_bad_email_field(self):
        form = ContactForm(
            name='Test Name',
            email='test@mail',
            subject='Hello World',
            body='This should not validate'
        )
        response = self.client.post('/contact', data=form.data)
        self.assertIsNot(form.validate(), True)  # email is not valid
        self.assertEqual(response.status_code, 200)

    def test_contact_page_bad_subject_field(self):
        form = ContactForm(
            name='Test Name',
            email='test@mail.com',
            subject=None,
            body='This should not validate'
        )
        response = self.client.post('/contact', data=form.data)
        self.assertIsNot(form.validate(), True)  # subject is required to validate
        self.assertEqual(response.status_code, 200)

    def test_contact_page_bad_body_field(self):
        form = ContactForm(
            name='Test Name',
            email='test@mail.com',
            subject='Hello World',
            body=None
        )
        response = self.client.post('/contact', data=form.data)
        self.assertIsNot(form.validate(), True)  # body is required to validate
        self.assertEqual(response.status_code, 200)
        form = ContactForm(
            name='Test Name',
            email='test@mail.com',
            subject='Hello World',
            body='a'
        )
        response = self.client.post('/contact', data=form.data)
        self.assertIsNot(form.validate(), True)  # body should be too short to validate
        self.assertEqual(response.status_code, 200)

    def test_success_page_get(self):
        response = self.client.get('/success')
        self.assertEqual(response.status_code, 200)

    def test_404_page_not_found_error(self):
        response = self.client.get('/bad-url')
        self.assertEqual(response.status_code, 404)
        self.assertTrue(b'404 error' in response.data)

    def test_500_internal_server_error(self):
        # https://scotch.io/tutorials/build-a-crud-web-app-with-python-and-flask-part-three
        # first example I could find that shows how to actually test that 500 routes correctly
        @self.app.route('/500')
        def internal_server_error():
            from flask import abort
            abort(500)
        response = self.client.get('/500')
        self.assertEqual(response.status_code, 500)
        self.assertTrue(b'500 error' in response.data)
