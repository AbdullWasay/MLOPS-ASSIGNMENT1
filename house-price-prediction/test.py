import unittest
from app import app

class FlaskTestCase(unittest.TestCase):

    # Test if the homepage loads
    def test_homepage(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)

    # Test if the API prediction works with correct input
    def test_predict(self):
        tester = app.test_client(self)
        response = tester.post('/api/predict', json={'area': 2000, 'bedrooms': 3, 'bathrooms': 2})
        self.assertEqual(response.status_code, 200)
        self.assertIn('predicted_price', response.get_json())

    # Test if the API handles missing fields
    def test_predict_missing_fields(self):
        tester = app.test_client(self)
        response = tester.post('/api/predict', json={'area': 2000})
        self.assertEqual(response.status_code, 400)  # Bad Request for missing data

    # Test if the API handles invalid inputs
    def test_predict_invalid_inputs(self):
        tester = app.test_client(self)
        response = tester.post('/api/predict', json={'area': 'invalid', 'bedrooms': 'three', 'bathrooms': 2})
        self.assertEqual(response.status_code, 400)  # Bad Request for invalid input types

if __name__ == '__main__':
    unittest.main()
