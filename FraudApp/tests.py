from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
import os

class FraudDetectionSystemTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.dataset_path = 'Dataset/Fraud_transaction.csv'
        self.test_data_path = 'Dataset/testData.csv'

    def test_01_index_view(self):
        """Test home page renders successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "A Multi-perspective Fraud Detection Method")

    def test_02_login_page_view(self):
        """Test user login page loads."""
        response = self.client.get('/UserLogin.html')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "User Login Module")

    def test_03_login_action(self):
        """Test authentication with admin credentials."""
        # Valid login
        response = self.client.post('/UserLoginAction', {'t1': 'admin', 't2': 'admin'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Welcome, admin!")

        # Invalid login
        bad_response = self.client.post('/UserLoginAction', {'t1': 'wrong', 't2': 'pass'})
        self.assertEqual(bad_response.status_code, 200)
        self.assertContains(bad_response, "Login failed")

    def test_04_full_fraud_detection_pipeline(self):
        """Test complete workflow: upload dataset, process mining, run ML algorithms, and detect fraud."""
        # 1. Upload dataset
        with open(self.dataset_path, 'rb') as f:
            upload_file = SimpleUploadedFile('Fraud_transaction.csv', f.read(), content_type='text/csv')
        load_resp = self.client.post('/LoadDatasetAction', {'t1': upload_file})
        self.assertEqual(load_resp.status_code, 200)
        self.assertContains(load_resp, "Dataset Loaded Successfully")

        # 2. Process Mining PCA Visualization
        pm_resp = self.client.get('/ProcessMining')
        self.assertEqual(pm_resp.status_code, 200)
        self.assertContains(pm_resp, "User Behaviour Process Mining Graph Generated")
        self.assertIn('img', pm_resp.context)
        self.assertTrue(len(pm_resp.context['img']) > 0)

        # 3. Run ML Algorithms (Proposed SVM & Extension Random Forest)
        ml_resp = self.client.get('/RunML')
        self.assertEqual(ml_resp.status_code, 200)
        self.assertContains(ml_resp, "Propose SVM")
        self.assertContains(ml_resp, "Extension Random Forest")
        self.assertIn('img', ml_resp.context)

        # 4. Detect Fraud on Test Dataset
        with open(self.test_data_path, 'rb') as f:
            test_file = SimpleUploadedFile('testData.csv', f.read(), content_type='text/csv')
        detect_resp = self.client.post('/DetectFraudAction', {'t1': test_file})
        self.assertEqual(detect_resp.status_code, 200)
        self.assertContains(detect_resp, "Fraud Detection Results from Test Data")
        self.assertContains(detect_resp, "NORMAL")
