"""
Unit tests for data preprocessing module
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor, preprocess_pipeline


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5

        # Create synthetic data
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, 2, self.n_samples)

        # Create DataFrame
        self.df = pd.DataFrame(
            self.X,
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.df['target'] = self.y

        # Add some missing values and duplicates
        self.df.loc[5:10, 'feature_0'] = np.nan
        self.df.loc[95:100] = self.df.loc[0:5].values

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(
            scaling_method='standard',
            feature_selection_method='pca',
            n_components=3
        )

        self.assertEqual(preprocessor.scaling_method, 'standard')
        self.assertEqual(preprocessor.feature_selection_method, 'pca')
        self.assertEqual(preprocessor.n_components, 3)
        self.assertFalse(preprocessor._fitted)

    def test_clean_data_drop_missing(self):
        """Test data cleaning with dropping missing values."""
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.clean_data(
            self.df,
            drop_duplicates=True,
            handle_missing='drop'
        )

        # Check no missing values
        self.assertEqual(df_clean.isnull().sum().sum(), 0)

        # Check duplicates removed
        self.assertEqual(df_clean.duplicated().sum(), 0)

        # Check shape is smaller
        self.assertLess(df_clean.shape[0], self.df.shape[0])

    def test_clean_data_fill_missing(self):
        """Test data cleaning with filling missing values."""
        preprocessor = DataPreprocessor()

        # Test mean filling
        df_mean = preprocessor.clean_data(
            self.df,
            drop_duplicates=False,
            handle_missing='mean'
        )
        self.assertEqual(df_mean.isnull().sum().sum(), 0)

        # Test median filling
        df_median = preprocessor.clean_data(
            self.df,
            drop_duplicates=False,
            handle_missing='median'
        )
        self.assertEqual(df_median.isnull().sum().sum(), 0)

    def test_scaling_methods(self):
        """Test different scaling methods."""
        scaling_methods = ['standard', 'minmax', 'robust']

        for method in scaling_methods:
            with self.subTest(method=method):
                preprocessor = DataPreprocessor(scaling_method=method)
                preprocessor.fit(self.X)
                X_scaled = preprocessor.transform(self.X)

                # Check shape preserved
                self.assertEqual(X_scaled.shape, self.X.shape)

                # Check scaling worked (for standard scaling)
                if method == 'standard':
                    np.testing.assert_almost_equal(
                        X_scaled.mean(axis=0), np.zeros(self.n_features), decimal=10
                    )
                    np.testing.assert_almost_equal(
                        X_scaled.std(axis=0), np.ones(self.n_features), decimal=10
                    )

                # Check minmax scaling
                if method == 'minmax':
                    self.assertTrue((X_scaled >= 0).all())
                    self.assertTrue((X_scaled <= 1).all())

    def test_feature_selection_pca(self):
        """Test PCA feature selection."""
        n_components = 3
        preprocessor = DataPreprocessor(
            scaling_method='standard',
            feature_selection_method='pca',
            n_components=n_components
        )

        X_transformed = preprocessor.fit_transform(self.X)

        # Check dimensions reduced
        self.assertEqual(X_transformed.shape[1], n_components)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])

        # Check feature importance available
        importance = preprocessor.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), n_components)

    def test_feature_selection_kbest(self):
        """Test SelectKBest feature selection."""
        k = 3
        preprocessor = DataPreprocessor(
            scaling_method='standard',
            feature_selection_method='kbest',
            n_components=k
        )

        X_transformed = preprocessor.fit_transform(self.X, self.y)

        # Check dimensions reduced
        self.assertEqual(X_transformed.shape[1], k)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])

        # Check feature importance available
        scores = preprocessor.get_feature_importance()
        self.assertIsNotNone(scores)
        self.assertEqual(len(scores), self.n_features)

    def test_engineer_features(self):
        """Test feature engineering."""
        preprocessor = DataPreprocessor()

        # Test interaction features
        df_interactions = preprocessor.engineer_features(
            self.df.drop('target', axis=1),
            create_interactions=True,
            create_polynomials=False
        )

        # Check new features created
        self.assertGreater(df_interactions.shape[1], self.df.shape[1] - 1)

        # Test polynomial features
        df_poly = preprocessor.engineer_features(
            self.df.drop('target', axis=1),
            create_interactions=False,
            create_polynomials=True,
            degree=2
        )

        # Check polynomial features created
        self.assertGreater(df_poly.shape[1], self.df.shape[1] - 1)

        # Test both
        df_both = preprocessor.engineer_features(
            self.df.drop('target', axis=1),
            create_interactions=True,
            create_polynomials=True,
            degree=2
        )

        # Check even more features created
        self.assertGreater(df_both.shape[1], df_poly.shape[1])
        self.assertGreater(df_both.shape[1], df_interactions.shape[1])

    def test_fit_transform(self):
        """Test fit_transform method."""
        preprocessor = DataPreprocessor(
            scaling_method='standard',
            feature_selection_method='pca',
            n_components=3
        )

        X_transformed = preprocessor.fit_transform(self.X)

        # Check fitted flag
        self.assertTrue(preprocessor._fitted)

        # Check transformation
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertEqual(X_transformed.shape[1], 3)

    def test_transform_without_fit(self):
        """Test that transform fails without fit."""
        preprocessor = DataPreprocessor()

        with self.assertRaises(ValueError):
            preprocessor.transform(self.X)

    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        config = {
            'scaling_method': 'standard',
            'feature_selection_method': 'pca',
            'n_components': 3,
            'handle_missing': 'drop',
            'drop_duplicates': True,
            'engineer_features': True,
            'create_interactions': True,
            'create_polynomials': False
        }

        result = preprocess_pipeline(
            self.df,
            target_col='target',
            test_size=0.2,
            preprocessing_config=config
        )

        # Check all required keys present
        required_keys = ['X_train', 'X_test', 'y_train', 'y_test',
                        'preprocessor', 'feature_names']
        for key in required_keys:
            self.assertIn(key, result)

        # Check shapes
        self.assertEqual(result['X_train'].shape[1], 3)  # n_components
        self.assertEqual(result['X_test'].shape[1], 3)

        # Check train/test split proportions
        total_samples = len(result['y_train']) + len(result['y_test'])
        test_ratio = len(result['y_test']) / total_samples
        self.assertAlmostEqual(test_ratio, 0.2, places=1)

    def test_preprocess_pipeline_unsupervised(self):
        """Test preprocessing pipeline for unsupervised learning."""
        df_no_target = self.df.drop('target', axis=1)

        result = preprocess_pipeline(
            df_no_target,
            target_col=None,
            preprocessing_config={'scaling_method': 'minmax'}
        )

        # Check required keys for unsupervised
        required_keys = ['X', 'preprocessor', 'feature_names']
        for key in required_keys:
            self.assertIn(key, result)

        # Check no train/test split
        self.assertNotIn('X_train', result)
        self.assertNotIn('y_train', result)

    def test_invalid_scaling_method(self):
        """Test invalid scaling method raises error."""
        with self.assertRaises(ValueError):
            preprocessor = DataPreprocessor(scaling_method='invalid')
            preprocessor.fit(self.X)

    def test_kbest_without_labels(self):
        """Test that SelectKBest fails without labels."""
        preprocessor = DataPreprocessor(
            feature_selection_method='kbest',
            n_components=3
        )

        with self.assertRaises(ValueError):
            preprocessor.fit_transform(self.X)


class TestIntegration(unittest.TestCase):
    """Integration tests for preprocessing module."""

    def test_end_to_end_supervised(self):
        """Test end-to-end supervised preprocessing."""
        # Create larger dataset
        np.random.seed(42)
        n_samples = 1000
        n_features = 20

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)  # 3 classes

        df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(n_features)])
        df['label'] = y

        # Add some noise
        df.loc[100:110, 'f_0'] = np.nan
        df.loc[200:205] = df.loc[0:5].values

        # Run pipeline
        result = preprocess_pipeline(
            df,
            target_col='label',
            test_size=0.3,
            preprocessing_config={
                'scaling_method': 'robust',
                'feature_selection_method': 'pca',
                'n_components': 10,
                'handle_missing': 'mean',
                'engineer_features': True,
                'create_polynomials': True,
                'degree': 2
            }
        )

        # Verify results
        self.assertEqual(result['X_train'].shape[1], 10)
        self.assertEqual(result['X_test'].shape[1], 10)
        self.assertTrue(len(result['y_train']) > 0)
        self.assertTrue(len(result['y_test']) > 0)

        # Check no NaN values
        self.assertFalse(np.isnan(result['X_train']).any())
        self.assertFalse(np.isnan(result['X_test']).any())


if __name__ == '__main__':
    unittest.main()