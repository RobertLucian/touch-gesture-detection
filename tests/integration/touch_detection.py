import unittest
import kiki
import pandas as pd
from keras.metrics import categorical_accuracy
from time import time

class TestPretrainedModel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestPretrainedModel, self).__init__(*args, **kwargs)

        datasets = [
            'datasets/horizontal_swipe.csv',
            'datasets/vertical_swipe.csv',
            'datasets/tapping.csv',
            'datasets/double_tapping.csv'
        ]

        datasets = list(map(lambda ds: kiki.detection.get_resource_path(ds), datasets))
        datasets = list(map(lambda file: pd.read_csv(file), datasets))
        data = pd.concat(datasets, sort=False)

        # do the one-hot-encoding for the output
        data['Label'] = pd.Categorical(data['Label'])
        data_dummies = pd.get_dummies(data['Label'])
        data = pd.concat([data, data_dummies], axis=1)
        data.drop('Label', inplace=True, axis=1)

        # separate labels from data for the test set
        self.x_validation = data.loc[:, 'A':'F'].to_numpy().reshape((-1, 11, 6)) / 4095
        self.y_validation = data.drop(list('ABCDEF'), axis=1)[(data.index + 1) % 11 == 0].to_numpy()
        self.samples = self.y_validation.shape[0]

        # load the pretrained model
        self.model = kiki.detection.load_pretrained_model('models/touch', inpackage_data=True)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy])

    def test_pretrained_model(self):
        """
        Test the pretrained model against the original datasets.
        """

        _, acc = self.model.evaluate(
            self.x_validation,
            self.y_validation,
            batch_size=self.samples)

        self.assertGreaterEqual(acc, 0.95, 'Model accuracy under 95%')

    def test_run_time(self):
        """
        Test the average time required to run a single sample against the model.
        """

        start = time()
        self.model.predict(
            self.x_validation,
            batch_size=1)
        end = time()
        total = end - start
        period = total / self.samples

        self.assertLessEqual(period, 0.025, f'It took {period} seconds to run a sample, which is more than 0.025 seconds.')


if __name__ == '__main__':
    unittest.main()