import unittest
import os
from tests.config.definitions import ROOT_DIR
from app.app import App
from sdk.moveapps_io import MoveAppsIo
import pandas as pd
import movingpandas as mpd


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        os.environ['APP_ARTIFACTS_DIR'] = os.path.join(ROOT_DIR, 'tests/resources/output')
        self.sut = App(moveapps_io=MoveAppsIo())

    def test_app_runs(self):
        # prepare
        data: mpd.TrajectoryCollection = pd.read_pickle(os.path.join(ROOT_DIR, 'resources/samples/input1.pickle'))
        config: dict = {}

        # execute
        self.sut.execute(data=data, config=config)
