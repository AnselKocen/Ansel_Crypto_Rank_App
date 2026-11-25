import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import time
from unittest.mock import patch

import pandas as pd

from web.services import prediction_service as service


class LivePricesDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.publication = {'path':self.root, 'period':'2026-11-11',
                            'manifest':{'generated_at':'2026-11-12T01:05:00+00:00'}}
        self.updates = SimpleNamespace(published=lambda: self.publication)
        self.catalog_patch = patch.object(service, 'read_catalog', return_value={'providers':{
            'BINANCE':{'updated_at':time.time(), 'pairs':{'BTC':{'symbol':'BTCUSDT','quote':'USDT'}}}}})
        self.catalog_patch.start()

    def tearDown(self):
        self.catalog_patch.stop()
        self.tmp.cleanup()

    def write(self, frame):
        frame.to_csv(self.root / 'pred_extra_all.csv', index=False)

    def load(self):
        with patch.object(service, 'get_updates', return_value=self.updates):
            return service.load_predictions('extra_all')

    def test_score_column_not_metadata_determines_real_universe_ranking(self):
        self.write(pd.DataFrame({'symbol':[f'C{i}' for i in range(50)],
                                'decision_date':['2026-11-11']*50,
                                'y_pred':[i*.001 for i in range(50)]}))
        data = self.load()
        self.assertEqual(data['top_20'][0]['symbol'], 'C49')
        self.assertEqual(data['top_20'][0]['rank'], 1)
        self.assertEqual([c['rank'] for c in data['bottom_20']], list(range(31,51)))
        self.assertEqual(data['universe_count'], 50)
        self.assertEqual(data['value_unit'], 'decimal_return')
        self.assertAlmostEqual(data['top_20'][0]['value'], .049)

    def test_positive_bottom_forecasts_are_preserved_and_unmapped_assets_not_hidden(self):
        symbols = ['BTC','QNT']+[f'C{i}' for i in range(48)]
        self.write(pd.DataFrame({'symbol':symbols,'y_pred':[.01+i*.001 for i in range(50)]}))
        data = self.load()
        qnt = next(x for x in data['bottom_20'] if x['symbol']=='QNT')
        self.assertGreater(qnt['value'], 0)
        self.assertIsNone(qnt['tradingview'])
        self.assertIn('未确认',qnt['chart_unavailable_reason'])

    def test_data_date_and_prediction_window_bound_to_single_publication(self):
        self.write(pd.DataFrame({'symbol':['BTC'],'y_pred':[.03]}))
        from unittest.mock import Mock
        self.updates.published = Mock(return_value=self.publication)
        data = self.load()
        self.updates.published.assert_called_once()
        self.assertEqual(data['data_period'], '2026-11-11')
        self.assertEqual(data['forecast_start'], '2026-11-12')
        self.assertEqual(data['forecast_end'], '2026-11-18')

    def test_legacy_predictions_are_not_falsely_stamped_with_current_week(self):
        self.write(pd.DataFrame({'symbol':['BTC'],'y_pred':[.03]}))
        self.updates.published = lambda: None
        with patch.object(service, 'BASE_DIR', self.root):
            data = self.load()
        self.assertIsNone(data['data_period'])
        self.assertIsNone(data['forecast_end'])

    def test_missing_published_file_never_falls_back_to_stale_root_score(self):
        legacy = self.root / 'legacy'
        legacy.mkdir()
        pd.DataFrame({'symbol':['BTC'],'y_pred':[.99]}).to_csv(legacy/'pred_extra_all.csv',index=False)
        with patch.object(service, 'BASE_DIR', legacy):
            self.assertIsNone(self.load())

    def test_unusable_prediction_values_and_duplicates_rejected(self):
        for frame in [pd.DataFrame({'symbol':['BTC'],'y_pred':[float('inf')]}),
                      pd.DataFrame({'symbol':['BTC','BTC'],'y_pred':[.1,.2]}),
                      pd.DataFrame({'symbol':['BTC'],'metadata':['not a forecast']})]:
            self.write(frame)
            self.assertIsNone(self.load())

    def test_malformed_chart_symbol_does_not_create_embed_mapping(self):
        self.assertIsNone(service.map_symbol_to_tradingview("BTC' onclick='alert(1)"))
        self.assertEqual(service.map_symbol_to_tradingview(' btc '), 'BINANCE:BTCUSDT')


if __name__ == '__main__':
    unittest.main()
