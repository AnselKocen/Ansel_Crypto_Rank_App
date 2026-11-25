import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests
from web.services import market_catalog as catalog


class MarketCatalogTests(unittest.TestCase):
    def binance(self):
        return {'symbols':[
            {'baseAsset':'BTC','symbol':'BTCUSDT','quoteAsset':'USDT','status':'TRADING','isSpotTradingAllowed':True},
            {'baseAsset':'QNT','symbol':'QNTUSDT','quoteAsset':'USDT','status':'TRADING','isSpotTradingAllowed':True},
            {'baseAsset':'OLD','symbol':'OLDUSDT','quoteAsset':'USDT','status':'BREAK'},
            {'baseAsset':'MARGIN','symbol':'MARGINUSDT','quoteAsset':'USDT','status':'TRADING','isSpotTradingAllowed':False},
            {'baseAsset':'USDCOIN','symbol':'USDCOINBTC','quoteAsset':'BTC','status':'TRADING'}]}

    def coinbase(self):
        return [{'id':'HNT-USD','base_currency':'HNT','quote_currency':'USD','status':'online','trading_disabled':False},
                {'id':'HALT-USD','base_currency':'HALT','quote_currency':'USD','status':'online','trading_disabled':True}]

    def test_only_active_spot_quote_pairs_are_offered(self):
        self.assertEqual(set(catalog.parse_pairs('BINANCE',self.binance())), {'BTC','QNT'})
        self.assertEqual(set(catalog.parse_pairs('COINBASE',self.coinbase())), {'HNT'})

    def test_validated_coinbase_fallback_and_previously_blacklisted_asset(self):
        data={'providers':{p:{'updated_at':time.time(),'pairs':catalog.parse_pairs(p,body)}
                           for p,body in [('BINANCE',self.binance()),('COINBASE',self.coinbase())]}}
        self.assertEqual(catalog.resolve('QNT',data)['tradingview'],'BINANCE:QNTUSDT')
        self.assertEqual(catalog.resolve('HNT',data)['tradingview'],'COINBASE:HNTUSD')
        self.assertIsNone(catalog.resolve('MWC',data))
        self.assertIsNone(catalog.resolve('LUNA',{'providers':{'BINANCE':{'updated_at':time.time(),'pairs':{'LUNC':{'symbol':'LUNCUSDT','quote':'USDT'}}}}}))

    def test_old_directory_does_not_claim_current_listing(self):
        data={'providers':{'BINANCE':{'updated_at':time.time()-49*3600,'pairs':{'BTC':{'symbol':'BTCUSDT','quote':'USDT'}}}}}
        self.assertIsNone(catalog.resolve('BTC',data))

    def test_directory_refresh_is_shared_cached_and_preserves_prior_data_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'catalog.json'
            responses=[Mock(json=lambda:self.binance()),Mock(json=lambda:self.coinbase())]
            with patch.object(catalog,'catalog_file',return_value=path), patch.object(catalog.requests,'get',side_effect=responses) as get:
                first=catalog.refresh_catalog()
                second=catalog.refresh_catalog()
                self.assertEqual(get.call_count,2)
                self.assertEqual(first,second)
            with patch.object(catalog,'catalog_file',return_value=path), patch.object(catalog.requests,'get',side_effect=requests.ConnectionError()):
                failed=catalog.refresh_catalog(force=True)
                self.assertEqual(catalog.resolve('BTC',failed)['tradingview'],'BINANCE:BTCUSDT')
                self.assertEqual(failed['providers']['BINANCE']['status'],'unavailable')


if __name__=='__main__':
    unittest.main()
