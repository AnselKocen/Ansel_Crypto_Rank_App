"""Browser integration checks against a disposable Chrome debugging profile.

Run Chrome separately with --headless --remote-debugging-port=9223 and a fresh
--user-data-dir. This driver only connects to the local test browser/site.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/ui_checks/live_prices"


class CDP:
    def __init__(self, url):
        parsed = urlparse(url)
        if parsed.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('Only the disposable local browser is allowed')
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=20)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f'GET {parsed.path} HTTP/1.1\r\nHost: {parsed.netloc}\r\n'
                           f'Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n'
                           'Sec-WebSocket-Version: 13\r\n\r\n').encode())
        data = b''
        while b'\r\n\r\n' not in data:
            data += self.sock.recv(4096)
        header, self.buffer = data.split(b'\r\n\r\n', 1)
        if b'101' not in header.split(b'\r\n', 1)[0]:
            raise RuntimeError(header.decode())
        self.sequence, self.events = 0, []

    def take(self, count):
        while len(self.buffer) < count:
            part = self.sock.recv(max(4096, count-len(self.buffer)))
            if not part:
                raise EOFError('Browser connection closed')
            self.buffer += part
        result, self.buffer = self.buffer[:count], self.buffer[count:]
        return result

    def send(self, body, opcode=1):
        mask = os.urandom(4)
        size = len(body)
        length = bytes([size | 128]) if size < 126 else (bytes([254])+struct.pack('!H', size)
                   if size <= 65535 else bytes([255])+struct.pack('!Q', size))
        self.sock.sendall(bytes([128 | opcode])+length+mask+bytes(b ^ mask[i % 4] for i,b in enumerate(body)))

    def receive(self):
        payload = b''
        while True:
            first, second = self.take(2)
            size = second & 127
            if size == 126:
                size = struct.unpack('!H', self.take(2))[0]
            elif size == 127:
                size = struct.unpack('!Q', self.take(8))[0]
            mask = self.take(4) if second & 128 else None
            part = self.take(size)
            if mask:
                part = bytes(b ^ mask[i % 4] for i,b in enumerate(part))
            opcode = first & 15
            if opcode == 9:
                self.send(part, 10)
                continue
            if opcode == 8:
                raise EOFError('Browser closed WebSocket')
            payload += part
            if first & 128:
                return json.loads(payload)

    def call(self, method, **params):
        self.sequence += 1
        request_id = self.sequence
        self.send(json.dumps({'id':request_id, 'method':method, 'params':params}).encode())
        while True:
            result = self.receive()
            if result.get('id') == request_id:
                if 'error' in result:
                    raise RuntimeError(result['error'])
                return result.get('result', {})
            self.events.append(result)

    def evaluate(self, expression):
        result = self.call('Runtime.evaluate', expression=expression, returnByValue=True, awaitPromise=True)
        if 'exceptionDetails' in result:
            raise RuntimeError(result['exceptionDetails'])
        return result.get('result', {}).get('value')

    def wait(self, expression, seconds=15):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if self.evaluate(expression):
                return
            time.sleep(.1)
        raise TimeoutError(expression)

    def screenshot(self, filename):
        result = self.call('Page.captureScreenshot', format='png', captureBeyondViewport=False)
        (OUTPUT / filename).write_bytes(base64.b64decode(result['data']))


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with urlopen(Request('http://127.0.0.1:9223/json/new?about:blank', method='PUT')) as response:
        target = json.load(response)
    browser = CDP(target['webSocketDebuggerUrl'])
    browser.call('Page.enable')
    browser.call('Runtime.enable')
    browser.call('Network.enable')
    browser.call('Network.setCacheDisabled', cacheDisabled=True)
    # Deterministic application checks do not require external market widgets.
    browser.call('Network.setBlockedURLs', urls=['*://s.tradingview.com/*', '*://s3.tradingview.com/*'])
    browser.call('Page.addScriptToEvaluateOnNewDocument', source="""
        window.__errors=[]; window.__fetches=[];
        window.addEventListener('error',e=>{if(e.filename?.includes('/static/js/live_prices.js'))window.__errors.push(e.message)});
        const original=window.fetch;
        window.fetch=(...args)=>{window.__fetches.push(String(args[0]));return original(...args)};
    """)
    browser.call('Emulation.setDeviceMetricsOverride', width=1440, height=1100, deviceScaleFactor=1, mobile=False)
    browser.call('Page.navigate', url='http://127.0.0.1:8000/live-prices?model=extra_all&side=top&interval=60&coin=NO_SUCH_COIN')
    browser.wait('!!window.liveMarket?.data && !window.liveMarket.loading && document.querySelectorAll(".coin-row").length===20')
    checks = {}
    def check(name, expression):
        value = browser.evaluate(expression)
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)
    check('initial_prediction_request_once', 'window.__fetches.filter(x=>x.includes("/api/predictions/data/")).length===1')
    check('simulator_not_loaded_on_market_visit', '!window.__fetches.some(x=>x.includes("paper-trading"))')
    check('dates_and_global_rank_visible', 'document.getElementById("market-data-period").textContent.includes(liveMarket.data.data_period) && document.querySelector(".coin-row .rank").textContent==="#1"')
    browser.evaluate('window.__loads=liveMarket.chartLoads; document.querySelector(".coin-row.active").click()')
    check('same_selection_keeps_chart', 'liveMarket.chartLoads===window.__loads')
    browser.evaluate('document.getElementById("coin-search").value="NO_MATCH_AT_ALL";document.getElementById("coin-search").dispatchEvent(new Event("input"))')
    check('search_filters_without_request_or_chart_reload', 'document.querySelectorAll(".coin-row").length===0 && liveMarket.chartLoads===window.__loads && window.__fetches.filter(x=>x.includes("/api/predictions/data/")).length===1')
    browser.evaluate('document.getElementById("coin-search").value="";document.getElementById("coin-search").dispatchEvent(new Event("input")); document.querySelector("[data-view=bottom]").click()')
    check('bottom_list_uses_actual_universe_rank', 'document.querySelector(".coin-row .rank").textContent==="#"+liveMarket.data.universe_count')
    browser.evaluate('window.__beforeFilter=liveMarket.chartLoads;document.getElementById("chart-available-filter").click()')
    check('available_filter_keeps_only_validated_pairs_without_reloading_chart', 'Array.from(document.querySelectorAll(".coin-row")).every(n=>!n.classList.contains("unavailable")) && liveMarket.chartLoads===window.__beforeFilter')
    browser.evaluate('document.getElementById("chart-available-filter").click()')
    browser.evaluate('document.querySelector("[data-interval=D]").click()')
    check('interval_persisted_in_chart_and_url', 'location.search.includes("interval=D") && document.querySelector("#main-chart iframe").src.includes("interval=D")')
    browser.evaluate('window.__selected=liveMarket.state.coin;window.__model=liveMarket.state.model')
    browser.call('Page.reload')
    browser.wait('!!window.liveMarket?.data && !window.liveMarket.loading')
    check('reload_restores_side_interval_and_selected_coin', 'liveMarket.state.side==="bottom" && liveMarket.state.interval==="D" && new URLSearchParams(location.search).get("coin")===liveMarket.state.coin')
    # Explicitly exercise the unmapped-asset path without removing its prediction.
    browser.evaluate("""
        window.__saved=liveMarket.data;
        liveMarket.data={...liveMarket.data,top_20:[{symbol:'UNMAPPED',rank:1,value:.0123,tradingview:null,chart_unavailable_reason:'未配置交易对'}]};
        liveMarket.state.side='top';liveMarket.state.coin='UNMAPPED';liveMarket.syncControls();liveMarket.applyData();
    """)
    check('unmapped_coin_keeps_prediction_and_clear_empty_chart', 'document.querySelectorAll(".coin-row").length===1 && document.getElementById("selected-prediction").textContent==="+1.23%" && !document.querySelector("#main-chart iframe") && !document.getElementById("chart-placeholder").hidden')
    browser.evaluate('liveMarket.data=window.__saved;liveMarket.state.side="bottom";liveMarket.applyData()')
    # Ignore AbortSignal on purpose: late response must still be rejected.
    browser.evaluate("""
        window.__fetch=window.fetch;window.__pending=[];liveMarket.cache.clear();
        window.fetch=(url,options)=>String(url).includes('/api/predictions/data/')?new Promise(resolve=>__pending.push(resolve)):__fetch(url,options);
        liveMarket.state.model='enet_all';window.__first=liveMarket.loadPredictions();
        liveMarket.state.model='extra_market';window.__second=liveMarket.loadPredictions();
        __pending[1](new Response(JSON.stringify({...__saved,model:'extra_market'})));
    """)
    browser.wait('liveMarket.data.model==="extra_market" && !liveMarket.loading')
    browser.evaluate("__pending[0](new Response(JSON.stringify({...__saved,model:'enet_all'})))")
    browser.evaluate('Promise.resolve(window.__first).then(()=>{window.fetch=window.__fetch;return true})')
    check('late_model_response_cannot_override_newer_selection', 'liveMarket.data.model==="extra_market" && liveMarket.state.model==="extra_market"')
    browser.evaluate("liveMarket.state.model='extra_all';liveMarket.state.side='top';liveMarket.cache.clear();liveMarket.syncControls();liveMarket.loadPredictions(true)")
    browser.wait('liveMarket.data.model==="extra_all" && !liveMarket.loading')
    # Exercise a failed GET and recovery; neither action may start model training.
    browser.evaluate("""
        window.__fetch=window.fetch;
        window.fetch=(url,options)=>String(url).includes('/api/predictions/data/')?Promise.resolve(new Response('unavailable',{status:503})):__fetch(url,options);
        window.__failedLoad=liveMarket.loadPredictions(true);
    """)
    browser.evaluate('Promise.resolve(window.__failedLoad).then(()=>true)')
    check('prediction_failure_offers_retry_instead_of_stuck_spinner', '!!document.querySelector("#coins-grid .list-message button") && !document.getElementById("refresh-market-list").disabled')
    browser.evaluate('window.fetch=window.__fetch;liveMarket.loadPredictions(true)')
    browser.wait('!liveMarket.loading && document.querySelectorAll(".coin-row").length===20')
    check('prediction_retry_recovers', '!document.querySelector("#coins-grid .list-message")')
    browser.evaluate('liveMarket.chartSlow(liveMarket.chartKey)')
    check('slow_chart_has_recovery_actions', '!document.getElementById("chart-loading").hidden && !document.getElementById("retry-chart").disabled && !document.getElementById("open-tradingview").hidden')
    check('page_interactions_never_request_model_regeneration', '!window.__fetches.some(x=>x.includes("/api/updates/run") || x.includes("/api/predictions/run"))')
    check('no_application_javascript_errors', 'window.__errors.length===0')
    # Collect actual network UI screenshots, not a generated mockup.
    browser.call('Network.setBlockedURLs', urls=[])
    coin = browser.evaluate('liveMarket.coins().find(c=>c.tradingview).symbol')
    browser.call('Page.navigate', url=f'http://127.0.0.1:8000/live-prices?model=extra_all&side=top&coin={coin}&interval=D')
    browser.wait('!!window.liveMarket?.data && !window.liveMarket.loading')
    time.sleep(6)
    browser.screenshot('desktop.png')
    browser.evaluate('document.getElementById("market-overview").scrollIntoView();liveMarket.loadOverview()')
    browser.wait('!!document.querySelector("#tradingview-market-widget iframe") || /暂未连接|加载较慢/.test(document.getElementById("overview-status").textContent)', seconds=20)
    time.sleep(3)
    check('market_overview_embeds_or_reports_external_failure', '!!document.querySelector("#tradingview-market-widget iframe") || /暂未连接|加载较慢/.test(document.getElementById("overview-status").textContent)')
    check('ticker_embeds_or_hides_unavailable_widget', '!!document.querySelector("#ticker-tape-container iframe") || document.getElementById("ticker-tape-container").hidden')
    browser.screenshot('overview.png')
    browser.call('Emulation.setDeviceMetricsOverride', width=390, height=844, deviceScaleFactor=1, mobile=True)
    browser.evaluate('window.scrollTo(0,0)')
    time.sleep(.4)
    check('mobile_has_no_horizontal_overflow', 'document.documentElement.scrollWidth<=window.innerWidth')
    check('mobile_navigation_does_not_fill_first_screen', 'document.querySelector(".navbar").getBoundingClientRect().height<130')
    check('mobile_chart_precedes_watchlist', 'document.querySelector(".market-chart-panel").getBoundingClientRect().top<document.querySelector(".market-watchlist").getBoundingClientRect().top')
    browser.screenshot('mobile.png')
    result = {'passed':all(checks.values()), 'checks':checks,
              'limitations':'Widget transport was re-enabled for screenshots; cross-origin iframe load cannot certify price freshness or every instrument mapping.'}
    (OUTPUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    browser.call('Page.close')


if __name__ == '__main__':
    run()
