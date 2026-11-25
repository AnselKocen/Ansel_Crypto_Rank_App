// ========== Paper Trading Functions ==========

async function loadPaperTrading() {
    const loadingEl = document.getElementById('paper-trading-loading');
    const errorEl = document.getElementById('paper-trading-error');
    const metricsEl = document.getElementById('pt-metrics');
    const posEl = document.getElementById('pt-positions');

    try {
        const resp = await fetch('/api/predictions/paper-trading?capital=10000');
        if (!resp.ok) throw new Error('Failed to load paper trading data');
        const data = await resp.json();

        loadingEl.style.display = 'none';

        // Render metrics
        if (data.strategies) {
            renderPTMetrics(data.strategies, data.initial_capital);
            metricsEl.style.display = 'grid';
        }

        // Render positions
        if (data.current_positions && data.current_positions.extra_all) {
            renderPTPositions(data.current_positions.extra_all);
            posEl.style.display = 'block';
        }

    } catch (err) {
        loadingEl.style.display = 'none';
        errorEl.textContent = `Error: ${err.message}`;
        errorEl.style.display = 'block';
    }
}

function renderPTMetrics(strategies, initialCapital) {
    const mapping = [
        { prefix: 'ls', key: 'extra_ls' },
        { prefix: 'ew', key: 'extra_EW' },
        { prefix: 'bl', key: 'baseline' },
    ];

    mapping.forEach(({ prefix, key }) => {
        const s = strategies[key];
        if (!s) return;

        const totalReturnEl = document.getElementById(`${prefix}-total-return`);
        const val = s.total_return_pct;
        totalReturnEl.textContent = `${val >= 0 ? '+' : ''}${val.toFixed(2)}%`;
        totalReturnEl.className = `pt-value big ${val >= 0 ? 'positive' : 'negative'}`;

        // Capital — use last point in cumulative series
        const lastCap = s.cumulative_series && s.cumulative_series.length > 0
            ? s.cumulative_series[s.cumulative_series.length - 1].capital
            : initialCapital;
        document.getElementById(`${prefix}-capital`).textContent = `$${lastCap.toLocaleString()}`;

        document.getElementById(`${prefix}-sharpe`).textContent = s.sharpe_ratio.toFixed(3);
        const ddEl = document.getElementById(`${prefix}-maxdd`);
        ddEl.textContent = `${s.max_drawdown_pct.toFixed(2)}%`;
        ddEl.className = 'pt-value negative';

        document.getElementById(`${prefix}-winrate`).textContent = `${s.win_rate_pct}%`;
        document.getElementById(`${prefix}-vol`).textContent = `${s.volatility_annual_pct}%`;
        document.getElementById(`${prefix}-weeks`).textContent = s.total_weeks;
    });
}

function renderPTPositions(positions) {
    const longList = document.getElementById('pt-long-list');
    const shortList = document.getElementById('pt-short-list');

    if (positions.long) {
        longList.innerHTML = positions.long.map((p, i) => `
            <div class="pt-pos-item">
                <span class="sym">#${i+1} ${p.symbol}</span>
                <span class="pred ${p.pred_return >= 0 ? 'positive' : 'negative'}">${p.pred_return >= 0 ? '+' : ''}${p.pred_return.toFixed(2)}%</span>
            </div>
        `).join('');
    }

    if (positions.short) {
        shortList.innerHTML = positions.short.map((p, i) => `
            <div class="pt-pos-item">
                <span class="sym">#${i+1} ${p.symbol}</span>
                <span class="pred ${p.pred_return >= 0 ? 'positive' : 'negative'}">${p.pred_return >= 0 ? '+' : ''}${p.pred_return.toFixed(2)}%</span>
            </div>
        `).join('');
    }
}


// Keep the old simulator available without running it for every market-page visit.
document.addEventListener('DOMContentLoaded', () => {
    const panel = document.getElementById('legacy-simulator');
    let loaded = false;
    panel?.addEventListener('toggle', () => {
        if (panel.open && !loaded) {
            loaded = true;
            loadPaperTrading();
        }
    });
});
