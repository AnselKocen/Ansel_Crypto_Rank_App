/* One shared weekly publication; page loads only read status and cached files. */
window.WeeklyUpdatesUI = {
    state: null,
    timer: null,
    async refresh() {
        const bar = document.getElementById('weekly-update-bar');
        if (!bar) return;
        const component = bar.dataset.component;
        try {
            const response = await fetch(`/api/updates/status?component=${component}`, {cache: 'no-store'});
            if (!response.ok) throw new Error('状态暂不可用');
            const state = await response.json();
            const previous = this.state;
            this.state = state;
            this.render();
            if (!previous || previous.data_period !== state.data_period || previous.report_current !== state.report_current ||
                    (previous.running && !state.running)) {
                document.dispatchEvent(new CustomEvent('weekly:status', {detail: state}));
            }
        } catch (error) {
            document.getElementById('weekly-update-label').textContent = window.SiteLocale?.text('暂时无法读取更新状态；已保存内容仍可查看', 'Update status unavailable; saved content is still available') ?? 'Update status unavailable; saved content is still available';
            document.getElementById('weekly-update-button').disabled = true;
        }
        clearTimeout(this.timer);
        this.timer = setTimeout(() => this.refresh(), this.state?.running ? 3000 : 60000);
    },
    render() {
        const state = this.state;
        const bar = document.getElementById('weekly-update-bar');
        if (!state || !bar) return;
        const component = bar.dataset.component;
        const textFor = (zh, en) => window.SiteLocale?.text(zh, en) ?? en;
        const label = document.getElementById('weekly-update-label');
        const button = document.getElementById('weekly-update-button');
        const current = component === 'report' ? state.report_current : state.data_current;
        let text = state.data_period ? textFor(`数据截至 ${state.data_period} UTC`, `Data through ${state.data_period} UTC`) : textFor('尚未发布完整周数据', 'No complete weekly data published');
        if (state.running) text += textFor(` · 更新中：${state.job.stage}`, ' · Update in progress');
        else if (current) text += textFor(' · 已是最新完整周', ' · Latest complete week');
        else if (component === 'report' && !state.report_outbound_enabled) text += textFor(' · AI周报外部连接待管理员授权', ' · Weekly report connection requires administrator approval');
        else if (state.retry_after_seconds) text += textFor(` · 更新未完成，${Math.ceil(state.retry_after_seconds / 60)}分钟后可重试`, ` · Update incomplete; retry in ${Math.ceil(state.retry_after_seconds / 60)} minutes`);
        else if (component === 'report' && !state.report_configured) text += textFor(' · AI报告服务尚未配置', ' · AI report service not configured');
        else text += textFor(' · 有更新可生成', ' · New update available');
        label.textContent = text;
        button.disabled = !state.can_update;
        button.textContent = state.running ? textFor('所有访客共用更新中', 'Shared update running') : current ? textFor('本周已更新', 'Up to date') : textFor('更新到最新一期', 'Update now');
        const detail = document.getElementById('weekly-update-detail');
        const coverage = state.coverage;
        const schedule = textFor(state.schedule, "Updates start Thursdays at 09:00 Beijing time (one hour after Wednesday closes in UTC); missed updates resume on startup.");
        detail.textContent = coverage
            ? textFor(`周度历史：${coverage.history_start} 至 ${coverage.history_end}，${coverage.history_weeks}周；原始日线自${coverage.daily_start}起。${schedule} `, `Weekly history: ${coverage.history_start} to ${coverage.history_end}, ${coverage.history_weeks} weeks; daily data since ${coverage.daily_start}. ${schedule} `)
            : schedule;
        if (state.coverage_url) {
            const link = document.createElement('a');
            link.href = state.coverage_url;
            link.textContent = textFor('查看逐币覆盖与缺口', 'View per-token coverage and gaps');
            detail.appendChild(link);
        }
        const reportButton = document.getElementById('generate-btn');
        if (reportButton) {
            reportButton.disabled = state.running || state.report_current || !state.can_update || !state.report_configured;
            reportButton.textContent = state.running ? textFor('⏳ 共享报告生成中', '⏳ Generating shared report') : state.report_current ? textFor('✓ 本周报告已生成', '✓ Weekly report ready')
                : !state.report_outbound_enabled ? textFor('待管理员启用周报', 'Awaiting administrator setup') : textFor('🔄 生成最新一期', '🔄 Generate latest report');
        }
    },
    async request(component) {
        const bar = document.getElementById('weekly-update-bar');
        const button = document.getElementById('weekly-update-button');
        button.disabled = true;
        try {
            const response = await fetch('/api/updates/run', {method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({component: component || bar.dataset.component})});
            if (!response.ok) throw new Error(window.SiteLocale?.text('更新请求未成功', 'Update request failed') ?? 'Update request failed');
            await response.json();
        } catch (error) {
            document.getElementById('weekly-update-label').textContent = error.message;
        }
        await this.refresh();
    }
};

/* Each chart section enters once, after its selected publication has loaded. */
window.SentimentMedia = (() => {
    const sections = [...document.querySelectorAll('.sentiment-workspace .insight-panel')];
    const settled = new WeakSet();
    let selected = false;

    function showVisible() {
        for (const section of sections) {
            if (section.hidden || section.style.display === 'none') {
                // Cancel an unfinished entrance so showing this tab cannot replay it.
                section.classList.remove('sentiment-entering');
                continue;
            }
            if (section.dataset.entrancePlayed || !selected ||
                    [...section.querySelectorAll('img')].some(img => !settled.has(img))) continue;
            section.dataset.entrancePlayed = 'true';
            section.classList.remove('sentiment-awaiting-media');
            section.setAttribute('aria-busy', 'false');
            if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                section.classList.add('sentiment-entering');
                setTimeout(() => section.classList.remove('sentiment-entering'), 800);
            }
        }
    }

    function showImage(img, available) {
        img.style.display = available ? '' : 'none';
        img.nextElementSibling.style.display = available ? 'none' : 'block';
    }

    async function settleImage(img) {
        if (img.dataset.mediaAvailable === 'false') {
            showImage(img, false);
        } else {
            if (!img.complete) return;
            if (img.naturalWidth > 0) {
                const source = img.src;
                if (img.decode) await img.decode().catch(() => {});
                if (img.src !== source || !img.complete) return;
                showImage(img, true);
            } else {
                showImage(img, false);
            }
        }
        settled.add(img);
        showVisible();
    }

    for (const section of sections) {
        section.classList.add('sentiment-awaiting-media');
        section.setAttribute('aria-busy', 'true');
        for (const img of section.querySelectorAll('img')) {
            img.addEventListener('load', () => settleImage(img));
            img.addEventListener('error', () => settleImage(img));
        }
    }

    function selectImages(entries) {
        selected = false;
        for (const [id, asset] of entries || []) {
            const img = document.getElementById(id);
            if (!img) continue;
            img.dataset.mediaAvailable = asset?.exists && asset.url ? 'true' : 'false';
            if (img.dataset.mediaAvailable === 'true') {
                const source = new URL(asset.url, document.baseURI).href;
                // Polling the same publication leaves the loaded image untouched.
                if (img.src !== source) {
                    settled.delete(img);
                    img.src = source;
                }
            }
        }
        selected = true;
        for (const section of sections) {
            for (const img of section.querySelectorAll('img')) settleImage(img);
        }
        showVisible();
    }
    return {showVisible, selectImages};
})();

let weeklyImagesRequest = null;
function refreshWeeklyImages() {
    if (!document.getElementById('wordcloud-positive')) return Promise.resolve();
    if (weeklyImagesRequest) return weeklyImagesRequest;
    const readImage = async url => {
        const response = await fetch(url, {signal: AbortSignal.timeout(8000)});
        if (!response.ok) throw new Error('Image metadata unavailable');
        return response.json();
    };
    weeklyImagesRequest = Promise.all([
        readImage('/api/sentiment/wordcloud'), readImage('/api/sentiment/fear-greed')
    ]).then(([clouds, gauge]) => {
        window.SentimentMedia.selectImages([
            ['wordcloud-positive', clouds.positive], ['wordcloud-negative', clouds.negative],
            ['fear-greed-gauge', gauge]
        ]);
    }).catch(() => {
        // A failed status request still reveals the saved image or its error message.
        window.SentimentMedia.selectImages(null);
    }).finally(() => { weeklyImagesRequest = null; });
    return weeklyImagesRequest;
}

document.addEventListener('weekly:status', () => {
    refreshWeeklyImages().catch(() => {});
    if (typeof window.loadExistingReport === 'function') window.loadExistingReport();
    if (typeof window.loadModel === 'function') {
        const model = document.getElementById('model-select')?.value;
        if (model) window.loadModel(model);
    }
});
document.addEventListener('DOMContentLoaded', () => {
    window.WeeklyUpdatesUI.refresh();
    refreshWeeklyImages();
});

window.addEventListener('site:localechange', () => window.WeeklyUpdatesUI.render());
