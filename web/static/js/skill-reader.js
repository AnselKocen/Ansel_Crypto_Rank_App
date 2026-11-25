/* Skill selection and an expanded view share one article node and its state. */
(() => {
    'use strict';
    const entries = new Map();
    let sequence = 0;
    let controller = null;
    let selectedName = null;
    let article = null;
    let inlineScroll = null;
    const $ = id => document.getElementById(id);
    const textFor = (zh, en) => window.SiteLocale?.text(zh, en) ?? zh;
    const currentLang = () => window.SiteLocale?.language === 'en' ? 'en' : 'zh';
    const scrollFraction = node => node.scrollTop / Math.max(1, node.scrollHeight - node.clientHeight);

    function localized(tag, className, zh, en) {
        const element = document.createElement(tag);
        element.className = className;
        element.dataset.zh = zh;
        element.dataset.en = en;
        element.textContent = textFor(zh, en);
        return element;
    }

    function restoreInline() {
        if (inlineScroll === null) return;
        if (article && article.parentElement === $('skill-reader-content')) $('skill-detail').replaceChildren(article);
        $('skill-detail').scrollTop = inlineScroll;
        inlineScroll = null;
        document.body.classList.remove('skill-reader-open');
        $('skill-expand').focus({preventScroll: true});
    }

    function closeExpanded() {
        const dialog = $('skill-reader-dialog');
        if (dialog.open) dialog.close();
        restoreInline();
    }

    function expand() {
        const dialog = $('skill-reader-dialog');
        if (!article || dialog.open || $('skill-expand').disabled) return;
        inlineScroll = $('skill-detail').scrollTop;
        $('skill-reader-content').replaceChildren(article);
        document.body.classList.add('skill-reader-open');
        dialog.showModal();
        $('skill-reader-content').scrollTop = 0;
        $('skill-reader-close').focus({preventScroll: true});
    }

    async function load(name, {preserveView = false} = {}) {
        if (!preserveView) closeExpanded();
        controller?.abort();
        const request = ++sequence;
        const language = currentLang();
        const detail = $('skill-detail');
        const expanded = $('skill-reader-content');
        const positions = {inline: scrollFraction(detail), expanded: scrollFraction(expanded)};
        const visibleHost = () => $('skill-reader-dialog').open ? expanded : detail;
        selectedName = name;
        article = null;
        document.querySelectorAll('.skill-item').forEach(item => item.classList.toggle('active', item.dataset.name === name));
        detail.setAttribute('aria-busy', 'true');
        expanded.setAttribute('aria-busy', 'true');
        if (!preserveView) detail.scrollTop = 0;
        $('skill-expand').disabled = true;
        visibleHost().replaceChildren(localized('p', 'skill-reader-status', '正在载入知识内容…', 'Loading the article…'));
        const requestController = new AbortController();
        controller = requestController;
        const timeout = setTimeout(() => requestController.abort(), 15000);
        try {
            const response = await fetch(`/api/skills/${encodeURIComponent(name)}?lang=${language}`, {signal: requestController.signal});
            if (!response.ok) throw new Error('Skill unavailable');
            const data = await response.json();
            if (request !== sequence) return;
            if (typeof data.content !== 'string') throw new Error('Invalid skill content');
            const model = window.SkillDocument.prepare(data.content, entries.get(name) || {name});
            article = window.SkillDocument.render(model);
            const host = visibleHost();
            host.replaceChildren(article);
            host.scrollTop = preserveView ? positions[host === expanded ? 'expanded' : 'inline'] * Math.max(0, host.scrollHeight - host.clientHeight) : 0;
            $('skill-expand').disabled = false;
        } catch {
            if (request !== sequence) return;
            const message = localized('p', 'skill-reader-status error', '暂时无法载入该知识，请重试。', 'This article could not be loaded. Please retry.');
            const retry = localized('button', 'skill-reader-button', '重新加载', 'Retry');
            retry.type = 'button';
            retry.addEventListener('click', () => load(selectedName, {preserveView: true}));
            visibleHost().replaceChildren(message, retry);
        } finally {
            clearTimeout(timeout);
            if (request === sequence) {
                detail.setAttribute('aria-busy', 'false');
                expanded.setAttribute('aria-busy', 'false');
            }
        }
    }

    window.SkillReader = {
        register(categories) {
            for (const [category, skills] of Object.entries(categories || {})) {
                for (const skill of skills) entries.set(skill.name, {...skill, category});
            }
        },
        load,
        getSelectedId() {
            return article && selectedName ? `skill/${selectedName}` : null;
        }
    };

    $('skill-expand').addEventListener('click', expand);
    $('skill-reader-close').addEventListener('click', closeExpanded);
    $('skill-reader-dialog').addEventListener('close', () => {
        if (!$('skill-reader-dialog').open) restoreInline();
    });
    $('skill-reader-dialog').addEventListener('click', event => {
        const dialog = $('skill-reader-dialog');
        if (event.target !== dialog) return;
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeExpanded();
    });
    window.addEventListener('site:localechange', () => {
        if (selectedName) load(selectedName, {preserveView: true});
    });
})();
