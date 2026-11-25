/* Interior pages share the language chosen on the home page. */
(() => {
    'use strict';
    const key = 'ansel_locale';
    const normalize = value => value === 'en' ? 'en' : ['zh', 'zh-CN'].includes(value) ? 'zh' : null;
    const stored = () => { try { return normalize(localStorage.getItem(key)); } catch { return null; } };
    let language = normalize(new URLSearchParams(location.search).get('lang')) || stored() || 'zh';
    const selector = '[data-zh][data-en],[data-aria-zh][data-aria-en],[data-placeholder-zh][data-placeholder-en],[data-title-zh][data-title-en]';
    const locale = {
        get language() { return language; },
        text(zh, en) { return language === 'zh' ? zh : en; },
        translate(root = document) {
            const elements = [...(root.querySelectorAll?.(selector) || [])];
            if (root.matches?.(selector)) elements.unshift(root);
            for (const element of elements) {
                if (element.hasAttribute('data-en') && element.hasAttribute('data-zh')) {
                    const value = element.dataset[language];
                    if (element.textContent !== value) element.textContent = value;
                }
                for (const [prefix, attribute] of [['aria', 'aria-label'], ['placeholder', 'placeholder'], ['title', 'title']]) {
                    const value = element.getAttribute(`data-${prefix}-${language}`);
                    if (value !== null && element.getAttribute(attribute) !== value) element.setAttribute(attribute, value);
                }
            }
        },
        set(value, persist = true) {
            const next = normalize(value);
            if (!next) return;
            language = next;
            document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
            this.translate();
            document.querySelectorAll('[data-site-language]').forEach(button => {
                button.setAttribute('aria-pressed', String(button.dataset.siteLanguage === language));
            });
            if (persist) {
                try { localStorage.setItem(key, language); } catch { /* Storage is optional. */ }
                if (new URLSearchParams(location.search).has('lang')) {
                    const url = new URL(location.href);
                    url.searchParams.set('lang', language);
                    history.replaceState(null, '', url);
                }
            }
            window.dispatchEvent(new CustomEvent('site:localechange', {detail: {language}}));
        }
    };
    window.SiteLocale = locale;
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.addEventListener('DOMContentLoaded', () => {
        locale.set(language, false);
        document.querySelectorAll('[data-site-language]').forEach(button => {
            button.addEventListener('click', () => locale.set(button.dataset.siteLanguage));
        });
        new MutationObserver(records => {
            for (const record of records) for (const node of record.addedNodes) {
                if (node.nodeType === 1) locale.translate(node);
            }
        }).observe(document.body, {childList: true, subtree: true});
    });
    window.addEventListener('storage', event => {
        if (event.key === key || event.key === null) locale.set(stored() || 'zh', false);
    });
})();
