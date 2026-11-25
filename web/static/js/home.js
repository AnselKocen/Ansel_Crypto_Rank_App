/* Home page: first-party localization and the live cat collection. */
(() => {
    'use strict';
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const localeKey = 'ansel_locale';
    const normalizeLocale = value => value === 'en' ? 'en' : value === 'zh' || value === 'zh-CN' ? 'zh' : null;
    const storedLocale = () => {
        try { return normalizeLocale(localStorage.getItem(localeKey)); }
        catch { return null; }
    };
    let language = normalizeLocale(new URLSearchParams(location.search).get('lang')) || storedLocale() || 'zh';
    const locale = {
        get current() { return language; },
        text(zh, en) { return language === 'zh' ? zh : en; },
        translate(root = document) {
            root.querySelectorAll('[data-zh][data-en]').forEach(element => {
                const value = element.dataset[language];
                if (element.tagName === 'META') element.setAttribute('content', value);
                else element.textContent = value;
            });
            root.querySelectorAll('[data-aria-zh][data-aria-en]').forEach(element => {
                element.setAttribute('aria-label', language === 'zh' ? element.dataset.ariaZh : element.dataset.ariaEn);
            });
        },
        set(value, persist = true) {
            const next = normalizeLocale(value);
            if (!next) return;
            language = next;
            document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
            this.translate();
            document.querySelectorAll('[data-language]').forEach(button => {
                const selected = button.dataset.language === language;
                button.setAttribute('aria-pressed', String(selected));
                button.classList.toggle('is-active', selected);
            });
            if (persist) {
                try { localStorage.setItem(localeKey, language); }
                catch { /* The page remains usable when storage is unavailable. */ }
                if (new URLSearchParams(location.search).has('lang')) {
                    const url = new URL(location.href);
                    url.searchParams.set('lang', language);
                    history.replaceState(null, '', url);
                }
            }
            window.dispatchEvent(new CustomEvent('home:localechange', { detail: { language } }));
        }
    };
    window.HomePageLocale = locale;
    const lockIcon = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><rect x="5" y="10" width="14" height="11" rx="3"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/><path d="M12 14v3"/></svg>';
    const breeds = ['Orange tabby', 'Dragon Li', 'Cream cat', 'Calico', 'Orange tabby', 'Dragon Li', 'American shorthair', 'Ragdoll', 'Siamese', 'Cat family'];

    class CatCollection {
        constructor() {
            this.track = document.getElementById('cat-track');
            if (!this.track) return;
            this.previous = document.getElementById('cats-prev');
            this.next = document.getElementById('cats-next');
            this.pause = document.getElementById('cat-carousel-pause');
            this.error = document.getElementById('cat-error');
            this.counter = document.getElementById('cat-count');
            this.cats = [];
            this.paused = reducedMotion.matches;
            this.hovered = false;
            this.focused = false;
            this.interactingUntil = 0;
            this.direction = 1;
            this.frame = 0;
            this.lastTime = 0;
            this.scrollRemainder = 0;
            this.resumeTimer = 0;
            // Continuous motion must not compete with CSS scroll snapping.
            this.track.style.scrollSnapType = 'none';
            this.track.style.scrollBehavior = 'auto';
            this.track.setAttribute('aria-busy', 'true');
            this.track.addEventListener('pointerenter', () => { this.hovered = true; this.startAnimation(); });
            this.track.addEventListener('pointerleave', () => { this.hovered = false; this.startAnimation(); });
            this.track.addEventListener('focusin', () => { this.focused = true; this.startAnimation(); });
            this.track.addEventListener('focusout', event => {
                this.focused = this.track.contains(event.relatedTarget);
                this.startAnimation();
            });
            ['pointerdown', 'wheel', 'keydown', 'touchstart'].forEach(eventName => {
                this.track.addEventListener(eventName, () => this.hold(), { passive: true });
            });
            this.previous?.addEventListener('click', () => this.scroll(-1));
            this.next?.addEventListener('click', () => this.scroll(1));
            this.pause?.addEventListener('click', () => {
                this.paused = !this.paused;
                this.interactingUntil = 0;
                this.updateControls();
            });
            document.getElementById('cats-retry')?.addEventListener('click', () => this.load());
            this.track.addEventListener('scroll', () => this.updateBoundaries(), { passive: true });
            window.addEventListener('resize', () => { this.updateBoundaries(); this.startAnimation(); }, { passive: true });
            window.addEventListener('home:localechange', () => this.refresh());
            window.addEventListener('pageshow', () => { this.refresh(); this.startAnimation(); });
            window.addEventListener('storage', event => {
                if (event.key === 'crypto_cat_game' || event.key === null) this.refresh();
            });
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    cancelAnimationFrame(this.frame);
                    this.frame = 0;
                    this.lastTime = 0;
                } else {
                    this.refresh();
                    this.startAnimation();
                }
            });
            reducedMotion.addEventListener('change', event => {
                if (event.matches) this.paused = true;
                this.updateControls();
            });
            this.updateControls();
            this.load();
            this.startAnimation();
        }
        async load() {
            if (this.loading) return;
            this.loading = true;
            this.track.setAttribute('aria-busy', 'true');
            if (this.error) this.error.hidden = true;
            this.setAvailable(false);
            this.startAnimation();
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 10000);
            try {
                const response = await fetch('/api/game/cats', { signal: controller.signal });
                if (!response.ok) throw new Error('Cat collection unavailable');
                const data = await response.json();
                const cats = Array.isArray(data.cats) ? data.cats.filter(cat =>
                    Number.isInteger(cat.id) && cat.id >= 1 && cat.id <= 10 &&
                    Number.isInteger(cat.level) && cat.level >= 1 && cat.level <= 10 &&
                    /^cat_\d{2}\.png$/.test(cat.sprite)
                ) : [];
                if (!cats.length) throw new Error('Cat collection empty');
                this.cats = cats;
                this.render();
            } catch {
                this.cats = [];
                this.track.replaceChildren();
                if (this.error) this.error.hidden = false;
                if (this.counter) this.counter.textContent = locale.text('图鉴暂不可用', 'Collection unavailable');
            } finally {
                clearTimeout(timeout);
                this.loading = false;
                this.track.setAttribute('aria-busy', 'false');
                this.setAvailable(this.cats.length > 0);
                this.startAnimation();
            }
        }
        readProgress() {
            try {
                const saved = JSON.parse(localStorage.getItem('crypto_cat_game') || '{}');
                return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
            } catch { return {}; }
        }
        isUnlocked(cat, progress) {
            const savedCat = progress.cats?.[cat.id];
            const level = progress.levels?.[cat.level];
            return !!(savedCat?.unlocked || savedCat?.perfect || level?.completed || level?.perfect || Number(level?.bestScore) >= 5);
        }
        render() {
            const fragment = document.createDocumentFragment();
            this.cats.forEach(cat => {
                const card = document.createElement('a');
                card.className = 'cat-card';
                card.href = `/game?level=${cat.level}`;
                card.target = '_blank';
                card.rel = 'noopener noreferrer';
                card.dataset.catId = cat.id;
                card.style.setProperty('--cat-color', /^#[\da-f]{6}$/i.test(cat.css_color) ? cat.css_color : '#8c9aa8');
                const art = document.createElement('div');
                art.className = 'cat-art';
                const image = document.createElement('img');
                image.className = 'cat-image';
                image.src = `/static/sprites/cats/${cat.sprite}`;
                image.alt = '';
                image.loading = 'lazy';
                image.decoding = 'async';
                image.width = 160;
                image.height = 160;
                const lock = document.createElement('span');
                lock.className = 'cat-lock';
                lock.innerHTML = `${lockIcon}<span></span>`;
                art.append(image, lock);
                const name = document.createElement('h3');
                name.className = 'cat-name';
                const meta = document.createElement('p');
                meta.className = 'cat-meta';
                card.append(art, name, meta);
                fragment.append(card);
            });
            this.track.replaceChildren(fragment);
            this.refresh();
        }
        refresh() {
            if (!this.cats.length) {
                if (this.counter && this.error && !this.error.hidden) this.counter.textContent = locale.text('图鉴暂不可用', 'Collection unavailable');
                this.updateControls();
                return;
            }
            const progress = this.readProgress();
            let count = 0;
            this.track.querySelectorAll('.cat-card').forEach((card, index) => {
                const cat = this.cats[index];
                const unlocked = this.isUnlocked(cat, progress);
                if (unlocked) count++;
                const name = locale.text(cat.name_zh, cat.name);
                card.classList.toggle('is-unlocked', unlocked);
                card.classList.toggle('is-locked', !unlocked);
                card.querySelector('.cat-name').textContent = name;
                card.querySelector('.cat-meta').textContent = locale.text(`${cat.breed_zh} · 第 ${cat.level} 关`, `${breeds[cat.id - 1]} · Level ${cat.level}`);
                const lock = card.querySelector('.cat-lock');
                lock.hidden = unlocked;
                lock.querySelector('span').textContent = locale.text(`第 ${cat.level} 关解锁`, `Unlock in level ${cat.level}`);
                card.setAttribute('aria-label', locale.text(
                    `${name}，${unlocked ? '已解锁' : '尚未解锁'}，游玩第 ${cat.level} 关`,
                    `${name}, ${unlocked ? 'unlocked' : 'locked'}, play level ${cat.level}`
                ));
            });
            if (this.counter) this.counter.textContent = locale.text(`已解锁 ${count} / ${this.cats.length}`, `${count} / ${this.cats.length} unlocked`);
            this.updateControls();
        }
        setAvailable(available) {
            if (this.pause) this.pause.disabled = !available;
            if (this.previous) this.previous.disabled = !available;
            if (this.next) this.next.disabled = !available;
            if (available) this.updateBoundaries();
        }
        updateControls() {
            if (this.pause) {
                const label = this.paused ? locale.text('继续滚动', 'Play carousel') : locale.text('暂停滚动', 'Pause carousel');
                const text = this.pause.querySelector('[data-carousel-label]');
                if (text) text.textContent = label;
                else this.pause.textContent = label;
                this.pause.setAttribute('aria-label', label);
                this.pause.setAttribute('aria-pressed', String(this.paused));
                this.pause.classList.toggle('is-paused', this.paused);
            }
            this.updateBoundaries();
            this.startAnimation();
        }
        updateBoundaries() {
            if (!this.cats.length || this.loading) return;
            const max = this.track.scrollWidth - this.track.clientWidth;
            if (this.previous) this.previous.disabled = this.track.scrollLeft < 2;
            if (this.next) this.next.disabled = max < 2 || this.track.scrollLeft >= max - 2;
        }
        hold() {
            this.interactingUntil = performance.now() + 8000;
            clearTimeout(this.resumeTimer);
            this.startAnimation();
            this.resumeTimer = setTimeout(() => this.startAnimation(), 8100);
        }
        scroll(direction) {
            const first = this.track.querySelector('.cat-card');
            if (!first) return;
            this.hold();
            const gap = parseFloat(getComputedStyle(this.track).columnGap) || 16;
            this.track.scrollBy({ left: direction * (first.getBoundingClientRect().width + gap), behavior: reducedMotion.matches ? 'instant' : 'smooth' });
        }
        canAnimate() {
            return !document.hidden && !this.paused && !this.hovered && !this.focused && !this.loading &&
                this.cats.length > 0 && performance.now() >= this.interactingUntil &&
                this.track.scrollWidth - this.track.clientWidth > 1;
        }
        startAnimation() {
            if (!this.canAnimate()) {
                cancelAnimationFrame(this.frame);
                this.frame = 0;
                this.lastTime = 0;
                return;
            }
            if (this.frame) return;
            const step = time => {
                this.frame = 0;
                if (!this.canAnimate()) return;
                const elapsed = this.lastTime ? Math.min(time - this.lastTime, 50) : 0;
                this.lastTime = time;
                const max = this.track.scrollWidth - this.track.clientWidth;
                if (this.track.scrollLeft >= max - 1) this.direction = -1;
                if (this.track.scrollLeft <= 0) this.direction = 1;
                // Accumulate fractions before writing whole pixels for browsers
                // that round scrollLeft; movement remains consistent at any fps.
                this.scrollRemainder += elapsed * 0.024;
                const pixels = Math.floor(this.scrollRemainder);
                if (pixels > 0) {
                    this.track.scrollLeft += this.direction * pixels;
                    this.scrollRemainder -= pixels;
                }
                this.frame = requestAnimationFrame(step);
            };
            this.frame = requestAnimationFrame(step);
        }
    }

    function init() {
        locale.set(language);
        document.querySelectorAll('[data-language]').forEach(button => {
            button.addEventListener('click', () => locale.set(button.dataset.language));
        });
        window.addEventListener('storage', event => {
            if (event.key === localeKey || event.key === null) locale.set(storedLocale() || 'zh', false);
        });
        new CatCollection();
        if (window.matchMedia('(hover: hover) and (pointer: fine)').matches && !reducedMotion.matches) {
            document.querySelectorAll('.feature-card, .game-feature').forEach(card => {
                card.addEventListener('pointermove', event => {
                    const bounds = card.getBoundingClientRect();
                    card.style.setProperty('--mouse-x', `${event.clientX - bounds.left}px`);
                    card.style.setProperty('--mouse-y', `${event.clientY - bounds.top}px`);
                });
            });
        }
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
    else init();
})();
