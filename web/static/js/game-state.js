/* Versioned save shared by the new games and the original nine levels. */
const STORAGE_KEY = 'crypto_cat_game';
const GameState = {
    _state: null,
    _default() { return { version: 3, coins: 0, levels: {}, cats: {}, runs: {}, settings: {}, knowledge: {} }; },
    load() {
        if (this._state) return this._state;
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            const old = raw ? JSON.parse(raw) : {};
            if (!old || typeof old !== 'object' || Array.isArray(old)) throw Error('Invalid save');
            const s = { ...this._default(), ...old, version: 3 };
            for (const key of ['levels', 'cats', 'runs', 'settings', 'knowledge']) {
                if (!s[key] || typeof s[key] !== 'object' || Array.isArray(s[key])) s[key] = {};
            }
            s.coins = Number.isFinite(s.coins) ? Math.max(0, s.coins) : 0;
            for (let id = 1; id <= 10; id++) {
                const p = s.levels[id] || {}, c = s.cats[id] || {};
                const best = Math.max(0, Math.min(10, Number(p.bestScore) || 0));
                const perfect = !!(p.perfect || c.perfect || best === 10);
                const unlocked = !!(p.completed || c.unlocked || perfect || best >= 5);
                if (s.levels[id] || unlocked) s.levels[id] = { ...p, bestScore: perfect ? 10 : best, completed: unlocked, perfect };
                if (unlocked) s.cats[id] = { ...c, unlocked: true, perfect };
            }
            this._state = s;
        } catch {
            this._state = this._default();
            this.warning = '存档无法读取，本页仍可游玩。损坏的原存档未被自动覆盖。';
            this._readOnly = true;
        }
        return this._state;
    },
    save() {
        if (this._readOnly) return false;
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.load())); return true; }
        catch { this.warning = '浏览器暂时无法保存进度，请保持本页打开。'; this.notify(); return false; }
    },
    notify() {
        const el = document.getElementById('game-save-notice');
        if (el && this.warning) { el.textContent = this.warning; el.hidden = false; }
    },
    reset() { this._state = this._default(); this._readOnly = false; this.save(); },
    getCoins() { return this.load().coins; },
    addCoins(n) { this.load().coins += n; this.save(); },
    getLevelProgress(id) { return this.load().levels[id] || null; },
    setLevelResult(id, score, extra = {}) {
        const s = this.load(), prev = s.levels[id] || {};
        if (extra.runId && prev.lastRunId === extra.runId) return prev;
        const best = Math.max(prev.bestScore || 0, score);
        const perfect = !!(prev.perfect || best === 10 || s.cats[id]?.perfect);
        const completed = !!(prev.completed || score >= 5 || s.cats[id]?.unlocked);
        s.levels[id] = { ...prev, bestScore: best, completed, perfect,
            routeComplete: !!(prev.routeComplete || extra.routeComplete),
            attempts: (prev.attempts || 0) + 1, lastRunId: extra.runId || null };
        if (completed) s.cats[id] = { ...s.cats[id], unlocked: true, perfect };
        this.save();
        return s.levels[id];
    },
    // Keep all original levels accessible during the staged, one-game-at-a-time rebuild.
    isLevelUnlocked() { return true; },
    getTotalCats() { return Object.values(this.load().cats).filter(c => c?.unlocked).length; },
    isCatUnlocked(id) { return !!this.load().cats[id]?.unlocked; },
    isCatPerfect(id) { return !!this.load().cats[id]?.perfect; },
    awardRun(run, amount) {
        // The collected item/answer, run balance and total balance are written atomically.
        run.coins += amount;
        this.load().coins += amount;
        this.load().runs[run.levelId] = run;
        this.save();
    }
};
