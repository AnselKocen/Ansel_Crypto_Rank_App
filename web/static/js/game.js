/**
 * Crypto Cat Adventure - DOM Controller
 * Manages views (map/platformer/result/gallery), quiz overlay, state.
 */

/* ============================================
   View Management
   ============================================ */
const VIEWS = ['map', 'platformer', 'postal', 'castle', 'adventure', 'result', 'gallery'];

function showView(name) {
    VIEWS.forEach(v => {
        const el = document.getElementById('view-' + v);
        if (el) el.style.display = v === name ? 'block' : 'none';
    });

    QuizOverlay.dispose();
    // If quiz is open when switching views, close it cleanly
    const quizRoot = document.getElementById('quiz-isolated-root');
    if (quizRoot) quizRoot.remove();

    // Toggle nav buttons
    const btnMap = document.getElementById('btn-map');
    const btnGallery = document.getElementById('btn-gallery');
    const btnReset = document.getElementById('btn-reset');
    if (btnMap) btnMap.style.display = name === 'map' ? 'none' : 'inline-block';
    if (btnGallery) btnGallery.style.display = (name === 'gallery') ? 'none' : 'inline-block';
    if (btnReset) btnReset.style.display = (name === 'platformer') ? 'inline-block' : 'none';

    // Show/hide touch controls (only during active platformer play)
    const touchControls = document.getElementById('touch-controls');
    if (touchControls) touchControls.style.display = name === 'platformer' ? 'flex' : 'none';

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ============================================
   Stats Bar
   ============================================ */
function updateStats() {
    const coinsEl = document.getElementById('stats-coins');
    const catsEl = document.getElementById('stats-cats');
    if (coinsEl) coinsEl.textContent = 'B ' + GameState.getCoins();
    if (catsEl) catsEl.textContent = GameState.getTotalCats() + ' / 10';
}

/* ============================================
   API Helpers
   ============================================ */
async function fetchJSON(url) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 12000);
    try {
        const resp = await fetch(url, { signal: controller.signal });
        if (!resp.ok) throw new Error('API error: ' + resp.status);
        return await resp.json();
    } finally { clearTimeout(timer); }
}

/* ============================================
   World Map Rendering
   ============================================ */
function renderWorldMap(levels, cats) {
    const container = document.getElementById('world-map');
    if (!container) return;
    const t = (zh, en) => window.SiteLocale?.text(zh, en) ?? en;
    const escapeText = value => String(value ?? '').replace(/[&<>"']/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[character]);
    const text = (zh, en) => escapeText(t(zh, en));
    const englishAdventures = {
        3: ['Bitcoin Mine Lift', 'Bounce across the mine platforms and reach the sunlight.'],
        4: ['Blockchain Express', 'Deliver your parcels to the data terminal.'],
        5: ['Ethereum Garden Walk', 'Follow the stone paths to the garden tea party.'],
        6: ['Wallet Safety Workshop', 'Use your little shield to recover the safety handbook.'],
        7: ['Exchange Dock Treasure', 'Dig through the sand to recover the warehouse handbook.'],
        8: ['Strategy Starlight Flight', 'Fly your cat-eared craft to the observation camp.'],
        9: ['Market Sky Observatory', 'Use the claw to retrieve knowledge capsules.'],
        10: ['ICO Final Archives', 'Solve five puzzle rooms and reach the photo hall.']
    };

    container.innerHTML = levels.map(lv => {
        const lvl = lv.level;
        const progress = GameState.getLevelProgress(lvl);
        const unlocked = GameState.isLevelUnlocked(lvl);
        const cat = cats.find(c => c.level === lvl);
        const catUnlocked = GameState.isCatUnlocked(lvl);
        const perfect = progress && progress.perfect;

        let cls = 'level-node';
        if (!unlocked) cls += ' locked';
        else if (perfect) cls += ' completed perfect';
        else if (progress && progress.completed) cls += ' completed';

        const starStr = Array(lv.difficulty).fill('*').join('');

        const scoreHtml = progress
            ? `<div class="level-score">${text('最佳', 'Best')}: <span class="score-value">${progress.bestScore}/10</span></div>`
            : (unlocked ? '<div class="level-score">--</div>' : '');

        const catPreview = catUnlocked && cat
            ? `<img class="level-cat-preview" src="/static/sprites/cats/${cat.sprite}" alt="${text(cat.name_zh, cat.name)}">`
            : '';
        const crownHtml = perfect ? '<span class="level-crown">C</span>' : '';
        const lockHtml = !unlocked ? `<div class="level-lock">${text('未解锁', 'LOCKED')}</div>` : '';

        if (lvl === 1) return `<button class="level-node postal-feature" data-start-level="1">
            <img src="${PostalArt.preview()}" alt="${text('像素海岸旁，猫咪驾驶薄荷绿邮车驶向灯塔', 'A cat drives a mint-green mail van along a pixel coast toward the lighthouse')}" class="postal-map-art" width="640" height="400">
            <div class="postal-feature-copy"><small>${text('第 01 站 · 环岛邮路', 'STOP 01 · ISLAND POST')}</small><h2>${text('货币起源 · 环岛邮路', 'Money Origins · Island Post')}</h2>
            <p>${text('开一辆小邮车，把十封知识信送往海岛。', 'Drive a little mail van and deliver ten knowledge letters around the island.')}</p>
            <span>${progress ? text(`知识最佳 ${progress.bestScore}/10 · ${progress.routeComplete ? '邮路已送达' : '邮路待探索'}`, `Best quiz score ${progress.bestScore}/10 · ${progress.routeComplete ? 'Mail route complete' : 'Route to explore'}`) : text('驶入 P 停车位收信 · 已完成进度跨趟保留', 'Park in a P bay to read letters · Your progress carries over')}</span>
            <span class="postal-feature-start">${GameState.load().runs[1]?.settled ? text('查看旅程纪念', 'View journey keepsake') : GameState.load().runs[1] ? text('继续上次旅程', 'Continue your journey') : text('出发去海岛', 'Set off for the island')} →</span>
            <small>${catUnlocked ? text('聪聪已收藏', 'Satoshi collected') + (perfect ? ' ♛' : '') : text('答对 5 题，认识聪聪', 'Answer 5 correctly to meet Satoshi')}</small></div></button>`;
        if (lvl === 2) return `<button class="level-node castle-feature" data-start-level="2">
            <img class="castle-map-art" width="640" height="400" src="${CastleArt.preview()}" alt="${text('暮色城堡中，围巾猫跳向暖光拱窗前的知识箱', 'A scarf-wearing cat jumps toward a knowledge chest beneath a glowing castle window')}">
            <div class="postal-feature-copy"><small>${text('第 02 站 · 城堡送信', 'STOP 02 · CASTLE MAIL')}</small><h2>${text('加密城堡送信记', 'Crypto Castle Mail')}</h2><p>${text('跑过窗廊，跳过缺口，把邀请函送进书房。', 'Run through the galleries, jump across gaps, and deliver an invitation to the study.')}</p>
            <span>${text('横版冒险 · 落下就近恢复 · 答对进度保留', 'A side-scrolling adventure · Safe checkpoints · Quiz progress saved')}</span><span class="postal-feature-start">${GameState.load().runs[2]?.settled ? text('查看城堡纪念', 'View castle keepsake') : GameState.load().runs[2] ? text('继续城堡旅程', 'Continue the castle journey') : text('带上信封，出发', 'Take your letter and set off')} →</span>
            <small>${catUnlocked ? text('V仔已收藏', 'Vitalik collected') : text('累计答对 5 题，认识 V仔', 'Answer 5 correctly to meet Vitalik')}</small></div></button>`;
        if (lvl >= 3) {
            const meta = AdventureCatalog[lvl];
            const [titleEn, goalEn] = englishAdventures[lvl];
            const stop = String(lvl).padStart(2, '0');
            return `<button class="adventure-feature" data-start-level="${lvl}"><img class="adventure-map-art" width="640" height="400" src="${AdventureArt.preview(lvl)}" alt="${text(`${meta.title}像素场景`, `${titleEn} pixel scene`)}"><div class="adventure-card-copy"><small>${text(`第 ${stop} 站`, `STOP ${stop}`)} · ${escapeText(meta.en)}</small><h3>${text(meta.title, titleEn)}</h3><p>${text(meta.goal, goalEn)}</p><p>${progress ? text(`历史最佳 ${progress.bestScore}/10`, `Best score ${progress.bestScore}/10`) : text('十份原知识信 · 答对进度保留', 'Ten knowledge letters · Quiz progress saved')}${perfect ? ' · ♛' : ''}</p><strong>${GameState.load().runs[lvl]?.settled ? text('查看旅行纪念', 'View journey keepsake') : GameState.load().runs[lvl] ? text('继续旅程', 'Continue your journey') : text('出发去探索', 'Start exploring')} →</strong></div></button>`;
        }
        return `
            <div class="${cls}" onclick="GameApp.startLevel(${lvl})" data-level="${lvl}">
                ${crownHtml}
                ${catPreview}
                ${lockHtml}
                <div class="level-number">${lockHtml ? '' : lvl}</div>
                <div class="level-world">${text(lv.world_name_zh, lv.world_name || `Level ${lvl}`)}</div><small class="legacy-label">${text('原版关卡 · 等待重做', 'Original level · Redesign pending')}</small>
                <div class="level-difficulty">${starStr}</div>
                ${scoreHtml}
            </div>`;
    }).join('');
}

/* ============================================
   Quiz Overlay — fully isolated from page DOM.
   Creates a brand-new element on document.body
   with all styles inlined via <style> tag scoped
   by ID. Nothing in the existing page is touched
   (no display changes, no class toggles).
   On close the element is removed entirely.
   ============================================ */
const QuizOverlay = {
    levelData: null,
    currentQuestion: null,
    answered: false,
    shuffledOptions: null,
    _visible: false,
    _el: null,

    dispose() {
        this._visible = false;
        this._el?.remove();
        this._el = null;
        this.currentQuestion = null;
    },

    async init(levelId) {
        this.levelData = await fetchJSON('/api/game/levels/' + levelId);
    },

    show(questionIndex) {
        if (this._visible) return;
        if (!this.levelData) return;

        const q = this.levelData.questions[questionIndex];
        if (!q) return;

        this.currentQuestion = q;
        this.answered = false;
        this._visible = true;

        // Shuffle options
        const options = q.options.map((text, i) => ({ text, isCorrect: i === q.answer }));
        shuffleArray(options);
        this.shuffledOptions = options;

        const optionsHtml = options.map((opt, i) => {
            const letter = String.fromCharCode(65 + i);
            return `<button class="qo-btn" data-idx="${i}" onclick="QuizOverlay.selectAnswer(${i})">${letter}. ${opt.text}</button>`;
        }).join('');

        // Build a completely independent element
        const el = document.createElement('div');
        el.id = 'quiz-isolated-root';
        el.innerHTML = `
<style>
#quiz-isolated-root {
    position:fixed; top:0; left:0; width:100%; height:100%; z-index:99999;
    background:rgba(10,22,40,0.92); display:flex; align-items:center; justify-content:center;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}
#quiz-isolated-root * { box-sizing:border-box; }
#quiz-isolated-root .qo-popup { background:#1a2332; border:2px solid rgba(78,205,196,0.3);
    border-radius:16px; padding:2rem; max-width:650px; width:90%; }
#quiz-isolated-root .qo-qcard { background:#151d2b; border:1px solid rgba(78,205,196,0.1);
    border-radius:12px; padding:1.5rem; margin-bottom:1rem; }
#quiz-isolated-root .qo-qtext { font-size:1.1rem; color:#e6f1ff; line-height:1.7; margin:0; }
#quiz-isolated-root .qo-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; }
@media(max-width:600px){ #quiz-isolated-root .qo-grid { grid-template-columns:1fr; } }
#quiz-isolated-root .qo-btn { display:block; width:100%; background:#1e293b;
    border:2px solid rgba(78,205,196,0.15); border-radius:8px; padding:0.9rem 1rem;
    color:#e6f1ff; font-size:0.9rem; cursor:pointer; text-align:left;
    transition:all 0.2s; line-height:1.5; font-family:inherit; }
#quiz-isolated-root .qo-btn:hover:not(.disabled) { border-color:#4ecdc4; background:#1e3a4a; }
#quiz-isolated-root .qo-btn.correct { border-color:#10b981; background:#132b23; color:#34d399; }
#quiz-isolated-root .qo-btn.incorrect { border-color:#ef4444; background:#2b1313; color:#f87171; }
#quiz-isolated-root .qo-btn.disabled { cursor:default; opacity:0.7; }
#quiz-isolated-root .qo-fb { padding:1rem 1.25rem; border-radius:8px; margin-top:1rem;
    animation:qoSlide 0.3s ease; }
#quiz-isolated-root .qo-fb.correct { background:#132b23; border:1px solid rgba(16,185,129,0.3); color:#34d399; }
#quiz-isolated-root .qo-fb.incorrect { background:#2b1313; border:1px solid rgba(239,68,68,0.3); color:#f87171; }
#quiz-isolated-root .qo-fb .qo-explain { color:#cbd5e1; margin-top:0.4rem; font-size:0.9rem; line-height:1.5; }
#quiz-isolated-root .qo-next { display:block; margin:1.25rem auto 0; padding:0.6rem 2rem;
    border-radius:8px; border:none; background:linear-gradient(135deg,#4ecdc4,#44a08d);
    color:#0a0e1a; font-weight:600; font-size:1rem; cursor:pointer; font-family:inherit; }
#quiz-isolated-root .qo-next:hover { box-shadow:0 4px 15px rgba(78,205,196,0.3); }
@keyframes qoSlide { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
</style>
<div class="qo-popup">
    <div class="qo-qcard"><p class="qo-qtext">${q.question}</p></div>
    <div class="qo-grid">${optionsHtml}</div>
    <div id="qo-fb-area"></div>
</div>`;

        document.body.appendChild(el);
        this._el = el;
    },

    selectAnswer(idx) {
        if (this.answered || !this._el) return;
        this.answered = true;

        const correctIdx = this.shuffledOptions.findIndex(o => o.isCorrect);
        const isCorrect = idx === correctIdx;

        const btns = this._el.querySelectorAll('.qo-btn');
        btns.forEach((btn, i) => {
            btn.classList.add('disabled');
            if (i === correctIdx) btn.classList.add('correct');
            if (i === idx && !isCorrect) btn.classList.add('incorrect');
            if (i !== idx && i !== correctIdx) btn.style.opacity = '0.4';
        });

        const area = this._el.querySelector('#qo-fb-area');
        if (area) {
            const cls = isCorrect ? 'correct' : 'incorrect';
            area.innerHTML = `
                <div class="qo-fb ${cls}">
                    <strong>${isCorrect ? 'Correct!' : 'Wrong!'}</strong>
                    <div class="qo-explain">${this.currentQuestion.explanation}</div>
                </div>
                <button class="qo-next" onclick="QuizOverlay.close(${isCorrect})">Continue</button>`;
        }
    },

    close(isCorrect) {
        this._visible = false;

        // Destroy the element completely — zero impact on existing page
        if (this._el) {
            this._el.remove();
            this._el = null;
        }

        // Dispatch answer back to Phaser
        window.dispatchEvent(new CustomEvent('quiz-answered', {
            detail: { correct: isCorrect }
        }));
    }
};

/* ============================================
   Results Rendering
   ============================================ */
function renderResults(detail) {
    showView('result');
    const area = document.getElementById('result-area');
    if (!area) return;

    const { levelId, score, totalQuestions, coins, hearts } = detail;
    const cats = window._gameData ? window._gameData.cats : [];
    const cat = cats.find(c => c.level === levelId);

    // Save state
    const totalCoins = coins;
    GameState.addCoins(totalCoins);
    const result = GameState.setLevelResult(levelId, score);
    const catUnlocked = score >= 5;
    const isPerfect = score === 10;

    let title, message;
    if (hearts <= 0) {
        title = 'Game Over!';
        message = 'You ran out of hearts. Try again!';
    } else if (isPerfect) {
        title = 'PERFECT!';
        message = 'Amazing! You mastered this level completely!';
    } else if (score >= 5) {
        title = 'Level Complete!';
        message = 'Great job! You unlocked a new cat!';
    } else if (score >= 3) {
        title = 'Keep Going!';
        message = 'Need 5/10 to pass. Try again!';
    } else {
        title = 'Study More~';
        message = 'Learn more about crypto and try again!';
    }

    let catHtml = '';
    if (catUnlocked && cat) {
        catHtml = `
            <div class="cat-reveal">
                <div class="cat-reveal-title">${isPerfect ? 'PERFECT Unlock!' : 'New Cat Unlocked!'}</div>
                <div class="cat-reveal-img">
                    <img src="/static/sprites/cats/${cat.sprite}" alt="${cat.name}">
                </div>
                <div class="cat-reveal-name">${cat.name} (${cat.name_zh})</div>
                <div class="cat-reveal-breed">${cat.breed_zh} - ${cat.personality_zh}</div>
                <div class="cat-reveal-quote">"${cat.quote}"</div>
            </div>`;
    }

    const nextLevelUnlocked = score >= 5 && levelId < 10 && GameState.isLevelUnlocked(levelId + 1);

    area.innerHTML = `
        <div class="result-card">
            <div class="result-title">${title}</div>
            <div class="score-display">${score} <span class="score-total">/ ${totalQuestions}</span></div>
            <div class="coins-earned">+ ${totalCoins} B</div>
            <div class="result-message">${message}</div>
            ${catHtml}
            <div class="result-buttons">
                <button class="btn-game" onclick="GameApp.showMap()">Back to Map</button>
                <button class="btn-game" onclick="GameApp.startLevel(${levelId})">Retry</button>
                ${nextLevelUnlocked ? `<button class="btn-game btn-primary-game" onclick="GameApp.startLevel(${levelId + 1})">Next Level</button>` : ''}
            </div>
        </div>`;

    updateStats();
}

/* ============================================
   Gallery Rendering
   ============================================ */
function renderGallery(cats) {
    const grid = document.getElementById('gallery-grid');
    const subtitle = document.getElementById('gallery-subtitle');
    if (!grid) return;
    const t = (zh, en) => window.SiteLocale?.text(zh, en) ?? en;

    const totalUnlocked = GameState.getTotalCats();
    if (subtitle) subtitle.textContent = t(
        `已收藏 ${totalUnlocked} / ${cats.length} · 完成关卡，点亮猫咪伙伴`,
        `${totalUnlocked} / ${cats.length} collected · Complete journeys to reveal your cat companions`
    );

    const escapeText = value => String(value ?? '').replace(/[&<>"']/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[character]);
    const text = (zh, en) => escapeText(t(zh, en));
    const lockIcon = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="10" width="14" height="11" rx="3"/><path d="M8 10V7a4 4 0 0 1 8 0v3M12 14v3"/></svg>';
    const englishCats = {
        1: ['Orange tabby', 'A strict instructor', "I'll teach you a lesson!"],
        2: ['Dragon Li', 'Thinking through a meltdown', 'My head feels chilly... must be the mood.'],
        3: ['Cream cat', 'An affectionate companion', 'Your screen needs a lick. Prprpr!'],
        4: ['Calico', 'A startled gamer', "Wait, isn't that my desk?!"],
        5: ['Orange tabby', 'Mysterious and graceful', 'Hia~ hia~ hia~'],
        6: ['Dragon Li', 'A master of side-eye', 'Damn!'],
        7: ['American shorthair', 'Effortlessly adorable', 'Meow~?'],
        8: ['Ragdoll', 'Your biggest hype cat', 'Boss, whoever made pdf2skills is amazing!'],
        9: ['Siamese', 'A little coal miner', 'Are my paws dark from all that mining?'],
        10: ['Cat family', 'All your favorite cat memes', 'Collect them all. Nothing happens, meow.']
    };

    grid.innerHTML = cats.map(cat => {
        const unlocked = GameState.isCatUnlocked(cat.level);
        const perfect = GameState.isCatPerfect(cat.level);
        const cls = 'cat-card ' + (unlocked ? 'unlocked' : 'locked');
        const [breedEn, personalityEn, quoteEn] = englishCats[cat.id] || ['Cat', 'Your crypto companion', 'Meow!'];
        const alternateName = t(cat.name, '');

        const stars = Array.from({ length: 5 }, (_, index) =>
            `<span class="${index < cat.rarity ? 'rarity-filled' : 'rarity-empty'}" aria-hidden="true">✦</span>`
        ).join('');
        const color = /^#[\da-f]{6}$/i.test(cat.css_color) ? cat.css_color : '#a9dfff';
        const sprite = /^cat_\d{2}\.png$/.test(cat.sprite) ? cat.sprite : 'cat_01.png';
        return `
            <article class="${cls}" style="--cat-color: ${color};" aria-labelledby="gallery-cat-${cat.id}">
                <div class="cat-img-display${unlocked ? '' : ' locked-img'}">
                    <img src="/static/sprites/cats/${sprite}" alt="${text(`${cat.name_zh} 猫 meme`, `${cat.name} cat meme`)}" width="256" height="256" loading="lazy" decoding="async">
                    ${unlocked ? `<span class="cat-gallery-status">${text('已收藏', 'Collected')}</span>` : `<span class="cat-preview-lock">${lockIcon}<span>${text(`第 ${cat.level} 关解锁`, `Unlock in level ${cat.level}`)}</span></span>`}
                    ${unlocked && perfect ? `<span class="crown-badge" aria-label="${text('满分通关', 'Perfect score')}" title="${text('满分通关', 'Perfect score')}">✧</span>` : ''}
                </div>
                <div class="cat-gallery-content">
                    <h3 class="cat-name" id="gallery-cat-${cat.id}">${text(cat.name_zh, cat.name)}${alternateName ? ` <span class="cat-name-en">${escapeText(alternateName)}</span>` : ''}</h3>
                    <p class="cat-breed">${text(cat.breed_zh, breedEn)}</p>
                    <p class="cat-personality">${text(cat.personality_zh, personalityEn)}</p>
                    <div class="cat-rarity" role="img" aria-label="${text(`稀有度 ${cat.rarity} / 5`, `Rarity ${cat.rarity} / 5`)}">${stars}</div>
                    <p class="cat-quote${unlocked ? '' : ' cat-locked-text'}">${unlocked ? `“${text(cat.quote, quoteEn)}”` : text(`完成第 ${cat.level} 关，收藏专属猫 meme。`, `Complete level ${cat.level} to collect your cat meme.`)}</p>
                </div>
            </article>`;
    }).join('');
}

/* ============================================
   Shuffle Helper
   ============================================ */
function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}

/* ============================================
   Main App Controller
   ============================================ */
const GameApp = {
    async init() {
        document.body.classList.add('cat-game-page');
        GameState.load();
        GameState.notify();

        const [levelsResp, catsResp] = await Promise.all([
            fetchJSON('/api/game/levels'),
            fetchJSON('/api/game/cats')
        ]);

        window._gameData = {
            levels: levelsResp.levels,
            cats: catsResp.cats
        };

        updateStats();
        this.showMap();

        if (this._listening) return;
        this._listening = true;
        document.getElementById('world-map').addEventListener('click', event => {
            const level = event.target.closest('[data-start-level]');
            if (level) this.startLevel(Number(level.dataset.startLevel));
        });
        // Listen for Phaser events
        window.addEventListener('quiz-trigger', (e) => {
            const { levelId, questionIndex } = e.detail;
            if (this.currentLevelId !== levelId || PostalGame.scene || CastleGame.scene || AdventureGames.scene || document.getElementById('view-platformer').style.display === 'none') return;
            QuizOverlay.show(questionIndex);
        });

        window.addEventListener('level-complete', (e) => {
            if (this.currentLevelId !== e.detail.levelId || PostalGame.scene || CastleGame.scene || AdventureGames.scene || document.getElementById('view-platformer').style.display === 'none') return;
            window.PlatformerGame?.destroy();
            renderResults(e.detail);
        });
        const directLevel=Number(new URLSearchParams(location.search).get('level'));
        if (Number.isInteger(directLevel) && directLevel>=1 && directLevel<=10) await this.startLevel(directLevel);
    },

    showMap() {
        this.cancelScene();
        window.PlatformerGame?.destroy();
        showView('map');
        renderWorldMap(window._gameData.levels, window._gameData.cats);
    },

    showGallery() {
        this.cancelScene();
        window.PlatformerGame?.destroy();
        showView('gallery');
        renderGallery(window._gameData.cats);
    },

    currentLevelId: null,
    _loadId: 0,
    cancelScene() {
        this._loadId++;
        PostalGame.destroy();
        CastleGame.destroy();
        AdventureGames.destroy();
        QuizOverlay.dispose();
        window.PlatformerGame?.destroy();
    },
    async startLevel(levelId) {
        if (!GameState.isLevelUnlocked(levelId)) return;
        this.cancelScene();
        const token = this._loadId;
        this.currentLevelId = levelId;
        if (levelId === 1) {
            showView('postal');
            await PostalGame.launch(token);
            return;
        }
        if (levelId === 2) {
            showView('castle');
            await CastleGame.launch(token);
            return;
        }
        if (levelId>=3 && levelId<=10) {
            showView('adventure');
            await AdventureGames.launch(levelId,token);
        }

    },
    resetGame() {
        if (this.currentLevelId) this.startLevel(this.currentLevelId);
    }

};

// Expose globally
window.QuizOverlay = QuizOverlay;
window.GameApp = GameApp;

// Refresh only the visible selection or collection view; leave active games intact.
window.addEventListener('site:localechange', () => {
    if (!window._gameData) return;
    if (document.getElementById('view-map')?.style.display !== 'none') {
        renderWorldMap(window._gameData.levels, window._gameData.cats);
    }
    if (document.getElementById('view-gallery')?.style.display !== 'none') {
        renderGallery(window._gameData.cats);
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Clean up any leftover overlay elements from previous code versions
    ['quiz-overlay-dynamic', 'quiz-page'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.remove();
    });

    GameApp.init().catch(err => {
        console.error('Game init failed:', err);
        document.getElementById('world-map').innerHTML = '<p>旅行地图加载失败。</p><button class="btn-game" onclick="GameApp.init()">重新加载</button>';
    });
});
