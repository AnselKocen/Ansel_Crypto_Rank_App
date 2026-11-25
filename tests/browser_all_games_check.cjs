/*
 * All ten games, one fresh disposable Chrome context, one earned collection.
 * Run: node tests/browser_all_games_check.cjs
 * Optional: GAME_TEST_URL, PLAYWRIGHT_MODULE and CHROME_BIN.
 *
 * No localStorage seeds, score/answer/position assignments, or unlock shortcuts.
 * Travel is accelerated through real scene updates and normal steering/actions;
 * every answer and continue action is clicked in the actual DOM.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || '/Users/anselkocen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core');

const BASE = process.env.GAME_TEST_URL || 'http://127.0.0.1:8000';
const OUTPUT = path.resolve(__dirname, '../artifacts/ui_checks/all_games');
const DRIVER = path.join(__dirname, 'adventure_test_driver.js');
const report = {
    passed: false,
    baseUrl: BASE,
    browser: 'Google Chrome, headless, disposable context',
    contextCount: 1,
    checks: {},
    levels: [],
    errors: [],
    failedRequests: [],
    limitations: [
        'Long routes are accelerated through real scene update logic using steering, jump input and game actions; this is not a real-time manual playthrough.',
        'The existing adventure input driver uses the player-facing aim/help action for the brick and claw games and reads the question bank to choose DOM answers.',
        'Desktop Chrome at 1360 × 900 only; mobile gestures, physical-device performance and human puzzle difficulty are covered by separate checks.'
    ]
};

(async () => {
    fs.mkdirSync(OUTPUT, {recursive: true});
    const browser = await chromium.launch({
        executablePath: process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        headless: true
    });
    let context;
    let page;
    const started = Date.now();
    try {
        context = await browser.newContext({viewport: {width: 1360, height: 900}});
        page = await context.newPage();
        page.setDefaultTimeout(15000);
        page.on('pageerror', error => report.errors.push(error.message));
        page.on('requestfailed', request => report.failedRequests.push({url: request.url(),error: request.failure()?.errorText}));
        const check = (name, value) => {
            assert.ok(value, name);
            report.checks[name] = true;
        };
        const sceneState = () => page.evaluate(() => {
            const scene = PostalGame.scene || CastleGame.scene || AdventureGames.scene;
            return scene && {
                level: scene.levelId, phase: scene.phase, score: scene.score(),
                count: scene.count(), questionId: scene.run.questionId,
                routeComplete: scene.run.routeComplete, settled: scene.run.settled,
                coins: scene.run.coins, missed: scene.run.missed,
                progress: GameState.getLevelProgress(scene.levelId),
                unlocked: GameState.isCatUnlocked(scene.levelId),
                perfect: GameState.isCatPerfect(scene.levelId)
            };
        });
        const snapshot = () => page.evaluate(() => {
            const state = GameState.load();
            return {
                coins: state.coins, levels: state.levels, cats: state.cats,
                knowledge: state.knowledge,
                runs: Object.fromEntries(Object.entries(state.runs).map(([id, run]) => [id, {
                    id: run.id, settled: run.settled, routeComplete: run.routeComplete,
                    coins: run.coins, answers: run.answers
                }])),
                totalCats: GameState.getTotalCats()
            };
        });

        // The browser creates an isolated temporary profile. Never attach to a daily browser.
        await page.goto(BASE + '/game');
        await page.locator('[data-start-level="10"]').waitFor();
        check('single_context_starts_without_an_existing_save', await page.evaluate(() =>
            localStorage.getItem('crypto_cat_game') === null && GameState.getCoins() === 0 &&
            GameState.getTotalCats() === 0 && Object.keys(GameState.load().knowledge).length === 0));
        check('all_ten_games_are_available_on_the_map', await page.locator('[data-start-level]').count() === 10);
        await page.addScriptTag({path: DRIVER});

        // Only steering/input state and the scene clock are supplied by this driver.
        // Distance, collisions, parking, questions and settlement come from update().
        const travelPostal = id => page.evaluate(id => {
            const scene = PostalGame.scene;
            const station = PostalArt.stations.find(station => station.id === id);
            for (let tick = 0; tick < 15000 && ['driving', 'docking'].includes(scene.phase); tick++) {
                if (scene.phase === 'driving') {
                    scene.keys.clear();
                    const target = station.side * 50;
                    if (Math.abs(target - scene.run.offset) > 2) scene.keys.add(target > scene.run.offset ? 'ArrowRight' : 'ArrowLeft');
                }
                scene.cooldown = Math.max(0, scene.cooldown - .05);
                scene.time += .05;
                scene.update(.05);
                if (scene.phase === 'driving' && scene.run.distance > station.s + 72) break;
            }
            scene.clearInput();
            return {phase: scene.phase, questionId: scene.run.questionId, distance: scene.run.distance};
        }, id);
        const finishPostal = () => page.evaluate(() => {
            const scene = PostalGame.scene;
            for (let tick = 0; tick < 15000 && scene.phase === 'driving'; tick++) {
                scene.keys.clear();
                if (Math.abs(scene.run.offset) > 2) scene.keys.add(scene.run.offset > 0 ? 'ArrowLeft' : 'ArrowRight');
                scene.cooldown = Math.max(0, scene.cooldown - .05);
                scene.time += .05;
                scene.update(.05);
            }
            scene.clearInput();
            return {phase: scene.phase};
        });
        const walkCastle = () => page.evaluate(() => {
            const scene = CastleGame.scene;
            for (let tick = 0; tick < 30000 && ['playing', 'recovering'].includes(scene.phase); tick++) {
                scene.keys.clear();
                scene.keys.add('ArrowRight');
                const gap = CastleArt.gaps.find(gap => gap.x + gap.w > scene.run.x);
                if (scene.phase === 'playing' && gap && scene.run.grounded && gap.x - scene.run.x < 28 && gap.x - scene.run.x > -10) {
                    // The real key handler creates the jump buffer; no physics fields are assigned.
                    window.dispatchEvent(new KeyboardEvent('keydown', {code: 'Space', bubbles: true}));
                }
                scene.cooldown = Math.max(0, scene.cooldown - .02);
                scene.time += .02;
                scene.update(.02);
            }
            scene.clearInput();
            return {phase: scene.phase, questionId: scene.run.questionId, x: scene.run.x, falls: scene.run.falls};
        });
        const driveNext = async (level, questionNumber) => {
            if (level === 1) return questionNumber ? travelPostal(questionNumber) : finishPostal();
            if (level === 2) return walkCastle();
            return page.evaluate(() => {
                const result = driveAdventure();
                return {phase: result.phase, questionId: result.question, world: result.world};
            });
        };

        for (let level = 1; level <= 10; level++) {
            const previous = await snapshot();
            const levelStarted = Date.now();
            const entry = {level, questionIds: [], correctAnswers: 0};
            report.levels.push(entry);
            await page.locator(`[data-start-level="${level}"]`).click();
            await page.locator('[data-action="start"]').waitFor();
            const initial = await sceneState();
            check(`level_${level}_starts_with_no_answers_or_unlocked_cat`, initial.level === level && initial.score === 0 && initial.count === 0 && !initial.unlocked);
            await page.locator('[data-action="start"]').click();

            for (let number = 1; number <= 10; number++) {
                const movement = await driveNext(level, number);
                assert.equal(movement.phase, 'quiz', `Level ${level}, expected question ${number}: ${JSON.stringify(movement)}`);
                await page.locator('.post-option').first().waitFor();
                const question = await page.evaluate(() => {
                    const scene = PostalGame.scene || CastleGame.scene || AdventureGames.scene;
                    return scene.data.questions[scene.questionIndex];
                });
                assert.ok(!entry.questionIds.includes(question.id), `Duplicate question ${level}:${question.id}`);
                assert.equal(question.id, number, `Wrong question order in level ${level}`);
                assert.equal(await page.locator('.post-question').innerText(), question.question);
                assert.equal(await page.locator('[data-option]').count(), 4);
                assert.deepEqual(await page.locator('[data-option] > span:last-child').allTextContents(), question.options);
                await page.locator(`[data-option="${question.answer}"]`).click();
                assert.equal(await page.locator('.post-answer p').first().innerText(), question.explanation);
                const answered = await sceneState();
                assert.equal(answered.score, number, `Answer did not add one mastered question in level ${level}`);
                assert.equal(answered.count, number);
                assert.equal(answered.settled, false, 'Answering must not skip route completion');
                entry.questionIds.push(question.id);
                entry.correctAnswers++;
                await page.locator('.post-continue').click();
            }

            const finish = await driveNext(level);
            assert.equal(finish.phase, 'result', `Level ${level} did not reach its real destination: ${JSON.stringify(finish)}`);
            await page.locator('[data-map]').waitFor();
            const result = await sceneState();
            check(`level_${level}_ten_real_answers_finish_route_unlock_cat_and_perfect`,
                result.count === 10 && result.score === 10 && result.routeComplete && result.settled &&
                result.unlocked && result.perfect && result.progress.bestScore === 10 && result.progress.attempts === 1);
            const current = await snapshot();
            for (let prior = 1; prior < level; prior++) {
                assert.deepEqual(current.levels[prior], previous.levels[prior], `Level ${level} changed level ${prior}'s result`);
                assert.deepEqual(current.knowledge[prior], previous.knowledge[prior], `Level ${level} changed level ${prior}'s knowledge`);
                assert.deepEqual(current.cats[prior], previous.cats[prior], `Level ${level} changed level ${prior}'s cat`);
            }
            check(`level_${level}_preserves_all_previous_levels`, current.totalCats === level);
            entry.result = result;
            entry.elapsedMs = Date.now() - levelStarted;
            console.log(`Completed ${level}/10: ${entry.questionIds.length} real DOM answers; collection ${current.totalCats}/10`);
            await page.locator('[data-map]').click();
            await page.locator(`[data-start-level="${Math.min(10, level + 1)}"]`).waitFor();
        }

        check('all_one_hundred_questions_answered_through_dom', report.levels.length === 10 && report.levels.every(level =>
            level.correctAnswers === 10 && level.questionIds.length === 10 && new Set(level.questionIds).size === 10));
        const beforeReload = await snapshot();
        check('all_ten_cats_collected_and_all_ten_levels_perfect', beforeReload.totalCats === 10 &&
            Object.values(beforeReload.levels).filter(level => level.perfect && level.bestScore === 10).length === 10);
        await page.locator('#btn-gallery').click();
        const inspectGallery = async () => {
            await page.locator('#gallery-grid .cat-card').last().scrollIntoViewIfNeeded();
            await page.waitForFunction(() => {
                const images = [...document.querySelectorAll('#gallery-grid .cat-card img')];
                return images.length === 10 && images.every(image => image.complete && image.naturalWidth > 0);
            });
            return page.evaluate(() => ({
                collected: document.querySelectorAll('#gallery-grid .cat-card.unlocked').length,
                locked: document.querySelectorAll('#gallery-grid .cat-card.locked').length,
                perfect: document.querySelectorAll('#gallery-grid .crown-badge').length,
                subtitle: document.getElementById('gallery-subtitle').textContent,
                images: [...document.querySelectorAll('#gallery-grid .cat-card img')].map(image => ({
                    src: image.getAttribute('src'), complete: image.complete,
                    width: image.naturalWidth, height: image.naturalHeight, alt: image.alt,
                    objectFit: getComputedStyle(image).objectFit
                }))
            }));
        };
        report.gallery = await inspectGallery();
        check('gallery_has_ten_unlocked_cards_ten_crowns_and_no_locked_cards',
            report.gallery.collected === 10 && report.gallery.perfect === 10 && report.gallery.locked === 0);
        check('all_ten_unique_gallery_images_loaded', report.gallery.images.every(image => image.complete && image.width > 0 && image.height > 0) &&
            new Set(report.gallery.images.map(image => image.src)).size === 10);
        const familyPhoto = report.gallery.images.find(image => image.src.endsWith('/cat_10.png'));
        check('wide_family_photo_is_contained_without_cropping', familyPhoto &&
            familyPhoto.width > familyPhoto.height && familyPhoto.objectFit === 'contain');

        await page.reload();
        await page.locator('[data-start-level="10"]').waitFor();
        const afterReload = await snapshot();
        assert.deepEqual(afterReload, beforeReload, 'Refreshing changed earned progress or coins');
        check('refresh_keeps_every_answer_result_cat_and_coin_without_repeated_rewards', true);
        await page.locator('#btn-gallery').click();
        report.galleryAfterReload = await inspectGallery();
        check('refresh_keeps_full_gallery_and_ten_loaded_images', report.galleryAfterReload.collected === 10 &&
            report.galleryAfterReload.perfect === 10 && report.galleryAfterReload.images.every(image => image.complete && image.width > 0));
        const familyAfterReload = report.galleryAfterReload.images.find(image => image.src.endsWith('/cat_10.png'));
        check('refresh_keeps_wide_family_photo_without_cropping', familyAfterReload &&
            familyAfterReload.width > familyAfterReload.height && familyAfterReload.objectFit === 'contain');
        await page.evaluate(() => window.scrollTo({top: 0, behavior: 'instant'}));
        await page.waitForTimeout(100);
        await page.screenshot({path: path.join(OUTPUT, 'gallery.png'), fullPage: true});
        check('no_application_javascript_errors', report.errors.length === 0);
        check('no_failed_network_requests', report.failedRequests.length === 0);
        report.finalState = afterReload;
        report.passed = true;
        fs.rmSync(path.join(OUTPUT, 'failure.png'), {force: true});
    } catch (error) {
        report.failure = {message: error.message, stack: error.stack};
        if (page && !page.isClosed()) {
            try { await page.screenshot({path: path.join(OUTPUT, 'failure.png'), fullPage: true}); } catch {}
        }
        throw error;
    } finally {
        report.elapsedMs = Date.now() - started;
        fs.writeFileSync(path.join(OUTPUT, 'verification.json'), JSON.stringify(report, null, 2));
        console.log(JSON.stringify({passed: report.passed, levels: report.levels.length,
            answered: report.levels.reduce((sum, level) => sum + level.correctAnswers, 0),
            checks: Object.keys(report.checks).length, errors: report.errors, output: OUTPUT}, null, 2));
        await context?.close();
        await browser.close();
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
