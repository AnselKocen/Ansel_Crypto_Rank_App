/*
 * Real-time input smoke checks, not an accelerated or complete playthrough.
 * Uses a fresh, disposable Chrome profile. Never attaches to the user's browser.
 * No update() calls, position/answer/score assignments, storage seeds or unlocks.
 * Run: node tests/browser_games_realtime_check.cjs
 * Optional: GAME_TEST_URL, PLAYWRIGHT_MODULE, CHROME_BIN.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || '/Users/anselkocen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core');

const BASE = process.env.GAME_TEST_URL || 'http://127.0.0.1:8000';
const OUTPUT = path.resolve(__dirname, '../artifacts/ui_checks/realtime');
const SAMPLE_MS = 4500;
const report = {
    passed: false,
    baseUrl: BASE,
    browser: 'Google Chrome, headless, fresh disposable profile and context',
    viewport: {width: 1360, height: 900},
    minimumActiveSampleMs: 4000,
    levels: [],
    checks: {},
    errors: [],
    limitations: [
        'Each game receives a short real-time keyboard/pointer input sample, not a complete route or a complete quiz playthrough.',
        'Frame timing measures this desktop headless Chrome run only; it does not certify performance on a physical phone.',
        'Frame timings are reported without a machine-dependent FPS pass threshold. Inputs, pause/resume and runtime errors are asserted.'
    ]
};

function summarizeIntervals(intervals) {
    if (!intervals.length) return {samples: 0, meanMs: null, p95Ms: null, maxMs: null, over50ms: 0};
    const sorted = [...intervals].sort((a, b) => a - b);
    const round = value => Math.round(value * 100) / 100;
    return {
        samples: intervals.length,
        meanMs: round(intervals.reduce((sum, value) => sum + value, 0) / intervals.length),
        p95Ms: round(sorted[Math.ceil(sorted.length * .95) - 1]),
        maxMs: round(sorted[sorted.length - 1]),
        over50ms: intervals.filter(value => value > 50).length
    };
}

(async () => {
    fs.mkdirSync(OUTPUT, {recursive: true});
    const browser = await chromium.launch({
        executablePath: process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        headless: true
    });
    let context;
    let page;
    let currentLevel = 0;
    const started = Date.now();
    try {
        context = await browser.newContext({viewport: report.viewport});
        page = await context.newPage();
        page.setDefaultTimeout(15000);
        page.on('pageerror', error => report.errors.push({level: currentLevel, message: error.message}));
        const check = (name, value) => {
            report.checks[name] = Boolean(value);
            assert.ok(value, name);
        };
        const sceneSnapshot = () => page.evaluate(() => {
            const scene = PostalGame.scene || CastleGame.scene || AdventureGames.scene;
            if (!scene) return null;
            const physical = scene.run.world || Object.fromEntries(
                ['distance', 'offset', 'x', 'y', 'vx', 'vy', 'grounded', 'camera', 'docking', 'falls']
                    .filter(key => key in scene.run).map(key => [key, scene.run[key]])
            );
            return {
                level: scene.levelId,
                phase: scene.phase,
                time: scene.time,
                physical: JSON.parse(JSON.stringify(physical)),
                answered: scene.count(),
                keys: scene.keys.size,
                pointers: scene.pointers.size
            };
        });
        const hold = async (key, milliseconds) => {
            await page.keyboard.down(key);
            try { await page.waitForTimeout(milliseconds); }
            finally { await page.keyboard.up(key); }
        };
        const gridStep = async key => {
            const before = await sceneSnapshot();
            await page.keyboard.down(key);
            try {
                await page.waitForFunction(({x, y}) => {
                    const world = AdventureGames.scene?.run.world;
                    return world && (world.x !== x || world.y !== y);
                }, before.physical);
            } finally { await page.keyboard.up(key); }
        };
        const pressButton = action => page.locator(`[data-drive="${action}"]`).click();

        for (let level = 1; level <= 10; level++) {
            currentLevel = level;
            const entry = {level, actions: [], checks: {}, frameTiming: null};
            report.levels.push(entry);
            await page.goto(`${BASE}/game?level=${level}`);
            await page.locator('[data-action="start"]').waitFor();
            await page.waitForFunction(level => {
                const scene = PostalGame.scene || CastleGame.scene || AdventureGames.scene;
                return scene?.levelId === level && scene.phase === 'intro';
            }, level);

            // This observer owns only its own measurement data. The game's rAF,
            // clock, input maps, physics and saved state remain untouched.
            await page.evaluate(() => {
                const sample = {running: true, intervals: [], activeMs: 0, activeFrames: 0, inactivePhases: [], last: null, lastActive: false};
                window.__realtimeAuditSample = sample;
                const observe = now => {
                    if (!sample.running) return;
                    const scene = PostalGame.scene || CastleGame.scene || AdventureGames.scene;
                    const active = ['playing', 'driving', 'docking', 'recovering'].includes(scene?.phase);
                    if (active) {
                        sample.activeFrames++;
                        if (sample.lastActive && sample.last !== null) {
                            const interval = now - sample.last;
                            sample.intervals.push(interval);
                            sample.activeMs += interval;
                        }
                    } else if (scene?.phase !== 'intro' && !sample.inactivePhases.includes(scene?.phase)) {
                        sample.inactivePhases.push(scene?.phase || 'missing');
                    }
                    sample.last = now;
                    sample.lastActive = active;
                    requestAnimationFrame(observe);
                };
                requestAnimationFrame(observe);
            });
            await page.locator('[data-action="start"]').click();
            const sampleStarted = Date.now();
            const beforeInput = await sceneSnapshot();
            let gardenReturnKey;
            let afterInput;

            if (level === 1) {
                await hold('ArrowRight', 300);
                afterInput = await sceneSnapshot();
                check('level_1_keyboard_steers_car', afterInput.physical.offset > beforeInput.physical.offset + 5);
                await hold('ArrowLeft', 300);
                entry.actions.push('Hold right, then left to return toward the road center.');
            } else if (level === 2) {
                await hold('ArrowRight', 450);
                afterInput = await sceneSnapshot();
                check('level_2_keyboard_moves_cat', afterInput.physical.x > beforeInput.physical.x + 5);
                await hold('ArrowLeft', 250);
                await page.keyboard.press('Space');
                entry.actions.push('Walk right and left, then perform a real keyboard jump near the starting bench.');
            } else if (level === 3) {
                await page.keyboard.down('ArrowLeft');
                await page.waitForTimeout(300);
                afterInput = await sceneSnapshot();
                check('level_3_keyboard_adjusts_landing', afterInput.physical.x < beforeInput.physical.x - 5);
                entry.actions.push('Hold left while automatic bouncing and safety-net recovery run normally.');
            } else if (level === 4) {
                await page.keyboard.press('ArrowRight');
                afterInput = await sceneSnapshot();
                check('level_4_keyboard_changes_lane', afterInput.physical.lane === 1 && beforeInput.physical.lane === 0);
                await page.keyboard.press('Space');
                await page.waitForTimeout(150);
                entry.actions.push('Change to the right lane and jump using keyboard input.');
            } else if (level === 5) {
                const step = await page.evaluate(() => {
                    const scene = AdventureGames.scene, world = scene.run.world;
                    const directions = [['ArrowRight', 1, 0, 'ArrowLeft'], ['ArrowUp', 0, -1, 'ArrowDown']];
                    return directions.find(([, dx, dy]) => scene.game.walk(world, world.x + dx, world.y + dy));
                });
                assert.ok(step, 'Garden has a walkable neighbor at its starting tile');
                gardenReturnKey = step[3];
                await gridStep(step[0]);
                afterInput = await sceneSnapshot();
                check('level_5_keyboard_walks_on_path', afterInput.physical.x !== beforeInput.physical.x || afterInput.physical.y !== beforeInput.physical.y);
                entry.actions.push('Take one real keyboard step onto a walkable neighboring tile.');
            } else if (level === 6) {
                await hold('ArrowRight', 300);
                afterInput = await sceneSnapshot();
                check('level_6_keyboard_moves_paddle', afterInput.physical.paddle > beforeInput.physical.paddle + 5);
                await hold('ArrowLeft', 300);
                entry.actions.push('Move the paddle right and left; click the launch button near the end of the sample.');
            } else if (level === 7) {
                await page.keyboard.press('ArrowRight');
                await pressButton('dig');
                afterInput = await sceneSnapshot();
                check('level_7_direction_and_dig_move_cat', afterInput.physical.x === 1 && afterInput.physical.y === 5 && afterInput.physical.depths['1,5'] === 0);
                entry.actions.push('Select an adjacent sand tile with the keyboard and click the dig button.');
            } else if (level === 8) {
                await hold('ArrowRight', 300);
                afterInput = await sceneSnapshot();
                check('level_8_keyboard_flies_ship', afterInput.physical.x > beforeInput.physical.x + 5);
                await hold('ArrowDown', 200);
                entry.actions.push('Move right and down in the flight corridor using the keyboard.');
            } else if (level === 9) {
                await hold('ArrowRight', 250);
                afterInput = await sceneSnapshot();
                check('level_9_keyboard_moves_claw', afterInput.physical.x > beforeInput.physical.x + 5);
                await hold('ArrowLeft', 250);
                await pressButton('grab');
                await page.waitForTimeout(150);
                const grabbed = await sceneSnapshot();
                check('level_9_pointer_button_releases_claw', grabbed.physical.mode !== 'swing' && grabbed.physical.y > 115);
                entry.actions.push('Adjust the claw left and right and click the grab button without using aim help.');
            } else {
                await gridStep('ArrowRight');
                afterInput = await sceneSnapshot();
                check('level_10_keyboard_moves_in_puzzle', afterInput.physical.x === 2 && afterInput.physical.y === 5);
                entry.actions.push('Take one real keyboard step in the archive room.');
            }

            if (level === 6) {
                await page.waitForTimeout(Math.max(0, SAMPLE_MS - 450 - (Date.now() - sampleStarted)));
                await pressButton('launch');
                await page.waitForTimeout(100);
                check('level_6_pointer_button_launches_ball', !(await sceneSnapshot()).physical.ball.attached);
            }
            await page.waitForTimeout(Math.max(0, SAMPLE_MS - (Date.now() - sampleStarted)));
            if (level === 3) await page.keyboard.up('ArrowLeft');
            const sample = await page.evaluate(() => {
                const sample = window.__realtimeAuditSample;
                sample.running = false;
                return {intervals: sample.intervals, activeMs: sample.activeMs, activeFrames: sample.activeFrames, inactivePhases: sample.inactivePhases};
            });
            entry.frameTiming = {...summarizeIntervals(sample.intervals), activeMs: Math.round(sample.activeMs), activeFrames: sample.activeFrames};
            entry.inactivePhasesDuringSample = sample.inactivePhases;
            entry.beforeInput = beforeInput;
            entry.afterInput = afterInput;
            entry.afterSample = await sceneSnapshot();
            check(`level_${level}_at_least_four_seconds_of_active_raf`, sample.activeMs >= 4000 && sample.intervals.length > 0);
            check(`level_${level}_sample_keeps_gameplay_active`, sample.inactivePhases.length === 0 && ['playing', 'driving', 'recovering', 'docking'].includes(entry.afterSample.phase));
            check(`level_${level}_sample_changes_world`, JSON.stringify(entry.afterSample.physical) !== JSON.stringify(beforeInput.physical));
            check(`level_${level}_no_quiz_answers_or_unlock_shortcuts`, entry.afterSample.answered === 0);
            const screenshot = `level-${level}.png`;
            await page.screenshot({path: path.join(OUTPUT, screenshot)});
            entry.screenshot = screenshot;

            await page.locator('[data-action="pause"]').click();
            const paused = await sceneSnapshot();
            await page.waitForTimeout(250);
            const stillPaused = await sceneSnapshot();
            check(`level_${level}_pause_freezes_world_and_clock`, paused.phase === 'paused' && stillPaused.phase === 'paused' && paused.time === stillPaused.time && JSON.stringify(paused.physical) === JSON.stringify(stillPaused.physical));
            check(`level_${level}_pause_clears_held_input`, stillPaused.keys === 0 && stillPaused.pointers === 0);
            await page.locator('[data-resume]').click();
            const resumed = await sceneSnapshot();
            if (level === 5) await gridStep(gardenReturnKey);
            else if (level === 7) { await page.keyboard.press('ArrowRight'); await pressButton('dig'); }
            else if (level === 10) await gridStep('ArrowLeft');
            else await hold(level === 2 ? 'ArrowRight' : level === 8 ? 'ArrowDown' : 'ArrowLeft', 200);
            await page.waitForTimeout(120);
            const afterResume = await sceneSnapshot();
            check(`level_${level}_resume_restores_clock_and_input`, ['playing', 'driving', 'recovering', 'docking'].includes(afterResume.phase) && afterResume.time > paused.time && JSON.stringify(afterResume.physical) !== JSON.stringify(resumed.physical));
            entry.afterResume = afterResume;
            entry.checks = Object.fromEntries(Object.entries(report.checks).filter(([name]) => name.startsWith(`level_${level}_`)));
            console.log(`Real-time level ${level}/10: ${entry.frameTiming.activeMs} ms active, mean ${entry.frameTiming.meanMs} ms, p95 ${entry.frameTiming.p95Ms} ms, max ${entry.frameTiming.maxMs} ms, >50 ms ${entry.frameTiming.over50ms}`);
        }
        check('all_ten_games_sampled_in_real_time', report.levels.length === 10 && report.levels.every(level => level.frameTiming?.activeMs >= 4000));
        check('no_application_javascript_errors', report.errors.length === 0);
        report.passed = true;
    } catch (error) {
        report.failure = {level: currentLevel, message: error.message, stack: error.stack};
        if (page && !page.isClosed()) {
            try { await page.screenshot({path: path.join(OUTPUT, 'failure.png')}); } catch {}
        }
        throw error;
    } finally {
        report.elapsedMs = Date.now() - started;
        fs.writeFileSync(path.join(OUTPUT, 'verification.json'), JSON.stringify(report, null, 2));
        console.log(JSON.stringify({passed: report.passed, levels: report.levels.length, checks: Object.keys(report.checks).length, errors: report.errors, output: OUTPUT}, null, 2));
        await context?.close();
        await browser.close();
    }
})().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
