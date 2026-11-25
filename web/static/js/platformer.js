/**
 * Crypto Cat Adventure - Phaser 3 Platformer Engine
 * Single-screen levels with physics, coins, question blocks.
 * Visual style: Blue sky, green grass, hero with red cape.
 */

/* global Phaser */

const GAME_W = 800;
const GAME_H = 600;
const TILE = 32;           // Map coordinate tile size (matches level JSONs)
const DISPLAY_SCALE = 0.75; // Sprites are 64px → display at 48px (1.5x bigger than 32)
const PLAYER_SCALE = 0.85;  // Player slightly bigger than tiles
const COIN_SCALE = 0.55;    // Coins smaller than tiles
const FLAG_SCALE = 0.9;     // Flag large and visible
const PLAYER_SPEED = 220;
const JUMP_VELOCITY = -480;
const GRAVITY = 850;

// ============================
// Boot Scene - Load assets
// ============================
class BootScene extends Phaser.Scene {
    constructor() {
        super('Boot');
    }

    init(data) {
        this.levelId = data.levelId;
    }

    preload() {
        // Loading screen with sky-colored background
        this.cameras.main.setBackgroundColor('#87CEFA');
        const cx = GAME_W / 2, cy = GAME_H / 2;
        this.add.text(cx, cy - 20, 'Loading...', {
            fontSize: '28px', fill: '#ffffff', fontFamily: 'Courier New',
            stroke: '#4a8a4a', strokeThickness: 3
        }).setOrigin(0.5);

        const bar = this.add.graphics();
        this.load.on('progress', (v) => {
            bar.clear();
            bar.fillStyle(0x4CAF50, 1);
            bar.fillRect(cx - 150, cy + 20, 300 * v, 18);
            bar.lineStyle(2, 0x388E3C, 1);
            bar.strokeRect(cx - 150, cy + 20, 300, 18);
        });

        // Background
        this.load.image('background', '/static/sprites/background.png');

        // Player sprite sheet: 12 frames, each 64x64
        this.load.spritesheet('player', '/static/sprites/player.png', {
            frameWidth: 64, frameHeight: 64
        });

        // Tileset: 5 tiles, each 64x64
        this.load.spritesheet('tiles', '/static/sprites/tiles.png', {
            frameWidth: 64, frameHeight: 64
        });

        // Coin: 6 frames
        this.load.spritesheet('coin', '/static/sprites/coin.png', {
            frameWidth: 64, frameHeight: 64
        });

        // Single images
        this.load.image('flag', '/static/sprites/flag.png');
        this.load.image('heart', '/static/sprites/heart.png');
        this.load.image('heart_empty', '/static/sprites/heart_empty.png');
    }

    create() {
        this.scene.start('Game', { levelId: this.levelId });
    }
}

// ============================
// Game Scene - Main gameplay
// ============================
class GameScene extends Phaser.Scene {
    constructor() {
        super('Game');
    }

    init(data) {
        this.levelId = data.levelId;
        this.score = 0;
        this.hearts = 3;
        this.coinsCollected = 0;
        this.questionsAnswered = 0;
        this.totalQuestions = 10;
        this.paused = false;
        this.levelComplete = false;
        this.questionBlocks = [];
    }

    create() {
        // Sky background
        this.add.image(GAME_W / 2, GAME_H / 2, 'background').setDisplaySize(GAME_W, GAME_H);

        // Create animations
        this.createAnimations();

        // Load level map
        this.loadLevel();

        // Input
        this.cursors = this.input.keyboard.createCursorKeys();
        this.wasd = this.input.keyboard.addKeys({
            up: Phaser.Input.Keyboard.KeyCodes.W,
            left: Phaser.Input.Keyboard.KeyCodes.A,
            right: Phaser.Input.Keyboard.KeyCodes.D,
            space: Phaser.Input.Keyboard.KeyCodes.SPACE
        });

        // Listen for quiz answers from DOM (remove any previous listener first)
        if (window._quizAnsweredHandler) {
            window.removeEventListener('quiz-answered', window._quizAnsweredHandler);
        }
        if (window._touchControlHandler) {
            window.removeEventListener('touch-control', window._touchControlHandler);
        }
        this._onQuizAnswered = (e) => this.handleQuizAnswer(e.detail);
        this._onTouchControl = (e) => this.handleTouchControl(e.detail);
        window._quizAnsweredHandler = this._onQuizAnswered;
        window._touchControlHandler = this._onTouchControl;
        window.addEventListener('quiz-answered', this._onQuizAnswered);
        window.addEventListener('touch-control', this._onTouchControl);

        // HUD
        this.createHUD();
    }

    createAnimations() {
        if (this.anims.exists('player-idle')) return;

        this.anims.create({
            key: 'player-idle',
            frames: this.anims.generateFrameNumbers('player', { start: 0, end: 3 }),
            frameRate: 6,
            repeat: -1
        });
        this.anims.create({
            key: 'player-run',
            frames: this.anims.generateFrameNumbers('player', { start: 4, end: 9 }),
            frameRate: 10,
            repeat: -1
        });
        this.anims.create({
            key: 'player-jump',
            frames: this.anims.generateFrameNumbers('player', { start: 10, end: 11 }),
            frameRate: 4,
            repeat: 0
        });
        this.anims.create({
            key: 'coin-spin',
            frames: this.anims.generateFrameNumbers('coin', { start: 0, end: 5 }),
            frameRate: 8,
            repeat: -1
        });
    }

    async loadLevel() {
        this._mapRequest = new AbortController();
        try {
            const resp = await fetch(`/api/game/levels/${this.levelId}/map`, { signal: this._mapRequest.signal });
            if (!resp.ok) throw new Error('Failed to load map');
            const mapData = await resp.json();
            if (this._disposed) return;
            this.buildLevel(mapData);
        } catch (err) {
            if (!this._disposed) console.error('Level load error:', err);
        }
    }

    buildLevel(mapData) {
        // Platforms group (static)
        this.platforms = this.physics.add.staticGroup();

        mapData.platforms.forEach(p => {
            const tileIndex = p.tile === 'ground' ? 0 : 1;
            const tilesNeeded = Math.ceil(p.width / TILE);
            for (let i = 0; i < tilesNeeded; i++) {
                const tile = this.platforms.create(
                    p.x + i * TILE + TILE / 2,
                    p.y + TILE / 2,
                    'tiles',
                    tileIndex
                );
                tile.setScale(DISPLAY_SCALE);
                tile.refreshBody();
                // Match collision body to displayed size (64 * 0.75 = 48)
                const displaySize = 64 * DISPLAY_SCALE;
                tile.body.setSize(displaySize, displaySize);
                tile.body.setOffset((64 - displaySize) / 2, (64 - displaySize) / 2);
            }
        });

        // Coins
        this.coins = this.physics.add.group();
        mapData.coins.forEach(c => {
            const coin = this.coins.create(c.x + TILE / 2, c.y + TILE / 2, 'coin', 0);
            coin.setScale(COIN_SCALE);
            coin.body.setAllowGravity(false);
            coin.play('coin-spin');
            // Gentle float animation
            this.tweens.add({
                targets: coin,
                y: coin.y - 4,
                duration: Phaser.Math.Between(800, 1200),
                yoyo: true,
                repeat: -1,
                ease: 'Sine.easeInOut'
            });
        });

        // Question blocks
        this.qBlockGroup = this.physics.add.staticGroup();
        this.questionBlocks = [];

        mapData.questionBlocks.forEach((qb) => {
            const block = this.qBlockGroup.create(
                qb.x + TILE / 2,
                qb.y + TILE / 2,
                'tiles',
                2 // question block tile
            );
            block.setScale(DISPLAY_SCALE);
            block.refreshBody();
            // Tight collision body — player must be right against the block
            const qbSize = 64 * DISPLAY_SCALE - 8; // 40px, slightly smaller than display
            block.body.setSize(qbSize, qbSize);
            block.body.setOffset((64 - qbSize) / 2, (64 - qbSize) / 2);
            block.setData('questionIndex', qb.questionIndex);
            block.setData('answered', false);
            this.questionBlocks.push(block);
        });

        // Flag
        const flag = mapData.flag;
        this.flag = this.physics.add.sprite(
            flag.x + TILE / 2,
            flag.y + TILE / 2,
            'flag'
        );
        this.flag.setScale(FLAG_SCALE);
        this.flag.body.setAllowGravity(false);
        this.flag.body.setImmovable(true);

        // Player
        const ps = mapData.playerStart;
        this.player = this.physics.add.sprite(ps.x + TILE / 2, ps.y, 'player', 0);
        this.player.setScale(PLAYER_SCALE);
        this.player.setBounce(0.1);
        this.player.setCollideWorldBounds(true);
        this.player.body.setGravityY(GRAVITY);
        this.player.body.setSize(32, 48);
        this.player.body.setOffset(16, 12);
        this.player.play('player-idle');

        // Collisions
        this.physics.add.collider(this.player, this.platforms);
        // Question blocks use overlap — trigger on contact from any direction
        this.physics.add.overlap(this.player, this.qBlockGroup, (player, block) => {
            this.hitQuestionBlock(player, block);
        });
        this.physics.add.overlap(this.player, this.coins, (player, coin) => {
            this.collectCoin(coin);
        });
        this.physics.add.overlap(this.player, this.flag, () => {
            this.reachFlag();
        });

        // World bounds
        this.physics.world.setBounds(0, 0, GAME_W, GAME_H);

        // Move player to top of render order
        this.children.bringToTop(this.player);
    }

    createHUD() {
        // Semi-transparent HUD bar at top
        const hudBg = this.add.graphics();
        hudBg.fillStyle(0x000000, 0.35);
        hudBg.fillRoundedRect(8, 6, GAME_W - 16, 36, 8);
        hudBg.setScrollFactor(0);
        hudBg.setDepth(100);

        // Hearts
        this.heartIcons = [];
        for (let i = 0; i < 3; i++) {
            const h = this.add.image(32 + i * 38, 24, 'heart');
            h.setScale(0.45);
            h.setScrollFactor(0);
            h.setDepth(101);
            this.heartIcons.push(h);
        }

        // Coin icon + counter
        this.coinText = this.add.text(GAME_W - 130, 13, 'B 0', {
            fontSize: '18px',
            fill: '#FFD700',
            fontFamily: 'Courier New',
            fontStyle: 'bold',
            stroke: '#000000',
            strokeThickness: 2
        }).setScrollFactor(0).setDepth(101);

        // Question progress
        this.questionText = this.add.text(GAME_W / 2, 13, '? 0 / 10', {
            fontSize: '16px',
            fill: '#FFFFFF',
            fontFamily: 'Courier New',
            stroke: '#000000',
            strokeThickness: 2
        }).setOrigin(0.5, 0).setScrollFactor(0).setDepth(101);

        this.updateHUD();
    }

    updateHUD() {
        // Hearts
        this.heartIcons.forEach((h, i) => {
            h.setTexture(i < this.hearts ? 'heart' : 'heart_empty');
        });

        if (this.coinText) {
            this.coinText.setText('B ' + this.coinsCollected);
        }

        if (this.questionText) {
            this.questionText.setText('? ' + this.questionsAnswered + ' / ' + this.totalQuestions);
        }
    }

    collectCoin(coin) {
        coin.destroy();
        this.coinsCollected += 5;
        this.updateHUD();

        // Floating score text
        const text = this.add.text(coin.x, coin.y, '+5', {
            fontSize: '18px', fill: '#FFD700', fontFamily: 'Courier New',
            fontStyle: 'bold', stroke: '#000', strokeThickness: 2
        }).setOrigin(0.5);
        this.tweens.add({
            targets: text,
            y: text.y - 40,
            alpha: 0,
            duration: 700,
            onComplete: () => text.destroy()
        });
    }

    hitQuestionBlock(player, block) {
        // Prevent re-trigger: already paused / answering / cooldown / already answered
        if (this.paused || this._quizCooldown || this.levelComplete) return;
        if (block.getData('answered')) return;

        // Lock immediately to prevent duplicate triggers in same frame
        this._quizCooldown = true;

        // Mark block as answered immediately to prevent re-trigger
        block.setData('answered', true);

        // Bump animation
        this.tweens.add({
            targets: block,
            y: block.y - 10,
            duration: 120,
            yoyo: true
        });

        // Trigger quiz
        const qIdx = block.getData('questionIndex');
        this.currentBlock = block;
        this.pauseGame();

        window.dispatchEvent(new CustomEvent('quiz-trigger', {
            detail: { levelId: this.levelId, questionIndex: qIdx }
        }));
    }

    _isPlayerOverlappingAnyUnansweredBlock() {
        if (!this.player || !this.player.body) return false;
        const pb = this.player.body;
        const pLeft = pb.x;
        const pRight = pb.x + pb.width;
        const pTop = pb.y;
        const pBottom = pb.y + pb.height;

        for (const block of this.questionBlocks) {
            if (block.getData('answered')) continue;
            const bb = block.body;
            if (!bb) continue;
            const bLeft = bb.x;
            const bRight = bb.x + bb.width;
            const bTop = bb.y;
            const bBottom = bb.y + bb.height;
            // AABB overlap check
            if (pLeft < bRight && pRight > bLeft && pTop < bBottom && pBottom > bTop) {
                return true;
            }
        }
        return false;
    }

    handleQuizAnswer(detail) {
        const { correct } = detail;

        if (this.currentBlock) {
            if (correct) {
                this.currentBlock.setFrame(3); // gold/check block
                this.score++;
                this.coinsCollected += 10;

                // Success effect
                const text = this.add.text(this.currentBlock.x, this.currentBlock.y - 24, '+10', {
                    fontSize: '20px', fill: '#4CAF50', fontFamily: 'Courier New',
                    fontStyle: 'bold', stroke: '#000', strokeThickness: 2
                }).setOrigin(0.5);
                this.tweens.add({
                    targets: text,
                    y: text.y - 50,
                    alpha: 0,
                    duration: 900,
                    onComplete: () => text.destroy()
                });
            } else {
                this.currentBlock.setFrame(4); // used/X block
                this.hearts--;
                // Screen shake on wrong answer
                this.cameras.main.shake(200, 0.005);
                if (this.hearts <= 0) {
                    this._quizCooldown = false;
                    this.resumeGame();
                    this.gameOver();
                    return;
                }
            }
            this.currentBlock.refreshBody();
            this.currentBlock = null;
        }

        this.questionsAnswered++;
        this.updateHUD();
        this.resumeGame();

        // Zone-based cooldown: keep cooldown active until player leaves all
        // unanswered question blocks. Check every frame via update().
        // Also set a minimum cooldown of 300ms as a safety net.
        this._cooldownMinExpiry = this.time.now + 300;

        // Auto-complete when 5 correct answers reached
        if (this.score >= 5) {
            this._quizCooldown = false;
            this.time.delayedCall(500, () => this.completeLevel());
        }
    }

    reachFlag() {
        if (this.levelComplete) return;

        // Pass condition: at least 70% of answered questions are correct.
        // Player does NOT need to answer every question block.
        const needed = 5;
        if (this.score < needed) {
            if (!this._flagHintShown) {
                this._flagHintShown = true;
                const hint = this.add.text(this.flag.x, this.flag.y - 50,
                    `Need ${needed} correct answers! (${this.score}/${needed})`, {
                    fontSize: '14px', fill: '#FFFFFF', fontFamily: 'Courier New',
                    stroke: '#000', strokeThickness: 2
                }).setOrigin(0.5);
                this.tweens.add({
                    targets: hint,
                    alpha: 0,
                    duration: 2000,
                    onComplete: () => {
                        hint.destroy();
                        this._flagHintShown = false;
                    }
                });
            }
            return;
        }

        this.completeLevel();
    }

    completeLevel() {
        if (this.levelComplete) return;
        this.levelComplete = true;

        this.player.body.setVelocity(0, 0);
        this.player.body.setAllowGravity(false);
        this.player.play('player-idle');

        // Victory animation
        this.tweens.add({
            targets: this.player,
            y: this.player.y - 60,
            duration: 600,
            yoyo: true,
            onComplete: () => {
                window.dispatchEvent(new CustomEvent('level-complete', {
                    detail: {
                        levelId: this.levelId,
                        score: this.score,
                        totalQuestions: this.totalQuestions,
                        coins: this.coinsCollected,
                        hearts: this.hearts
                    }
                }));
            }
        });
    }

    gameOver() {
        if (this.levelComplete) return;
        this.levelComplete = true;

        this.player.setTint(0xff0000);
        this.cameras.main.shake(400, 0.01);
        this.tweens.add({
            targets: this.player,
            alpha: 0,
            duration: 800,
            onComplete: () => {
                window.dispatchEvent(new CustomEvent('level-complete', {
                    detail: {
                        levelId: this.levelId,
                        score: this.score,
                        totalQuestions: this.totalQuestions,
                        coins: this.coinsCollected,
                        hearts: 0
                    }
                }));
            }
        });
    }

    pauseGame() {
        this.paused = true;
        this.physics.pause();
    }

    resumeGame() {
        this.paused = false;
        this.physics.resume();
    }

    handleTouchControl(detail) {
        this._touchAction = detail.action;
    }

    update() {
        if (this.paused || this.levelComplete || !this.player || !this.player.body) return;

        // Zone-based cooldown: only allow next quiz trigger once player has
        // left all unanswered question blocks AND minimum cooldown has passed
        if (this._quizCooldown && this._cooldownMinExpiry) {
            if (this.time.now >= this._cooldownMinExpiry &&
                !this._isPlayerOverlappingAnyUnansweredBlock()) {
                this._quizCooldown = false;
                this._cooldownMinExpiry = null;
            }
        }

        const left = this.cursors.left.isDown || this.wasd.left.isDown || this._touchAction === 'left';
        const right = this.cursors.right.isDown || this.wasd.right.isDown || this._touchAction === 'right';
        const jump = Phaser.Input.Keyboard.JustDown(this.cursors.up) ||
                     Phaser.Input.Keyboard.JustDown(this.wasd.up) ||
                     Phaser.Input.Keyboard.JustDown(this.wasd.space) ||
                     Phaser.Input.Keyboard.JustDown(this.cursors.space) ||
                     this._touchAction === 'jump';

        if (left) {
            this.player.setVelocityX(-PLAYER_SPEED);
            this.player.setFlipX(true);
        } else if (right) {
            this.player.setVelocityX(PLAYER_SPEED);
            this.player.setFlipX(false);
        } else {
            this.player.setVelocityX(0);
        }

        if (jump && this.player.body.touching.down) {
            this.player.setVelocityY(JUMP_VELOCITY);
        }

        if (!this.player.body.touching.down) {
            this.player.play('player-jump', true);
        } else if (left || right) {
            this.player.play('player-run', true);
        } else {
            this.player.play('player-idle', true);
        }

        if (this._touchAction === 'jump') {
            this._touchAction = 'none';
        }
    }

    shutdown() {
        this._disposed = true;
        this._mapRequest?.abort();
        window.removeEventListener('quiz-answered', this._onQuizAnswered);
        window.removeEventListener('touch-control', this._onTouchControl);
    }
}

// ============================
// PlatformerGame - Factory
// ============================
const PlatformerGame = {
    instance: null,

    launch(levelId, containerId) {
        this.destroy();

        const config = {
            type: Phaser.AUTO,
            width: GAME_W,
            height: GAME_H,
            parent: containerId,
            pixelArt: true,
            physics: {
                default: 'arcade',
                arcade: {
                    gravity: { y: 0 },
                    debug: false
                }
            },
            scene: [BootScene, GameScene],
            scale: {
                mode: Phaser.Scale.FIT,
                autoCenter: Phaser.Scale.CENTER_BOTH
            }
        };

        this.instance = new Phaser.Game(config);
        this.instance.scene.start('Boot', { levelId: levelId });
    },

    destroy() {
        if (this.instance) {
            this.instance.scene.scenes.forEach(scene => { if (scene instanceof GameScene) scene.shutdown(); });
            this.instance.destroy(true);
            this.instance = null;
        }
    }
};

window.PlatformerGame = PlatformerGame;
