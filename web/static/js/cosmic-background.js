/* Shared cosmic background. Independent of page controllers and game canvases. */
(() => {
    'use strict';
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const instanceKey = Symbol.for('ansel.cosmicBackground');

    class StarField {
        constructor(canvas) {
            this.canvas = canvas;
            this.context = canvas.getContext('2d');
            if (!this.context) return;
            this.frame = 0;
            this.lastFrame = 0;
            this.pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };
            this.colors = ['236,245,255', '135,176,255', '105,230,229', '191,155,255', '255,194,118'];
            // A fixed noise lattice gives resized clouds a consistent structure.
            this.noiseGrid = Float32Array.from({ length: 65536 }, () => Math.random());
            this.glows = this.colors.map(color => this.createGlow(color));
            this.stars = [];
            this.resize();
            this.start();
            window.addEventListener('resize', () => {
                clearTimeout(this.resizeTimer);
                this.resizeTimer = setTimeout(() => this.resize(), 120);
            }, { passive: true });
            document.addEventListener('visibilitychange', () => this.start());
            reducedMotion.addEventListener('change', () => {
                this.pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };
                this.start();
            });
            if (window.matchMedia('(hover: hover) and (pointer: fine)').matches) {
                window.addEventListener('pointermove', event => {
                    if (reducedMotion.matches) return;
                    this.pointer.targetX = (event.clientX / this.width - 0.5) * 14;
                    this.pointer.targetY = (event.clientY / this.height - 0.5) * 10;
                }, { passive: true });
                window.addEventListener('blur', () => {
                    this.pointer.targetX = 0;
                    this.pointer.targetY = 0;
                });
            }
        }
        surface(width, height) {
            const surface = document.createElement('canvas');
            surface.width = width;
            surface.height = height;
            return surface;
        }
        createGlow(color) {
            const surface = this.surface(40, 40);
            const context = surface.getContext('2d');
            const glow = context.createRadialGradient(20, 20, 0, 20, 20, 20);
            glow.addColorStop(0, `rgba(${color},0.8)`);
            glow.addColorStop(0.12, `rgba(${color},0.3)`);
            glow.addColorStop(0.4, `rgba(${color},0.09)`);
            glow.addColorStop(1, `rgba(${color},0)`);
            context.fillStyle = glow;
            context.fillRect(0, 0, 40, 40);
            return surface;
        }
        noise(x, y) {
            const ix = Math.floor(x), iy = Math.floor(y);
            const fx = x - ix, fy = y - iy;
            const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
            const row = (iy & 255) << 8, next = ((iy + 1) & 255) << 8;
            const left = ix & 255, right = (ix + 1) & 255;
            const a = this.noiseGrid[row | left], b = this.noiseGrid[row | right];
            const c = this.noiseGrid[next | left], d = this.noiseGrid[next | right];
            return a + (b - a) * sx + (c - a + (a - b - c + d) * sx) * sy;
        }
        fbm(x, y, octaves) {
            let sum = 0, weight = 0.53, total = 0;
            for (let octave = 0; octave < octaves; octave++) {
                sum += this.noise(x, y) * weight;
                total += weight;
                x = x * 2.03 + 13.17;
                y = y * 2.03 + 7.61;
                weight *= 0.49;
            }
            return sum / total;
        }
        makeNebula() {
            // Domain-warped five-octave clouds and three-octave dust are baked
            // once per resize; the animation loop only composites cached images.
            const scale = Math.min(1, (this.mobile ? 240 : 600) / this.width, (this.mobile ? 400 : 500) / this.height);
            const width = Math.max(1, Math.round(this.width * scale));
            const height = Math.max(1, Math.round(this.height * scale));
            const surface = this.surface(width, height);
            const context = surface.getContext('2d');
            const image = context.createImageData(width, height);
            const pixels = image.data;
            const palette = [[29, 66, 162], [94, 48, 171], [164, 48, 142], [42, 99, 191], [36, 164, 184], [205, 145, 81]];
            const horizontal = new Float32Array(width * 4);
            for (let px = 0; px < width; px++) {
                const x = px / width;
                const left = (x + 0.02) / 0.52, right = (x - 1.02) / 0.49;
                const upper = (x - 0.54) / 0.7, center = (x - 0.5) / 0.21;
                horizontal[px * 4] = Math.exp(-left * left);
                horizontal[px * 4 + 1] = Math.exp(-right * right);
                horizontal[px * 4 + 2] = Math.exp(-upper * upper);
                horizontal[px * 4 + 3] = 1 - 0.66 * Math.exp(-center * center);
            }
            for (let py = 0; py < height; py++) {
                const y = py / height;
                const leftY = (y - 0.30) / 0.47, rightY = (y - 0.29) / 0.46, upperY = (y + 0.11) / 0.32;
                const verticalLeft = Math.exp(-leftY * leftY), verticalRight = Math.exp(-rightY * rightY), verticalUpper = Math.exp(-upperY * upperY) * 0.58;
                for (let px = 0; px < width; px++) {
                    const x = px / width;
                    const nx = x * 5.5, ny = y * 4.1;
                    const warpX = this.noise(nx * 0.8 + 11, ny * 0.8 + 7) - 0.5;
                    const warpY = this.noise(nx * 0.8 + 61, ny * 0.8 + 29) - 0.5;
                    const clouds = this.fbm(nx + warpX * 3.2, ny + warpY * 3.2, 5);
                    const dust = this.fbm(nx * 2.1 - warpY * 2 + 91, ny * 2.1 + warpX * 2 + 64, 3);
                    const colorNoise = this.noise(nx * 0.9 + warpX * 1.5 + 23, ny * 0.9 + warpY + 41);
                    const horizontalIndex = px * 4;
                    const envelope = Math.min(1, horizontal[horizontalIndex] * verticalLeft + horizontal[horizontalIndex + 1] * verticalRight + horizontal[horizontalIndex + 2] * verticalUpper);
                    const centralShade = horizontal[horizontalIndex + 3];
                    const mass = Math.pow(Math.max(0, Math.min(1, (clouds - 0.28) * 2)), 1.65);
                    const filamentBase = Math.max(0, 1 - Math.abs(clouds - 0.56) * 10);
                    const filaments = filamentBase * filamentBase * filamentBase;
                    const dustDistance = (y - (0.20 + x * 0.38 + warpY * 0.14)) / 0.065;
                    const dustRidge = Math.exp(-dustDistance * dustDistance);
                    const dustAmount = Math.min(0.88, Math.max(0, (dust - 0.42) * 2.5) + dustRidge * 0.32);
                    const light = (mass * 0.95 + filaments * 0.20) * (1 - dustAmount) * envelope * centralShade * 1.4;
                    const tone = Math.max(0, Math.min(0.999, 0.15 + x * 0.48 + colorNoise * 0.5 - y * 0.15)) * (palette.length - 1);
                    const first = Math.floor(tone), blend = tone - first;
                    const a = palette[first], b = palette[first + 1];
                    const index = (py * width + px) * 4;
                    const red = (a[0] + (b[0] - a[0]) * blend) * light;
                    const green = (a[1] + (b[1] - a[1]) * blend) * light;
                    const blue = (a[2] + (b[2] - a[2]) * blend) * light;
                    const luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722;
                    // Muting only emitted cloud light preserves the deep-space base.
                    pixels[index] = 3 + (red * 0.63 + luminance * 0.37) * 0.24;
                    pixels[index + 1] = 7 + (green * 0.63 + luminance * 0.37) * 0.24;
                    pixels[index + 2] = 17 + (blue * 0.63 + luminance * 0.37) * 0.24;
                    pixels[index + 3] = 255;
                }
            }
            context.putImageData(image, 0, 0);
            this.nebula = surface;
            const dustSurface = this.surface(width, height);
            const dustContext = dustSurface.getContext('2d');
            for (let i = 0; i < (this.mobile ? 220 : 820); i++) {
                const x = Math.random();
                const spread = (Math.random() + Math.random() + Math.random() - 1.5) * 0.055;
                const y = 0.19 + x * 0.38 + Math.sin(x * 8) * 0.025 + spread;
                const alpha = (0.04 + Math.random() * 0.14) * (0.35 + Math.abs(x - 0.5));
                dustContext.fillStyle = `rgba(160,186,234,${alpha})`;
                dustContext.fillRect(x * width, y * height, 0.3 + Math.random() * 0.3, 0.3 + Math.random() * 0.3);
            }
            this.dust = dustSurface;
        }
        resize() {
            this.width = window.innerWidth;
            this.height = window.innerHeight;
            this.mobile = this.width < 680;
            const ratio = Math.min(window.devicePixelRatio || 1, this.mobile ? 1.5 : 2);
            const count = this.mobile ? 175 : Math.min(510, Math.max(320, Math.round(this.width * this.height / 3500)));
            if (this.stars.length !== count) {
                this.stars = Array.from({ length: count }, (_, index) => {
                    const layer = index < count * 0.64 ? 0 : index < count * 0.91 ? 1 : 2;
                    const colorChance = Math.random();
                    return {
                        x: Math.random(), y: Math.random(), layer,
                        r: [0.28, 0.65, 1.2][layer] + Math.random() * [0.43, 0.5, 0.85][layer],
                        alpha: [0.27, 0.52, 0.75][layer] + Math.random() * 0.18,
                        depth: [0.18, 0.48, 1][layer],
                        phase: Math.random() * Math.PI * 2,
                        speed: Math.random() * 0.00035 + 0.00013,
                        color: colorChance < 0.58 ? 0 : colorChance < 0.80 ? 1 : colorChance < 0.89 ? 2 : colorChance < 0.95 ? 3 : 4,
                        rays: layer === 2 && Math.random() < 0.2
                    };
                });
            }
            this.canvas.width = this.width * ratio;
            this.canvas.height = this.height * ratio;
            this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
            this.makeNebula();
            this.draw(reducedMotion.matches ? 0 : performance.now());
        }
        draw(time) {
            const context = this.context;
            const pointer = this.pointer;
            pointer.x += (pointer.targetX - pointer.x) * 0.06;
            pointer.y += (pointer.targetY - pointer.y) * 0.06;
            context.globalAlpha = 1;
            context.globalCompositeOperation = 'source-over';
            context.drawImage(this.nebula, -this.width * 0.02 + pointer.x * 0.12, -this.height * 0.02 + pointer.y * 0.12, this.width * 1.04, this.height * 1.04);
            context.globalCompositeOperation = 'screen';
            context.drawImage(this.dust, -pointer.x * 0.2, -pointer.y * 0.2, this.width, this.height);
            this.stars.forEach(star => {
                const drift = reducedMotion.matches ? 0 : Math.sin(star.phase + time * 0.000025) * star.depth * 2.5;
                const x = star.x * this.width + pointer.x * star.depth + drift;
                const y = star.y * this.height + pointer.y * star.depth + drift * 0.55;
                const alpha = star.alpha * (0.72 + Math.sin(star.phase + time * star.speed) * 0.22);
                const color = this.colors[star.color];
                if (star.layer === 2) {
                    const size = star.r * 9;
                    context.globalAlpha = alpha * 0.65;
                    context.drawImage(this.glows[star.color], x - size / 2, y - size / 2, size, size);
                }
                if (star.rays) {
                    context.globalAlpha = alpha * 0.4;
                    context.strokeStyle = `rgb(${color})`;
                    context.lineWidth = 0.6;
                    context.beginPath();
                    context.moveTo(x - star.r * 3.2, y);
                    context.lineTo(x + star.r * 3.2, y);
                    context.moveTo(x, y - star.r * 4.5);
                    context.lineTo(x, y + star.r * 4.5);
                    context.stroke();
                }
                context.globalAlpha = alpha;
                context.beginPath();
                context.arc(x, y, star.r, 0, Math.PI * 2);
                context.fillStyle = `rgb(${color})`;
                context.fill();
            });
            context.globalAlpha = 1;
            context.globalCompositeOperation = 'source-over';
        }
        start() {
            cancelAnimationFrame(this.frame);
            this.frame = 0;
            this.lastFrame = 0;
            if (document.hidden) return;
            if (reducedMotion.matches) { this.draw(0); return; }
            const step = time => {
                const interval = 1000 / 30;
                const elapsed = time - this.lastFrame;
                if (elapsed >= interval) {
                    this.draw(time);
                    this.lastFrame = time - (elapsed % interval);
                }
                this.frame = requestAnimationFrame(step);
            };
            this.frame = requestAnimationFrame(step);
        }
    }
    function init() {
        const canvas = document.getElementById('starfield');
        if (!canvas || canvas.tagName !== 'CANVAS' || canvas[instanceKey]) return;
        const instance = new StarField(canvas);
        if (instance.context) canvas[instanceKey] = instance;
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
    else init();
})();
