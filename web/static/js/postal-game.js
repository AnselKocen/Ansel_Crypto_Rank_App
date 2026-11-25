/* First complete mini-game. Scene, input, quiz and lifecycle have a single owner. */
const PostalGame = {
    scene: null,
    async launch(token) {
        const root=document.getElementById('view-postal');
        root.innerHTML='<div class="postal-loading" role="status">正在给小邮车装上十封信…</div>';
        try {
            const data=await fetchJSON('/api/game/levels/1');
            if(token!==GameApp._loadId) return;
            if(data.level!==1 || data.questions?.length!==10 || data.questions.some(q=>!q.id || !Array.isArray(q.options) || q.options.length!==4 || !Number.isInteger(q.answer) || q.answer<0 || q.answer>3)) throw Error('Invalid question data');
            this.scene=new PostalScene(root,data);
        } catch {
            if(token!==GameApp._loadId) return;
            root.innerHTML='<div class="postal-loading"><p>信件暂时没有装好，请重新加载。</p><button class="post-btn primary" onclick="GameApp.startLevel(1)">重试</button> <button class="post-btn" onclick="GameApp.showMap()">回地图</button></div>';
        }
    },
    destroy() { this.scene?.destroy(); this.scene=null; }
};

class PostalScene {
    constructor(root,data) {
        this.root=root;this.data=data;this.levelId=data.level;this.rewardName=window._gameData?.cats.find(c=>c.level===data.level)?.name_zh||'猫咪朋友';this.alive=true; this.phase='intro';
        this.controller=new AbortController(); this.signal=this.controller.signal;
        this.keys=new Set(); this.pointers=new Map(); this.time=0; this.cooldown=0; this.hitDelay=0;
        this.lastSave=0; this.lastFrame=0; this.hudKey=''; this.soundStep=-1;
        this.items=[];
        for(let i=0;i<10;i++) for(let j=0;j<3;j++) this.items.push({id:`coin-${i}-${j}`,kind:'coin',s:100+i*400+j*46,x:(i%3-1)*30});
        [490,1310,2370,3260].forEach((s,i)=>this.items.push({id:`crate-${i}`,kind:'crate',s,x:i%2?30:-30}));
        const settings=GameState.load().settings;
        this.settings={reduced:window.matchMedia('(prefers-reduced-motion: reduce)').matches,sfx:true,music:false,...settings};
        this.voices=new Set();
        this.run=this.validRun(GameState.load().runs[this.levelId]) || this.newRun();
        this.mastery=this.restoreKnowledge(this.run);
        if(this.run.mode==='quiz') this.run=this.newRun();
        this.renderShell(); this.bindInput(); this.resizeCanvas();
        document.body.classList.add('postal-playing');
        if(this.run.settled) { this.renderResult(); }
        else { this.showIntro(); this.prepareAudio(); }
        this.tick=this.tick.bind(this); this.frame=requestAnimationFrame(this.tick);
    }
    newRun() {
        return {id:globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,levelId:1,
            mode:'drive',rulesVersion:2,distance:0,offset:0,answers:{},passed:[],missed:[],collected:[],coins:0,bumps:0,assisted:false,
            routeComplete:false,settled:false,questionId:null,docking:null};
    }
    validRun(r) {
        if(!r || typeof r.id!=='string' || r.levelId!==1 || !['drive','quiz'].includes(r.mode) || !r.answers || typeof r.answers!=='object' || !Array.isArray(r.collected) || !Number.isFinite(r.coins)) return null;
        for(const [id,a] of Object.entries(r.answers)) {
            const q=this.data.questions.find(q=>String(q.id)===id);
            if(!q || !a || !Number.isInteger(a.selected) || a.selected<0 || a.selected>=q.options.length || a.correct!==(a.selected===q.answer)) return null;
        }
        r.distance=Math.max(0,Math.min(4200,Number(r.distance)||0));
        r.offset=Math.max(-84,Math.min(84,Number(r.offset)||0));
        r.bumps=Math.max(0,Number(r.bumps)||0);r.coins=Math.max(0,r.coins);
        r.passed=Array.isArray(r.passed)?r.passed:[];r.missed=Array.isArray(r.missed)?r.missed:[];
        if(!this.data.questions.some(q=>q.id===r.questionId)) r.questionId=null;
        if(!r.questionId) r.questionId=this.data.questions.find(q=>r.answers[q.id] && !r.answers[q.id].continued)?.id || null;
        if(r.docking && (!PostalArt.stations.some(p=>p.id===r.docking.id) || !['fromDistance','fromOffset','elapsed'].every(k=>Number.isFinite(r.docking[k])))) r.docking=null;
        // Missed stations never rewind the route. They remain available next lap.
        r.rulesVersion=2;
        return r;
    }
    restoreKnowledge(run) {
        const state=GameState.load(), raw=state.knowledge[this.levelId] || {}, known={};
        for(const q of this.data.questions) {
            const a=raw[q.id], previous=run.answers[q.id];
            if(a && Number.isInteger(a.selected) && a.selected>=0 && a.selected<4 && a.correct===(a.selected===q.answer)) known[q.id]={...a};
            if(previous && (!known[q.id] || previous.correct)) known[q.id]={selected:previous.selected,correct:previous.correct};
            if(GameState.isCatPerfect(this.levelId)) known[q.id]={selected:q.answer,correct:true};
        }
        state.knowledge[this.levelId]=known;
        return known;
    }
    save() { if(!this.alive) return; GameState.load().runs[this.levelId]=this.run; GameState.save(); updateStats(); }
    count() { return Object.keys(this.run.answers).length; }
    score() { return this.data.questions.filter(q=>this.mastery[q.id]?.correct).length; }
    roundScore() { return Object.values(this.run.answers).filter(a=>a.correct).length; }
    el(selector) { return this.root.querySelector(selector); }
    listen(target,type,handler,options={}) { target.addEventListener(type,handler,{...options,signal:this.signal}); }
    renderShell() {
        const letter='<svg viewBox="0 0 20 16" aria-hidden="true"><path d="M1 2h18v12H1z M1 3l9 6 9-6"/></svg>';
        this.root.innerHTML=`<section class="postal-app" aria-label="货币起源环岛邮路">
          <header class="postal-top"><button class="post-back" data-action="map" aria-label="保存并返回地图">← 地图</button>
            <div><small>ISLAND POST · 01</small><h2>货币起源环岛邮路</h2></div>
            <button class="post-btn" data-action="handbook">知识手册</button></header>
          <div class="postal-frame">
            <canvas id="postal-canvas" width="640" height="400" tabindex="0" aria-label="海岛邮路：左右方向键或 A、D 转向，向下减速。驶入带感叹号邮筒旁的 P 停车位后自动停靠。错过可在下一趟补做。"></canvas>
            <div class="post-corner-progress"><div class="post-progress-title">已完成信件 <b><span id="post-count">0</span> / 10</b></div><div class="post-mini-letters" aria-label="跨局保留的信件完成进度">${this.data.questions.map(q=>`<span data-stamp="${q.id}" class="post-mini-letter">${letter}<i></i></span>`).join('')}</div></div>
            <div class="post-corner-actions"><span class="post-coin-pill">金币 <b id="post-coins">0</b></span><button class="post-hud-button" data-action="sound" aria-label="切换音效" aria-pressed="true">♪</button><button class="post-hud-button" data-action="pause" aria-label="暂停与设置">Ⅱ</button></div>
            <div class="post-location" id="post-location">01 · 贝壳海滩</div>
            <div class="post-toast" id="post-toast" role="status"></div>
            <div class="post-start-curtain"><div class="post-guide"><small>你的第一份海岛差事</small><h3>去找带 ! 的邮筒</h3>
              <div class="post-guide-drawing" aria-hidden="true"><span class="guide-van">✉</span><span class="guide-arrow">→</span><span class="guide-parking">P</span><span class="guide-mailbox">!</span></div>
              <p>左右转向，驶入邮筒旁的 <b>P 停车位</b>，小邮车就会停稳，打开知识信。</p><p class="post-guide-small">错过了？下一趟再来。答对的信件会一直打勾。</p>
              <button class="post-btn primary" data-action="start">知道了，出发送信 →</button></div></div>
          </div>
          <div class="post-drive-bar"><span class="post-key-hint">← → / A D 转向 · ↓ 减速 · 驶入 P 位收信</span><div class="postal-controls" aria-label="驾驶按钮"><button class="post-drive" data-drive="left" aria-label="向左转向">←</button><button class="post-brake" data-drive="brake" aria-label="按住减速">减速</button><button class="post-drive" data-drive="right" aria-label="向右转向">→</button></div><span class="post-lap-label" id="post-lap-label">寻找邮筒，慢慢开。</span></div>
        </section>`;
        this.canvas=this.el('canvas');this.ctx=this.canvas.getContext('2d');
        this.listen(this.root,'click',e=>{
            const action=e.target.closest('[data-action]')?.dataset.action;
            if(action==='map') GameApp.showMap();
            if(action==='start') this.start();
            if(action==='pause') this.pause();
            if(action==='handbook') this.handbook();
            if(action==='sound') {this.settings.sfx=!this.settings.sfx;this.saveSettings();if(this.settings.sfx) this.playSound('coin');}
        });
    }
    resizeCanvas() {
        const portrait=window.innerWidth<=650 && window.innerHeight>window.innerWidth;
        const width=portrait?320:640;
        const height=portrait?Math.max(400,Math.min(680,Math.round((window.innerHeight-124)*320/Math.max(240,window.innerWidth-24)))):400;
        if(this.canvas.width!==width) this.canvas.width=width;
        if(this.canvas.height!==height) this.canvas.height=height;
        this.el('.postal-app').style.setProperty('--post-ratio',String(width/height));
        this.root.classList.toggle('postal-portrait',portrait);
    }
    showIntro() {
        this.phase='intro';this.root.classList.remove('post-started');
        const resumed=this.run.distance>0 || this.count()>0;
        this.el('[data-action="start"]').textContent=resumed?'继续这趟送信 →':'知道了，出发送信 →';
        this.updateHUD();
    }
    start() {
        if(!this.alive || this.phase!=='intro') return;
        this.root.classList.add('post-started');this.phase=this.run.docking?'docking':'driving';this.clearInput();this.cooldown=1.5;
        this.ensureAudio();this.playSound('start');this.save();this.canvas.focus({preventScroll:true});
        if(this.run.questionId && !this.run.docking) this.openQuestion(this.data.questions.findIndex(q=>q.id===this.run.questionId));
        else this.toast(this.run.docking?'小邮车正在停稳，稍后打开知识信。':'寻找带 ! 的邮筒，驶入旁边的 P 停车位就能收信。',4);
    }
    bindInput() {
        const actions={ArrowLeft:'left',KeyA:'left',ArrowRight:'right',KeyD:'right',ArrowDown:'brake',KeyS:'brake'};
        this.listen(window,'keydown',e=>{
            if(this.phase!=='driving') return;
            if(actions[e.code]) { e.preventDefault(); this.keys.add(e.code); }
            if(['KeyP','Escape'].includes(e.code) && !e.repeat) { e.preventDefault(); this.pause(); }
        });
        this.listen(window,'keyup',e=>this.keys.delete(e.code));
        this.root.querySelectorAll('[data-drive]').forEach(b=>{
            this.listen(b,'pointerdown',e=>{
                if(this.phase!=='driving') return;
                e.preventDefault(); b.setPointerCapture(e.pointerId); this.pointers.set(e.pointerId,b.dataset.drive); b.classList.add('pressed');
            });
            const release=e=>{ this.pointers.delete(e.pointerId); b.classList.remove('pressed'); };
            for(const type of ['pointerup','pointercancel','lostpointercapture']) this.listen(b,type,release);
        });
        this.listen(window,'resize',()=>this.resizeCanvas());
        const hide=()=>{ this.clearInput();this.stopVoices();if(['driving','docking'].includes(this.phase)) this.pause('旅程已暂停，回来后再出发。'); };
        this.listen(window,'blur',hide);
        this.listen(document,'visibilitychange',()=>{if(document.hidden) hide();});
        this.listen(window,'pagehide',()=>this.save());
    }
    input(action) {
        if(action==='left') return this.keys.has('ArrowLeft')||this.keys.has('KeyA')||[...this.pointers.values()].includes('left');
        if(action==='right') return this.keys.has('ArrowRight')||this.keys.has('KeyD')||[...this.pointers.values()].includes('right');
        return this.keys.has('ArrowDown')||this.keys.has('KeyS')||[...this.pointers.values()].includes('brake');
    }
    clearInput() { this.keys.clear(); this.pointers.clear(); this.root.querySelectorAll('.pressed').forEach(el=>el.classList.remove('pressed')); }
    tick(now) {
        if(!this.alive) return;
        const dt=Math.min(.05,Math.max(0,(now-(this.lastFrame||now))/1000)); this.lastFrame=now;
        if(this.phase==='driving' || this.phase==='docking') {
            this.time+=dt; this.cooldown=Math.max(0,this.cooldown-dt);
            this.update(dt);
            if(now-this.lastSave>1200) { this.lastSave=now; this.save(); }
            if(this.settings.music) {
                const step=Math.floor(this.time*2);
                if(step!==this.soundStep) { this.soundStep=step; this.note([262,330,392,330,294,349,440,349][step%8],.22,.018); }
            }
        }
        PostalArt.draw(this.ctx,this.run.distance,this.run.offset,this.settings.reduced?0:this.time,
            {mastery:this.mastery,items:this.items,collected:this.run.collected,reduced:this.settings.reduced,steer:this.steer,protectedTime:this.cooldown});
        if(this.toastExpiry && this.time>this.toastExpiry) this.el('#post-toast').classList.remove('visible');
        this.updateHUD(); this.frame=requestAnimationFrame(this.tick);
    }
    update(dt) {
        if(this.phase==='docking') {
            const dock=this.run.docking, station=PostalArt.stations.find(p=>p.id===dock?.id);
            if(!station) {this.run.docking=null;this.phase='driving';return;}
            dock.elapsed+=dt;
            const t=Math.min(1,dock.elapsed/.65), ease=1-(1-t)**3;
            this.run.distance=dock.fromDistance+(station.s-dock.fromDistance)*ease;
            this.run.offset=dock.fromOffset+(station.bayOffset-dock.fromOffset)*ease;
            if(t===1) {this.run.docking=null;this.openQuestion(station.id-1);}
            return;
        }
        if(this.phase!=='driving') return;
        if(this.hitDelay>0) {this.hitDelay-=dt;return;}
        this.steer=Number(this.input('right'))-Number(this.input('left'));
        this.run.offset+=this.steer*(this.run.assisted?68:80)*dt;
        const nearby=PostalArt.stations.find(p=>Math.abs(p.s-this.run.distance)<72);
        const leftLimit=nearby?.side===-1?84:54,rightLimit=nearby?.side===1?84:54;
        if(this.run.offset>rightLimit || this.run.offset< -leftLimit) {
            this.run.offset=this.run.offset>0?rightLimit-4:-leftLimit+4;
            if(!this.cooldown) this.bump('路沿轻轻碰了一下，信件和金币都在。');
        }
        this.run.distance=Math.min(4200,this.run.distance+(this.input('brake')?22:54)*dt);
        for(const station of PostalArt.stations) {
            const attempted=!!this.run.answers[station.id] || this.run.passed.includes(station.id);
            if(!attempted && !this.mastery[station.id]?.correct && Math.abs(this.run.distance-station.s)<=station.halfLength && Math.abs(PostalArt.center(this.run.distance)+this.run.offset-PostalArt.center(station.s)-station.bayOffset)<=station.halfWidth) {
                this.beginDock(station);return;
            }
            if(this.run.distance>station.s+station.halfLength && !this.run.passed.includes(station.id)) {
                this.run.passed.push(station.id);
                if(!this.mastery[station.id]?.correct && !this.run.answers[station.id]) {
                    this.run.missed.push(station.id);this.toast(`错过了第 ${station.id} 封信，下一趟再来。`,2.5);
                }
                this.save();
            }
        }
        for(const item of this.items) {
            if(this.run.collected.includes(item.id) || Math.abs(this.run.distance-item.s)>20 || Math.abs(this.run.offset-item.x)>23) continue;
            this.run.collected.push(item.id);
            if(item.kind==='coin') {GameState.awardRun(this.run,5);this.playSound('coin');this.toast('+5 金币',1.1);}
            else if(!this.cooldown) {this.bump('货箱轻轻碰了一下，慢慢继续。');}
        }
        if(this.run.distance>=4200) {this.run.routeComplete=true;this.finish();}
    }
    beginDock(station) {
        this.run.docking={id:station.id,fromDistance:this.run.distance,fromOffset:this.run.offset,elapsed:0};
        this.run.questionId=station.id;this.phase='docking';this.steer=0;this.clearInput();this.save();
        this.playSound('dock');this.toast('已进入停车位，小邮车正在停稳…',1.2);
    }
    bump(message) {
        this.run.bumps++; this.hitDelay=.5; this.cooldown=1.5;
        if(this.run.bumps>=2 && !this.run.assisted) { this.run.assisted=true; message='转向已放缓，停车位仍由你选择。'; }
        this.playSound('bump');this.toast(message,3);this.save();
    }
    updateHUD() {
        const region=Math.min(3,Math.floor(this.run.distance/1050));
        const key=[this.score(),this.count(),this.run.missed.join('.'),this.run.coins,region,this.settings.sfx].join(',');
        if(this.hudKey===key) return;this.hudKey=key;
        this.el('#post-count').textContent=this.score();this.el('#post-coins').textContent=this.run.coins;
        this.el('#post-location').textContent=['01 · 贝壳海滩','02 · 铸币小镇','03 · 纸币邮局','04 · 数字灯塔'][region];
        this.root.querySelectorAll('[data-stamp]').forEach(el=>{
            const id=Number(el.dataset.stamp),known=this.mastery[id],missed=this.run.missed.includes(id);
            const status=known?.correct?'correct':this.run.answers[id]?'incorrect':missed?'missed':'pending';
            el.className='post-mini-letter '+status;el.querySelector('i').textContent=status==='correct'?'✓':status==='incorrect'?'·':'';
            el.setAttribute('aria-label',`第 ${id} 封信：${status==='correct'?'已完成':status==='incorrect'?'答错，下趟再试':missed?'本趟错过':'未完成'}`);
        });
        const sound=this.el('[data-action="sound"]');sound.textContent=this.settings.sfx?'♪':'♪×';sound.setAttribute('aria-pressed',String(this.settings.sfx));sound.setAttribute('aria-label',this.settings.sfx?'关闭音效':'开启音效');
        this.el('#post-lap-label').textContent=this.score()===10?'十封信已盖章，享受海风吧。':`本趟错过 ${this.run.missed.length} 封 · 进度跨趟保留`;
    }
    toast(message,seconds=3) {this.el('#post-toast').textContent=message;this.toastExpiry=this.time+seconds;this.el('#post-toast').classList.add('visible');}
    saveSettings() {
        GameState.load().settings=this.settings;GameState.save();
        if(!this.settings.sfx && !this.settings.music) this.stopVoices();
        this.ensureAudio();this.hudKey='';this.updateHUD();
    }
    prepareAudio() {
        if(!(this.settings.music||this.settings.sfx)) return;
        try {
            // Prepare the device before gameplay, while the introduction is loading.
            this.audio ||= new (window.AudioContext||window.webkitAudioContext)();
            return this.audio;
        } catch { /* Audio is optional; a failed device must not block the game. */ }
    }
    ensureAudio() {
        try { return this.prepareAudio()?.resume().catch(()=>{}); } catch { /* Audio is optional. */ }
    }
    note(freq,length=.1,volume=.035,delay=0,endFreq=freq,type='triangle') {
        if(!this.audio || this.audio.state!=='running') return;
        const o=this.audio.createOscillator(),g=this.audio.createGain(),t=this.audio.currentTime+delay;
        o.type=type;o.frequency.setValueAtTime(freq,t);o.frequency.exponentialRampToValueAtTime(endFreq,t+length);
        g.gain.setValueAtTime(.0001,t);g.gain.linearRampToValueAtTime(volume,t+.008);g.gain.exponentialRampToValueAtTime(.0001,t+length);
        o.connect(g);g.connect(this.audio.destination);this.voices.add(o);
        o.onended=()=>{this.voices.delete(o);o.disconnect();g.disconnect();};o.start(t);o.stop(t+length+.01);
    }
    stopVoices() {for(const voice of this.voices) {try {voice.stop();} catch {}} this.voices.clear();}
    playSound(kind) {
        if(!this.settings.sfx) return;const ready=this.ensureAudio();
        const sounds={
            start:[[330,.12,.03,0],[440,.12,.03,.11],[660,.18,.025,.22]],
            coin:[[880,.12,.025,0,1175]],
            dock:[[180,.22,.03,0,85],[660,.16,.03,.24]],
            letter:[[920,.055,.012,0,500],[760,.06,.012,.07,420]],
            correct:[[660,.13,.035,0],[880,.22,.03,.11]],
            wrong:[[294,.16,.025,0],[247,.2,.02,.13]],
            bump:[[100,.13,.03,0,65]],
            finish:[[523,.15,.03,0],[659,.15,.03,.14],[784,.15,.03,.28],[1047,.32,.025,.42]]
        };
        const play=()=>{if(this.alive && this.settings.sfx && !['paused','handbook'].includes(this.phase)) for(const args of sounds[kind] || []) this.note(...args);};
        if(this.audio?.state==='running') play();else ready?.then(play);
    }
    modal(html,label) {
        this.closeModal(); this.clearInput();
        const d=document.createElement('dialog'); d.className='postal-dialog'; d.setAttribute('aria-label',label);
        d.innerHTML=`<div class="post-dialog-paper">${html}</div>`;
        document.body.appendChild(d); this.dialog=d;
        d.addEventListener('cancel',e=>e.preventDefault(),{signal:this.signal}); d.showModal();
        return d;
    }
    closeModal() { if(this.dialog) { this.dialog.close(); this.dialog.remove(); this.dialog=null; } }
    openQuestion(index) {
        const question=this.data.questions[index];
        if(!this.alive || this.run.settled || !question || (this.mastery[question.id]?.correct && !this.run.answers[question.id])) return;
        this.run.questionId=question.id;this.run.docking=null;this.playSound('letter');
        this.phase='quiz'; this.questionIndex=index; this.clearInput(); this.save();
        const q=this.data.questions[index], a=this.run.answers[q.id];
        const d=this.modal(`<div class="post-letter-head"><span>${this.levelId===2?'CASTLE POST':'ISLAND POST'} · 知识来信</span><b>${String(index+1).padStart(2,'0')} / 10</b></div><p class="post-safe">${this.levelId===2?'已在知识箱旁站稳':'邮车已安全靠站'} · 阅读不限时</p><h2 class="post-question"></h2><div class="post-options"></div><div class="post-answer" aria-live="polite"></div><button class="post-text-button" data-quit>保存并回地图</button>`,'第 '+(index+1)+' 封知识来信');
        d.querySelector('.post-question').textContent=q.question;
        q.options.forEach((text,i)=>{
            const b=document.createElement('button'); b.className='post-option'; b.dataset.option=i;
            const letter=document.createElement('span'); letter.textContent=String.fromCharCode(65+i);
            const content=document.createElement('span'); content.textContent=text; b.append(letter,content);
            b.addEventListener('click',()=>this.answer(q,i,d),{signal:this.signal}); d.querySelector('.post-options').appendChild(b);
        });
        d.querySelector('[data-quit]').addEventListener('click',()=>GameApp.showMap(),{signal:this.signal});
        if(a) this.renderAnswer(q,a,d);
        const heading=d.querySelector('.post-question');heading.tabIndex=-1;heading.focus({preventScroll:true});
        d.addEventListener('keydown',event=>{if(event.target===heading && ['Space','ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.code)) event.preventDefault();},{signal:this.signal});
    }
    answer(q,selected,d) {
        if(!this.alive || this.phase!=='quiz' || this.dialog!==d || this.data.questions[this.questionIndex]?.id!==q.id || this.run.answers[q.id]) return;
        const correct=selected===q.answer;
        this.run.answers[q.id]={selected,correct,continued:false};
        const wasComplete=this.mastery[q.id]?.correct;
        if(!wasComplete) this.mastery[q.id]={selected,correct};
        if(correct && !wasComplete) GameState.awardRun(this.run,10);else this.save();
        this.save(); this.playSound(correct?'correct':'wrong'); this.updateHUD(); this.renderAnswer(q,this.run.answers[q.id],d);
    }
    renderAnswer(q,a,d) {
        d.querySelectorAll('[data-option]').forEach(b=>{
            const idx=Number(b.dataset.option); b.disabled=true;
            if(idx===q.answer) b.classList.add('correct');
            if(idx===a.selected && !a.correct) b.classList.add('incorrect');
            if(idx===q.answer) b.setAttribute('aria-label','正确选项：'+q.options[idx]);
        });
        const area=d.querySelector('.post-answer'); area.replaceChildren();
        const title=document.createElement('strong'); title.textContent=a.correct?'✓ 答对了，信件盖上了邮戳。':'◆ 已收到。正确答案：'+q.options[q.answer];
        const explanation=document.createElement('p'); explanation.textContent=q.explanation;
        area.append(title,explanation);
        if(this.score()===5 && a.correct) {
            const note=document.createElement('p'); note.className='post-unlock-note'; note.textContent=`累计完成 5 封信，已达到${this.rewardName}的解锁条件！到终点见。`; area.appendChild(note);
        }
        const button=document.createElement('button'); button.className='post-btn primary post-continue'; button.textContent='读完了，继续送信 →';
        button.addEventListener('click',()=>{
            if(!this.alive || this.dialog!==d || a.continued) return;
            this.continueQuestion(q,a);
        },{signal:this.signal}); area.appendChild(button);
        requestAnimationFrame(()=>{if(this.dialog===d) button.scrollIntoView({block:'nearest',behavior:'instant'});});
    }
    continueQuestion(q,a) {
        a.continued=true;this.run.questionId=null;
        if(!this.run.passed.includes(q.id)) this.run.passed.push(q.id);
        this.run.offset=PostalArt.stations.find(p=>p.id===q.id).side*40;
        this.save();this.closeModal();this.clearInput();
        this.phase='driving';this.cooldown=1.5;
        this.canvas.focus({preventScroll:true});this.toast(a.correct?'邮戳已盖好，这封信下趟也不用重答。':'信件先收好，下一趟再试这道题。',2.5);
    }
    resume() {
        if(!this.alive) return; this.closeModal(); this.clearInput();
        this.phase=this.run.docking?'docking':'driving';this.cooldown=1.5;this.hitDelay=0;this.canvas.focus({preventScroll:true});
    }
    pause(message='小邮车在这里等你。') {
        if(!['driving','docking'].includes(this.phase)) return;
        this.phase='paused';this.stopVoices();this.clearInput();this.save();
        const d=this.modal(`<small>TAKE A LITTLE BREAK</small><h2>在海边歇一会儿</h2><p>${message}</p><div class="post-settings"><label><input type="checkbox" data-setting="sfx"> 停车、收信与答题音效</label><label><input type="checkbox" data-setting="music"> 轻音乐</label><label><input type="checkbox" data-setting="reduced"> 减少动态效果</label></div><button class="post-btn primary" data-resume>继续旅程 →</button><button class="post-btn" data-handbook>知识手册</button><button class="post-text-button" data-quit>保存并回地图</button>`,'暂停与设置');
        d.querySelectorAll('[data-setting]').forEach(input=>{
            input.checked=!!this.settings[input.dataset.setting];
            input.addEventListener('change',()=>{
                this.settings[input.dataset.setting]=input.checked;
                this.saveSettings();
            },{signal:this.signal});
        });
        d.querySelector('[data-resume]').addEventListener('click',()=>this.resume(),{signal:this.signal});
        d.querySelector('[data-handbook]').addEventListener('click',()=>this.handbook(),{signal:this.signal});
        d.querySelector('[data-quit]').addEventListener('click',()=>GameApp.showMap(),{signal:this.signal});
    }
    handbook(fromResult=false) {
        if(!this.alive) return;
        const wasIntro=this.phase==='intro';this.stopVoices();this.phase='handbook'; this.clearInput(); this.save();
        const d=this.modal('<small>LETTERS WE HAVE READ</small><h2>海岛知识手册</h2><div class="post-notebook"></div><button class="post-btn primary" data-return>返回旅程</button>','知识手册');
        const area=d.querySelector('.post-notebook');
        for(const q of this.data.questions) {
            const a=this.mastery[q.id];if(!a) continue;
            const item=document.createElement('details'), summary=document.createElement('summary'), answer=document.createElement('p'),explain=document.createElement('p');
            summary.textContent=`${a.correct?'✓':'◆'} ${q.id}. ${q.question}`;
            answer.textContent='正确答案：'+q.options[q.answer]; explain.textContent=q.explanation;
            item.append(summary,answer,explain); area.appendChild(item);
        }
        if(!area.children.length) area.textContent='送达第一封信后，题目与解析会收进这里。';
        d.querySelector('[data-return]').textContent=fromResult?'回到旅程纪念页':'返回旅程';
        d.querySelector('[data-return]').addEventListener('click',()=>{
            this.closeModal();
            if(fromResult) this.renderResult(); else if(wasIntro) this.showIntro(); else this.resume();
        },{signal:this.signal});
    }
    finish() {
        if(!this.alive || this.run.settled || !this.run.routeComplete) return;
        this.clearInput(); this.run.newCat=!GameState.isCatUnlocked(this.levelId);
        GameState.setLevelResult(this.levelId,this.score(),{routeComplete:true,runId:this.run.id});
        this.run.settled=true;this.playSound('finish');this.save();this.renderResult();
    }
    renderResult() {
        this.phase='result'; this.closeModal(); this.clearInput();
        this.root.classList.add('post-started');
        const score=this.score(), cat=window._gameData.cats.find(c=>c.level===this.levelId);
        const d=this.modal(`<small>DELIVERED WITH A LITTLE PAW</small><h2>${score===10?'十封信，十枚完美邮戳。':'这一趟，先送到这里。'}</h2><p>本趟答对 ${this.roundScore()} 题，错过 ${this.run.missed.length} 个邮筒。已经答对的信件会一直保留。</p><div class="post-result-score"><b>${score}<small> / 10 已完成</small></b><span>本轮获得 ${this.run.coins} 金币<br><small>已保存到旅行钱包</small></span></div><div class="post-cat-result"></div><div class="post-result-actions"><button class="post-btn primary" data-drive-again>${score===10?'再兜一圈海岛':`下一趟，继续找剩下 ${10-score} 封信`}</button><button class="post-btn" data-map>回旅行地图</button><button class="post-btn" data-notes>知识手册</button></div>`,'环岛邮路结算');
        const reward=d.querySelector('.post-cat-result');
        if(GameState.isCatUnlocked(this.levelId) && cat) {
            const img=document.createElement('img'); img.src='/static/sprites/cats/'+cat.sprite; img.alt=cat.name_zh;
            const title=document.createElement('h3'); title.textContent=`${score===10?'♛ ':''}${this.run.newCat?'认识新朋友':'再次拜访'} · ${cat.name_zh} / ${cat.name}`;
            const desc=document.createElement('p'); desc.textContent=`${cat.breed_zh} · ${cat.personality_zh} · 稀有度 ${cat.rarity}/5`;
            const quote=document.createElement('blockquote'); quote.textContent=cat.quote;
            reward.append(img,title,desc,quote);
        } else {
            reward.textContent=`累计完成 ${score}/10 封信。下一趟补做漏题和错题，累计答对 5 题就能认识${this.rewardName}。`;
        }
        d.querySelector('[data-map]').addEventListener('click',()=>GameApp.showMap(),{signal:this.signal});
        d.querySelector('[data-notes]').addEventListener('click',()=>this.handbook(true),{signal:this.signal});
        d.querySelector('[data-drive-again]').addEventListener('click',()=>this.restart(),{signal:this.signal});
    }
    restart() {
        this.closeModal();this.run=this.newRun();this.hitDelay=0;this.toastExpiry=0; this.hudKey=''; this.time=0; this.save();
        this.phase='intro'; this.showIntro(); this.start();
    }
    destroy() {
        if(!this.alive) return;
        this.save(); this.alive=false; cancelAnimationFrame(this.frame); this.clearInput();
        this.controller.abort();this.stopVoices();this.closeModal();this.audio?.close().catch(()=>{});
        this.root.replaceChildren();
        document.body.classList.remove('postal-playing');
    }
}
