/* Eight independent mini-games share the accepted interface, quiz and save lifecycle. */
const AdventureCatalog={
 3:{kind:'mine',title:'比特币矿井升空记',en:'MINE LIFT',theme:'矿灯金与水晶青',goal:'踩着矿板，跳向阳光',guide:'小猫会自动弹跳，用左右键调整落点。落到带 ! 的水晶平台就能读信。',tip:'失足回最近安全平台，已答进度保留。',controls:['left','right'],hint:'← → / A D 调整落点 · 自动弹跳',colors:['#35445f','#77d4cf','#f4cf82']},
 4:{kind:'runner',title:'区块链快递专线',en:'BLOCK EXPRESS',goal:'把快递送到数据总站',guide:'左右换轨，↑／空格跳跃，↓下滑。进入带 ! 的驿站车道后才会停靠读信。',tip:'错过的驿站下一趟补，碰撞只短暂停顿。',controls:['left','right','jump','slide'],hint:'← → 换轨 · ↑ / 空格 跳跃 · ↓ 下滑',colors:['#344b68','#58b7b0','#df9a5f']},
 5:{kind:'garden',title:'以太坊花园漫游',en:'ETHER GARDENS',goal:'沿石径，找到茶会花园',guide:'方向键／WASD 移动，也可点石径寻路。接触带 ! 的书亭收信，走进花门前往下一座花园。',tip:'猫爪提示会标出路线，答错也能继续旅行。',controls:['left','up','down','right'],hint:'方向键 / WASD 移动 · 点石径寻路',colors:['#83a584','#d9a0b5','#88c9cf']},
 6:{kind:'bricks',title:'钱包安全修复站',en:'WALLET WORKSHOP',goal:'用小盾牌找回安全手册',guide:'左右移动挡板，空格／发球按钮放出小球。击开带 ! 的书页砖后，小球会停稳，打开知识手册。',tip:'漏球不限次数；提示可瞄准剩余知识砖，再由你发球。',controls:['left','launch','right'],hint:'← → / 拖动 挡板 · 空格 发球',colors:['#526c81','#6bc5ba','#caa35d']},
 7:{kind:'dig',title:'交易所码头旧仓寻宝',en:'DOCK TREASURE',goal:'挖开沙土，找回旧仓手册',guide:'点相邻格子挖掘，或方向键选择、空格落铲。浅沙一铲，木板两铲；走到栈道出口前往下一仓间。',tip:'没有挖掘次数限制。纸标签和提示会指出知识货箱。',controls:['left','up','dig','down','right'],hint:'方向键 选格 · 空格 挖掘 · 点相邻格',colors:['#9f7855','#477e7a','#e9bc73']},
 8:{kind:'flight',title:'投资策略星光峡谷',en:'STARLIGHT FLIGHT',goal:'驾驶猫耳艇，飞向观测营地',guide:'方向键微调飞船，手机也可在画面内拖动。靠近带 ! 的铜环浮标后才会靠泊读信。',tip:'不用射击；碰到岩石只轻轻弹开，错过可下趟补。',controls:['left','up','down','right'],hint:'方向键 / WASD 飞行 · 手机可拖动',colors:['#273657','#73b9bf','#f1d9a0']},
 9:{kind:'claw',title:'市场风云塔观测室',en:'SKY OBSERVATORY',goal:'用钩爪取回知识胶囊',guide:'钩爪缓慢摆动，空格／放钩按钮向下抓取；左右键也可手动调整。胶囊拉回软垫后才会打开题目。',tip:'抓空不扣金币；提示可帮你对准，仍由你确认放钩。',controls:['left','grab','right'],hint:'← → 手动横移 · 空格 放钩',colors:['#61506b','#7fc8c6','#bc9459']},
 10:{kind:'puzzle',title:'ICO 冒险终点档案馆',en:'FINAL ARCHIVES',goal:'打开五间机关房，走进合影厅',guide:'方向键移动，空格／交互操作机关。推箱、转镜、踩编号、用传送垫、调开关，每间只解一个小机关。',tip:'可以撤销、复位或看三级提示，已答题目始终保留。',controls:['left','up','interact','down','right'],hint:'方向键 移动 · 空格 交互 · 撤销 / 复位',colors:['#36565a','#d8b66e','#957257']}
};
const AdventureGames={scene:null,async launch(level,token){
 const root=document.getElementById('view-adventure');root.innerHTML='<div class="postal-loading" role="status">正在准备这段猫咪旅程…</div>';
 try{const data=await fetchJSON('/api/game/levels/'+level);if(token!==GameApp._loadId)return;
  if(data.level!==level||data.questions?.length!==10||data.questions.some(q=>!q.id||q.options?.length!==4||!Number.isInteger(q.answer)||q.answer<0||q.answer>3))throw Error('Invalid questions');
  this.scene=new AdventureScene(root,data);
 }catch(error){if(token!==GameApp._loadId)return;console.error('Adventure load:',error);root.innerHTML='<div class="postal-loading"><p>这段旅程暂时没有准备好。</p><button class="post-btn primary" data-retry>重试</button> <button class="post-btn" data-map>回地图</button></div>';root.querySelector('[data-retry]').onclick=()=>GameApp.startLevel(level);root.querySelector('[data-map]').onclick=()=>GameApp.showMap();}
},destroy(){this.scene?.destroy();this.scene=null;}};
class AdventureScene extends PostalScene{
 get meta(){return AdventureCatalog[this.levelId];} get game(){return AdventureWorlds[this.meta.kind];}
 newRun(){return {id:globalThis.crypto?.randomUUID?.()||`${Date.now()}-${Math.random()}`,levelId:this.levelId,mode:'adventure-'+this.meta.kind,world:this.game.init(),answers:{},passed:[],missed:[],collected:[],coins:0,routeComplete:false,settled:false,questionId:null};}
 validRun(r){if(!r||r.levelId!==this.levelId||r.mode!=='adventure-'+this.meta.kind||typeof r.id!=='string'||!r.world||typeof r.world!=='object'||!r.answers||!Array.isArray(r.collected)||!Number.isFinite(r.coins))return null;
  for(const [id,a]of Object.entries(r.answers)){const q=this.data.questions.find(q=>String(q.id)===id);if(!q||!a||!Number.isInteger(a.selected)||a.selected<0||a.selected>3||a.correct!==(a.selected===q.answer))return null;}
  r.passed=Array.isArray(r.passed)?r.passed:[];r.missed=Array.isArray(r.missed)?r.missed:[];if(!this.data.questions.some(q=>q.id===r.questionId))r.questionId=null;
  if(!r.questionId)r.questionId=this.data.questions.find(q=>r.answers[q.id]&&!r.answers[q.id].continued)?.id||null;
  if(this.game.validate&&!this.game.validate(r.world))return null;return r;
 }
 renderShell(){super.renderShell();document.body.classList.add('adventure-playing');this.root.dataset.kind=this.meta.kind;
  this.el('.postal-app').setAttribute('aria-label',this.meta.title);this.el('.postal-app').style.setProperty('--game-accent',this.meta.colors[0]);
  this.el('.postal-top h2').textContent=this.meta.title;this.el('.postal-top small').textContent=this.meta.en+' · '+this.levelId.toString().padStart(2,'0');
  this.canvas.setAttribute('aria-label',this.meta.title+'。'+this.meta.guide);this.el('.post-guide>small').textContent='第 '+this.levelId+' 站 · 新的猫咪差事';this.el('.post-guide h3').textContent=this.meta.goal;
  this.el('.post-guide p').textContent=this.meta.guide;this.el('.post-guide-small').textContent=this.meta.tip;this.el('.guide-parking').textContent='!';this.el('.guide-mailbox').textContent='✓';this.el('.post-key-hint').textContent=this.meta.hint;
  const labels={left:'←',right:'→',up:'↑',down:'↓',jump:'跳跃 ↑',slide:'下滑 ↓',launch:'发球',grab:'放钩',dig:'挖掘',interact:'交互'};
  this.el('.postal-controls').innerHTML=this.meta.controls.map(action=>`<button class="post-drive adventure-control" data-drive="${action}" aria-label="${labels[action]}">${labels[action]}</button>`).join('');
  const tools=document.createElement('div');tools.className='adventure-tools';tools.innerHTML='<button class="post-btn" data-help>猫爪提示</button>'+(this.levelId===10?'<button class="post-btn" data-undo>撤销一步</button><button class="post-btn" data-reset-room>本房复位</button>':'');this.el('.post-drive-bar').after(tools);
  this.listen(tools,'click',e=>{if(this.phase!=='playing')return;if(e.target.closest('[data-help]'))this.game.help?.(this);if(e.target.closest('[data-undo]'))this.game.action?.(this,'undo');if(e.target.closest('[data-reset-room]'))this.game.action?.(this,'reset');this.save();});
 }
 resizeCanvas(){super.resizeCanvas();}
 showIntro(){super.showIntro();this.el('[data-action="start"]').textContent='准备好了，出发 →';}
 start(){if(!this.alive||this.phase!=='intro')return;this.phase='playing';this.root.classList.add('post-started');this.clearInput();this.ensureAudio();this.playSound('start');this.save();this.canvas.focus({preventScroll:true});if(this.run.questionId)this.openQuestion(this.run.questionId-1);else{this.game.start?.(this);this.toast(this.meta.hint,4);}}
 bindInput(){this.stepClock=0;const directions={ArrowLeft:'left',KeyA:'left',ArrowRight:'right',KeyD:'right',ArrowUp:'up',KeyW:'up',ArrowDown:'down',KeyS:'down'};
  this.listen(window,'keydown',e=>{if(this.phase!=='playing')return;const action=directions[e.code];if(action){e.preventDefault();this.keys.add(e.code);if(!e.repeat)this.game.key?.(this,action);}if(['Space','KeyE'].includes(e.code)){e.preventDefault();if(!e.repeat)this.perform('interact');}if(['KeyP','Escape'].includes(e.code)&&!e.repeat){e.preventDefault();this.pause();}});
  this.listen(window,'keyup',e=>this.keys.delete(e.code));
  this.root.querySelectorAll('[data-drive]').forEach(button=>{this.listen(button,'pointerdown',e=>{if(this.phase!=='playing')return;e.preventDefault();button.setPointerCapture(e.pointerId);this.pointers.set(e.pointerId,button.dataset.drive);button.classList.add('pressed');this.perform(button.dataset.drive);});for(const type of ['pointerup','pointercancel','lostpointercapture'])this.listen(button,type,e=>{this.pointers.delete(e.pointerId);button.classList.remove('pressed');});this.listen(button,'click',e=>{if(e.detail===0)this.perform(button.dataset.drive);});});
  this.listen(this.canvas,'pointerdown',e=>{if(this.phase!=='playing')return;e.preventDefault();this.canvas.setPointerCapture(e.pointerId);this.dragId=e.pointerId;this.dragStart={x:e.clientX,y:e.clientY};this.game.pointer?.(this,this.point(e),'down');});
  this.listen(this.canvas,'pointermove',e=>{if(this.phase==='playing'&&this.dragId===e.pointerId)this.game.pointer?.(this,this.point(e),'move');});
  this.listen(this.canvas,'pointerup',e=>{if(this.phase==='playing'&&this.dragId===e.pointerId){const moved=Math.hypot(e.clientX-this.dragStart.x,e.clientY-this.dragStart.y);this.game.pointer?.(this,this.point(e),moved<8?'tap':'up');}this.dragId=null;});
  this.listen(this.canvas,'pointercancel',()=>{this.dragId=null;this.clearInput();});
  this.listen(window,'resize',()=>this.resizeCanvas());const hide=()=>{this.clearInput();this.stopVoices();if(this.phase==='playing')this.pause('旅程已暂停，回来后主动继续。');};this.listen(window,'blur',hide);this.listen(document,'visibilitychange',()=>{if(document.hidden)hide();});this.listen(window,'pagehide',()=>this.save());
 }
 point(e){const r=this.canvas.getBoundingClientRect();return {x:(e.clientX-r.x)*this.canvas.width/r.width-(this.canvas.width-320)/2,y:(e.clientY-r.y)*this.canvas.height/r.height-Math.round((this.canvas.height-400)/2)};}
 input(a){const codes={left:['ArrowLeft','KeyA'],right:['ArrowRight','KeyD'],up:['ArrowUp','KeyW'],down:['ArrowDown','KeyS']};return (codes[a]||[]).some(k=>this.keys.has(k))||[...this.pointers.values()].includes(a);}
 clearInput(){super.clearInput();this.dragId=null;}
 perform(action){if(this.phase!=='playing')return;this.game.action?.(this,action);this.save();}
 available(id){return !this.mastery[id]?.correct&&!this.run.answers[id];}
 attempted(id){return !!this.mastery[id]?.correct||!!this.run.answers[id];}
 interact(id){if(this.phase!=='playing'||!this.available(id))return false;this.openQuestion(id-1);return true;}
 openQuestion(index){super.openQuestion(index);if(this.dialog){this.dialog.querySelector('.post-letter-head span').textContent=this.meta.en+' · 知识来信';this.dialog.querySelector('.post-safe').textContent='已在安全处停下 · 阅读不限时';}}
 award(id,amount=5){if(this.run.collected.includes(id))return;this.run.collected.push(id);GameState.awardRun(this.run,amount);this.playSound('coin');}
 update(dt){if(this.phase!=='playing')return;this.game.update?.(this,Math.min(dt,.04));}
 tick(now){if(!this.alive)return;const dt=Math.min(.04,Math.max(0,(now-(this.lastFrame||now))/1000));this.lastFrame=now;if(this.phase==='playing'){this.time+=dt;this.cooldown=Math.max(0,this.cooldown-dt);this.update(dt);if(now-this.lastSave>1200){this.lastSave=now;this.save();}if(this.settings.music){const step=Math.floor(this.time*1.5);if(step!==this.soundStep){this.soundStep=step;this.note([262,330,392,440,349,330][step%6],.22,.015);}}}AdventureArt.draw(this.ctx,this,this.settings.reduced?0:this.time);if(this.toastExpiry&&this.time>this.toastExpiry)this.el('#post-toast').classList.remove('visible');this.updateHUD();this.frame=requestAnimationFrame(this.tick);}
 continueQuestion(q,a){a.continued=true;this.run.questionId=null;if(!this.run.passed.includes(q.id))this.run.passed.push(q.id);this.closeModal();this.clearInput();this.phase='playing';this.cooldown=1.5;this.game.afterQuestion?.(this,q.id);this.save();this.canvas.focus({preventScroll:true});if(this.phase==='playing')this.toast(a.correct?'已盖章，这道题下次也不用重答。':'解析已收入手册，下次再试。',2.5);}
 updateHUD(){super.updateHUD();this.el('#post-location').textContent=this.game.location?.(this)||this.meta.title;this.el('#post-lap-label').textContent=this.game.status?.(this)||'已答对跨局保留 · 随时可暂停';}
 resume(){this.closeModal();this.clearInput();this.phase='playing';this.cooldown=1.5;this.canvas.focus({preventScroll:true});}
 pause(message='歇一会儿，进度都在。'){if(this.phase!=='playing')return;this.phase='driving';super.pause(message);this.dialog.querySelector('h2').textContent='在这里歇一会儿';this.dialog.querySelector('[data-setting="sfx"]').parentElement.lastChild.textContent=' 操作、拾取与答题音效';const p=document.createElement('p');p.className='adventure-pause-guide';p.textContent=this.meta.guide+' '+this.meta.tip;this.dialog.querySelector('.post-settings').before(p);const help=document.createElement('button');help.className='post-btn';help.textContent='猫爪提示';help.onclick=()=>{this.resume();this.game.help?.(this);this.save();};this.dialog.querySelector('[data-handbook]').after(help);}
 complete(){if(this.phase!=='playing')return;this.run.routeComplete=true;this.finish();}
 renderResult(){super.renderResult();this.dialog.querySelector('h2').textContent=this.score()===10?'十枚知识邮戳，全部收齐。':'这一站，留下新的旅行纪念。';this.dialog.querySelector('.post-dialog-paper>p').textContent=`${this.meta.title}已完成。本趟新答对 ${this.roundScore()} 题，累计 ${this.score()}/10。`;this.dialog.querySelector('[data-drive-again]').textContent='再访这一站';if(this.levelId===10){this.dialog.classList.add('adventure-final');const album=document.createElement('div');album.className='final-cat-album';for(const cat of window._gameData.cats.filter(c=>GameState.isCatUnlocked(c.level))){const img=document.createElement('img');img.src='/static/sprites/cats/'+cat.sprite;img.alt=cat.name_zh;album.append(img);}this.dialog.querySelector('.post-cat-result').after(album);}}
 restart(){this.closeModal();this.run=this.newRun();this.hudKey='';this.time=0;this.save();this.phase='intro';this.showIntro();this.start();}
 destroy(){super.destroy();document.body.classList.remove('adventure-playing');this.root.removeAttribute('data-kind');}
}
