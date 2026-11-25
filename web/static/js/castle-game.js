/* The castle owns platform physics; save, quiz, rewards and audio share the accepted postal flow. */
const CastleGame={
 scene:null,
 async launch(token){
  const root=document.getElementById('view-castle');
  root.innerHTML='<div class="postal-loading" role="status">正在点亮城堡的窗灯…</div>';
  try{
   const data=await fetchJSON('/api/game/levels/2');if(token!==GameApp._loadId)return;
   if(data.level!==2||data.questions?.length!==10||data.questions.some(q=>!q.id||q.options?.length!==4||!Number.isInteger(q.answer)||q.answer<0||q.answer>3))throw Error('Invalid castle questions');
   this.scene=new CastleScene(root,data);
  }catch(error){
   if(token!==GameApp._loadId)return;
   console.error('Castle load failed:',error);
   root.innerHTML='<div class="postal-loading"><p>城堡暂时没有打开，请重新加载。</p><button class="post-btn primary" data-castle-retry>重试</button> <button class="post-btn" data-castle-map>回地图</button></div>';
   root.querySelector('[data-castle-retry]').onclick=()=>GameApp.startLevel(2);root.querySelector('[data-castle-map]').onclick=()=>GameApp.showMap();
  }
 },
 destroy(){this.scene?.destroy();this.scene=null;}
};
class CastleScene extends PostalScene {
 newRun(){return {id:globalThis.crypto?.randomUUID?.()||`${Date.now()}-${Math.random()}`,levelId:2,mode:'castle',x:70,y:326,vx:0,vy:0,facing:1,grounded:true,camera:0,checkpoint:{id:'start',x:70,y:326},answers:{},passed:[],missed:[],collected:[],coins:0,falls:0,gapFalls:{},bridges:[],routeComplete:false,settled:false,questionId:null};}
 validRun(r){
  if(!r||r.levelId!==2||r.mode!=='castle'||typeof r.id!=='string'||!r.answers||typeof r.answers!=='object'||!Array.isArray(r.collected)||!Number.isFinite(r.coins))return null;
  for(const [id,a]of Object.entries(r.answers)){const q=this.data.questions.find(q=>String(q.id)===id);if(!q||!a||!Number.isInteger(a.selected)||a.selected<0||a.selected>3||a.correct!==(a.selected===q.answer))return null;}
  r.x=Math.max(24,Math.min(CastleArt.length-24,Number(r.x)||70));r.y=Number.isFinite(r.y)?r.y:326;
  const checkpoint=CastleArt.benches.find(b=>b.id===r.checkpoint?.id)||CastleArt.boxes.find(b=>'box-'+b.id===r.checkpoint?.id);
  r.checkpoint=checkpoint?{id:r.checkpoint.id,x:checkpoint.x,y:checkpoint.y}:{id:'start',x:70,y:326};
  if(!r.grounded||r.y>400){r.x=r.checkpoint.x;r.y=r.checkpoint.y;}
  r.vx=0;r.vy=0;r.grounded=true;r.camera=Math.max(0,r.x-180);r.facing=r.facing===-1?-1:1;
  for(const key of ['passed','missed','bridges'])if(!Array.isArray(r[key]))r[key]=[];
  r.bridges=r.bridges.filter(id=>CastleArt.gaps.some(g=>g.id===id));r.gapFalls=r.gapFalls&&typeof r.gapFalls==='object'?r.gapFalls:{};r.falls=Math.max(0,Number(r.falls)||0);
  if(!this.data.questions.some(q=>q.id===r.questionId))r.questionId=this.data.questions.find(q=>r.answers[q.id]&&!r.answers[q.id].continued)?.id||null;
  return r;
 }
 renderShell(){
  super.renderShell();document.body.classList.add('castle-playing');
  this.el('.postal-app').setAttribute('aria-label','加密城堡送信记');this.el('.postal-top small').textContent='CASTLE POST · 02';this.el('.postal-top h2').textContent='加密城堡送信记';
  this.canvas.setAttribute('aria-label','左右键或 A、D 移动，空格、上键或 W 跳跃，E 打开附近知识箱，P 暂停。掉下后回到最近长椅，已答题目保留。');
  this.el('.post-guide>small').textContent='一封寄往塔顶的邀请函';this.el('.post-guide h3').textContent='把信送进暖灯书房';
  this.el('.guide-arrow').textContent='↗';this.el('.guide-parking').textContent='↑';
  this.el('.post-guide p').innerHTML='左右移动，按<b>空格／↑</b>跳过缺口；碰到带 <b>!</b> 的知识箱，站稳后读信。';
  this.el('.post-guide-small').textContent='落下回最近长椅，进度都在。答对的箱子下次也不用重答。';
  this.el('.post-key-hint').textContent='← → / A D 移动 · 空格 / ↑ 跳跃 · E 开箱';
  this.el('.postal-controls').innerHTML='<div class="castle-direction"><button class="post-drive" data-drive="left" aria-label="向左移动">←</button><button class="post-drive" data-drive="right" aria-label="向右移动">→</button></div><button class="post-drive post-jump" data-drive="jump" aria-label="跳跃">跳跃 ↑</button>';
 }
 resizeCanvas(){super.resizeCanvas();this.run.camera=Math.max(0,Math.min(CastleArt.length-this.canvas.width,this.run.x-this.canvas.width*.35));}
 showIntro(){super.showIntro();this.el('[data-action="start"]').textContent=this.run.x>100?'继续城堡旅程 →':'带上信封，进城堡 →';}
 start(){
  if(!this.alive||this.phase!=='intro')return;
  this.phase='playing';this.root.classList.add('post-started');this.clearInput();this.cooldown=1.5;this.ensureAudio();this.playSound('start');this.save();this.canvas.focus({preventScroll:true});
  if(this.run.questionId)this.openQuestion(this.data.questions.findIndex(q=>q.id===this.run.questionId));
  else this.toast('左右移动，空格／↑ 跳跃。寻找带 ! 的知识箱。',4);
 }
 bindInput(){
  this.jumpBuffer=0;this.coyote=.12;this.recoverTimer=0;
  const dirs=['ArrowLeft','KeyA','ArrowRight','KeyD'],jump=['Space','ArrowUp','KeyW'];
  this.listen(window,'keydown',e=>{
   if(this.phase!=='playing')return;
   if(dirs.includes(e.code)){e.preventDefault();this.keys.add(e.code);}
   if(jump.includes(e.code)){e.preventDefault();if(!e.repeat)this.jumpBuffer=.12;}
   if(e.code==='KeyE'&&!e.repeat){e.preventDefault();this.tryBox();}
   if(['KeyP','Escape'].includes(e.code)&&!e.repeat){e.preventDefault();this.pause();}
  });
  this.listen(window,'keyup',e=>this.keys.delete(e.code));
  this.root.querySelectorAll('[data-drive]').forEach(b=>{
   this.listen(b,'pointerdown',e=>{if(this.phase!=='playing')return;e.preventDefault();b.setPointerCapture(e.pointerId);this.pointers.set(e.pointerId,b.dataset.drive);b.classList.add('pressed');if(b.dataset.drive==='jump')this.jumpBuffer=.12;});
   const release=e=>{this.pointers.delete(e.pointerId);b.classList.remove('pressed');};
   for(const type of ['pointerup','pointercancel','lostpointercapture'])this.listen(b,type,release);
  });
  this.listen(window,'resize',()=>this.resizeCanvas());
  const hide=()=>{this.clearInput();this.stopVoices();if(['playing','recovering'].includes(this.phase))this.pause('小猫在长椅旁等你，准备好再出发。');};
  this.listen(window,'blur',hide);this.listen(document,'visibilitychange',()=>{if(document.hidden)hide();});this.listen(window,'pagehide',()=>this.save());
 }
 clearInput(){super.clearInput();this.jumpBuffer=0;}
 input(action){if(action==='left')return this.keys.has('ArrowLeft')||this.keys.has('KeyA')||[...this.pointers.values()].includes('left');if(action==='right')return this.keys.has('ArrowRight')||this.keys.has('KeyD')||[...this.pointers.values()].includes('right');return [...this.pointers.values()].includes(action);}
 solids(){return CastleArt.platforms.concat(CastleArt.gaps.filter(g=>this.run.bridges.includes(g.id)).map(g=>({x:g.x,y:g.y,w:g.w,h:14,ground:true})));}
 tick(now){
  if(!this.alive)return;
  const dt=Math.min(.04,Math.max(0,(now-(this.lastFrame||now))/1000));this.lastFrame=now;
  if(['playing','recovering'].includes(this.phase)){
   this.time+=dt;this.cooldown=Math.max(0,this.cooldown-dt);this.update(dt);
   if(now-this.lastSave>1200){this.lastSave=now;this.save();}
   if(this.settings.music){const step=Math.floor(this.time*1.6);if(step!==this.soundStep){this.soundStep=step;this.note([294,349,440,349,262,330,392,330][step%8],.25,.018);}}
  }
  CastleArt.draw(this.ctx,this.run,this.mastery,this.settings.reduced?0:this.time);
  if(this.toastExpiry&&this.time>this.toastExpiry)this.el('#post-toast').classList.remove('visible');this.updateHUD();this.frame=requestAnimationFrame(this.tick);
 }
 update(dt){
  this.stepPhysics(dt);
  const target=Math.max(0,Math.min(CastleArt.length-this.canvas.width,this.run.x-this.canvas.width*.35));
  this.run.camera+=(target-this.run.camera)*Math.min(1,dt*6);
 }
 stepPhysics(dt){
  const r=this.run;
  if(this.phase==='recovering'){this.recoverTimer-=dt;if(this.recoverTimer<=0)this.phase='playing';return;}
  if(this.phase!=='playing')return;
  this.jumpBuffer=Math.max(0,this.jumpBuffer-dt);this.coyote=r.grounded?.12:Math.max(0,this.coyote-dt);r.landing=Math.max(0,(r.landing||0)-dt);
  const dir=Number(this.input('right'))-Number(this.input('left'));r.vx=dir*140;if(dir)r.facing=dir;
  if(this.jumpBuffer>0&&this.coyote>0){r.vy=-370;r.grounded=false;this.coyote=0;this.jumpBuffer=0;this.noteJump();}
  const solids=this.solids(),nextX=Math.max(24,Math.min(CastleArt.length-24,r.x+r.vx*dt));
  if(r.grounded){const step=solids.find(p=>p.ground&&nextX+9>p.x&&nextX-9<p.x+p.w&&Math.abs(r.y-p.y)<=20);if(step)r.y=step.y;}
  r.x=nextX;const beforeY=r.y,wasGrounded=r.grounded;r.vy=Math.min(680,r.vy+1000*dt);let nextY=r.y+r.vy*dt;r.grounded=false;
  if(r.vy<0){for(const p of solids){if(!p.ground&&r.x+10>p.x&&r.x-10<p.x+p.w&&beforeY-34>=p.y+p.h&&nextY-34<=p.y+p.h){nextY=p.y+p.h+34;r.vy=0;}}}
  if(r.vy>=0){const landing=solids.filter(p=>r.x+9>p.x&&r.x-9<p.x+p.w&&beforeY<=p.y+1&&nextY>=p.y).sort((a,b)=>a.y-b.y)[0];if(landing){nextY=landing.y;r.vy=0;r.grounded=true;if(!wasGrounded){r.landing=.09;if(this.settings.sfx)this.note(155,.045,.012);}}}
  r.y=nextY;
  if(r.y>480){this.recover();return;}
  for(const c of CastleArt.coins){if(r.collected.includes(c.id)||Math.abs(r.x-c.x)>17||c.y<r.y-40||c.y>r.y+4)continue;r.collected.push(c.id);GameState.awardRun(r,5);this.playSound('coin');this.toast('+5 金币',1);}
  if(r.grounded){
   for(const b of CastleArt.benches){if(Math.abs(r.x-b.x)<18&&Math.abs(r.y-b.y)<3&&r.checkpoint.id!==b.id){r.checkpoint={...b};this.save();}}
   if(this.tryBox())return;
   if(r.x>=CastleArt.length-95){r.routeComplete=true;this.finish();return;}
  }
  for(const b of CastleArt.boxes){if(r.x>b.x+40&&!this.mastery[b.id]?.correct&&!r.answers[b.id]&&!r.missed.includes(b.id))r.missed.push(b.id);}
 }
 noteJump(){if(this.settings.sfx){this.ensureAudio();this.note(240,.13,.025,0,520);}}
 tryBox(){
  if(this.phase!=='playing'||!this.run.grounded)return false;
  const b=CastleArt.boxes.find(b=>!this.mastery[b.id]?.correct&&!this.run.answers[b.id]&&Math.abs(this.run.x-b.x)<=28&&Math.abs(this.run.y-b.y)<4);
  if(!b)return false;
  this.run.vx=0;this.run.vy=0;this.run.checkpoint={id:'box-'+b.id,x:b.x,y:b.y};this.run.missed=this.run.missed.filter(id=>id!==b.id);this.openQuestion(b.id-1);return true;
 }
 recover(){
  const r=this.run,g=CastleArt.gaps.reduce((a,b)=>Math.abs(r.x-b.x-b.w/2)<Math.abs(r.x-a.x-a.w/2)?b:a);
  r.falls++;r.gapFalls[g.id]=(r.gapFalls[g.id]||0)+1;
  if(r.gapFalls[g.id]>=2&&!r.bridges.includes(g.id)){r.bridges.push(g.id);this.toast('软垫桥已经铺好，这次直接走过去就行。',4);}else this.toast('回到最近的长椅，信件和金币都在。',3);
  r.x=r.checkpoint.x;r.y=r.checkpoint.y;r.vx=0;r.vy=0;r.grounded=true;r.camera=Math.max(0,Math.min(CastleArt.length-this.canvas.width,r.x-this.canvas.width*.35));
  this.clearInput();this.coyote=.12;this.phase='recovering';this.recoverTimer=.5;this.playSound('bump');this.save();
 }
 continueQuestion(q,a){
  a.continued=true;this.run.questionId=null;if(!this.run.passed.includes(q.id))this.run.passed.push(q.id);this.run.vx=0;this.run.vy=0;
  this.save();this.closeModal();this.clearInput();this.phase='playing';this.cooldown=1.5;this.canvas.focus({preventScroll:true});this.toast(a.correct?'知识箱盖上邮戳，下次也不用重答。':'解析已收进手册，下一趟再试这道题。',2.5);
 }
 updateHUD(){super.updateHUD();const z=Math.min(4,Math.floor(this.run.x/1040));this.el('#post-location').textContent=`0${z+1} · ${CastleArt.zones[z]}`;this.el('#post-lap-label').textContent=this.phase==='recovering'?'正在回到安全长椅…':`已到 ${Math.min(100,Math.round(this.run.x/CastleArt.length*100))}% · 落下就近恢复`;}
 resume(){if(!this.alive)return;this.closeModal();this.clearInput();this.phase='playing';this.cooldown=1.5;this.canvas.focus({preventScroll:true});}
 pause(message='把信放稳，歇一会儿。'){
  if(!['playing','recovering'].includes(this.phase))return;
  this.phase='driving';super.pause(message);
  const p=document.createElement('p');p.className='castle-help';p.textContent='左右键／A、D 移动，空格／↑ 跳跃。碰知识箱收信；长椅会记录恢复点。同一缺口落下两次会铺出软垫桥。';this.dialog.querySelector('.post-settings').before(p);
  const label=this.dialog.querySelector('[data-setting="sfx"]').parentElement;label.lastChild.textContent=' 跳跃、拾取与答题音效';
  const help=document.createElement('button');help.className='post-btn';help.textContent='帮我铺好前方的缺口';help.addEventListener('click',()=>{const g=CastleArt.gaps.find(g=>g.x+g.w>this.run.x);if(g&&!this.run.bridges.includes(g.id))this.run.bridges.push(g.id);this.save();this.resume();this.toast('前方软垫桥已经铺好，知识箱仍由你自己打开。',3);},{signal:this.signal});this.dialog.querySelector('[data-handbook]').after(help);
 }
 renderResult(){super.renderResult();const score=this.score();this.dialog.querySelector('h2').textContent=score===10?'邀请函送达，十枚邮戳收齐。':'邀请函已送进暖灯书房。';this.dialog.querySelector('.post-dialog-paper>p').textContent=`本趟新完成 ${this.roundScore()} 道题，累计 ${score}/10。未完成的知识箱可以下次再访。`;this.dialog.querySelector('[data-drive-again]').textContent=score===10?'再去城堡散散步':`再访城堡，补做剩下 ${10-score} 题`;}
 restart(){this.closeModal();this.run=this.newRun();this.hudKey='';this.time=0;this.jumpBuffer=0;this.coyote=.12;this.recoverTimer=0;this.save();this.phase='intro';this.showIntro();this.start();}
 destroy(){super.destroy();document.body.classList.remove('castle-playing');}
}
