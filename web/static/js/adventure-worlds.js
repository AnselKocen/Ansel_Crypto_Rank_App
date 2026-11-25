/* Each object owns a distinct mechanic. State is JSON so checkpoints survive refresh. */
const MiniGrid={
 dirs:{left:[-1,0],right:[1,0],up:[0,-1],down:[0,1]},
 direction(s){return ['left','right','up','down'].find(d=>s.input(d));},
 path(start,end,walk){const q=[{x:start.x,y:start.y,path:[]}],seen=new Set([start.x+','+start.y]);while(q.length){const p=q.shift();if(p.x===end.x&&p.y===end.y)return p.path;for(const [dx,dy]of Object.values(this.dirs)){const x=p.x+dx,y=p.y+dy,k=x+','+y;if(!seen.has(k)&&walk(x,y)){seen.add(k);q.push({x,y,path:[...p.path,{x,y}]});}}}return [];},
 maze(room){const a=Array.from({length:7},()=>Array(9).fill(1)),stack=[{x:1,y:5}];a[5][1]=0;let seed=room+17;const rand=()=>{seed=(seed*16807)%2147483647;return seed/2147483647;};while(stack.length){const p=stack[stack.length-1],next=Object.values(this.dirs).map(([dx,dy])=>({x:p.x+dx*2,y:p.y+dy*2,dx,dy})).filter(n=>n.x>0&&n.x<8&&n.y>0&&n.y<6&&a[n.y][n.x]);if(!next.length){stack.pop();continue;}const n=next[Math.floor(rand()*next.length)];a[n.y][n.x]=0;a[p.y+n.dy][p.x+n.dx]=0;stack.push(n);}return a;},
 roomIds(room){return [room*2+1,room*2+2];},
 nextStage(s,reset){const w=s.run.world;while(w.room<5&&this.roomIds(w.room).every(id=>s.attempted(id))){w.room++;if(w.room<5)reset(w);}if(w.room>=5)s.complete();},
 valid(w,fields){return w&&fields.every(k=>Number.isFinite(w[k]));}
};
const AdventureWorlds={
 mine:{
  init(){return {x:160,y:0,vy:340,camera:0,peak:0,checkpoint:0,falls:0,recover:0,guide:false};},validate:w=>MiniGrid.valid(w,['x','y','vy','camera','peak','checkpoint','falls']),
  platforms(){return Array.from({length:37},(_,i)=>({i,x:i===0?160:160+Math.sin(i*.65)*65,y:i*62,w:i%3===0?138:108,id:i>0&&i%3===0&&i<=30?i/3:null}));},
  update(s,dt){const w=s.run.world;if(w.recover>0){w.recover-=dt;return;}const dir=Number(s.input('right'))-Number(s.input('left'));w.x=Math.max(20,Math.min(300,w.x+dir*130*dt));
   const platforms=this.platforms(),next=platforms.find(p=>p.y>w.y+4);if(w.guide&&next)w.x+=(next.x-w.x)*Math.min(1,dt*4);
   const before=w.y;w.vy-=480*dt;w.y+=w.vy*dt;w.peak=Math.max(w.peak,w.y);w.camera=Math.max(w.camera,w.y-185);
   if(w.vy<=0){const p=platforms.filter(p=>before>=p.y&&w.y<=p.y&&Math.abs(w.x-p.x)<p.w/2+(w.falls>=2?20:0)).sort((a,b)=>b.y-a.y)[0];if(p){w.y=p.y;w.vy=340;if(p.i%3===0)w.checkpoint=Math.max(w.checkpoint,p.i);if(p.i===36){s.complete();return;}if(p.id&&s.available(p.id)){w.vy=0;s.interact(p.id);return;}s.award('mine-landing-'+p.i);}}
   if(w.y<w.camera-95){w.falls++;const p=platforms[w.checkpoint];w.x=p.x;w.y=p.y;w.vy=340;w.camera=Math.max(0,p.y-145);w.recover=.55;s.playSound('bump');s.toast(w.falls>=2?'矿板已加宽，猫爪会标出下一落点。':'安全网把你送回最近的水晶平台，进度都在。',3);}
  },afterQuestion(s){s.run.world.vy=340;},help(s){s.run.world.guide=!s.run.world.guide;s.toast(s.run.world.guide?'猫爪落点引导已开启，题目仍由你亲自作答。':'已关闭落点引导。',3);},location:s=>'矿井 · '+Math.min(100,Math.round(s.run.world.peak/2232*100))+'%',status:s=>'安全平台 '+Math.floor(s.run.world.checkpoint/3)+' · 失足 '+s.run.world.falls+' 次'
 },
 runner:{
  init(){return {distance:0,lane:0,laneX:0,jump:0,slide:0,stumble:0,hits:0,seen:[]};},validate:w=>MiniGrid.valid(w,['distance','lane','laneX','jump','slide','hits'])&&Array.isArray(w.seen),
  stations(){return Array.from({length:10},(_,i)=>({id:i+1,s:250+i*330,lane:i%3-1}));},obstacles(){return Array.from({length:13},(_,i)=>({id:'obstacle-'+i,s:420+i*230,lane:(i+1)%3-1,type:i%2?'beam':'crate'}));},
  action(s,a){const w=s.run.world;if(a==='left'||a==='right')w.lane=Math.max(-1,Math.min(1,w.lane+(a==='left'?-1:1)));if(a==='jump'||a==='interact')w.jump=.85;if(a==='slide')w.slide=.8;},key(s,a){this.action(s,a==='up'?'jump':a==='down'?'slide':a);},
  update(s,dt){const w=s.run.world;w.laneX+=(w.lane-w.laneX)*Math.min(1,dt*12);w.jump=Math.max(0,w.jump-dt);w.slide=Math.max(0,w.slide-dt);if(w.stumble>0){w.stumble-=dt;return;}w.distance+=(w.hits>=2?48:58)*dt;
   for(const p of this.stations()){if(Math.abs(w.distance-p.s)<12&&Math.abs(w.laneX-p.lane)<.3&&s.available(p.id)){w.distance=p.s;s.interact(p.id);return;}if(w.distance>p.s+15&&!s.attempted(p.id)&&!s.run.missed.includes(p.id))s.run.missed.push(p.id);}
   for(const o of this.obstacles()){if(w.seen.includes(o.id)||Math.abs(w.distance-o.s)>14||Math.abs(w.laneX-o.lane)>.3)continue;w.seen.push(o.id);if(o.type==='crate'?w.jump>.15:w.slide>.1)s.award('dodge-'+o.id);else if(!s.cooldown){w.hits++;w.stumble=.45;s.cooldown=1.5;s.playSound('bump');s.toast('轻轻踉跄了一下，快递和金币都在。',2);}}
   if(w.distance>=3600)s.complete();
  },afterQuestion(s){const w=s.run.world;w.distance+=20;w.jump=0;w.slide=0;},help(s){const w=s.run.world,p=this.stations().find(p=>p.s>w.distance&&s.available(p.id));s.toast(p?'下一座知识驿站在'+['左','中','右'][p.lane+1]+'轨，提前换道进入。':'沿着安全轨道前往总站。',4);},location:s=>'快递专线 · '+Math.min(100,Math.round(s.run.world.distance/3600*100))+'%',status:s=>'碰撞不丢进度 · 本趟错过 '+s.run.missed.length+' 站'
 },
 garden:{
  init(){return {room:0,x:1,y:5,move:0,path:[],hint:[]};},validate:w=>MiniGrid.valid(w,['room','x','y','move'])&&w.room>=0&&w.room<5&&Array.isArray(w.path),
  nodes(room){return [{id:room*2+1,x:3,y:5},{id:room*2+2,x:5,y:1}];},walk(w,x,y){return x>=0&&x<9&&y>=0&&y<7&&MiniGrid.maze(w.room)[y][x]===0;},
  update(s,dt){const w=s.run.world;w.move=Math.max(0,w.move-dt);if(w.move>0)return;const a=MiniGrid.direction(s);let next;if(a){w.path=[];const [dx,dy]=MiniGrid.dirs[a];next={x:w.x+dx,y:w.y+dy};}else if(w.path.length)next=w.path.shift();if(!next||!this.walk(w,next.x,next.y))return;w.x=next.x;w.y=next.y;w.move=.13;
   for(const n of this.nodes(w.room))if(w.x===n.x&&w.y===n.y&&s.interact(n.id))return;
   if((w.x===1&&w.y===1)||(w.x===7&&w.y===5))s.award('garden-'+w.room+'-'+w.x+'-'+w.y);
   if(w.x===7&&w.y===1){if(w.room===4)s.complete();else{w.room++;w.x=1;w.y=5;w.path=[];w.hint=[];s.save();s.toast('走进下一座小花园。',2);}}
  },pointer(s,p,type){if(type!=='tap')return;const w=s.run.world,x=Math.floor((p.x-25)/30),y=Math.floor((p.y-110)/30);if(!this.walk(w,x,y)){s.toast('树篱不能穿过，点石径就能寻路。',2);return;}w.path=MiniGrid.path(w,{x,y},(x,y)=>this.walk(w,x,y));},help(s){const w=s.run.world,n=this.nodes(w.room).find(n=>s.available(n.id))||{x:7,y:1};w.hint=MiniGrid.path(w,n,(x,y)=>this.walk(w,x,y));s.toast('猫爪脚印已标出通往'+(n.id?'知识书亭':'花门')+'的石径。',3);},location:s=>['入口喷泉','紫藤花架','玻璃温室','花溪露台','猫咪茶会'][s.run.world.room]+' · '+(s.run.world.room+1)+'/5',status:()=> '点石径寻路 · 猫爪提示可标出路线'
 },
 bricks:{
  init(){return {room:0,paddle:160,ball:{x:160,y:338,vx:80,vy:-150,attached:true},broken:[],misses:0,aim:null};},validate:w=>MiniGrid.valid(w,['room','paddle','misses'])&&w.ball&&MiniGrid.valid(w.ball,['x','y','vx','vy'])&&Array.isArray(w.broken),
  layout(room){return Array.from({length:18},(_,i)=>({id:'brick-'+room+'-'+i,x:20+(i%6)*47,y:118+Math.floor(i/6)*23,w:44,h:18,q:i===7?room*2+1:i===10?room*2+2:null}));},
  reset(w){w.broken=[];w.ball={x:w.paddle,y:338,vx:80,vy:-150,attached:true};w.aim=null;},
  action(s,a){if(a!=='launch'&&a!=='interact')return;const w=s.run.world,b=w.ball;if(!b.attached)return;b.x=w.paddle;b.y=338;const target=this.layout(w.room).find(p=>p.q&&s.available(p.q)&&!w.broken.includes(p.id));const dx=w.aim&&target?target.x+22-w.paddle:70,speed=w.misses>=2?145:175;const norm=Math.hypot(dx,338-(target?.y||130));b.vx=dx/norm*speed;b.vy=-(338-(target?.y||130))/norm*speed;b.attached=false;},
  update(s,dt){const w=s.run.world;MiniGrid.nextStage(s,w=>this.reset(w));if(s.phase!=='playing')return;const width=w.misses>=2?120:90;w.paddle=Math.max(width/2+8,Math.min(312-width/2,w.paddle+(Number(s.input('right'))-Number(s.input('left')))*175*dt));const b=w.ball;if(b.attached){b.x=w.paddle;b.y=338;return;}const py=b.y;b.x+=b.vx*dt;b.y+=b.vy*dt;if(b.x<10||b.x>310){b.x=Math.max(10,Math.min(310,b.x));b.vx*=-1;}if(b.y<88){b.y=88;b.vy=Math.abs(b.vy);}
   if(b.vy>0&&py<346&&b.y>=346&&Math.abs(b.x-w.paddle)<width/2+4){b.y=345;b.vy=-Math.abs(b.vy);b.vx+=(b.x-w.paddle)*1.1;const sp=Math.hypot(b.vx,b.vy);if(sp>185){b.vx*=185/sp;b.vy*=185/sp;}s.playSound('coin');}
   for(const p of this.layout(w.room)){if(w.broken.includes(p.id)||(p.q&&s.mastery[p.q]?.correct)||b.x+4<p.x||b.x-4>p.x+p.w||b.y+4<p.y||b.y-4>p.y+p.h)continue;w.broken.push(p.id);b.vy*=-1;if(p.q){b.attached=true;if(s.interact(p.q))return;}else s.award(p.id);break;}
   if(b.y>385){w.misses++;b.attached=true;b.x=w.paddle;b.y=338;s.toast(w.misses>=2?'挡板已经加宽，小球也慢了下来。':'小球回到挡板，继续发球就好。',3);}
  },pointer(s,p,type){const w=s.run.world;if(['down','move'].includes(type))w.paddle=Math.max(55,Math.min(265,p.x));if(type==='tap')this.action(s,'launch');},afterQuestion(s){MiniGrid.nextStage(s,w=>this.reset(w));},help(s){const w=s.run.world;w.aim=true;w.ball.attached=true;const target=this.layout(w.room).find(p=>p.q&&s.available(p.q)&&!w.broken.includes(p.id));if(target)w.paddle=target.x+22;s.toast('已标出知识砖并准备瞄准，由你点击“发球”。',3);},location:s=>'修复墙 '+Math.min(5,s.run.world.room+1)+' / 5',status:s=>'漏球 '+s.run.world.misses+' 次 · 不限重试'
 },
 dig:{
  init(){return {room:0,x:0,y:5,selected:{x:1,y:5},depths:{},swing:0,reveal:false};},validate:w=>MiniGrid.valid(w,['room','x','y','swing'])&&w.selected&&w.depths&&w.room>=0&&w.room<5,
  nodes(room){return [{id:room*2+1,x:2,y:1},{id:room*2+2,x:5,y:3}];},rock(w,x,y){return (x===3&&y===2)||(x===1&&y===3&&w.room%2===0);},inside(x,y){return x>=0&&x<7&&y>=0&&y<6;},depth(w,x,y){const key=x+','+y;if(x===0&&y===5)return 0;if(w.depths[key]!==undefined)return w.depths[key];return this.nodes(w.room).some(n=>n.x===x&&n.y===y)||(x+y+w.room)%5===0?2:1;},
  action(s,a){const w=s.run.world;if(MiniGrid.dirs[a]){const [dx,dy]=MiniGrid.dirs[a];w.selected={x:w.x+dx,y:w.y+dy};return;}if(!['dig','interact'].includes(a))return;const p=w.selected;if(!this.inside(p.x,p.y)||this.rock(w,p.x,p.y)){s.toast('这是硬石边界，换个方向挖。',2);return;}if(Math.abs(p.x-w.x)+Math.abs(p.y-w.y)>1){s.toast('先挖相邻的一格，再慢慢走过去。',2);return;}const key=p.x+','+p.y,depth=this.depth(w,p.x,p.y);w.depths[key]=Math.max(0,depth-1);w.swing=.22;s.playSound('letter');if(w.depths[key]>0)return;w.x=p.x;w.y=p.y;const node=this.nodes(w.room).find(n=>n.x===w.x&&n.y===w.y);if(node&&s.interact(node.id))return;if(!node&&(w.x+w.y)%4===0)s.award('dig-'+w.room+'-'+key);
   if(w.x===6&&w.y===0){if(w.room===4)s.complete();else{w.room++;w.x=0;w.y=5;w.selected={x:1,y:5};w.depths={};s.save();}}
  },key(s,a){this.action(s,a);},update(s,dt){s.run.world.swing=Math.max(0,s.run.world.swing-dt);},pointer(s,p,type){if(type!=='tap')return;const w=s.run.world;w.selected={x:Math.floor((p.x-41)/34),y:Math.floor((p.y-112)/34)};this.action(s,'dig');},help(s){s.run.world.reveal=!s.run.world.reveal;s.toast('纸标签已标出知识货箱，仍需挖相邻格子靠近。',3);},location:s=>'旧仓 '+(s.run.world.room+1)+' / 5',status:()=> '浅沙一铲 · 木板两铲 · 没有次数限制'
 },
 flight:{
  init(){return {distance:0,x:90,y:240,hits:0,invincible:0,stall:0,seen:[]};},validate:w=>MiniGrid.valid(w,['distance','x','y','hits','invincible','stall'])&&Array.isArray(w.seen),
  buoys(){return Array.from({length:10},(_,i)=>({id:i+1,s:240+i*335,y:[160,265,205][i%3]}));},rocks(){return Array.from({length:11},(_,i)=>({id:'rock-'+i,s:390+i*280,y:[295,135,230][i%3]}));},
  update(s,dt){const w=s.run.world;w.invincible=Math.max(0,w.invincible-dt);if(w.stall>0){w.stall-=dt;return;}w.x=Math.max(35,Math.min(270,w.x+(Number(s.input('right'))-Number(s.input('left')))*125*dt));w.y=Math.max(103,Math.min(335,w.y+(Number(s.input('down'))-Number(s.input('up')))*125*dt));w.distance+=(w.hits>=3?45:55)*dt;
   for(const b of this.buoys()){const x=90+(b.s-w.distance)*.6;if(Math.hypot(w.x-x,w.y-b.y)<29&&s.available(b.id)){s.interact(b.id);return;}if(x<0&&!s.attempted(b.id)&&!s.run.missed.includes(b.id))s.run.missed.push(b.id);}
   for(const r of this.rocks()){const x=90+(r.s-w.distance)*.6;if(Math.hypot(w.x-x,w.y-r.y)<24&&!w.invincible){w.hits++;w.invincible=1.6;w.stall=.3;w.y=Math.max(105,Math.min(330,w.y+22));s.playSound('bump');s.toast('船身轻轻弹开，航程和信件都还在。',2);}}
   for(let i=0;i<32;i++){const x=90+(120+i*110-w.distance)*.6,y=i%2?255:220;if(Math.abs(w.x-x)<16&&Math.abs(w.y-y)<18)s.award('flight-star-'+i);}if(w.distance>=3600)s.complete();
  },pointer(s,p,type){if(['down','move'].includes(type)){const w=s.run.world;w.x=Math.max(35,Math.min(270,p.x+18));w.y=Math.max(103,Math.min(335,p.y-24));}},afterQuestion(s){const w=s.run.world;w.distance+=65;w.invincible=1.8;},help(s){const w=s.run.world,b=this.buoys().find(b=>b.s>w.distance&&s.available(b.id));s.toast(b?'下一座浮标在'+(b.y<210?'上方':'下方')+'航道，靠近铜环才会停靠。':'沿星光航道前往营地。',4);},location:s=>'星光峡谷 · '+Math.min(100,Math.round(s.run.world.distance/3600*100))+'%',status:s=>'保护恢复 · 本趟错过 '+s.run.missed.length+' 座浮标'
 },
 claw:{
  init(){return {room:0,x:160,y:115,mode:'swing',manual:0,caught:null,misses:0,hold:0};},validate:w=>MiniGrid.valid(w,['room','x','y','manual','misses','hold'])&&['swing','down','up','close'].includes(w.mode),
  capsules(room){return [{id:room*2+1,x:88,y:270},{id:room*2+2,x:232,y:286}];},reset(w){w.y=115;w.mode='swing';w.caught=null;},
  action(s,a){const w=s.run.world;if(a==='left'||a==='right'){w.x=Math.max(36,Math.min(284,w.x+(a==='left'?-15:15)));w.manual=3;}if(['grab','interact'].includes(a)&&w.mode==='swing'){w.mode='down';w.caught=null;s.playSound('dock');}},
  update(s,dt){const w=s.run.world;MiniGrid.nextStage(s,w=>this.reset(w));if(s.phase!=='playing')return;
   if(w.mode==='swing'){const dir=Number(s.input('right'))-Number(s.input('left'));w.manual=Math.max(0,w.manual-dt);if(dir){w.x=Math.max(36,Math.min(284,w.x+dir*95*dt));w.manual=3;}else if(!w.manual)w.x=160+Math.sin(s.time*(w.misses>=2?.4:.65))*105;}
   if(w.mode==='down'){w.y+=155*dt;const capsule=this.capsules(w.room).find(c=>s.available(c.id)&&Math.hypot(w.x-c.x,w.y-c.y)<(w.misses>=2?52:35));if(capsule){w.caught=capsule.id;w.x=capsule.x;w.mode='close';w.hold=.2;}else if(w.y>320){w.misses++;w.mode='up';s.toast('抓空了也没关系，钩爪正在回来。',2);}}
   if(w.mode==='close'){w.hold-=dt;if(w.hold<=0)w.mode='up';}if(w.mode==='up'){w.y-=180*dt;if(w.y<=115){w.y=115;w.mode='swing';if(w.caught&&s.interact(w.caught))return;}}
  },afterQuestion(s){s.run.world.caught=null;MiniGrid.nextStage(s,w=>this.reset(w));},help(s){const w=s.run.world,c=this.capsules(w.room).find(c=>s.available(c.id));if(w.mode!=='swing'){s.toast('等钩爪回到轨道，再帮你对准。',2);return;}if(c){w.x=c.x;w.manual=5;s.toast('已经对准胶囊，由你按“放钩”。',3);}},location:s=>'展示槽 '+Math.min(5,s.run.world.room+1)+' / 5',status:s=>'抓空 '+s.run.world.misses+' 次 · 不扣金币'
 },
 puzzle:{
  fresh(room){return {room,x:1,y:5,crate:{x:3,y:3},mirror:0,sequence:0,usedPortal:false,switches:[false,false,false,false],move:0,portalLock:false,hint:0,history:[]};},init(){return this.fresh(0);},validate:w=>MiniGrid.valid(w,['room','x','y','mirror','sequence','move','hint'])&&w.crate&&Array.isArray(w.switches)&&Array.isArray(w.history)&&w.room>=0&&w.room<5,
  nodes(room){return [{id:room*2+1,x:1,y:2},{id:room*2+2,x:5,y:4}];},floor(w,x,y){return x>=1&&x<=5&&y>=1&&y<=5&&!(w.room===3&&x===3&&(y===2||y===3));},
  solved(w){return [()=>w.crate.x===5&&w.crate.y===3,()=>w.mirror===2,()=>w.sequence===3,()=>w.usedPortal,()=>w.switches[0]&&!w.switches[1]&&w.switches[2]&&!w.switches[3]][w.room]();},
  remember(w){const copy={...w,history:[]};w.history.push(JSON.parse(JSON.stringify(copy)));if(w.history.length>80)w.history.shift();},
  move(s,direction){const w=s.run.world,[dx,dy]=MiniGrid.dirs[direction],nx=w.x+dx,ny=w.y+dy;if(!this.floor(w,nx,ny))return;this.remember(w);
   if(w.room===0&&nx===w.crate.x&&ny===w.crate.y){const cx=nx+dx,cy=ny+dy;if(!this.floor(w,cx,cy)){w.history.pop();return;}w.crate={x:cx,y:cy};}w.x=nx;w.y=ny;w.move=.15;
   if(w.room===2&&w.sequence<3&&ny===2&&nx>=2&&nx<=4){const n=nx-1;if(n===w.sequence+1)w.sequence++;else if(n!==w.sequence){w.sequence=0;s.toast('按地上的 1 → 2 → 3 顺序走就好。',2);}}
   if(w.room===3){const portal=[{x:2,y:4},{x:4,y:2}],i=portal.findIndex(p=>p.x===w.x&&p.y===w.y);if(i>=0&&!w.portalLock){w.x=portal[1-i].x;w.y=portal[1-i].y;w.usedPortal=true;w.portalLock=true;s.playSound('dock');}else if(i<0)w.portalLock=false;}
   for(const n of this.nodes(w.room))if(w.x===n.x&&w.y===n.y&&s.interact(n.id))return;
   if(w.x===5&&w.y===1){if(this.solved(w)){if(w.room===4)s.complete();else{s.run.world=this.fresh(w.room+1);s.save();s.toast('机关门打开了，走进下一间档案室。',3);}}else s.toast('门上的灯还没亮，先完成本房的小机关。',3);}
  },
  action(s,a){let w=s.run.world;if(MiniGrid.dirs[a]){if(w.move<=0)this.move(s,a);return;}if(a==='undo'){const last=w.history.pop();if(last)s.run.world={...last,history:w.history};return;}if(a==='reset'){s.run.world=this.fresh(w.room);s.toast('本房机关复位，已答题目都保留。',3);return;}if(a!=='interact')return;
   if(w.room===1&&Math.abs(w.x-3)+Math.abs(w.y-3)<=1){this.remember(w);w.mirror=(w.mirror+1)%4;s.playSound('letter');}
   if(w.room===4){const switches=[{x:2,y:2},{x:4,y:2},{x:2,y:4},{x:4,y:4}];const near=switches.findIndex(p=>Math.abs(w.x-p.x)+Math.abs(w.y-p.y)<=1);if(near>=0){this.remember(w);w.switches[near]=!w.switches[near];s.playSound('letter');}}
  },update(s,dt){const w=s.run.world;w.move=Math.max(0,w.move-dt);const d=MiniGrid.direction(s);if(d&&w.move<=0)this.move(s,d);},
  pointer(s,p,type){if(type!=='tap')return;const w=s.run.world,x=Math.floor((p.x-41)/34),y=Math.floor((p.y-100)/34),dx=x-w.x,dy=y-w.y;if(Math.abs(dx)+Math.abs(dy)===1){this.action(s,dx===1?'right':dx===-1?'left':dy===1?'down':'up');}else if(x===w.x&&y===w.y)this.action(s,'interact');else s.toast('用方向键靠近机关，再按“交互”。',2);},
  help(s){const w=s.run.world;w.hint++;const hints=['把木箱推到右侧的金色压板。','靠近镜子按交互，朝上的光线要照到接收器。','依次踩下标着 1、2、3 的三块地砖。','踏上传送垫，让 A、B 两端连接起来。','让左上和左下开关亮起，右边两只保持熄灭。'];if(w.hint<3){s.toast(hints[w.room]+(w.hint===2?' 再点一次可帮助完成动作机关。':''),5);return;}if(w.room===0)w.crate={x:5,y:3};if(w.room===1)w.mirror=2;if(w.room===2)w.sequence=3;if(w.room===3)w.usedPortal=true;if(w.room===4)w.switches=[true,false,true,false];s.toast('动作机关已解开，知识档案仍要亲自接触并作答。',4);},
  location:s=>['推箱压板室','镜光档案室','编号门锁室','传送书房','组合开关室'][s.run.world.room],status:s=>AdventureWorlds.puzzle.solved(s.run.world)?'机关已亮 · 前往右上角的门':'可撤销 / 复位 · 三级帮助不代答'
 }
};
