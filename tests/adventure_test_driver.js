/* Virtual input driver: never assigns player positions, scores, answers or completion flags. */
function driveAdventure(limit=30000){
 const s=AdventureGames.scene,g=s.game,dt=.02;let w=s.run.world;
 const moveGrid=(target,walk)=>{if(w.move>0)return;const path=MiniGrid.path(w,target,walk),p=path[0];if(!p)return;const d=p.x>w.x?'right':p.x<w.x?'left':p.y>w.y?'down':'up';g.action(s,d);};
 for(let i=0;i<limit&&s.phase==='playing';i++){
  w=s.run.world;s.keys.clear();
  if(s.levelId===3){const peak=w.y+(w.vy>0?w.vy*w.vy/(2*480):0),index=Math.max(0,Math.min(36,Math.floor((peak+.1)/62))),p=g.platforms()[index];if(Math.abs(w.x-p.x)>3)s.keys.add(w.x<p.x?'ArrowRight':'ArrowLeft');}
  if(s.levelId===4){const p=g.stations().find(p=>s.available(p.id)&&p.s>=w.distance-12);if(p&&w.lane!==p.lane)g.action(s,w.lane<p.lane?'right':'left');const o=g.obstacles().find(o=>!w.seen.includes(o.id)&&o.s-w.distance<32&&o.s-w.distance>0&&o.lane===w.lane);if(o){if(o.type==='crate'&&w.jump===0)g.action(s,'jump');if(o.type==='beam'&&w.slide===0)g.action(s,'slide');}}
  if(s.levelId===5){if(!w.path.length){const n=g.nodes(w.room).find(n=>s.available(n.id))||{x:7,y:1};g.pointer(s,{x:25+n.x*30+15,y:110+n.y*30+15},'tap');}}
  if(s.levelId===6){if(w.ball.attached){g.help(s);g.action(s,'launch');}else g.pointer(s,{x:w.ball.x,y:350},'move');}
  if(s.levelId===7){const n=g.nodes(w.room).find(n=>s.available(n.id))||{x:6,y:0};const path=MiniGrid.path(w,n,(x,y)=>g.inside(x,y)&&!g.rock(w,x,y)),p=path[0];if(p){g.action(s,p.x>w.x?'right':p.x<w.x?'left':p.y>w.y?'down':'up');g.action(s,'dig');}}
  if(s.levelId===8){const b=g.buoys().find(b=>s.available(b.id)&&90+(b.s-w.distance)*.6>=w.x-29);if(b&&Math.abs(w.y-b.y)>3)s.keys.add(w.y<b.y?'ArrowDown':'ArrowUp');}
  if(s.levelId===9&&w.mode==='swing'){g.help(s);g.action(s,'grab');}
  if(s.levelId===10){
   const node=g.nodes(w.room).find(n=>s.available(n.id));
   const floor=(x,y)=>g.floor(w,x,y)&&!(w.room===0&&w.crate.x===x&&w.crate.y===y);
   if(node)moveGrid(node,floor);
   else if(!g.solved(w)){
    if(w.room===0){const target={x:w.crate.x-1,y:3};if(w.x===target.x&&w.y===target.y&&w.move<=0)g.action(s,'right');else moveGrid(target,floor);}
    if(w.room===1){if(w.x===3&&w.y===4)g.action(s,'interact');else moveGrid({x:3,y:4},floor);}
    if(w.room===2){const target={x:2+w.sequence,y:2};moveGrid(target,(x,y)=>floor(x,y)&&(!(y===2&&x>=2&&x<=4)||(x===target.x&&y===target.y)));}
    if(w.room===3)moveGrid({x:2,y:4},floor);
    if(w.room===4){const positions=[{x:2,y:2},{x:4,y:2},{x:2,y:4},{x:4,y:4}],targetState=[true,false,true,false],idx=w.switches.findIndex((value,j)=>value!==targetState[j]),target=positions[idx];if(target){if(w.x===target.x&&w.y===target.y)g.action(s,'interact');else moveGrid(target,floor);}}
   }else moveGrid({x:5,y:1},floor);
  }
  if(s.phase==='playing'){s.time+=dt;s.cooldown=Math.max(0,s.cooldown-dt);s.update(dt);}
 }
 s.clearInput();return {phase:s.phase,question:s.run.questionId,score:s.score(),world:s.run.world};
}
