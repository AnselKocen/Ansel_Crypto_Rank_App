/* Editable castle layers and the physical course share one world definition. */
const CastleArt=(()=>{
 const P={sky:'#b4a7b5',mist:'#c9b9b8',far:'#8e8fa5',mid:'#72839b',stone:'#7186a0',light:'#9cacc0',edge:'#bfcbcf',shadow:'#303b55',deep:'#40516e',gold:'#f5cb7d',paper:'#fff1cd',roof:'#76638e',leaf:'#668e62',wood:'#ac855e',copper:'#c98a58'};
 const length=5200,zones=['城堡庭院','暖窗回廊','钟楼前厅','常春藤露台','塔顶书房'];
 const platforms=[],gaps=[],boxes=[],benches=[],coins=[];
 const floors=[326,310,294,310,326];
 for(let i=0;i<5;i++) {
  const x=i*1040,y=floors[i];
  platforms.push({id:`floor-${i}-a`,x,y,w:600,h:160,ground:true},{id:`floor-${i}-b`,x:x+664,y,w:376,h:160,ground:true},{id:`shelf-${i}`,x:x+380,y:y-56,w:130,h:14,ground:false});
  gaps.push({id:i,x:x+600,w:64,y});
  [270,830].forEach((dx,j)=>boxes.push({id:i*2+j+1,x:x+dx,y}));
  [120,530,915].forEach((dx,j)=>benches.push({id:`bench-${i}-${j}`,x:x+dx,y}));
  [145,185,225,735,775].forEach((dx,j)=>coins.push({id:`castle-coin-${i}-${j}`,x:x+dx,y:y-18}));
  [410,450,490].forEach((dx,j)=>coins.push({id:`castle-bonus-${i}-${j}`,x:x+dx,y:y-73}));
 }
 const rect=(c,x,y,w,h,color)=>{c.fillStyle=color;c.fillRect(Math.round(x),Math.round(y),Math.round(w),Math.round(h));};
 const line=(c,pts,color,w=1)=>{c.strokeStyle=color;c.lineWidth=w;c.beginPath();pts.forEach(([x,y],i)=>i?c.lineTo(Math.round(x),Math.round(y)):c.moveTo(Math.round(x),Math.round(y)));c.stroke();};
 function arch(c,x,y,w=50,h=116,lit=true) {
  rect(c,x-6,y+14,w+12,h-12,P.deep);rect(c,x-4,y+7,w+8,h-5,P.light);rect(c,x+4,y,w-8,h,P.light);
  rect(c,x+2,y+16,w-4,h-18,P.shadow);rect(c,x+8,y+7,w-16,h-10,P.shadow);
  rect(c,x+6,y+18,w-12,h-24,lit?'#be9871':'#71839a');rect(c,x+12,y+11,w-24,h-18,lit?'#d9b57e':'#8296ac');
  rect(c,x+w/2-2,y+11,4,h-16,P.deep);rect(c,x+7,y+49,w-14,4,P.deep);rect(c,x+7,y+85,w-14,3,P.deep);
  rect(c,x-9,y+h-3,w+18,7,P.edge);rect(c,x-8,y+h+4,w+16,5,P.deep);
  if(lit) {rect(c,x+12,y+24,6,19,'#efd08d');rect(c,x+w-18,y+56,5,18,'#efd08d');}
 }
 function tower(c,x,y,w,h,t=0,clock=false) {
  rect(c,x,y,w,h,P.mid);rect(c,x+w-9,y,9,h,'#677891');
  for(let i=0;i<w;i+=12) rect(c,x+i,y-7,7,8,P.mid);
  for(let yy=y+20;yy<y+h;yy+=27) {rect(c,x+9,yy,5,10,P.gold);rect(c,x+w-16,yy,5,10,'#a4b4b5');}
  rect(c,x-6,y-11,w+12,5,P.roof);
  if(clock) {rect(c,x+w/2-16,y+19,32,32,P.deep);rect(c,x+w/2-13,y+22,26,26,P.paper);line(c,[[x+w/2,y+25],[x+w/2,y+35],[x+w/2+8,y+39]],P.deep,2);}
  rect(c,x+w/2-1,y-30,2,18,P.deep);rect(c,x+w/2+1,y-29,16,7+Math.floor(Math.sin(t*2)*1.5),'#b68982');
 }
 function lantern(c,x,y,t) {
  rect(c,x-2,y-8,4,8,P.deep);rect(c,x-6,y,12,18,P.deep);rect(c,x-4,y+3,8,12,'#c19b6d');
  rect(c,x-2,y+5,4,8,P.gold);rect(c,x-1,y+5-Math.floor(Math.sin(t*5)),2,4,P.paper);rect(c,x-7,y+18,14,3,P.shadow);
 }
 function plant(c,x,y,t=0) {
  rect(c,x-8,y-12,16,13,'#a57c65');rect(c,x-10,y-15,20,5,'#c39c7b');
  for(let k=0;k<4;k++){const dx=(k%2?1:-1)*(4+k);rect(c,x+dx,y-26-k*3,6,4,P.leaf);line(c,[[x,y-15],[x+dx+2,y-25-k*3]],P.leaf,2);}
  rect(c,x+5+Math.floor(Math.sin(t)),y-32,4,4,P.gold);
 }
 function bookBox(c,b,answer,t) {
  const x=b.x,y=b.y;
  rect(c,x-16,y-5,34,5,'#536178');rect(c,x-14,y-29,28,26,'#715c50');rect(c,x-12,y-28,24,23,P.wood);
  rect(c,x-10,y-25,20,3,'#d5b680');rect(c,x-10,y-21,20,12,'#b99970');
  for(const dx of [-12,8]){rect(c,x+dx,y-27,4,5,P.gold);rect(c,x+dx,y-8,4,4,P.gold);}
  rect(c,x-6,y-21,12,10,P.paper);line(c,[[x-6,y-21],[x,y-17],[x+6,y-21]],P.copper);
  if(answer?.correct) {line(c,[[x-5,y-41],[x-1,y-37],[x+7,y-46]],'#a5c5a4',2);rect(c,x-12,y-28,24,3,'#b1c69c');}
  else {const bob=Math.round(Math.sin(t*2+b.id)*2);rect(c,x-7,y-57+bob,14,19,P.shadow);rect(c,x-5,y-55+bob,10,15,P.paper);rect(c,x-1,y-52+bob,2,7,P.copper);rect(c,x-1,y-42+bob,2,2,P.copper);}
 }
 function cat(c,x,y,t=0,facing=1,moving=false,grounded=true,landing=0) {
  c.save();c.translate(Math.round(x),Math.round(y));c.scale(facing,1);
  const step=moving&&grounded?Math.floor(t*12)%4:0,bob=grounded&&moving?step%2:0;
  rect(c,-11,-2,24,3,'#53617a');
  line(c,[[-8,-12],[-16,-14],[-18,-23+Math.floor(Math.sin(t*3)*2)],[-14,-26]],'#bf895c',4);
  const squash=landing>0?2:0;
  rect(c,-8,-22+bob+squash,16,18-squash,'#b77d53');rect(c,-6,-23+bob+squash,13,16-squash,'#d5a46d');rect(c,-3,-20+bob,8,11,'#edca91');
  rect(c,-11,-35+bob+squash,6,9,'#a97050');rect(c,5,-35+bob+squash,6,9,'#a97050');rect(c,-9,-33+bob+squash,3,4,'#e4b18b');rect(c,7,-33+bob+squash,3,4,'#e4b18b');
  rect(c,-11,-29+bob+squash,22,14,P.shadow);rect(c,-10,-30+bob+squash,20,13,'#d5a46d');rect(c,-6,-22+bob+squash,14,6,P.paper);
  rect(c,-5,-26+bob+squash,2,3,P.shadow);rect(c,5,-26+bob+squash,2,3,P.shadow);rect(c,1,-22+bob+squash,2,2,'#986750');
  rect(c,-8,-17+bob,16,4,'#b57469');rect(c,-12,-15+bob-(step%2),7,3,'#b57469');
  rect(c,-10,-17+bob,4,9,'#765e54');rect(c,-12,-14+bob,8,7,P.paper);line(c,[[-12,-14+bob],[-8,-10+bob],[-4,-14+bob]],P.copper);
  if(grounded){rect(c,-7-(step===1?2:0),-6,5,6,P.shadow);rect(c,3+(step===3?2:0),-6,5,6,P.shadow);}
  else {rect(c,-9,-10,6,4,P.shadow);rect(c,4,-11,6,4,P.shadow);}
  c.restore();
 }
 function door(c,x,y,t) {
  arch(c,x-32,y-143,64,142,true);rect(c,x-22,y-103,44,98,'#5e6772');
  for(let k=0;k<3;k++){rect(c,x-19,y-95+k*26,38,2,P.gold);for(let j=0;j<5;j++)rect(c,x-17+j*7,y-113+k*26,5,18,[P.roof,'#7c9b8a',P.copper][j%3]);}
  rect(c,x-10,y-50,20,46,P.gold);rect(c,x-6,y-45,12,40,P.paper);rect(c,x-32,y-3,64,5,P.edge);
  rect(c,x-19,y+3,38,4,'#ac9b8c');lantern(c,x-49,y-77,t);lantern(c,x+49,y-77,t);
 }
 function draw(c,run,mastery,time=0) {
  const w=c.canvas.width,h=c.canvas.height,shift=h-400,cam=run.camera||0,t=time;
  rect(c,0,0,w,h,P.sky);rect(c,0,Math.max(60,h-310),w,180,P.mist);
  for(let k=-1;k<8;k++) {
   const x=k*160-(cam*.08)%160,y=66+(k%3)*15;
   rect(c,x,y,160,180,'#a1a0ae');rect(c,x+35,y-20,92,24,'#a1a0ae');rect(c,x+65,y-40,35,22,'#a1a0ae');
  }
  rect(c,w-58,28,19,19,'#e7d5b8');rect(c,w-62,24,13,15,P.sky);
  for(let k=-1;k<w/120+2;k++) {const x=k*120-(cam*.28)%120;tower(c,x,26+(k%2)*18,57,210,t,k%4===2);}
  c.save();c.translate(-Math.round(cam),Math.round(shift));
  const first=Math.floor(cam/1040),last=Math.min(4,Math.floor((cam+w)/1040));
  for(let z=Math.max(0,first);z<=last;z++) {
   const start=z*1040,base=floors[z];
   rect(c,start,94,1040,base-94,'#6b7f99');rect(c,start,94,1040,6,'#9cabb9');
   for(let y=108;y<base;y+=20){rect(c,start,y,1040,1,'#60748e');for(let x=start+((y/20)%2)*24;x<start+1040;x+=48)rect(c,x,y,1,20,'#60748e');}
   for(let x=start+80;x<start+1040;x+=184) {
    arch(c,x,126,50,base-151,true);rect(c,x-18,base-31,5,31,P.light);rect(c,x+64,base-31,5,31,P.light);
    if(z===0||z===3) plant(c,x+64,base,t);
    if(z===1||z===4) {rect(c,x+74,base-39,31,4,P.wood);for(let j=0;j<4;j++)rect(c,x+77+j*7,base-57,5,18,[P.roof,P.leaf,P.copper][j%3]);}
   }
   if(z===2) {tower(c,start+430,18,82,250,t,true);rect(c,start+454,250,34,8,P.light);}
   if(z===3) for(let n=0;n<6;n++) {const x=start+48+n*142;line(c,[[x,97],[x+5,130],[x-2,163]],P.leaf,2);for(let q=0;q<4;q++)rect(c,x+(q%2?2:-5),105+q*13,7,3,P.leaf);}
  }
  for(const p of platforms) {
   if(p.x+p.w<cam-30||p.x>cam+w+30)continue;
   rect(c,p.x,p.y,p.w,p.h,P.deep);rect(c,p.x,p.y,p.w,5,P.edge);rect(c,p.x,p.y+5,p.w,5,P.light);
   if(p.ground)for(let y=p.y+14;y<Math.max(h-shift,p.y+80);y+=19){rect(c,p.x,y,p.w,1,'#536682');for(let x=p.x+((y%2)*18);x<p.x+p.w;x+=37)rect(c,x,y,1,19,'#536682');}
   else {rect(c,p.x+8,p.y+p.h,7,13,P.deep);rect(c,p.x+p.w-15,p.y+p.h,7,13,P.deep);}
   rect(c,p.x,p.y+5,3,Math.min(p.h,18),P.light);rect(c,p.x+p.w-3,p.y+5,3,Math.min(p.h,18),P.shadow);
  }
  for(const g of gaps) {
   if(g.x>cam+w||g.x+g.w<cam)continue;
   line(c,[[g.x-4,g.y+40],[g.x+g.w/2,g.y+53],[g.x+g.w+4,g.y+40]],'#9fa99a',2);
   for(let k=0;k<5;k++)line(c,[[g.x+k*15,g.y+40],[g.x+k*15+7,g.y+48]],'#9fa99a');
   if(run.bridges?.includes(g.id)){rect(c,g.x,g.y,g.w,7,'#b6c5ad');rect(c,g.x,g.y+7,g.w,7,'#708f7e');for(let x=g.x+3;x<g.x+g.w;x+=12)rect(c,x,g.y+2,8,2,P.paper);}
  }
  for(const b of benches){if(b.x<cam-40||b.x>cam+w+40)continue;rect(c,b.x-18,b.y-15,36,5,P.wood);rect(c,b.x-16,b.y-24,32,5,'#c1a17c');rect(c,b.x-14,b.y-11,4,11,P.shadow);rect(c,b.x+10,b.y-11,4,11,P.shadow);lantern(c,b.x-26,b.y-48,t);if(run.checkpoint?.id===b.id)rect(c,b.x+24,b.y-27,5,5,'#b8d3aa');}
  for(const b of boxes)if(b.x>cam-40&&b.x<cam+w+40)bookBox(c,b,mastery[b.id],t);
  for(const coin of coins){if(coin.x<cam-20||coin.x>cam+w+20||run.collected?.includes(coin.id))continue;const d=3+Math.abs(Math.round(Math.sin(t*4+coin.x)*3));rect(c,coin.x-d/2-1,coin.y-6,d+2,12,'#b38a4e');rect(c,coin.x-d/2,coin.y-7,d,11,P.gold);rect(c,coin.x-d/2+1,coin.y-5,1,6,P.paper);}
  if(cam+w>length-180)door(c,length-95,326,t);
  cat(c,run.x,run.y,t,run.facing||1,Math.abs(run.vx||0)>3,run.grounded,run.landing||0);
  c.restore();
 }
 let preview;
 return {length,zones,platforms,gaps,boxes,benches,coins,floors,draw,cat,preview(){if(!preview){const c=document.createElement('canvas');c.width=640;c.height=400;draw(c.getContext('2d'),{x:2360,y:294,vx:140,grounded:true,camera:2080,collected:[],checkpoint:{},bridges:[]},{},1);preview=c.toDataURL();}return preview;}};
})();
