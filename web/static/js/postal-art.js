/* Editable, deterministic pixel art. Wide desktop and adaptive portrait canvas; all edges snap to pixels. */
const PostalArt = (() => {
    const P = { sea:'#3b9c9c', deep:'#307e88', foam:'#bce4cd', sand:'#ebcb91', light:'#fff1cd',
        ink:'#244454', road:'#bfae87', curb:'#e8d8ae', red:'#c76b55', darkRed:'#914f45',
        green:'#587b5d', leaf:'#80a16a', mint:'#90cbb3', shade:'#497b78', gold:'#e8b852' };
    const stations = Object.freeze(Array.from({length:10}, (_,i) => Object.freeze({id:i+1, s:250+i*400, side:i%3===1?-1:1, bayOffset:(i%3===1?-1:1)*64, halfWidth:18, halfLength:34})));
    const center = s => 165 + Math.sin(s / 360) * 27 + Math.sin(s / 145) * 7;
    const rect = (c, x, y, w, h, color) => { c.fillStyle=color; c.fillRect(Math.round(x),Math.round(y),Math.round(w),Math.round(h)); };
    const line = (c, points, color, width=1) => {
        c.strokeStyle=color; c.lineWidth=width; c.beginPath();
        points.forEach(([x,y],i)=>i ? c.lineTo(Math.round(x),Math.round(y)) : c.moveTo(Math.round(x),Math.round(y))); c.stroke();
    };
    function tree(c,x,y,t=0) {
        rect(c,x+4,y+18,5,16,'#816951'); rect(c,x-7,y+7,27,17,P.green);
        rect(c,x-3,y,19,24,P.green); rect(c,x-5,y+6,19,12,P.leaf); rect(c,x,y+2,10,5,'#a3b979');
        rect(c,x+10,y+19,8,6,'#456a53'); rect(c,x-4+(t%2),y+11,4,4,'#bdd08b');
    }
    function house(c,x,y,type=0,t=0) {
        const roof = type===2 ? '#638e82' : P.red;
        rect(c,x+3,y+10,42,60,'#c3a26c'); rect(c,x,y+8,42,57,'#f0dbab');
        rect(c,x+31,y+8,11,57,'#d1b886'); rect(c,x-3,y+3,49,17,P.darkRed);
        rect(c,x-5,y,53,13,roof); rect(c,x-1,y+2,45,3,'#e09370');
        for(let i=0;i<5;i++) rect(c,x+i*10,y+7,7,2,'#a55d4e');
        rect(c,x+4,y+22,11,16,P.ink); rect(c,x+6,y+24,7,11,'#91b8ac');
        rect(c,x+22,y+22,11,16,P.ink); rect(c,x+24,y+24,7,11,'#91b8ac');
        rect(c,x+8,y+42,13,23,'#8f7258'); rect(c,x+10,y+44,9,21,'#a88965'); rect(c,x+16,y+53,2,3,P.gold);
        rect(c,x-4,y+38,43,7,roof); for(let i=0;i<4;i++) rect(c,x+i*10,y+38,5,7,P.light);
        rect(c,x-6,y+66,31,4,'#ddc494'); rect(c,x+27,y+56,10,10,'#96765a');
        rect(c,x+27,y+53,11,5,P.green); rect(c,x+30,y+51,3,4,t%2?P.red:'#e6b96c');
        if(type===1) { rect(c,x+4,y+13,26,7,P.light); rect(c,x+9,y+15,16,2,P.gold); }
        if(type===2) { envelope(c,x+15,y+14,10); rect(c,x+34,y-5,5,5,P.mint); }
    }
    function envelope(c,x,y,w=14) {
        rect(c,x,y,w,Math.round(w*.65),P.light); line(c,[[x,y],[x+w/2,y+w*.35],[x+w,y]],P.darkRed);
    }
    function lighthouse(c,x,y,t) {
        rect(c,x-25,y+103,73,9,'#beae83'); rect(c,x-22,y+99,67,6,P.light);
        rect(c,x-12,y+18,46,82,'#d9cda8'); rect(c,x-8,y+12,36,88,P.light);
        rect(c,x-9,y+40,38,14,P.red); rect(c,x-10,y+72,40,14,P.red);
        rect(c,x+20,y+18,8,82,'#ddc49b'); rect(c,x-16,y+9,52,5,P.ink);
        rect(c,x-11,y-13,42,22,P.ink); rect(c,x-7,y-9,34,16,'#f4d484');
        rect(c,x+5,y-8,5,14,'#be9762'); rect(c,x-14,y-18,48,5,P.red); rect(c,x-6,y-23,31,5,P.darkRed);
        rect(c,x+6,y-30,5,8,P.ink); rect(c,x+3,y+87,12,13,P.ink);
        rect(c,x+5,y+21,8,10,'#72a2a1'); rect(c,x+5,y+57,8,9,'#72a2a1');
        if(t!==0) { c.globalAlpha=.13; c.fillStyle=P.light; c.beginPath(); c.moveTo(x+10,y-3); c.lineTo(x-150,y-50+Math.sin(t*.3)*20); c.lineTo(x-150,y+20+Math.sin(t*.3)*20); c.fill(); c.globalAlpha=1; }
    }
    function mailbox(c,x,y,index,answer,t) {
        rect(c,x-6,y+19,25,6,'#a08c67'); rect(c,x+3,y+3,5,20,P.darkRed);
        rect(c,x-6,y-14,26,22,P.darkRed); rect(c,x-3,y-17,20,24,P.red); rect(c,x-1,y-15,16,3,'#e9a079');
        rect(c,x+1,y-7,12,3,P.ink); envelope(c,x+3,y-1,8);
        rect(c,x+18,y-12,3,14,'#816951'); rect(c,x+20,y-12,7,5,answer?.correct ? P.mint : P.gold);
        if(!answer?.correct) {
            const b=Math.round(Math.sin(t*2)*2); rect(c,x-1,y-44+b,17,20,P.ink); rect(c,x+1,y-43+b,13,17,P.light);
            rect(c,x+6,y-40+b,3,8,P.red); rect(c,x+6,y-30+b,3,3,P.red);
        } else {
            line(c,[[x+1,y-27],[x+5,y-23],[x+12,y-31]],'#335d51',2);
        }
        c.fillStyle=P.light; c.font='bold 7px monospace'; c.fillText(String(index+1).padStart(2,'0'),x+3,y+6);
    }
    function car(c,x,y,steer,t,protectedTime) {
        const bob=Math.floor(Math.sin(t*14)*.7); x=Math.round(x+steer*1.5); y=Math.round(y+bob);
        rect(c,x-18,y-19,37,48,'#8e926d'); rect(c,x-17,y-15,6,13,P.ink); rect(c,x+12,y-15,6,13,P.ink);
        rect(c,x-17,y+11,6,13,P.ink); rect(c,x+12,y+11,6,13,P.ink);
        rect(c,x-14,y-24,28,49,P.shade); rect(c,x-12,y-26,24,50,P.mint); rect(c,x-10,y-26,20,4,'#bee3bd');
        rect(c,x-10,y-19,20,16,P.ink); rect(c,x-8,y-17,16,12,'#8eb7b0');
        // The driver: orange ears, cream muzzle, scarf and paws behind the windscreen.
        rect(c,x-7,y-16,5,6,'#c79056'); rect(c,x+2,y-16,5,6,'#c79056');
        rect(c,x-6,y-12,12,9,'#e2b371'); rect(c,x-4,y-7,8,4,P.light);
        rect(c,x-4,y-10,2,2,P.ink); rect(c,x+3,y-10,2,2,P.ink); rect(c,x-1,y-7,2,2,P.darkRed);
        rect(c,x-8,y-2,16,4,P.red); rect(c,x-11,y+3,22,15,P.light); rect(c,x-8,y+4,16,2,'#e5d5b1');
        rect(c,x-3,y+10,6,5,P.shade); rect(c,x-6,y+7,3,3,P.shade); rect(c,x-1,y+6,3,3,P.shade); rect(c,x+4,y+7,3,3,P.shade);
        rect(c,x-13,y+20,26,5,P.shade); rect(c,x-11,y+21,5,3,P.red); rect(c,x+6,y+21,5,3,P.red);
        rect(c,x-15,y+26,30,3,P.ink); rect(c,x-14,y-21,4,3,P.light); rect(c,x+10,y-21,4,3,P.light);
        rect(c,x-10,y+29,20,4,'#987654'); envelope(c,x-6,y+29,11);
        if(protectedTime>0) { rect(c,x-21,y-31,5,2,P.light); rect(c,x+18,y+30,5,2,P.light); }
    }
    function shell(c,x,y) {
        rect(c,x+2,y,4,2,P.light);rect(c,x,y+2,8,4,P.light);rect(c,x+2,y+6,4,2,'#d6ae7b');
        rect(c,x+2,y+2,1,3,'#cda279');rect(c,x+5,y+2,1,3,'#cda279');
    }
    function coinSign(c,x,y) {
        rect(c,x-2,y+11,4,14,'#8b7255');rect(c,x-7,y-9,14,20,'#b28b4e');
        rect(c,x-9,y-6,18,14,'#b28b4e');rect(c,x-6,y-8,12,18,P.gold);rect(c,x-8,y-5,16,12,P.gold);
        rect(c,x-4,y-5,8,12,'#c49d52');rect(c,x-2,y-6,4,14,P.light);rect(c,x-5,y-2,10,3,P.light);
    }
    function draw(c, distance=0, offset=0, t=0, state={}) {
        const w=c.canvas.width,h=c.canvas.height,anchor=h-100, left=160-w/2, right=160+w/2;
        c.save(); c.translate(Math.round(w/2-160),0);
        c.imageSmoothingEnabled=false; rect(c,left,0,w,h,P.sand);
        for(let y=0;y<h;y+=4) {
            const s=distance+anchor-y, coast=35+Math.sin(s/87)*6+Math.sin(s/41)*3, cx=center(s);
            const region=Math.max(0,Math.min(3,Math.floor(s/1050)));
            rect(c,coast+7,y,right-coast,4,['#ebcb91','#d4c59b','#e3d4aa','#c3c69e'][region]);
            rect(c,left,y,coast-left,4,P.deep); rect(c,coast-13,y,13,4,P.sea); rect(c,coast,y,4,4,'#79bfac'); rect(c,coast+4,y,3,4,P.foam);
            rect(c,cx-78,y,156,4,'#b6a37c'); rect(c,cx-74,y,148,4,P.curb); rect(c,cx-68,y,136,4,['#bfae87','#b6ad94','#c1b799','#aaa991'][region]);
            if(region===1 || region===2) {
                if(Math.floor(s/4)%5===0) rect(c,cx-65,y,130,1,region===1?'#a8a089':'#b3a98e');
                for(let k=0;k<6;k++) {if(Math.floor(s/4)%5===1) rect(c,cx-58+k*23+(Math.floor(s/20)%2)*10,y,1,3,'#aaa38c');}
            }
            rect(c,cx-63,y,1,4,'#cdbd97'); rect(c,cx+62,y,1,4,'#cdbd97');
            if(Math.floor(s/22)%2===0) { rect(c,cx-74,y,4,4,P.light); rect(c,cx+70,y,4,4,P.light); }
            if(Math.floor(s/39)%2===0) rect(c,cx-1,y,3,4,'#dfcda4');
            if(Math.floor(s/4)%9===0) { rect(c,cx-50+(Math.floor(s/36)%4)*26,y,4,1,'#b4a27e'); }
        }
        // Quiet shoreline details and animated stepped foam, never on the drivable surface.
        for(let k=Math.floor((distance+anchor-h-40)/48);k<(distance+anchor+60)/48;k++) {
            const s=k*48, y=anchor-(s-distance), x=7+((k*13)%19+19)%19;
            rect(c,x,y,8,1,(k+Math.floor(t*2))%3===0?P.foam:'#69b6ad');
            if(k%3===0) { rect(c,48,y+8,4,2,'#d0af7d'); rect(c,50,y+6,2,2,P.light); }
            if(k%5===0) shell(c,58,y+16);
        }
        // Buildings, gardens and quays are placed along the route in world coordinates.
        for(let k=Math.floor((distance+anchor-h-100)/130);k<(distance+anchor+100)/130;k++) {
            if(k<0) continue;
            const s=k*130+50, y=anchor-(s-distance), cx=center(s), biome=Math.min(3,Math.floor(s/1050));
            if(k%3!==1) house(c,Math.max(266,cx+84),y-40,biome===2?2:biome===1?1:0,Math.floor(t));
            else { tree(c,cx+94,y-12,Math.floor(t)); rect(c,cx+84,y+30,29,3,'#b3a373'); rect(c,cx+86,y+24,3,12,'#8f795a'); }
            if(k%2===0) tree(c,Math.max(52,cx-100),y-20,Math.floor(t));
            else { rect(c,cx-101,y,18,8,'#96785a'); rect(c,cx-99,y-3,14,4,'#bca07a'); }
            if(biome===0 && k%4===1) {
                for(let p=0;p<4;p++) { rect(c,12+p*9,y+5,8,21,'#a18360'); rect(c,12+p*9,y+7,8,2,'#c4a477'); }
                rect(c,14,y+22,4,8,'#80674e'); rect(c,38,y+22,4,8,'#80674e');
                rect(c,7,y-16,15,26,P.light);rect(c,10,y-19,9,33,P.red);rect(c,13,y-12,4,16,'#c4c9a7');
            }
            if(biome===1 && k%2===0) coinSign(c,cx+99,y-48);
            if(biome===2) { envelope(c,cx+87,y-31,19);rect(c,cx+84,y+42,28,4,P.red); for(let p=0;p<3;p++) rect(c,cx+86+p*9,y+36,5,6,P.light); }
            if(biome===3) {
                rect(c,cx+88,y-40,4,43,P.shade);rect(c,cx+84,y-43,12,10,P.ink);rect(c,cx+86,y-41,8,6,P.mint);
                rect(c,cx+89,y-48,2,4,P.gold);rect(c,cx+84,y+1,13,4,'#93a788');
                for(let p=0;p<3;p++) rect(c,cx+100+p*4,y-40-p*3,2,3,P.shade);
            }
            // Washing / mail flags, with two wind frames.
            line(c,[[cx+81,y-45],[cx+112,y-43]],P.ink);
            for(let j=0;j<3;j++) rect(c,cx+83+j*10,y-44,5,6+((j+Math.floor(t*2))%2),j%2?P.light:P.red);
        }
        const landmarkY=anchor-(3990-distance);
        if(landmarkY>-160 && landmarkY<h+80) {
            lighthouse(c,47,landmarkY-100,t); house(c,255,landmarkY-20,2,Math.floor(t));
        }
        for(const station of stations) {
            const {s,side,id,bayOffset}=station, y=anchor-(s-distance), cx=center(s), x=cx+bayOffset;
            if(y<-80||y>h+60) continue;
            const answered=state.mastery?.[id]?.correct;
            rect(c,x-22,y-40,44,80,answered?'#c3cfad':'#e7d6ab');
            rect(c,x-19,y-36,38,72,answered?'#d7dfbb':'#f4e3b8');
            const edge=answered?'#7b9b7a':'#8b9d77';
            for(let row=0;row<7;row++) {rect(c,x-18,y-34+row*10,2,6,edge);rect(c,x+16,y-34+row*10,2,6,edge);}
            rect(c,x-18,y-35,36,2,edge);rect(c,x-18,y+33,36,2,edge);
            c.fillStyle=edge;c.font='bold 13px monospace';c.fillText('P',x-4,y+5);
            for(let n=0;n<3;n++) {rect(c,cx+side*(32+n*5),y+60-n*3,3,2,P.light);rect(c,cx+side*(32+n*5),y+60+n*3,3,2,P.light);}
            mailbox(c,cx+side*98-5,y,id-1,state.mastery?.[id],t);
        }
        for(const item of state.items||[]) {
            const y=anchor-(item.s-distance), x=center(item.s)+item.x;
            if(y<-20||y>h+30 || state.collected?.includes(item.id)) continue;
            if(item.kind==='coin') {
                const w=state.reduced?7:4+Math.abs(Math.round(Math.sin(t*4+item.s)*4));
                rect(c,x-w/2-1,y-6,w+2,13,'#a88043'); rect(c,x-w/2,y-7,w,12,P.gold);
                rect(c,x-w/2+1,y-5,2,7,P.light);
            } else {
                rect(c,x-12,y-9,25,24,'#8e7659'); rect(c,x-11,y-12,22,22,'#bd9264');
                rect(c,x-8,y-9,16,15,'#d4ac79'); line(c,[[x-8,y-9],[x+8,y+6]],'#94704f',2);
                line(c,[[x+8,y-9],[x-8,y+6]],'#94704f',2);
            }
        }
        const finishY=anchor-(4200-distance);
        if(finishY>-20&&finishY<h+20) {
            for(let y=0;y<3;y++) for(let x=0;x<18;x++) rect(c,center(4200)-72+x*8,finishY+y*8,8,8,(x+y)%2?P.light:P.ink);
            for(let x=0;x<10;x++) { rect(c,center(4200)-90+x*20,finishY-50,10,6,x%2?P.red:P.mint); }
        }
        car(c,center(distance)+offset,anchor,state.steer||0,t,state.protectedTime||0);
        // Additional scenery fills the wide desktop viewport, preserving the same road and hitboxes.
        if(w>400) {
            for(let k=Math.floor((distance+anchor-h-80)/135);k<(distance+anchor+50)/135;k++) {
                const y=anchor-(k*135-distance), region=Math.max(0,Math.min(3,Math.floor(k*135/1050)));
                rect(c,330,y-60,76,3,'#c4b58c');
                if(k%3===0) house(c,361,y-44,region===2?2:region===1?1:0,Math.floor(t));
                else { tree(c,359,y-15,Math.floor(t));tree(c,408,y+9,Math.floor(t)); }
                rect(c,434,y+15,28,3,'#9eab7e');
                if(k%2===0) {
                    rect(c,-73,y+12,36,7,'#9bc9b6'); rect(c,-68,y+19,26,4,'#e8d2a1');rect(c,-58,y-7,2,20,P.ink);
                    rect(c,-56,y-5,12,14,P.light);rect(c,-56,y+5,12,3,P.red);
                } else {rect(c,-133,y+30,19,5,'#509a95');rect(c,-128,y+27,9,3,'#66aaa0');}
            }
        }
        if(!state.reduced) {
            const bx=20+(t*7)%330, by=60+Math.sin(t*.4)*24;
            line(c,[[bx-6,by],[bx-2,by-2],[bx,by],[bx+2,by-2],[bx+6,by]],'#dae6c5');
        }
        c.restore();
    }
    let preview;
    return { draw, center, stations, palette:P, preview() {
        if(!preview) { const c=document.createElement('canvas'); c.width=640;c.height=400; draw(c.getContext('2d'),3650,0,1); preview=c.toDataURL(); }
        return preview;
    }};
})();
