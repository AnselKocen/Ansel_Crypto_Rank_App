/* Disposable Chrome integration: voluntary parking, persistent letters, responsive HUD and audio. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || '/Users/anselkocen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core');
const OUTPUT=path.resolve(__dirname,'../artifacts/ui_checks/postal');
const BASE=process.env.GAME_TEST_URL || 'http://127.0.0.1:8000';
const checks={};
(async()=>{
 fs.mkdirSync(OUTPUT,{recursive:true});
 const browser=await chromium.launch({executablePath:process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
 try {
  const context=await browser.newContext({viewport:{width:1360,height:900}}),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const state=fn=>page.evaluate(fn);
  const check=async(name,fn)=>{assert.ok(await fn(),name);checks[name]=true;};
  const shot=async(name)=>{await page.waitForTimeout(100);await page.screenshot({path:path.join(OUTPUT,name)});};
  // A virtual driver supplies steering through the same input state as the real keyboard.
  // Every position, parking transition and route outcome still comes from the scene update.
  const travel=async(id,park=true,stopAtDock=false)=>{
   return page.evaluate(({id,park,stopAtDock})=>{
    const s=PostalGame.scene,p=PostalArt.stations.find(p=>p.id===id);
    for(let n=0;n<15000 && ['driving','docking'].includes(s.phase);n++) {
     if(s.phase==='driving') {
      const target=park?p.side*50:0;s.keys.clear();
      if(Math.abs(target-s.run.offset)>2) s.keys.add(target>s.run.offset?'ArrowRight':'ArrowLeft');
     }
     s.cooldown=Math.max(0,s.cooldown-.05);s.time+=.05;s.update(.05);
     if(stopAtDock && s.phase==='docking') break;
     if(s.phase==='driving' && s.run.distance>p.s+72) break;
    }
    s.clearInput();return {phase:s.phase,distance:s.run.distance,offset:s.run.offset,questionId:s.run.questionId};
   },{id,park,stopAtDock});
  };
  const finishLap=()=>state(()=>{
   const s=PostalGame.scene;
   for(let n=0;n<15000 && s.phase==='driving';n++) {
    s.keys.clear();if(Math.abs(s.run.offset)>2) s.keys.add(s.run.offset>0?'ArrowLeft':'ArrowRight');
    s.cooldown=Math.max(0,s.cooldown-.05);s.time+=.05;s.update(.05);
   }
   s.clearInput();
  });
  const answer=async(correct=true)=>{
   const idx=await state(()=>PostalGame.scene.data.questions[PostalGame.scene.questionIndex].answer);
   await page.locator(`[data-option="${correct?idx:(idx+1)%4}"]`).click();
  };
  const continueLetter=()=>page.locator('.post-continue').click();
  const start=()=>page.locator('[data-action="start"]').click();
  await page.goto(BASE+'/game?level=1');await page.locator('#postal-canvas').waitFor();
  await check('intro_explains_mailbox_parking_misses_and_persistence',()=>page.locator('.post-guide').innerText().then(t=>t.includes('P 停车位') && t.includes('下一趟') && t.includes('打勾')));
  await check('new_game_has_no_phaser_dependency',()=>state(()=>!window.Phaser));
  await shot('desktop-intro.png');await start();await page.waitForTimeout(80);
  await check('audio_defaults_on_after_user_start',()=>state(()=>PostalGame.scene.settings.sfx && PostalGame.scene.audio?.state==='running' && PostalGame.scene.voices.size>0));
  await check('desktop_play_area_is_large_and_controls_fit',async()=>{
   const canvas=await page.locator('#postal-canvas').boundingBox(),controls=await page.locator('.postal-controls').boundingBox();
   return canvas.width>1000 && canvas.height>600 && controls.y+controls.height<=900;
  });
  await check('ten_envelope_icons_are_inside_top_left_of_playfield',()=>state(()=>{
   const frame=document.querySelector('.postal-frame').getBoundingClientRect(),hud=document.querySelector('.post-corner-progress').getBoundingClientRect();
   return document.querySelectorAll('[data-stamp]').length===10 && hud.left>frame.left && hud.top>frame.top && hud.bottom<frame.top+100;
  }));
  await page.keyboard.down('ArrowRight');await page.waitForTimeout(300);await page.keyboard.up('ArrowRight');
  await check('real_keyboard_steers_without_auto_return',async()=>{const x=await state(()=>PostalGame.scene.run.offset);await page.waitForTimeout(100);return x>10 && Math.abs(await state(()=>PostalGame.scene.run.offset)-x)<.01;});
  await page.locator('[data-action="pause"]').click();const frozen=await state(()=>PostalGame.scene.run.distance);await page.waitForTimeout(120);
  await check('pause_stops_world',async()=>await state(()=>PostalGame.scene.run.distance)===frozen);
  await page.locator('[data-resume]').click();
  await travel(1,false);
  await check('passing_on_road_does_not_pull_in_or_open_quiz',()=>state(()=>PostalGame.scene.phase==='driving' && PostalGame.scene.run.distance>284 && PostalGame.scene.run.missed.includes(1) && !PostalGame.scene.run.answers[1] && !document.querySelector('dialog[open]')));
  await shot('desktop-drive.png');await finishLap();
  await page.locator('[data-drive-again]').waitFor();
  await check('all_ten_mailboxes_can_be_missed_without_blocking_finish',()=>state(()=>PostalGame.scene.run.settled && PostalGame.scene.run.routeComplete && PostalGame.scene.count()===0 && PostalGame.scene.run.missed.length===10 && PostalGame.scene.score()===0));
  await check('missed_questions_are_not_available_as_quiz_only_retry',()=>page.locator('[data-retry]').count().then(n=>n===0));
  await shot('desktop-missed-result.png');
  await page.locator('[data-drive-again]').click();await travel(1,true,true);
  await check('parking_is_triggered_only_inside_the_painted_bay',()=>state(()=>{
   const s=PostalGame.scene,p=PostalArt.stations[0];return s.phase==='docking' && Math.abs(s.run.distance-p.s)<=p.halfLength && Math.abs(PostalArt.center(s.run.distance)+s.run.offset-PostalArt.center(p.s)-p.bayOffset)<=p.halfWidth;
  }));
  await check('docking_clears_controls',()=>state(()=>PostalGame.scene.keys.size===0 && PostalGame.scene.pointers.size===0));
  await page.locator('[data-action="pause"]').click();await check('docking_can_pause_safely',()=>state(()=>PostalGame.scene.phase==='paused' && !!PostalGame.scene.run.docking));
  await page.reload();await page.locator('#postal-canvas').waitFor();
  await check('reload_keeps_pending_docking_before_user_resumes',()=>state(()=>PostalGame.scene.phase==='intro' && PostalGame.scene.run.docking?.id===1 && PostalGame.scene.run.questionId===1));
  await start();await check('resuming_pending_docking_finishes_parking_before_quiz',()=>state(()=>PostalGame.scene.phase==='docking' && !!PostalGame.scene.run.docking && !document.querySelector('.post-option')));
  await travel(1,true);await page.locator('.post-option').first().waitFor();
  await check('dock_finishes_at_mailbox_before_opening_original_question',()=>state(()=>PostalGame.scene.phase==='quiz' && PostalGame.scene.run.distance===250 && PostalGame.scene.run.offset===64 && PostalGame.scene.questionIndex===0));
  await page.keyboard.press('ArrowRight');await page.waitForTimeout(80);
  await check('question_blocks_game_input',()=>state(()=>PostalGame.scene.keys.size===0 && PostalGame.scene.run.distance===250));
  await answer(true);const coinsOnce=await state(()=>GameState.getCoins());
  await state(()=>{const s=PostalGame.scene,q=s.data.questions[0];s.answer(q,q.answer,s.dialog);});
  await check('answer_and_letter_completion_are_awarded_once',async()=>await state(()=>GameState.getCoins())===coinsOnce && await state(()=>PostalGame.scene.score())===1);
  await check('original_question_and_explanation_remain_unchanged',async()=>{
   const q=await state(()=>PostalGame.scene.data.questions[0]);return await page.locator('.post-question').innerText()===q.question && await page.locator('.post-answer p').first().innerText()===q.explanation;
  });
  await shot('desktop-question.png');
  await page.reload();await page.locator('#postal-canvas').waitFor();await start();await page.locator('.post-continue').waitFor();
  await check('reload_restores_submitted_answer_without_double_award',async()=>await state(()=>GameState.getCoins())===coinsOnce && await state(()=>PostalGame.scene.score())===1);
  await continueLetter();await travel(2,true);await answer(false);await continueLetter();
  await check('wrong_letter_is_unfinished_and_waits_for_next_lap',()=>state(()=>!PostalGame.scene.mastery[2].correct && PostalGame.scene.run.passed.includes(2) && document.querySelector('[data-stamp="2"]').classList.contains('incorrect')));
  await finishLap();await page.locator('[data-drive-again]').click();
  await check('new_lap_preserves_completed_envelope_and_resets_run_answers',()=>state(()=>PostalGame.scene.score()===1 && PostalGame.scene.count()===0 && document.querySelector('[data-stamp="1"]').classList.contains('correct')));
  await travel(1,true);
  await check('previously_correct_mailbox_never_forces_another_question',()=>state(()=>PostalGame.scene.phase==='driving' && !PostalGame.scene.run.answers[1] && PostalGame.scene.score()===1));
  await travel(2,true);await answer(true);await continueLetter();
  await check('previous_wrong_question_can_be_completed_on_next_lap',()=>state(()=>PostalGame.scene.mastery[2].correct && PostalGame.scene.score()===2));
  for(let id=3;id<=10;id++) {
   await travel(id,true);assert.equal(await state(()=>PostalGame.scene.questionIndex),id-1);await answer(true);
   if(id===5) await check('cumulative_five_is_a_hint_not_an_early_finish',()=>state(()=>PostalGame.scene.score()===5 && !PostalGame.scene.run.settled && PostalGame.scene.phase==='quiz'));
   await continueLetter();
   if(id===7) await shot('desktop-post-office.png');
   if(id===10) await shot('desktop-lighthouse.png');
  }
  await finishLap();await page.locator('[data-map]').waitFor();
  await check('ten_letters_completed_across_laps_unlock_original_cat_and_crown',()=>state(()=>PostalGame.scene.score()===10 && PostalGame.scene.roundScore()===9 && GameState.isCatPerfect(1) && document.querySelector('.post-cat-result img').getAttribute('src')==='/static/sprites/cats/cat_01.png'));
  await shot('desktop-result.png');const settledCoins=await state(()=>GameState.getCoins()),attempts=await state(()=>GameState.getLevelProgress(1).attempts);
  await state(()=>{PostalGame.scene.finish();PostalGame.scene.finish();});await page.reload();await page.locator('[data-map]').waitFor();
  await check('reopening_result_does_not_repeat_coins_or_attempts',async()=>await state(()=>GameState.getCoins())===settledCoins && await state(()=>GameState.getLevelProgress(1).attempts)===attempts);
  await page.locator('[data-drive-again]').click();await travel(1,true);
  await check('completed_letters_do_not_award_again_on_revisit',()=>state(()=>PostalGame.scene.count()===0 && PostalGame.scene.run.coins===PostalGame.scene.run.collected.filter(id=>id.startsWith('coin-')).length*5 && PostalGame.scene.score()===10));
  await page.locator('[data-action="sound"]').click();
  await check('sound_toggle_mutes_and_persists',()=>state(()=>!PostalGame.scene.settings.sfx && !GameState.load().settings.sfx && document.querySelector('[data-action="sound"]').getAttribute('aria-pressed')==='false'));
  await page.locator('[data-action="sound"]').click();
  await check('sound_can_be_enabled_without_music',()=>state(()=>PostalGame.scene.settings.sfx && !PostalGame.scene.settings.music));
  await page.locator('[data-action="handbook"]').click();
  await check('handbook_contains_cross_lap_completed_letters',()=>page.locator('.post-notebook details').count().then(n=>n===10));await page.locator('[data-return]').click();
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(100);await shot('mobile-drive.png');
  await check('portrait_no_overflow_and_controls_fit',async()=>{const r=await page.locator('.postal-controls').boundingBox();return r.y+r.height<=844 && await state(()=>document.documentElement.scrollWidth<=innerWidth && document.querySelector('#postal-canvas').width===320);});
  await check('portrait_corner_widgets_do_not_overlap',async()=>{const a=await page.locator('.post-corner-progress').boundingBox(),b=await page.locator('.post-corner-actions').boundingBox();return a.x+a.width<=b.x;});
  const cdp=await context.newCDPSession(page),right=await page.locator('[data-drive="right"]').boundingBox(),brake=await page.locator('[data-drive="brake"]').boundingBox();
  const p1={x:right.x+right.width/2,y:right.y+right.height/2,id:1},p2={x:brake.x+brake.width/2,y:brake.y+brake.height/2,id:2};
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[p1]});await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[p1,p2]});
  await check('two_fingers_steer_and_brake_independently',()=>state(()=>PostalGame.scene.input('right') && PostalGame.scene.input('brake')));
  await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{...p1,x:15,y:15},p2]});await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[{...p1,x:15,y:15}]});
  await check('release_outside_clears_only_released_finger',()=>state(()=>!PostalGame.scene.input('right') && PostalGame.scene.input('brake')));
  await cdp.send('Input.dispatchTouchEvent',{type:'touchCancel',touchPoints:[]});await check('touch_cancel_clears_all_fingers',()=>state(()=>PostalGame.scene.pointers.size===0));
  await state(()=>window.dispatchEvent(new Event('blur')));await check('focus_loss_pauses_and_clears_input',()=>state(()=>PostalGame.scene.phase==='paused' && PostalGame.scene.keys.size===0));await page.locator('[data-resume]').click();
  await page.setViewportSize({width:844,height:390});await page.waitForTimeout(100);await shot('mobile-landscape-drive.png');
  await check('landscape_playfield_and_controls_fit',async()=>{const r=await page.locator('.postal-controls').boundingBox();return r.y+r.height<=390 && await state(()=>document.querySelector('#postal-canvas').width===640);});
  await state(()=>GameApp.showMap());await check('exit_removes_scene_modal_and_input_owner',()=>state(()=>!PostalGame.scene && !document.querySelector('dialog[open]') && !document.body.classList.contains('postal-playing')));
  await page.route('**/api/game/levels/1',async route=>{await new Promise(resolve=>setTimeout(resolve,250));await route.continue();});
  await state(()=>{GameApp.startLevel(1);GameApp.showMap();});await page.waitForTimeout(450);await check('late_response_cannot_reenter_exited_game',()=>state(()=>!PostalGame.scene));await page.unroute('**/api/game/levels/1');
  await page.route('**/api/game/levels/1',route=>route.fulfill({status:503,body:'unavailable'}));await page.locator('.postal-feature').click();await page.getByText('信件暂时没有装好').waitFor();
  await check('load_failure_has_retry_and_map',()=>page.locator('#view-postal button').count().then(n=>n===2));await page.unroute('**/api/game/levels/1');await page.getByRole('button',{name:'重试',exact:true}).click();await page.locator('#postal-canvas').waitFor();
  await check('retry_load_recovers',()=>state(()=>!!PostalGame.scene));
  await check('no_application_javascript_errors',async()=>errors.length===0);
  await context.close();
  // Migration and low-width checks use a second disposable context.
  const legacy=await browser.newContext({viewport:{width:390,height:667}}),lp=await legacy.newPage();await lp.goto(BASE+'/game');
  await lp.evaluate(()=>localStorage.setItem('crypto_cat_game',JSON.stringify({coins:123,levels:{1:{completed:true,bestScore:5},2:{perfect:true,bestScore:10}},cats:{1:{unlocked:true}},runs:{1:{id:'old-post-route',levelId:1,mode:'drive',distance:950,offset:0,coins:10,collected:[],answers:{1:{selected:1,correct:true,continued:true},2:{selected:1,correct:false,continued:true}}}}})));
  await lp.goto(BASE+'/game?level=1');await lp.locator('#postal-canvas').waitFor();
  await check('legacy_route_migrates_known_answers_without_inventing_question_ids',()=>lp.evaluate(()=>PostalGame.scene.mastery[1].correct && !PostalGame.scene.mastery[2].correct && PostalGame.scene.score()===1 && PostalGame.scene.run.distance===950 && GameState.getCoins()===123 && GameState.isCatUnlocked(1) && GameState.isCatPerfect(2)));
  await lp.screenshot({path:path.join(OUTPUT,'mobile-intro.png')});
  await lp.locator('[data-action="start"]').click();
  await check('starting_migrated_route_does_not_rewind_to_missed_questions',()=>lp.evaluate(()=>PostalGame.scene.phase==='driving' && PostalGame.scene.run.distance>=950));
  await lp.evaluate(()=>GameApp.showMap());await lp.evaluate(()=>{const s=GameState.load();s.runs[1]=null;s.knowledge[1]={};GameState.save();});await lp.evaluate(()=>GameApp.startLevel(1));await lp.locator('[data-action="start"]').click();
  await lp.evaluate(()=>{const s=PostalGame.scene;for(let n=0;n<500&&s.phase!=='quiz';n++){s.keys.clear();if(s.run.offset<50)s.keys.add('ArrowRight');s.cooldown=Math.max(0,s.cooldown-.05);s.time+=.05;s.update(.05);}s.clearInput();});
  await lp.locator('[data-option="0"]').click();await lp.waitForTimeout(100);await lp.screenshot({path:path.join(OUTPUT,'mobile-feedback.png')});
  await check('mobile_feedback_has_readable_scrollable_continue',async()=>{const d=await lp.locator('.postal-dialog').boundingBox();return d.x>=10 && d.y>=0 && d.y+d.height<=667 && await lp.locator('.post-continue').isVisible();});
  await lp.evaluate(()=>{GameApp.showMap();localStorage.setItem('crypto_cat_game','{broken');});await lp.reload();await lp.locator('#postal-canvas').waitFor();
  await check('corrupt_save_preserves_original_and_shows_notice',()=>lp.evaluate(()=>!document.getElementById('game-save-notice').hidden && localStorage.getItem('crypto_cat_game')==='{broken'));
  await legacy.close();
  const result={passed:true,checks,errors,limitations:'Desktop Chrome and emulated mobile viewports/touches. Keyboard entry and DOM buttons exercised; long drives accelerated through real scene updates. No physical-phone performance benchmark.'};
  fs.writeFileSync(path.join(OUTPUT,'verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
