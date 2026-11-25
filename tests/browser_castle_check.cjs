/* Complete second-game checks in disposable Chrome; no daily browser profile or user save access. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/Users/anselkocen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core');
const OUT=path.resolve(__dirname,'../artifacts/ui_checks/castle'),BASE=process.env.GAME_TEST_URL||'http://127.0.0.1:8000';
(async()=>{
 fs.mkdirSync(OUT,{recursive:true});const browser=await chromium.launch({executablePath:process.env.CHROME_BIN||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
 try{
  const context=await browser.newContext({viewport:{width:1360,height:900}}),page=await context.newPage(),checks={},errors=[];page.on('pageerror',e=>errors.push(e.message));
  const state=fn=>page.evaluate(fn),check=async(name,fn)=>{assert.ok(await fn(),name);checks[name]=true;},shot=async(name)=>{await page.waitForTimeout(100);await page.screenshot({path:path.join(OUT,name)});};
  const walk=async({target=5200,jump=true,stopOnRecover=false}={})=>page.evaluate(({target,jump,stopOnRecover})=>{
   const s=CastleGame.scene;
   for(let n=0;n<30000&&['playing','recovering'].includes(s.phase);n++){
    s.keys.clear();s.keys.add('ArrowRight');const gap=CastleArt.gaps.find(g=>g.x+g.w>s.run.x);
    if(jump&&gap&&s.run.grounded&&gap.x-s.run.x<28&&gap.x-s.run.x> -10)s.jumpBuffer=.12;
    s.cooldown=Math.max(0,s.cooldown-.02);s.time+=.02;s.update(.02);
    if(stopOnRecover&&s.phase==='recovering')break;if(s.run.x>=target)break;
   }
   s.clearInput();return {phase:s.phase,x:s.run.x,y:s.run.y,questionId:s.run.questionId,falls:s.run.falls};
  },{target,jump,stopOnRecover});
  const answer=async(correct)=>{const idx=await state(()=>CastleGame.scene.data.questions[CastleGame.scene.questionIndex].answer);await page.locator(`[data-option="${correct?idx:(idx+1)%4}"]`).click();};
  const advance=()=>page.locator('.post-continue').click();
  await page.goto(BASE+'/game');await page.locator('.castle-feature').waitFor();
  await state(()=>{GameState.load().knowledge[1]={1:{selected:1,correct:true}};GameState.save();});const postalBefore=await state(()=>JSON.stringify(GameState.load().knowledge[1]));
  await page.locator('.castle-feature').click();await page.locator('#postal-canvas').waitFor();
  await check('second_game_uses_new_castle_scene_without_legacy_engine',()=>state(()=>!!CastleGame.scene&&!PostalGame.scene&&!window.Phaser));
  await check('intro_explains_jumping_boxes_and_recovery',()=>page.locator('.post-guide').innerText().then(t=>t.includes('空格')&&t.includes('知识箱')&&t.includes('长椅')));await shot('desktop-intro.png');
  await page.locator('[data-action="start"]').click();await page.keyboard.down('ArrowRight');await page.waitForTimeout(350);await page.keyboard.press('Space');await page.waitForTimeout(80);
  await check('real_keyboard_can_run_and_jump_together',()=>state(()=>CastleGame.scene.run.vx>0&&CastleGame.scene.run.vy<0&&!CastleGame.scene.run.grounded));await page.keyboard.up('ArrowRight');await page.waitForTimeout(850);
  await check('jump_returns_to_a_solid_platform',()=>state(()=>CastleGame.scene.run.grounded&&CastleGame.scene.run.y===326));
  await check('desktop_canvas_is_large_and_controls_fit',async()=>{const c=await page.locator('canvas').boundingBox(),b=await page.locator('.postal-controls').boundingBox();return c.width>1000&&c.height>600&&b.y+b.height<=900;});
  await shot('desktop-drive.png');await walk({target:300});await page.locator('.post-option').first().waitFor();
  await check('knowledge_box_opens_on_actual_ground_contact',()=>state(()=>CastleGame.scene.questionIndex===0&&CastleGame.scene.run.grounded&&Math.abs(CastleGame.scene.run.x-270)<=28));
  const stopped=await state(()=>({x:CastleGame.scene.run.x,y:CastleGame.scene.run.y,time:CastleGame.scene.time}));await page.keyboard.press('ArrowRight');await page.keyboard.press('Space');await page.waitForTimeout(100);
  await check('quiz_freezes_player_and_scene_and_blocks_jump_input',async()=>await state(()=>CastleGame.scene.run.x)===stopped.x&&await state(()=>CastleGame.scene.run.y)===stopped.y&&await state(()=>CastleGame.scene.time)===stopped.time&&await state(()=>CastleGame.scene.jumpBuffer)===0);
  const q=await state(()=>CastleGame.scene.data.questions[0]);await answer(true);
  await check('uses_original_level_two_question_options_and_explanation',async()=>await page.locator('.post-question').innerText()===q.question&&await page.locator('.post-answer p').first().innerText()===q.explanation&&await page.locator('[data-option]').count()===q.options.length);
  await shot('desktop-question.png');const once=await state(()=>GameState.getCoins());
  await state(()=>{const s=CastleGame.scene,q=s.data.questions[0];s.answer(q,q.answer,s.dialog);});await check('duplicate_answer_does_not_award_twice',async()=>await state(()=>GameState.getCoins())===once);
  await page.goto(BASE+'/game?level=2');await page.locator('[data-action="start"]').click();await page.locator('.post-continue').waitFor();
  await check('reload_restores_level_two_explanation_without_double_award',async()=>await state(()=>CastleGame.scene.score())===1&&await state(()=>GameState.getCoins())===once);await advance();
  const beforeFall=await state(()=>({coins:GameState.getCoins(),answer:CastleGame.scene.mastery[1].correct}));await walk({jump:false,stopOnRecover:true});
  await check('fall_recovers_to_nearby_checkpoint_without_losing_answers_or_coins',async()=>await state(()=>CastleGame.scene.phase==='recovering'&&CastleGame.scene.run.x===CastleGame.scene.run.checkpoint.x&&CastleGame.scene.mastery[1].correct)&&await state(()=>GameState.getCoins())>=beforeFall.coins);
  await state(()=>{CastleGame.scene.update(.6);});await walk({jump:false,stopOnRecover:true});
  await check('two_falls_build_a_bridge_only_for_the_failed_gap',()=>state(()=>CastleGame.scene.run.falls===2&&CastleGame.scene.run.bridges.length===1&&CastleGame.scene.run.bridges[0]===0));
  await state(()=>CastleGame.scene.update(.6));await walk({jump:false});await page.locator('.post-option').first().waitFor();
  await check('bridge_allows_walking_to_next_box_without_a_jump',()=>state(()=>CastleGame.scene.questionIndex===1&&CastleGame.scene.run.x>664));await answer(false);await advance();
  await check('wrong_answer_continues_without_game_over',()=>state(()=>CastleGame.scene.phase==='playing'&&!CastleGame.scene.mastery[2].correct&&CastleGame.scene.count()===2));
  for(let id=3;id<=10;id++){
   await walk();await page.locator('.post-option').first().waitFor();assert.equal(await state(()=>CastleGame.scene.questionIndex),id-1);await answer(true);
   if(id===6)await check('fifth_correct_does_not_end_the_adventure',()=>state(()=>CastleGame.scene.score()===5&&!CastleGame.scene.run.settled));await advance();
   if(id===5)await shot('desktop-clocktower.png');if(id===9)await shot('desktop-study.png');
  }
  await walk();await page.locator('[data-map]').waitFor();
  await check('reaches_study_after_all_ten_original_questions',()=>state(()=>CastleGame.scene.count()===10&&CastleGame.scene.run.routeComplete&&CastleGame.scene.run.settled&&CastleGame.scene.run.x>=5105));
  await check('level_two_reward_is_original_vitalik_cat_and_preserves_postal_progress',async()=>await state(()=>GameState.isCatUnlocked(2)&&!GameState.isCatPerfect(2)&&document.querySelector('.post-cat-result img').getAttribute('src')==='/static/sprites/cats/cat_02.png')&&await state(()=>JSON.stringify(GameState.load().knowledge[1]))===postalBefore);await shot('desktop-result.png');
  const coins=await state(()=>GameState.getCoins()),attempts=await state(()=>GameState.getLevelProgress(2).attempts);await state(()=>{CastleGame.scene.finish();CastleGame.scene.finish();});await page.reload();await page.locator('[data-map]').waitFor();
  await check('result_reload_does_not_repeat_reward_or_attempt',async()=>await state(()=>GameState.getCoins())===coins&&await state(()=>GameState.getLevelProgress(2).attempts)===attempts);
  await page.locator('[data-drive-again]').click();await walk();await page.locator('.post-option').first().waitFor();
  await check('revisit_skips_completed_boxes_and_retries_only_unfinished_one',()=>state(()=>CastleGame.scene.questionIndex===1&&CastleGame.scene.count()===0&&CastleGame.scene.score()===9));await answer(true);await advance();await walk();await page.locator('[data-map]').waitFor();
  await check('cumulative_ten_across_runs_awards_level_two_crown',()=>state(()=>GameState.isCatPerfect(2)&&CastleGame.scene.score()===10&&CastleGame.scene.count()===1));
  await page.locator('[data-map]').click();await page.locator('#btn-gallery').click();
  await check('gallery_opens_after_unlock_and_shows_original_cat',()=>page.locator('.cat-card.unlocked img[src="/static/sprites/cats/cat_02.png"]').count().then(n=>n===1));
  await state(()=>GameApp.startLevel(2));await page.locator('[data-drive-again]').click();
  await page.locator('[data-action="pause"]').click();const paused=await state(()=>CastleGame.scene.run.x);await page.waitForTimeout(100);await check('pause_stops_platform_physics',async()=>await state(()=>CastleGame.scene.run.x)===paused);await page.locator('[data-resume]').click();
  await page.locator('[data-action="sound"]').click();await check('sound_can_be_muted_and_setting_is_saved',()=>state(()=>!CastleGame.scene.settings.sfx&&!GameState.load().settings.sfx));
  await page.setViewportSize({width:390,height:844});await page.waitForTimeout(120);await shot('mobile-drive.png');
  await check('portrait_canvas_and_controls_fit_without_overflow',async()=>{const r=await page.locator('.postal-controls').boundingBox();return r.y+r.height<=844&&await state(()=>document.documentElement.scrollWidth<=innerWidth);});
  const cdp=await context.newCDPSession(page),right=await page.locator('[data-drive="right"]').boundingBox(),jump=await page.locator('[data-drive="jump"]').boundingBox();
  const p1={id:1,x:right.x+right.width/2,y:right.y+right.height/2},p2={id:2,x:jump.x+jump.width/2,y:jump.y+jump.height/2};
  await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[p1]});await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[p1,p2]});await page.waitForTimeout(60);
  await check('two_finger_running_jump_keeps_direction_held',()=>state(()=>CastleGame.scene.input('right')&&CastleGame.scene.input('jump')&&CastleGame.scene.run.vx>0&&CastleGame.scene.run.vy<0));
  await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[p2]});await check('releasing_jump_keeps_the_other_finger_direction',()=>state(()=>CastleGame.scene.input('right')&&!CastleGame.scene.input('jump')));
  await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{...p1,x:10,y:10}]});await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[{...p1,x:10,y:10}]});await check('release_outside_clears_direction',()=>state(()=>!CastleGame.scene.input('right')&&CastleGame.scene.pointers.size===0));
  await state(()=>window.dispatchEvent(new Event('blur')));await check('focus_loss_pauses_and_clears_all_input',()=>state(()=>CastleGame.scene.phase==='paused'&&CastleGame.scene.keys.size===0&&CastleGame.scene.jumpBuffer===0));await page.locator('[data-resume]').click();
  await page.setViewportSize({width:844,height:390});await page.waitForTimeout(100);await shot('mobile-landscape.png');await check('landscape_controls_fit_screen',async()=>{const r=await page.locator('.postal-controls').boundingBox();return r.y+r.height<=390;});
  await state(()=>GameApp.showMap());await check('exit_cleans_castle_body_state_and_dialogs',()=>state(()=>!CastleGame.scene&&!document.body.classList.contains('castle-playing')&&!document.querySelector('dialog[open]')));
  await page.route('**/api/game/levels/2',async route=>{await new Promise(resolve=>setTimeout(resolve,300));await route.continue();});await state(()=>{GameApp.startLevel(2);GameApp.startLevel(1);});await page.waitForTimeout(500);
  await check('late_castle_load_cannot_replace_newer_postal_scene',()=>state(()=>!!PostalGame.scene&&!CastleGame.scene&&GameApp.currentLevelId===1));await page.unroute('**/api/game/levels/2');await state(()=>GameApp.showMap());
  await page.route('**/api/game/levels/2',route=>route.fulfill({status:503,body:'unavailable'}));await page.locator('.castle-feature').click();await page.getByText('城堡暂时没有打开').waitFor();await check('castle_load_failure_has_retry_and_map',()=>page.locator('#view-castle button').count().then(n=>n===2));await page.unroute('**/api/game/levels/2');await page.locator('[data-castle-retry]').click();await page.locator('canvas').waitFor();await check('castle_retry_recovers',()=>state(()=>!!CastleGame.scene));
  await check('no_application_javascript_errors',async()=>errors.length===0);await context.close();
  const mobile=await browser.newContext({viewport:{width:390,height:667}}),mp=await mobile.newPage();await mp.goto(BASE+'/game?level=2');await mp.locator('canvas').waitFor();await mp.screenshot({path:path.join(OUT,'mobile-intro.png')});await mp.locator('[data-action="start"]').click();
  await mp.evaluate(()=>{const s=CastleGame.scene;for(let i=0;i<500&&s.phase==='playing';i++){s.keys.add('ArrowRight');s.time+=.02;s.update(.02);}s.clearInput();});await mp.locator('[data-option="0"]').click();await mp.waitForTimeout(100);await mp.screenshot({path:path.join(OUT,'mobile-feedback.png')});
  await check('small_phone_feedback_scrolls_and_continue_is_reachable',async()=>{const r=await mp.locator('.postal-dialog').boundingBox();return r.y>=0&&r.y+r.height<=667&&await mp.locator('.post-continue').isVisible();});await mobile.close();
  const result={passed:true,checks,errors,limitations:'Desktop Chrome and emulated phones; long travel accelerated using platform physics and steering/jump inputs. No physical-device performance benchmark.'};fs.writeFileSync(path.join(OUT,'verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
