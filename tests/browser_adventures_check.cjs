const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'/Users/anselkocen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright-core');
const BASE=process.env.GAME_TEST_URL||'http://127.0.0.1:8000',OUT=path.resolve(__dirname,'../artifacts/ui_checks/adventures'),driver=fs.readFileSync(path.join(__dirname,'adventure_test_driver.js'),'utf8');
(async()=>{fs.mkdirSync(OUT,{recursive:true});const browser=await chromium.launch({executablePath:process.env.CHROME_BIN||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});try{
 const context=await browser.newContext({viewport:{width:1360,height:900}}),page=await context.newPage(),checks={},errors=[],questions={};page.on('pageerror',e=>errors.push(e.message));
 const check=async(name,fn)=>{assert.ok(await fn(),name);checks[name]=true;},state=fn=>page.evaluate(fn),inject=()=>page.addScriptTag({content:driver});
 const shot=async(name)=>{await page.waitForTimeout(70);await page.screenshot({path:path.join(OUT,name)});};
 const start=()=>page.locator('[data-action="start"]').click();
 for(let level=3;level<=10;level++){
  await page.goto(BASE+'/game?level='+level);await page.locator('[data-action="start"]').waitFor();await inject();
  const otherBefore=await state(()=>JSON.stringify(Object.fromEntries(Object.entries(GameState.load().knowledge).filter(([id])=>Number(id)!==AdventureGames.scene.levelId))));
  await check('level_'+level+'_starts_with_guide_and_no_legacy_engine',()=>state(()=>AdventureGames.scene.phase==='intro'&&!window.Phaser&&document.querySelectorAll('[data-stamp]').length===10));await start();await shot('level-'+level+'-play.png');
  let iterations=0;questions[level]=[];
  while(await state(()=>AdventureGames.scene.phase)!=='result'&&iterations++<18){
   const result=await state(()=>driveAdventure());
   if(result.phase==='playing')throw Error('Driver stuck level '+level+' '+JSON.stringify(result));
   if(result.phase==='result')break;
   assert.equal(result.phase,'quiz');const q=await state(()=>AdventureGames.scene.data.questions[AdventureGames.scene.questionIndex]);
   assert.ok(!questions[level].includes(q.id),'Duplicate question '+level+':'+q.id);questions[level].push(q.id);
   await check('original_question_'+level+'_'+q.id,async()=>await page.locator('.post-question').innerText()===q.question&&await page.locator('[data-option]').count()===4);
   if(questions[level].length===1){const frozen=await state(()=>JSON.stringify(AdventureGames.scene.run.world));await page.keyboard.press('Space');await page.keyboard.press('ArrowRight');await page.waitForTimeout(60);await check('level_'+level+'_quiz_freezes_world_and_blocks_game_keys',async()=>await state(()=>JSON.stringify(AdventureGames.scene.run.world))===frozen&&await state(()=>AdventureGames.scene.phase)==='quiz');}
   await page.locator(`[data-option="${q.answer}"]`).click();const balance=await state(()=>GameState.getCoins());await state(()=>{const s=AdventureGames.scene,q=s.data.questions[s.questionIndex];s.answer(q,q.answer,s.dialog);});await check('single_award_'+level+'_'+q.id,async()=>await state(()=>GameState.getCoins())===balance&&await page.locator('.post-answer p').first().innerText()===q.explanation);
   if(questions[level].length===1){await shot('level-'+level+'-question.png');await page.reload();await page.locator('[data-action="start"]').waitFor();await inject();await start();await page.locator('.post-continue').waitFor();await check('level_'+level+'_reload_keeps_analysis_and_coins',async()=>await state(()=>GameState.getCoins())===balance&&await state(()=>AdventureGames.scene.score())===1);}
   await page.locator('.post-continue').click();
  }
  await check('level_'+level+'_all_ten_questions_and_correct_original_reward',()=>state(()=>{const s=AdventureGames.scene,cat=window._gameData.cats.find(c=>c.level===s.levelId);return s.phase==='result'&&s.count()===10&&s.score()===10&&s.run.routeComplete&&GameState.isCatPerfect(s.levelId)&&document.querySelector('.post-cat-result img').getAttribute('src')==='/static/sprites/cats/'+cat.sprite;}));
  await check('level_'+level+'_cannot_overwrite_other_levels_progress',async()=>await state(()=>JSON.stringify(Object.fromEntries(Object.entries(GameState.load().knowledge).filter(([id])=>Number(id)!==AdventureGames.scene.levelId))))===otherBefore);
  await shot('level-'+level+'-result.png');const coins=await state(()=>GameState.getCoins()),attempts=await state(()=>GameState.getLevelProgress(AdventureGames.scene.levelId).attempts);await state(()=>{AdventureGames.scene.finish();AdventureGames.scene.finish();});await page.reload();await page.locator('[data-map]').waitFor();await check('level_'+level+'_result_reload_is_idempotent',async()=>await state(()=>GameState.getCoins())===coins&&await state(()=>GameState.getLevelProgress(AdventureGames.scene.levelId).attempts)===attempts);
  console.log('completed',level,questions[level]);
 }
 await check('all_eighty_new_questions_covered',async()=>Object.values(questions).every(ids=>ids.length===10&&new Set(ids).size===10));
 await check('family_photo_keeps_its_original_wide_proportions',()=>state(()=>{const img=document.querySelector('.post-cat-result img'),r=img.getBoundingClientRect();return img.naturalWidth>img.naturalHeight&&r.width/r.height>1.8&&getComputedStyle(img).objectFit==='contain';}));
 await page.locator('[data-map]').click();await page.locator('#btn-gallery').click();await check('gallery_displays_eight_new_original_cats',()=>page.locator('.cat-card.unlocked').count().then(n=>n===8));
 await state(()=>GameApp.showMap());await check('all_ten_map_entries_are_new_playable_games',()=>state(()=>document.querySelectorAll('[data-start-level]').length===10&&!document.querySelector('.legacy-label')));
 await check('no_application_javascript_errors',async()=>errors.length===0);const result={passed:true,checks,questions,errors,limitations:'Long movement and physical interactions accelerated through actual game update logic and input actions; answers via real DOM buttons. Mobile controls tested separately. No physical-device performance benchmark.'};fs.writeFileSync(path.join(OUT,'verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({passed:true,checks:Object.keys(checks).length,questions,errors},null,2));
 }finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
