// Synthetic observations only. No personal recordings or exported episode data.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {snapshot,poseSVG} = require('../src/care_vision/public_demo/caregiver.js');
const data={sourceWidth:1280,sourceHeight:720,poseDisplayResponse:0.3,
  bodyJointNames:Array.from({length:12},(_,i)=>'body '+i)};
const frame={t:1,fps:5,confirmed:true,bbox:[200,100,300,500],
  pose:{status:'ok',bodyPoints:Array.from({length:12},(_,i)=>[220+i*8,130+i*30,0.8])}};
const trial={duration:10,frames:[frame],states:[{t:0,state:'VERIFYING'},{t:3,state:'NORMAL'}],
  recordedStates:[{t:0,state:'VERIFYING'},{t:2,state:'UNRESOLVED'}],
  episodes:[{t:1,open:true,alert:false},{t:3,open:false,alert:false}],replay:true};
test('replay closure preserves the recorded state and original inputs',()=> {
  const before=JSON.stringify(trial), s=snapshot(data,trial,3);
  assert.equal(s.state,'NORMAL'); assert.equal(s.original,'UNRESOLVED');
  assert.equal(s.episode.open,false); assert.equal(s.alert,false);
  assert.equal(JSON.stringify(trial),before);
});
test('selects no future observation or episode',()=> {
  const s=snapshot(data,trial,0.9);
  assert.equal(s.frame,undefined);assert.equal(s.episode,undefined);assert.equal(s.count,0);
});
test('a stale observation cannot leave a held pose or box',()=> {
  assert.equal(snapshot(data,trial,2).count,12);
  const s=snapshot(data,trial,2.001);
  assert.equal(s.count,0);assert.equal(s.fresh,false);
  assert.doesNotMatch(poseSVG(data,s,true),/<circle|<line|<rect/);
});
test('unrequested pose is absent instead of copied from an earlier observation',()=> {
  const t={...trial,frames:[frame,{...frame,t:1.5,pose:{status:'not_requested',bodyPoints:null}}]};
  assert.equal(snapshot(data,t,1.49).count,12);
  assert.equal(snapshot(data,t,1.5).count,0);
});
test('rejects low response, nonfinite and out of bounds joints and their links',()=> {
  const points=structuredClone(frame.pose.bodyPoints);
  points[0]=[10,20,0.29]; points[3]=[NaN,20,0.9];points[9]=[1280,20,0.9];
  const t={...trial,frames:[{...frame,pose:{status:'ok',bodyPoints:points}}]};
  const s=snapshot(data,t,1); assert.equal(s.count,9);
  const svg=poseSVG(data,s,false);
  assert.equal((svg.match(/<circle /g)||[]).length,9);
  assert.equal((svg.match(/<line /g)||[]).length,5);
  assert.doesNotMatch(svg,/NaN|undefined|<image|<video|<audio/);
});
test('body-only schema cannot accidentally render a 17-point face pose',()=> {
  const t={...trial,frames:[{...frame,pose:{status:'ok',bodyPoints:Array(17).fill([10,20,0.9])}}]};
  assert.equal(snapshot(data,t,1).count,0);
});
test('the saved alert remains present through the end of the episode',()=> {
  const t={...trial,episodes:[{t:1,open:true,alert:false},{t:4,open:true,alert:true}],
    states:[{t:0,state:'VERIFYING'},{t:4,state:'ASSISTANCE_REQUIRED'}]};
  assert.equal(snapshot(data,t,3.999).alert,false);
  assert.equal(snapshot(data,t,4).alert,true);
  assert.equal(snapshot(data,t,10).state,'ASSISTANCE_REQUIRED');
});
