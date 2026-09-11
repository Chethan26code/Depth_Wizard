const canvas = document.getElementById('c');
const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true });
const scene  = new BABYLON.Scene(engine);
const $ = id => document.getElementById(id);

scene.clearColor = new BABYLON.Color4(0.478, 0.604, 0.706, 1);
scene.fogMode = BABYLON.Scene.FOGMODE_LINEAR;
scene.fogColor = new BABYLON.Color3(0.478, 0.604, 0.706);

let W = 512, H = 512, GSD = 1.0, exag = 1, relief = 0.70;
let base, structVals = null, positions, uvs, indices, normals, colors;
let hMin = 0, hMax = 1, mode = 'orbit';

function syntheticHeights(w, h){
  const a = new Float32Array(w*h);
  for(let z=0; z<h; z++) for(let x=0; x<w; x++){
    const u = x/w*8, v = z/h*8;
    let val = 18*Math.sin(u)*Math.cos(v) + 6*Math.sin(u*3.1)*Math.sin(v*2.7);
    if((x%97) < 26 && (z%89) < 22) val += 34;
    a[z*w+x] = val + 40;
  }
  return a;
}

const RAMP = [[.106,.227,.361],[.180,.490,.561],[.498,.690,.412],[.910,.769,.408],[.949,.949,.949]];
function ramp(t){
  t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
  const a = RAMP[i], b = RAMP[i+1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}

const ground = new BABYLON.Mesh('terrain', scene);
const mat = new BABYLON.StandardMaterial('mat', scene);
mat.specularColor = new BABYLON.Color3(0,0,0);
mat.diffuseColor  = new BABYLON.Color3(1,1,1);
mat.backFaceCulling = false;
ground.material = mat;
ground.receiveShadows = true;

function computeNormals(){
  BABYLON.VertexData.ComputeNormals(positions, indices, normals);
  let sy = 0; for(let i=1;i<normals.length;i+=3) sy += normals[i];
  if(sy < 0) for(let i=0;i<normals.length;i++) normals[i] = -normals[i];
}

function computeColors(){
  hMin = Infinity; hMax = -Infinity;
  for(let i=0;i<base.length;i++){
    if(base[i] < hMin) hMin = base[i];
    if(base[i] > hMax) hMax = base[i];
  }
  const span = (hMax - hMin) || 1;
  const hyp = $('cHyp').checked;
  const showStruct = $('cStruct').checked && structVals;
  for(let i=0;i<base.length;i++){
    const ny = Math.abs(normals[i*3+1]);
    const shade = (1 - relief) + relief * (0.30 + 0.70*ny);
    let c;
    if(showStruct){
      const s = structVals[i];               // 0 = ground, 1 = structure
      c = [0.55 + 0.4*s, 0.55 - 0.35*s, 0.55 - 0.35*s];  // reddish tint on structures
    } else {
      c = hyp ? ramp((base[i]-hMin)/span) : [1,1,1];
    }
    colors[i*4]   = c[0]*shade;
    colors[i*4+1] = c[1]*shade;
    colors[i*4+2] = c[2]*shade;
    colors[i*4+3] = 1;
  }
}

function buildTerrain(){
  positions = new Float32Array(W*H*3);
  uvs       = new Float32Array(W*H*2);
  colors    = new Float32Array(W*H*4);
  normals   = new Float32Array(W*H*3);

  for(let z=0; z<H; z++) for(let x=0; x<W; x++){
    const i = z*W + x;
    positions[i*3  ] = (x - W/2) * GSD;
    positions[i*3+1] = base[i] * exag;
    positions[i*3+2] = (z - H/2) * GSD;
    uvs[i*2]   = x/(W-1);
    uvs[i*2+1] = 1 - z/(H-1);
  }
  indices = new Uint32Array((W-1)*(H-1)*6);
  let k = 0;
  for(let z=0; z<H-1; z++) for(let x=0; x<W-1; x++){
    const a = z*W+x, b = a+1, c = a+W, d = c+1;
    indices[k++]=a; indices[k++]=c; indices[k++]=b;
    indices[k++]=b; indices[k++]=c; indices[k++]=d;
  }
  computeNormals();
  computeColors();

  const vd = new BABYLON.VertexData();
  vd.positions = positions; vd.indices = indices;
  vd.uvs = uvs; vd.normals = normals; vd.colors = colors;
  vd.applyToMesh(ground, true);

  const spanMax = Math.max(W, H) * GSD;
  scene.fogStart = spanMax*0.8;
  scene.fogEnd   = spanMax*2.6;
  refreshUI(); frameCamera(); drawMinimap(); placeSun();
}

function reheight(){
  for(let i=0;i<W*H;i++) positions[i*3+1] = base[i] * exag;
  computeNormals(); computeColors();
  ground.updateVerticesData(BABYLON.VertexBuffer.PositionKind, positions);
  ground.updateVerticesData(BABYLON.VertexBuffer.NormalKind, normals);
  ground.updateVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
}
function recolor(){
  computeColors();
  ground.updateVerticesData(BABYLON.VertexBuffer.ColorKind, colors);
}

const hemi = new BABYLON.HemisphericLight('h', new BABYLON.Vector3(0,1,0), scene);
hemi.intensity = 0.55;
hemi.diffuse = new BABYLON.Color3(0.78,0.86,0.95);
hemi.groundColor = new BABYLON.Color3(0.22,0.21,0.19);

const sun = new BABYLON.DirectionalLight('sun', new BABYLON.Vector3(-1,-1,-1), scene);
sun.intensity = 1.5;
sun.diffuse = new BABYLON.Color3(1, 0.96, 0.88);

const shadows = new BABYLON.ShadowGenerator(2048, sun);
shadows.usePercentageCloserFiltering = true;
shadows.filteringQuality = BABYLON.ShadowGenerator.QUALITY_MEDIUM;
shadows.bias = 0.0012;
shadows.normalBias = 0.02;
shadows.getShadowMap().renderList.push(ground);

function placeSun(){
  const az = parseFloat($('az').value) * Math.PI/180;
  const el = parseFloat($('el').value) * Math.PI/180;
  const dir = new BABYLON.Vector3(
    -Math.cos(el)*Math.sin(az), -Math.sin(el), -Math.cos(el)*Math.cos(az));
  sun.direction = dir;
  const spanMax = Math.max(W, H) * GSD;
  sun.position = dir.scale(-spanMax*1.4);
  sun.shadowMinZ = 1;
  sun.shadowMaxZ = spanMax*3.5;
  sun.autoUpdateExtends = true;
}

const orbit = new BABYLON.ArcRotateCamera('orbit', -Math.PI/2, 1.05, 500,
                  new BABYLON.Vector3(0,40,0), scene);
orbit.wheelPrecision = 0.6;
orbit.panningSensibility = 30;

const fly = new BABYLON.UniversalCamera('fly', new BABYLON.Vector3(0,180,-240), scene);
fly.angularSensibility = 3000; fly.minZ = 0.5;
fly.keysUp.push(87);     fly.keysDown.push(83);
fly.keysLeft.push(65);   fly.keysRight.push(68);
fly.keysUpward.push(69); fly.keysDownward.push(81);

function frameCamera(){
  const spanMax = Math.max(W, H) * GSD;
  orbit.target.set(0, (hMin+hMax)/2 * exag, 0);
  orbit.radius = spanMax * 0.85;
  orbit.alpha = -Math.PI/2; orbit.beta = 1.02;
  orbit.lowerRadiusLimit = spanMax * 0.05;
  orbit.upperRadiusLimit = spanMax * 4;
  orbit.maxZ = spanMax * 8; fly.maxZ = spanMax * 8;
  fly.speed = spanMax / 90;
  fly.position.set(0, hMax*exag + spanMax*0.22, -spanMax*0.45);
  fly.setTarget(new BABYLON.Vector3(0, hMin*exag, 0));
}

function setMode(m){
  mode = m;
  scene.activeCamera.detachControl();
  scene.activeCamera = (m === 'orbit') ? orbit : fly;
  scene.activeCamera.attachControl(canvas, true);
  $('bOrbit').classList.toggle('on', m === 'orbit');
  $('bFly').classList.toggle('on', m === 'fly');
}
scene.activeCamera = orbit; orbit.attachControl(canvas, true);
$('bOrbit').onclick = () => setMode('orbit');
$('bFly').onclick   = () => setMode('fly');
$('bReset').onclick = () => { frameCamera(); setMode('orbit'); };

const dotMat = new BABYLON.StandardMaterial('dm', scene);
dotMat.emissiveColor = new BABYLON.Color3(1,.54,.24);
dotMat.disableLighting = true;
const marks = []; let prev = null;

function clearMarks(){
  marks.forEach(m => m.dispose()); marks.length = 0; prev = null;
  $('readout').innerHTML = '<span class="hint">Click a rooftop to read its height.</span>';
}
$('bClear').onclick = clearMarks;

scene.onPointerObservable.add(pi => {
  if(pi.type !== BABYLON.PointerEventTypes.POINTERPICK || mode !== 'orbit') return;
  const p = pi.pickInfo;
  if(!p || !p.hit || p.pickedMesh !== ground) return;

  const spanMax = Math.max(W, H) * GSD;
  const d = BABYLON.MeshBuilder.CreateSphere('mk', { diameter: spanMax/140 }, scene);
  d.material = dotMat; d.isPickable = false;
  d.position.copyFrom(p.pickedPoint); marks.push(d);

  const trueH = p.pickedPoint.y / exag;
  const nrm   = p.getNormal(true);
  const sEx   = Math.acos(Math.min(1, Math.abs(nrm.y)));
  const slope = BABYLON.Tools.ToDegrees(Math.atan(Math.tan(sEx) / exag));

  let t = `Height <b>${trueH.toFixed(1)} m</b> &nbsp;·&nbsp; slope <b>${slope.toFixed(0)}°</b>`;
  if(prev){
    const dist = BABYLON.Vector3.Distance(
      new BABYLON.Vector3(prev.x, prev.y/exag, prev.z),
      new BABYLON.Vector3(p.pickedPoint.x, trueH, p.pickedPoint.z));
    t += ` &nbsp;·&nbsp; distance <b>${dist.toFixed(1)} m</b>`;
  }
  prev = p.pickedPoint.clone();
  $('readout').innerHTML = t;
});

function drawMinimap(){
  if(window._texUrl && $('cTex').checked){ $('miniImg').src = window._texUrl; return; }
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(W, H);
  const span = (hMax - hMin) || 1;
  for(let i=0;i<W*H;i++){
    const c = ramp((base[i]-hMin)/span);
    img.data[i*4]=c[0]*255; img.data[i*4+1]=c[1]*255;
    img.data[i*4+2]=c[2]*255; img.data[i*4+3]=255;
  }
  ctx.putImageData(img, 0, 0);
  $('miniImg').src = cv.toDataURL();
}

function refreshUI(){
  const spanX = W*GSD, spanZ = H*GSD;
  $('meta') && ($('meta').innerHTML = '');
  $('bMin').textContent = hMin.toFixed(1) + ' m';
  $('bMax').textContent = hMax.toFixed(1) + ' m';
  $('sExtent').textContent = `${spanX.toFixed(0)} × ${spanZ.toFixed(0)} m`;
  $('sGrid').textContent = `${W} × ${H} @ ${GSD} m/px`;
}

$('ex').oninput = e => {
  exag = parseFloat(e.target.value);
  $('exVal').textContent = exag.toFixed(1) + '×';
  reheight();
};
$('relief').oninput = e => {
  relief = parseFloat(e.target.value)/100;
  $('reliefVal').textContent = e.target.value + '%';
  recolor();
};
$('az').oninput = e => { $('azVal').textContent = e.target.value + '°'; placeSun(); };
$('el').oninput = e => { $('elVal').textContent = e.target.value + '°'; placeSun(); };

function syncMaterial(){
  const tex = $('cTex').checked;
  mat.diffuseTexture = (tex && window._tex) ? window._tex : null;
  ground.useVertexColors = true;
  mat.diffuseColor = (tex && window._tex)
      ? new BABYLON.Color3(1,1,1)
      : ($('cHyp').checked ? new BABYLON.Color3(1,1,1) : new BABYLON.Color3(.62,.66,.69));
  mat.wireframe = $('cWire').checked;
  shadows.getShadowMap().renderList.length = 0;
  if($('cShadow').checked) shadows.getShadowMap().renderList.push(ground);
  ground.receiveShadows = $('cShadow').checked;
  recolor(); drawMinimap();
}
$('cTex').onchange = $('cHyp').onchange = $('cWire').onchange =
  $('cShadow').onchange = $('cStruct').onchange = syncMaterial;

function useTexture(url){
  const t = new BABYLON.Texture(url, scene);
  t.anisotropicFilteringLevel = 8;
  t.onLoadObservable.addOnce(() => {
    window._tex = t; window._texUrl = url;
    $('cTex').checked = true;
    syncMaterial();
  });
}

function loadScene(w, h, gsd, heightArr, structArr, textureUrl){
  W = w; H = h; GSD = gsd;
  base = heightArr;
  structVals = structArr;
  buildTerrain();
  if(textureUrl) useTexture(textureUrl);
  syncMaterial();
  clearMarks();
}

/* ---------- FULL PIPELINE upload ---------- */
$('fProcess').onchange = async e => {
  const f = e.target.files[0]; if(!f) return;
  $('load').classList.remove('hide'); $('loadMsg').textContent = 'Processing image…';
  $('processStatus').textContent = 'Uploading and running pipeline…';

  const formData = new FormData();
  formData.append('image', f);

  try{
    const res = await fetch('/api/process', { method: 'POST', body: formData });
    const data = await res.json();
    if(data.error) throw new Error(data.error);

    const [dsmBuf, structBuf] = await Promise.all([
      fetch(data.dsm_url).then(r => r.arrayBuffer()),
      fetch(data.struct_url).then(r => r.arrayBuffer()),
    ]);
    const heightArr = new Float32Array(dsmBuf);
    const structArr = new Float32Array(structBuf);

    loadScene(data.width, data.height, data.gsd, heightArr, structArr, data.rgb_url);
    $('processStatus').textContent =
      `Done. ${data.georeferenced ? 'Georeferenced' : 'Non-georeferenced'} · ` +
      `${(data.valid_fraction*100).toFixed(0)}% valid data.`;
  }catch(err){
    $('processStatus').textContent = 'Error: ' + err.message;
  }
  $('load').classList.add('hide');
};

/* ---------- DIRECT (pre-made dsm) upload ---------- */
let directDsmFile = null, directRgbFile = null;
$('fDsmDirect').onchange = e => {
  directDsmFile = e.target.files[0];
  const isRaw = directDsmFile && /\.(f32|bin|raw)$/i.test(directDsmFile.name);
  $('rawDims').style.display = isRaw ? 'flex' : 'none';
};
$('fRgbDirect').onchange = e => { directRgbFile = e.target.files[0]; };

$('bLoadDirect').onclick = async () => {
  if(!directDsmFile || !directRgbFile){
    $('directStatus').textContent = 'Choose both a heights file and an imagery file first.';
    return;
  }
  $('load').classList.remove('hide'); $('loadMsg').textContent = 'Loading…';

  const formData = new FormData();
  formData.append('dsm', directDsmFile);
  formData.append('rgb', directRgbFile);
  formData.append('gsd', $('gsdDirect').value);
  if(/\.(f32|bin|raw)$/i.test(directDsmFile.name)){
    formData.append('width', $('rawWidth').value);
    formData.append('height', $('rawHeight').value);
  }

  try{
    const res = await fetch('/api/load_direct', { method: 'POST', body: formData });
    const data = await res.json();
    if(data.error) throw new Error(data.error);

    const dsmBuf = await fetch(data.dsm_url).then(r => r.arrayBuffer());
    const heightArr = new Float32Array(dsmBuf);

    loadScene(data.width, data.height, data.gsd, heightArr, null, data.rgb_url);
    $('directStatus').textContent = 'Loaded.';
  }catch(err){
    $('directStatus').textContent = 'Error: ' + err.message;
  }
  $('load').classList.add('hide');
};

/* ---------- boot with synthetic terrain ---------- */
(function boot(){
  W = 512; H = 512;
  base = syntheticHeights(W, H);
  buildTerrain();
  syncMaterial();
  $('load').classList.add('hide');
})();

let acc = 0;
engine.runRenderLoop(() => {
  scene.render();
  if(++acc % 12 === 0){
    const cam = scene.activeCamera;
    $('sAlt').textContent = cam.position.y.toFixed(0) + ' m';
    $('sFps').textContent = engine.getFps().toFixed(0) + ' fps';
    const spanX = W*GSD, spanZ = H*GSD;
    const px = (cam.position.x + spanX/2) / spanX;
    const pz = (cam.position.z + spanZ/2) / spanZ;
    $('dot').style.left = Math.max(0,Math.min(1,px))*100 + '%';
    $('dot').style.top  = Math.max(0,Math.min(1,pz))*100 + '%';
  }
});
addEventListener('resize', () => engine.resize());