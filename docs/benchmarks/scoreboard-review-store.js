'use strict';
// Preserve v1 verbatim. Migration carries old result-level findings into a draft incident for reconciliation.
(() => {
const KEY = 'tilawa.manual-reviews.v2', LEGACY_KEY = 'tilawa.manual-reviews.v1';
const object = v => !!v && typeof v === 'object' && !Array.isArray(v);
const validRecord = r => object(r) && typeof r.id === 'string' && object(r.run) && typeof r.setup === 'string' && object(r.finding) && ['draft','reviewed'].includes(r.finding.status) && Array.isArray(r.finding.areas) && Array.isArray(r.finding.categories) && typeof r.finding.comment === 'string';
function create(storage){
 let records = {}, legacy = {}, conflicts = [], recovery = [], writable = true, error = '';
 function read(key){
  try {
   const raw = storage.getItem(key); if(!raw) return {};
   try { const value = JSON.parse(raw); if(!object(value)) throw Error('Invalid review object'); return value; }
   catch(e){ recovery.push({key,raw}); writable = false; error = `Could not read saved reviews (${key}). Download a backup before changing browser storage.`; return {}; }
  }catch { writable = false; error = 'This browser cannot save reviews. Download a backup before leaving this page.'; return {}; }
 }
 const saved = read(KEY); legacy = read(LEGACY_KEY);
 if(Object.keys(saved).length){
  if(saved.schemaVersion === 2 && object(saved.records) && Object.values(saved.records).every(validRecord)){ records=saved.records; legacy={...legacy,...(saved.legacy || {})}; conflicts=saved.conflicts || []; recovery.push(...(saved.recovery || [])); }
  else { writable=false; recovery.push({key:KEY,raw:JSON.stringify(saved)}); error='The saved reviews have an unknown format. Download a backup. Your stored data has been kept.'; }
 }
 const payload = () => ({schemaVersion:2,records,legacy,conflicts,recovery});
 function save(){
  if(!writable) return false;
  try { storage.setItem(KEY,JSON.stringify(payload())); error=''; return true; }
  catch { error='Could not save in this browser. Download a backup of your changes before leaving.'; return false; }
 }
 function migrate(incident, model, baseUrl){
  const id=model.reviewKey(incident); if(records[id]) return records[id];
  const old=model.legacyFor(incident,legacy); if(!old.length) return null;
  records[id]={...model.snapshot(incident,baseUrl),finding:{status:'draft',areas:[],categories:[],certainty:'tentative',comment:''},legacyFindings:old,migration:{rule:'Run file + setup + original outcome key; reconcile all prior findings before marking reviewed.',evidenceSnapshotChanged:old.some(r=>r.run?.recordedAt!==incident.run.recordedAt)},createdAt:new Date().toISOString(),updatedAt:new Date().toISOString()};
  save(); return records[id];
 }
 function ingest(p){
  if(!object(p) || ![1,2].includes(p.schemaVersion) || !Array.isArray(p.reviews)) throw Error('Choose a Tilawa review backup (version 1 or 2).');
  // Validate the entire file before touching the live store.
  for(const r of p.reviews) if(!object(r) || typeof r.id!=='string' || !object(r.run) || (p.schemaVersion===2 ? !validRecord(r) : !object(r.review))) throw Error('A review is missing required data. Nothing was imported.');
  let added=0, duplicates=0, changed=0;
  const merge=(target,r) => {
   if(!target[r.id]) { target[r.id]=r; added++; }
   else if(JSON.stringify(target[r.id])===JSON.stringify(r)) duplicates++;
   else { conflicts.push({id:r.id,existing:target[r.id],incoming:r,importedAt:new Date().toISOString()}); changed++; }
  };
  p.reviews.forEach(r=>merge(p.schemaVersion===2?records:legacy,r));
  for(const r of p.legacyReviews || []) if(object(r) && typeof r.id==='string') merge(legacy,r);
  conflicts.push(...(Array.isArray(p.importConflicts)?p.importConflicts:[]));
  recovery.push(...(Array.isArray(p.storageRecovery)?p.storageRecovery:[]));
  save(); return {added,duplicates,conflicts:changed};
 }
 return {records,legacy,conflicts,recovery,save,migrate,ingest,get error(){return error;},get writable(){return writable;}};
}
globalThis.tilawaReviewStore = Object.freeze({KEY,LEGACY_KEY,create});
})();
