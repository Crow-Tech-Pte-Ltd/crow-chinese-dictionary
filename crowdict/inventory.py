"""Local-only inventory. Dictionary packs are read-only; no model discovers words."""
import argparse, collections, csv, gc, gzip, hashlib, json, re, sqlite3, time
from pathlib import Path
from .common import *

COMPOSITIONAL = re.compile(r'^(?:[一二两三四五六七八九十百千万亿零〇]+|几[十百千万亿]?)(?:年|月|天|日|周|分钟|秒|小时|元|块|个|位|项|种|杯|股|次|遍|本|张|辆|瓶|箱|支|只|件|条|名|人|公斤|米|公里)$')

SCHEMA='''
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE sources(id TEXT PRIMARY KEY,language TEXT,spoken TEXT,policy TEXT,family TEXT,quality TEXT,baseline INTEGER,catalog_json TEXT);
CREATE TABLE records(id INTEGER PRIMARY KEY,source_id TEXT,row_index INTEGER,canonical TEXT,traditional TEXT,simplified TEXT,aliases TEXT,reading TEXT,nr TEXT,spoken TEXT,language TEXT,has_definition INTEGER,sense_id TEXT,pos TEXT,proper INTEGER);
CREATE TABLE alias_claims(spoken TEXT,alias TEXT,nr TEXT,canonical TEXT,PRIMARY KEY(spoken,alias,nr,canonical)) WITHOUT ROWID;
CREATE TABLE raw_readings(id INTEGER PRIMARY KEY,source_id TEXT,word TEXT,reading TEXT,nr TEXT,spoken TEXT,provenance TEXT);
CREATE TABLE resolutions(record_id INTEGER PRIMARY KEY,canonical TEXT,reading TEXT,nr TEXT,spoken TEXT,reading_status TEXT,alias_status TEXT,lexeme_id TEXT);
CREATE TABLE lexical_readings(spoken TEXT,canonical TEXT,nr TEXT,reading TEXT,provenance TEXT,PRIMARY KEY(spoken,canonical,nr)) WITHOUT ROWID;
CREATE TABLE lexemes(id TEXT PRIMARY KEY,spoken TEXT,canonical TEXT,nr TEXT,reading TEXT,priority INTEGER,frequency INTEGER,hsk INTEGER,word_length INTEGER);
'''

def _record(source,index,row):
    if not isinstance(row,list) or len(row)<4 or any(not isinstance(v,str) for v in row[:4]):raise ValueError('Malformed pack entry '+source['id']+':'+str(index))
    meta=row[4] if len(row)>4 and isinstance(row[4],dict) else {}
    spoken={'mandarin':'cmn','cantonese':'yue'}.get(meta.get('chineseLanguage'),source['spoken'])
    canonical=nfc(meta.get('canonicalWord') or row[1]); aliases=sorted({nfc(v) for v in [row[0],row[1],canonical,*meta.get('lookupAliases',[])] if isinstance(v,str) and v})
    reading=(meta.get('jyutping','') if spoken=='yue' else row[2]).strip();nr=normalized_reading(reading) if spoken=='cmn' else reading.lower().replace(' ','')
    pos=meta.get('pos') or meta.get('sourcePos') or ''
    proper=bool(re.search(r'\b(?:proper|name|propn|surname|place name|person|patronym|toponym)\b',pos,re.I)) or bool(re.search(r'(?:^|/)\s*(?:surname|variant of|old variant of|abbr\.|abbreviation for)\b|\(Eig(?:[, )])|japanische Variante|nom propre|patronyme|prénom',row[3],re.I))
    return (source['id'],index,canonical,nfc(row[0]),nfc(row[1]),canonical_json(aliases),reading,nr,spoken,source['lang'],int(bool(row[3].strip())),str(meta.get('sourceSense') or meta.get('wordnetSynset') or meta.get('interlingualId') or ''),pos,int(proper))

def hsk_rows(web_root):
    path=data_path(web_root,'/data/hsk-source.json.gz')
    if not path.exists():return []
    rows=[]
    for item in load_json(path):
        for form in item.get('forms',[]):
            meanings=[x.strip().replace('/','∕') for x in form.get('meanings',[]) if isinstance(x,str) and x.strip()]
            raw=form.get('transcriptions',{});reading=raw.get('numeric') or raw.get('pinyin') or ''
            if meanings:rows.append([form.get('traditional') or item['simplified'],item['simplified'],reading,'/'.join(meanings),{'source':'hsk-open','pos':', '.join(item.get('pos',[]))}])
    return rows

def source_file(web_root,source):
    return data_path(source.get('sourceRoot',web_root),source['file'])

def source_rows(web_root,source):
    if source.get('dataFormat')=='entry-jsonl':
        def entries():
            opener=gzip.open if source_file(web_root,source).suffix=='.gz' else open
            with opener(source_file(web_root,source),'rt',encoding='utf-8') as stream:
                for number,line in enumerate(stream):
                    if line.strip():yield json.loads(line)
        return entries()
    if source.get('dataFormat')=='wiktionary-jsonl':
        from .raw_sources import iter_raw_source_rows
        return iter_raw_source_rows(web_root,source)
    return hsk_rows(web_root) if source['id']=='hsk-open' else load_json(data_path(web_root,source['file']))

def _sources(web_root,extra_catalog=None):
    catalog=load_json(Path(web_root)/'public/data/catalog.json'); result=[]
    for source in catalog:
        target=source['language'].split('→')[-1].strip(); lang=next((k for k,v in LANGUAGES.items() if v==target),None)
        if not lang:raise ValueError('Unknown dictionary target '+source['id'])
        source=dict(source);source['lang']=lang;source['spoken']='yue' if source.get('sourceLanguage')=='cantonese' or source['language'].startswith('Cantonese') else 'cmn';source['baseline']=True;result.append(source)
    hsk=data_path(web_root,'/data/hsk-source.json.gz')
    if hsk.exists():
        result.append(dict(id='hsk-open',name='Complete HSK Vocabulary',lang='en',spoken='cmn',language='Chinese → English',license='MIT',licenseUrl='https://opensource.org/license/mit',attribution='Yanis Zafirópulos; Complete HSK Vocabulary contributors',url='https://github.com/drkameleon/complete-hsk-vocabulary',snapshot=sha(hsk.read_bytes()),file='/data/hsk-source.json.gz',sha256=sha(hsk.read_bytes()),original='/data/hsk-source.json.gz',licenseNotice=(Path(web_root)/'public/data/hsk-LICENSE.txt').read_text(),baseline=False))
    from .raw_sources import raw_source_catalog
    result.extend(raw_source_catalog(web_root,[s for s in result if s['baseline']]))
    if extra_catalog:
        extra_catalog=Path(extra_catalog).resolve();extra=load_json(extra_catalog);source_root=Path(extra['root'])
        if not source_root.is_absolute():source_root=(extra_catalog.parent/source_root).resolve()
        ids={s['id'] for s in result}
        for supplied in extra['sources']:
            source=dict(supplied)
            if source['id'] in ids:raise ValueError('Supplemental source id conflicts with existing source')
            if source.get('lang') not in LANGUAGES or source.get('spoken') not in ('cmn','yue'):raise ValueError('Supplemental source language is unsupported')
            source['baseline']=False;source['sourceRoot']=str(source_root)
            if source.get('dataFormat')!='entry-jsonl':raise ValueError('Supplemental inputs must be explicitly distilled entry-jsonl')
            result.append(source);ids.add(source['id'])
    return result

def build(web_root,db_path,reports,raw_readings=True,extra_catalog=None):
    web_root=Path(web_root).resolve();db_path=Path(db_path);reports=Path(reports);db_path.parent.mkdir(parents=True,exist_ok=True);reports.mkdir(parents=True,exist_ok=True)
    pending=db_path.with_name(db_path.name+'.pending');pending.unlink(missing_ok=True);db=sqlite3.connect(pending);db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-65536');db.executescript(SCHEMA)
    started=time.time();sources=_sources(web_root,extra_catalog);fingerprint=sha(canonical_json({'inventoryVersion':'2026-10-23.2','sources':[{**{k:s.get(k) for k in ('id','sha256','snapshot','license')},'policy':source_policy(s)} for s in sources]}));db.execute('INSERT INTO metadata VALUES(?,?)',('sourceFingerprint',fingerprint));db.execute('INSERT INTO metadata VALUES(?,?)',('webRoot',str(web_root)))
    for source in sources:
        policy=source_policy(source);quality='machine_translated' if source.get('machineTranslated') else 'translation_only' if source.get('translationOnly') else 'full_definition'
        db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)',(source['id'],source['lang'],source['spoken'],policy,source_family(source),quality,int(source['baseline']),canonical_json(source)))
        path=source_file(web_root,source); actual=sha(path.read_bytes())
        if source.get('sha256') and source['sha256']!=actual:raise ValueError('Source checksum mismatch: '+source['id'])
        rows=source_rows(web_root,source)
        batch=[];claims=[];row_count=0;extra_readings=[]
        for index,row in enumerate(rows):
            record=_record(source,index,row);batch.append(record);row_count+=1
            meta=row[4] if len(row)>4 and isinstance(row[4],dict) else {}
            if policy=='compatible' and quality=='full_definition':
                for reading_candidate in meta.get('readingCandidates',[]):
                    if valid_reading(reading_candidate,record[8]):
                        provenance={'sourceId':source['id'],'word':record[2],'reading':reading_candidate,'sourceFile':source['file'],'sourceRowIndex':meta.get('rawOriginalRowIndex',index),'sourceSha256':source.get('originalSha256',source.get('sha256','')),'field':'raw.readingCandidates','spokenLanguage':record[8]}
                        extra_readings.append((source['id'],record[2],reading_candidate,normalized_reading(reading_candidate),record[8],canonical_json(provenance)))
            if policy=='compatible' and quality=='full_definition' and valid_reading(record[6],record[8]):
                for alias in json.loads(record[5]):
                    if alias!=record[2]:claims.append((record[8],alias,record[7],record[2]))
            if len(batch)>=2000:
                db.executemany('INSERT INTO records(source_id,row_index,canonical,traditional,simplified,aliases,reading,nr,spoken,language,has_definition,sense_id,pos,proper) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',batch);db.executemany('INSERT OR IGNORE INTO alias_claims VALUES(?,?,?,?)',claims);batch=[];claims=[]
        db.executemany('INSERT INTO records(source_id,row_index,canonical,traditional,simplified,aliases,reading,nr,spoken,language,has_definition,sense_id,pos,proper) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',batch);db.executemany('INSERT OR IGNORE INTO alias_claims VALUES(?,?,?,?)',claims);db.executemany('INSERT INTO raw_readings(source_id,word,reading,nr,spoken,provenance) VALUES(?,?,?,?,?,?)',extra_readings);db.commit();print('Imported',source['id'],row_count,policy,flush=True);
        if source.get('count') is not None and row_count!=source['count']:raise ValueError('Source count mismatch: '+source['id'])
        del rows;gc.collect()
    db.executescript('CREATE INDEX records_source ON records(source_id,row_index);CREATE INDEX records_identity ON records(spoken,canonical,nr);CREATE INDEX records_language ON records(language,spoken,canonical);')
    if raw_readings:
        from .raw_readings import iter_raw_readings
        batch=[]
        for item in iter_raw_readings(web_root,[s for s in sources if s['baseline'] and source_policy(s)=='compatible' and not s.get('machineTranslated')]):
            if not valid_reading(item['reading'],'cmn'):continue
            batch.append((item['sourceId'],nfc(item['word']),item['reading'],normalized_reading(item['reading']),'cmn',canonical_json(item)))
            if len(batch)>=2000:db.executemany('INSERT INTO raw_readings(source_id,word,reading,nr,spoken,provenance) VALUES(?,?,?,?,?,?)',batch);batch=[]
        db.executemany('INSERT INTO raw_readings(source_id,word,reading,nr,spoken,provenance) VALUES(?,?,?,?,?,?)',batch);db.commit()
    # Alias claims are conditional on language AND reading. Never union alias graphs.
    aliases=collections.defaultdict(set)
    for spoken,alias,nr,canonical in db.execute('SELECT * FROM alias_claims'):aliases[(spoken,alias,nr)].add(canonical)
    def canonicalize(spoken,word,nr):
        choices=aliases.get((spoken,word,nr),set())
        if not choices:return word,'unchanged'
        if len(choices)!=1:return word,'conflict'
        target=next(iter(choices)); downstream=aliases.get((spoken,target,nr),set())
        if downstream and downstream!={target}:return word,'conflict'
        return target,'mapped_explicit'
    # Phase one: retain direct readings and collect permitted non-AI attestations.
    batch=[];attest=[]
    for record in db.execute('SELECT r.id,r.source_id,r.row_index,r.canonical,r.reading,r.nr,r.spoken,s.policy,s.quality FROM records r JOIN sources s ON s.id=r.source_id'):
        rid,sid,idx,word,reading,nr,spoken,policy,quality=record
        good=valid_reading(reading,spoken);canonical,alias_status=canonicalize(spoken,word,nr) if good else (word,'unresolved')
        status='source_attested' if good else 'invalid' if reading else 'missing'
        batch.append((rid,canonical,reading if good else '',nr if good else '',spoken,status,alias_status,None))
        if good and policy=='compatible' and quality=='full_definition' and alias_status!='conflict':
            prov={'sourceId':sid,'rowIndex':idx,'field':'pack.lexicalReading','reading':reading};attest.append((spoken,canonical,nr,reading,canonical_json(prov)))
        if len(batch)>=2000:db.executemany('INSERT INTO resolutions VALUES(?,?,?,?,?,?,?,?)',batch);db.executemany('INSERT OR IGNORE INTO lexical_readings VALUES(?,?,?,?,?)',attest);batch=[];attest=[]
    db.executemany('INSERT INTO resolutions VALUES(?,?,?,?,?,?,?,?)',batch);db.executemany('INSERT OR IGNORE INTO lexical_readings VALUES(?,?,?,?,?)',attest)
    for sid,word,reading,nr,spoken,prov in db.execute('SELECT source_id,word,reading,nr,spoken,provenance FROM raw_readings'):
        canonical,alias_status=canonicalize(spoken,word,nr)
        if alias_status!='conflict':db.execute('INSERT OR IGNORE INTO lexical_readings VALUES(?,?,?,?,?)',(spoken,canonical,nr,reading,prov))
    db.executescript('CREATE INDEX resolved_identity ON resolutions(spoken,canonical,nr);CREATE INDEX resolved_lexeme ON resolutions(lexeme_id);CREATE INDEX readings_word ON lexical_readings(spoken,canonical);')
    # Only a SINGLE attested reading of this exact identity can fill a blank row.
    unique={};ambiguous=set()
    for spoken,word,count,reading,nr in db.execute('SELECT spoken,canonical,count(*),min(reading),min(nr) FROM lexical_readings GROUP BY spoken,canonical'):
        if count==1:unique[(spoken,word)]=(reading,nr)
        else:ambiguous.add((spoken,word))
    word_aliases=collections.defaultdict(set)
    for (spoken,alias,nr),targets in aliases.items():word_aliases[(spoken,alias)].update(targets)
    updates=[]
    for rid,word,reading,nr,spoken,status,alias_status in db.execute('SELECT record_id,canonical,reading,nr,spoken,reading_status,alias_status FROM resolutions WHERE reading_status!="source_attested"'):
        targets=word_aliases.get((spoken,word),set());target=next(iter(targets)) if len(targets)==1 else word
        key=(spoken,target)
        if len(targets)>1:updates.append((word,'','','ambiguous','conflict',rid));continue
        if key in unique and status!='invalid':
            found,norm=unique[key];updates.append((target,found,norm,'resolved_unique','mapped_explicit' if target!=word else 'unchanged',rid))
        elif key in ambiguous:updates.append((target,'','','ambiguous',alias_status,rid))
    db.executemany('UPDATE resolutions SET canonical=?,reading=?,nr=?,reading_status=?,alias_status=? WHERE record_id=?',updates);del aliases,unique,word_aliases;gc.collect()
    levels=load_json(web_root/'public/data/hsk-levels.json') if (web_root/'public/data/hsk-levels.json').exists() else {};frequencies=load_json(web_root/'public/data/frequency.json.gz').get('words',{}) if (web_root/'public/data/frequency.json.gz').exists() else {}
    lexemes=[];ids=[]
    for spoken,word,nr,reading in db.execute('SELECT spoken,canonical,nr,min(reading) FROM resolutions WHERE nr!="" AND alias_status!="conflict" GROUP BY spoken,canonical,nr'):
        lid=sha(canonical_json([spoken,word,nr]));freq=frequencies.get(word,0);hsk=min([int(v.rsplit('-',1)[-1]) for v in levels.get(word,[]) if v.rsplit('-',1)[-1].isdigit()] or [99]);length=len(word);priority=1000000*(hsk<99)+max(0,10-hsk)*10000+freq*100+max(0,12-length)
        lexemes.append((lid,spoken,word,nr,reading,priority,freq,hsk,length));ids.append((lid,spoken,word,nr))
        if len(lexemes)>=2000:db.executemany('INSERT INTO lexemes VALUES(?,?,?,?,?,?,?,?,?)',lexemes);db.executemany('UPDATE resolutions SET lexeme_id=? WHERE spoken=? AND canonical=? AND nr=? AND alias_status!="conflict"',ids);lexemes=[];ids=[]
    db.executemany('INSERT INTO lexemes VALUES(?,?,?,?,?,?,?,?,?)',lexemes);db.executemany('UPDATE resolutions SET lexeme_id=? WHERE spoken=? AND canonical=? AND nr=? AND alias_status!="conflict"',ids)
    db.executescript('CREATE TABLE head_coverage AS SELECT DISTINCT z.spoken,z.canonical,r.language,s.baseline,s.policy,s.quality FROM resolutions z JOIN records r ON r.id=z.record_id JOIN sources s ON s.id=r.source_id WHERE r.has_definition=1;CREATE INDEX coverage_lookup ON head_coverage(spoken,canonical,language,baseline);CREATE INDEX lexemes_priority ON lexemes(priority DESC,id);')
    db.execute('INSERT INTO metadata VALUES(?,?)',('buildSeconds',str(round(time.time()-started,2))));db.commit();report=report_inventory(db,reports);db.close();pending.replace(db_path);print('Inventory ready:',canonical_json(report),flush=True);return report

def report_inventory(db,out):
    source_counts=[dict(zip(('id','policy','quality','records'),r)) for r in db.execute('SELECT s.id,s.policy,s.quality,count(r.id) FROM sources s JOIN records r ON r.source_id=s.id GROUP BY s.id ORDER BY s.id')]
    reading=dict(db.execute('SELECT reading_status,count(*) FROM resolutions GROUP BY reading_status'));alias=dict(db.execute('SELECT alias_status,count(*) FROM resolutions GROUP BY alias_status'))
    union=dict(db.execute('SELECT spoken,count(DISTINCT canonical) FROM resolutions GROUP BY spoken'));rows=[]
    for language in LANGUAGES:
        for spoken,total in union.items():
            present=db.execute('SELECT count(DISTINCT canonical) FROM head_coverage WHERE language=? AND spoken=? AND baseline=1',(language,spoken)).fetchone()[0]
            nonmachine=db.execute('SELECT count(DISTINCT canonical) FROM head_coverage WHERE language=? AND spoken=? AND baseline=1 AND quality!="machine_translated"',(language,spoken)).fetchone()[0]
            ready=db.execute('SELECT count(DISTINCT z.canonical) FROM resolutions z JOIN records r ON r.id=z.record_id JOIN sources s ON s.id=r.source_id WHERE z.spoken=? AND z.lexeme_id IS NOT NULL AND s.policy="compatible" AND s.quality="full_definition" AND r.has_definition=1 AND NOT EXISTS(SELECT 1 FROM head_coverage h WHERE h.spoken=z.spoken AND h.canonical=z.canonical AND h.language=? AND h.baseline=1)',(spoken,language)).fetchone()[0]
            rows.append({'language':language,'spokenLanguage':spoken,'referenceHeads':total,'existingTargetHeads':present,'missingTargetHeads':total-present,'existingNonMachineHeads':nonmachine,'machineOnlyHeads':present-nonmachine,'attestedDefinitionAndReadingCandidates':ready})
    with (out/'language-coverage.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'sourceFingerprint':db.execute('SELECT value FROM metadata WHERE key="sourceFingerprint"').fetchone()[0],'referenceHeadwordsBySpokenLanguage':union,'records':db.execute('SELECT count(*) FROM records').fetchone()[0],'lexemesWithReadings':db.execute('SELECT count(*) FROM lexemes').fetchone()[0],'rawReadingAttestations':db.execute('SELECT count(*) FROM raw_readings').fetchone()[0],'readingStatusRows':reading,'aliasStatusRows':alias,'sources':source_counts,'coverage':rows,'limitations':['Reference universe is the supplied dictionaries, retained HSK data and explicitly catalogued supplemental snapshots; this is not an exact Pleco gap count or proof of completeness.','Counts are candidates before lexical/source-quality screening; related source families are not independent corroboration.','Unknown readings and headword-present reading differences are not automatically generated.','ND definitions are never retained in prompts; mixed component licences require review.']}
    (out/'inventory-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return {k:v for k,v in report.items() if k not in ('sources','coverage')}



def job_script_forms(refs, canonical, aliases):
    """Choose existing typed source forms without changing lexical evidence.

    Refs must already be in the job's source-priority order and belong to its
    resolved lexical reading. An untyped lookup alias alone is insufficient to
    decide which script it represents. The canonical source form is the fallback.
    """
    known = {value for value in aliases if isinstance(value, str)} | {canonical}
    traditional = canonical
    for ref in refs:
        ref = dict(ref)
        value = ref.get('traditional')
        if (ref.get('simplified') == canonical and isinstance(value, str)
                and value != canonical and value in known and 0 < len(value) <= 128
                and not re.search(r'[\s\x00-\x1f\x7f/\\]', value)):
            traditional = value
            break
    return {'traditionalWord': traditional, 'simplifiedWord': canonical}

def jobs(db_path,web_root,language,out,limit=1000,min_frequency=250,max_word_length=12):
    if language not in LANGUAGES:raise ValueError('Unknown language')
    db=sqlite3.connect(db_path);db.row_factory=sqlite3.Row;sources={r['id']:json.loads(r['catalog_json'])|dict(policy=r['policy'],family=r['family'],quality=r['quality']) for r in db.execute('SELECT * FROM sources')}
    # Existing ANY target definition (including AI) means no automatic paid replacement.
    candidates=list(db.execute('SELECT l.* FROM lexemes l WHERE NOT EXISTS(SELECT 1 FROM head_coverage h WHERE h.spoken=l.spoken AND h.canonical=l.canonical AND h.language=? AND h.baseline=1) AND EXISTS(SELECT 1 FROM resolutions z JOIN records r ON r.id=z.record_id JOIN sources s ON s.id=r.source_id WHERE z.lexeme_id=l.id AND s.policy="compatible" AND s.quality="full_definition" AND r.has_definition=1) ORDER BY l.priority DESC,l.id LIMIT ?',(language,limit)))
    baseline_aliases=collections.defaultdict(set)
    for r in db.execute('SELECT r.spoken,r.aliases FROM records r JOIN sources s ON s.id=r.source_id WHERE r.language=? AND s.baseline=1 AND r.has_definition=1',(language,)):
        baseline_aliases[r[0]].update(json.loads(r[1]))
    extra_target_aliases=collections.defaultdict(set)
    for r in db.execute('SELECT r.spoken,r.aliases FROM records r JOIN sources s ON s.id=r.source_id WHERE r.language=? AND s.baseline=0 AND s.policy="compatible" AND s.quality="full_definition" AND r.has_definition=1',(language,)):
        extra_target_aliases[r[0]].update(json.loads(r[1]))
    selected={};needed=collections.defaultdict(set)
    for lexeme in candidates:
        refs=list(db.execute('SELECT r.*,z.reading AS resolved_reading,z.reading_status FROM resolutions z JOIN records r ON r.id=z.record_id JOIN sources s ON s.id=r.source_id WHERE z.lexeme_id=? AND s.policy="compatible" AND s.quality="full_definition" AND r.has_definition=1 ORDER BY CASE WHEN r.language=? THEN 0 ELSE 1 END,CASE WHEN r.language=? THEN 0 WHEN r.language=? THEN 1 ELSE 2 END,s.baseline DESC,r.source_id,r.row_index',(lexeme['id'],language,'en','zh')))
        # Cap prompt size deterministically, favour independent families and keep sense anchors.
        chosen=[];families=collections.Counter()
        for r in refs:
            family=sources[r['source_id']]['family']
            if families[family]>=3 or len(chosen)>=8:continue
            chosen.append(dict(r));families[family]+=1;needed[r['source_id']].add(r['row_index'])
        selected[lexeme['id']]=(dict(lexeme),chosen)
    values={}
    for sid,indexes in needed.items():
        source=sources[sid];data=source_rows(web_root,source)
        for index,row in enumerate(data):
            if index in indexes:values[(sid,index)]=row
        del data;gc.collect()
    output=[];held=[];reuse=[]
    for lid,(lexeme,refs) in selected.items():
        evidence=[];aliases={lexeme['canonical']};proper=False;provenance=[]
        for r in refs:
            source=sources[r['source_id']];row=values[(r['source_id'],r['row_index'])];aliases.update(json.loads(r['aliases']));proper|=bool(r['proper'])
            e={'id':sha(canonical_json([r['source_id'],source.get('sha256'),r['row_index']]))[:24],'sourceId':r['source_id'],'language':r['language'],'definition':row[3],'reading':r['resolved_reading'],'license':source['license'],'licenseUrl':source.get('licenseUrl',''),'attribution':source.get('attribution',''),'sourceUrl':source.get('url',''),'snapshot':source.get('snapshot',''),'sourceSha256':source.get('sha256',''),'originalRowIndex':r['row_index'],'quality':'full_definition','sourceFamily':source['family'],'canonicalWord':lexeme['canonical'],'sourcePos':r['pos']}
            if source.get('licenseNotice'):e['licenseNotice']=source['licenseNotice']
            if r['sense_id']:e['senseId']=r['sense_id']
            if len(row)>4 and isinstance(row[4],dict) and row[4].get('sourceArticleUrl'):
                e['sourceUrl']=row[4]['sourceArticleUrl'];e['sourceDatasetUrl']=source.get('url','')
            if len(row)>4 and isinstance(row[4],dict) and 'rawOriginalRowIndex' in row[4]:
                e['rawOriginalRowIndex']=row[4]['rawOriginalRowIndex'];e['rawSourceFile']=source['file'];e['rawReadingFields']=row[4].get('rawReadingFields',[]);e['rawSourceSenses']=row[4].get('rawSourceSenses',[])
            evidence.append(e)
            if r['reading_status']=='resolved_unique':
                for pr in db.execute('SELECT provenance FROM lexical_readings WHERE spoken=? AND canonical=? AND nr=?',(lexeme['spoken'],lexeme['canonical'],lexeme['nr'])):provenance.append(json.loads(pr[0]))
        same_language=[e for e in evidence if e['language']==language]
        reasons=[]
        if not same_language and aliases & extra_target_aliases[lexeme['spoken']]:reasons.append('target_language_source_reading_alignment_needed')
        possible_compositional=bool(COMPOSITIONAL.fullmatch(lexeme['canonical']))
        if possible_compositional:reasons.append('productive_quantity_expression_review')
        if len(evidence)==1 and evidence[0]['language']=='fr' and re.fullmatch(r'[A-Z][a-zéèêëàâäùûüôöîïç-]{2,}',evidence[0]['definition'].strip()):reasons.append('unexplained_single_source_name_review')
        if aliases & baseline_aliases[lexeme['spoken']]:reasons.append('existing_target_alias_or_reading_review')
        if not valid_reading(lexeme['reading'],lexeme['spoken']):reasons.append('invalid_or_unverified_lexical_reading')
        if lexeme['spoken']=='cmn' and any(re.search(r'Cantonese|cantonais|cantonés|Kantonesisch',e['definition'],re.I) for e in evidence):reasons.append('source_spoken_variety_needs_review')
        if lexeme['spoken']!='cmn':reasons.append('spoken_variety_needs_separate_review')
        if not HAN.search(lexeme['canonical']):reasons.append('not_chinese_script_headword')
        if lexeme['word_length']>max_word_length:reasons.append('long_expression_needs_lexical_review')
        if lexeme['hsk']==99 and lexeme['frequency']<min_frequency and not same_language:reasons.append('low_frequency_candidate_needs_review')
        if proper or any(re.search(r'\(Eig(?:[, )])|japanische Variante|nom propre|patronyme|prénom',e['definition'],re.I) for e in evidence):reasons.append('proper_name_variant_or_abbreviation_needs_review')
        job={'id':sha(canonical_json([lexeme['spoken'],lexeme['canonical'],lexeme['nr'],language])),'canonicalWord':lexeme['canonical'],'aliases':sorted(aliases),'reading':lexeme['reading'],'normalizedReading':lexeme['nr'],'spokenLanguage':lexeme['spoken'],'targetLanguage':language,'gapKind':'missing_target_headword','candidateKind':'possible_productive_quantity_expression' if possible_compositional else 'word_or_fixed_expression_candidate','priority':lexeme['priority'],'frequencyScore':lexeme['frequency'],'hskLevel':None if lexeme['hsk']==99 else lexeme['hsk'],'evidence':evidence,'evidenceHash':sha(canonical_json(evidence)),'sourceLicenses':sorted({e['license'] for e in evidence}),'sourceFamilies':sorted({e['sourceFamily'] for e in evidence}),'readingProvenance':provenance,'reuseEligible':bool(same_language) and not reasons,'generationEligible':not reasons and not same_language,'holdReasons':reasons+(['same_language_source_reuse_available'] if same_language else [])}
        job.update(job_script_forms(refs, lexeme['canonical'], aliases))
        job['jobId']=job['id']
        if reasons:held.append(job)
        elif same_language:reuse.append(job)
        else:output.append(job)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,records in [('jobs',output),('held',held),('reuse',reuse)]:write_jsonl(out/f'{name}-{language}.jsonl.gz',records)
    summary={'language':language,'limit':limit,'selectedLexemes':len(selected),'generationEligible':len(output),'sourceReuseEligible':len(reuse),'held':len(held),'selectionPolicy':{'minimumFrequency':min_frequency,'maxWordLength':max_word_length,'HSKAlwaysPrioritized':True},'sourceFingerprint':db.execute('SELECT value FROM metadata WHERE key="sourceFingerprint"').fetchone()[0]}
    (out/f'jobs-{language}-summary.json').write_text(json.dumps(summary,indent=2)+'\n');db.close();print(canonical_json(summary));return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--web-root',type=Path,required=True);b.add_argument('--db',type=Path,default=Path('work/inventory.sqlite'));b.add_argument('--reports',type=Path,default=Path('reports'));b.add_argument('--no-raw-readings',action='store_true');b.add_argument('--extra-catalog',type=Path)
    j=sub.add_parser('jobs');j.add_argument('--web-root',type=Path,required=True);j.add_argument('--db',type=Path,default=Path('work/inventory.sqlite'));j.add_argument('--language',choices=list(LANGUAGES),default='en');j.add_argument('--out',type=Path,default=Path('work/jobs'));j.add_argument('--limit',type=int,default=1000);j.add_argument('--min-frequency',type=int,default=250);j.add_argument('--max-word-length',type=int,default=12)
    a=p.parse_args(argv)
    if a.command=='build':build(a.web_root,a.db,a.reports,not a.no_raw_readings,a.extra_catalog)
    else:jobs(a.db,a.web_root,a.language,a.out,a.limit,a.min_frequency,a.max_word_length)
if __name__=='__main__':main()
