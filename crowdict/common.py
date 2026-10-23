from pathlib import Path
import gzip, hashlib, json, re, unicodedata

SEMANTIC_REVIEW_VERSION = "crow-semantic-review-v5"

LANGUAGES = dict(en='English', es='Spanish', fr='French', de='German', ja='Japanese', ko='Korean', id='Indonesian', ms='Malay', vi='Vietnamese', ru='Russian', pt='Portuguese', it='Italian', nl='Dutch', pl='Polish', sv='Swedish', fi='Finnish', no='Norwegian', da='Danish', cs='Czech', tr='Turkish', el='Greek', bg='Bulgarian', ca='Catalan', hu='Hungarian', zh='Chinese', ar='Arabic', th='Thai', ro='Romanian', sl='Slovenian', hr='Croatian', sk='Slovak', lt='Lithuanian')
TONES = {'a':'āáǎà','e':'ēéěè','i':'īíǐì','o':'ōóǒò','u':'ūúǔù','ü':'ǖǘǚǜ'}
PINYIN_BASES = {v.replace('v','ü') for v in json.loads((Path(__file__).parent/'data/pinyin-bases.json').read_text())} | {'m','n','ng','hm','hng','ê'}
HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U000323af]')

def canonical_json(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def sha(value): return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()
def nfc(value): return unicodedata.normalize('NFC', value)
def normalized_reading(value):
    """Mirror website core.readingKey, retaining lexical tone distinctions."""
    def tone(m):
        word=m[1].replace('u:', 'ü').replace('U:', 'Ü').lower().replace('v','ü'); t=int(m[2])
        if t==5:return word
        at=word.find('a')
        if at<0:at=word.find('e')
        if at<0 and 'ou' in word:at=word.index('o')
        if at<0:at=next((i for i in range(len(word)-1,-1,-1) if word[i] in 'iouü'),-1)
        return word if at<0 else word[:at]+TONES[word[at]][t-1]+word[at+1:]
    return re.sub(r"[\s'’·-]",'',nfc(re.sub(r'([a-zA-ZüÜ:]+)([1-5])',tone,value)).lower())

def valid_reading(value, spoken):
    if not value or len(value)>256:return False
    if spoken=='yue':return bool(re.fullmatch(r'[a-z]+[1-6](?:\s+[a-z]+[1-6])*',value.strip(),re.I))
    # Reject source prose/IPA, placeholders and literal digits (e.g. year headwords).
    if not re.fullmatch(r"[a-zA-ZüÜvV:āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛêÊńňǹḿ1-5\s'’·-]+",value):return False
    if not re.search(r'[a-zA-ZüÜāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜêńňǹḿ]',value):return False
    # Mixing marked syllables with tone numbers (e.g. bù4 or huìr5) is corrupt source notation.
    if any(re.search(r'[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛ]', t) and re.search(r'[1-5]',t) for t in re.split(r"[\s'’·-]+",value)):
        return False
    if re.search(r'[1-5]{2}|(?:^|\s)[1-5](?:\s|$)',value):return False
    def base(word):
        decomposed=unicodedata.normalize('NFD',word.lower().replace('u:','ü').replace('v','ü')).replace('u\u0308','ü')
        return ''.join(c for c in decomposed if not unicodedata.combining(c))
    def syllable(word):return word in PINYIN_BASES or (word.endswith('r') and word!='er' and word[:-1] in PINYIN_BASES)
    for token in re.split(r"[\s'’·-]+",value.strip()):
        if not token:continue
        if re.search('[1-5]',token):
            parts=re.findall(r'([^1-5]+)([1-5])',token)
            if ''.join(w+t for w,t in parts)!=token or not all(syllable(base(w)) for w,t in parts):return False
        else:
            plain=base(token)
            if syllable(plain):continue
            # Marked, unspaced Pinyin may contain several syllables. Bare prose may not.
            if token.lower()==plain:return False
            reachable={0}
            for at in range(len(plain)):
                if at in reachable:
                    for end in range(at+1,min(len(plain),at+7)+1):
                        if syllable(plain[at:end]):reachable.add(end)
            if len(plain) not in reachable:return False
    return True

def is_derived_crow_source(source):
    """Crow exports count as coverage, never as independent primary evidence."""
    if isinstance(source, str):
        sid = source
        marker = family = ''
    else:
        sid = str(source.get('sourceId', source.get('id', '')))
        marker = str(source.get('source', ''))
        family = str(source.get('sourceFamily', source.get('family', '')))
    return (sid.casefold().startswith('crow-') or sid.casefold() == 'crow-chinese-dictionary'
            or marker.casefold() == 'crow-chinese-dictionary'
            or family.casefold() == 'derived-crow-family')


def source_policy(source):
    if is_derived_crow_source(source):return 'derived_corpus_not_evidence'
    licence=source.get('license','')
    if 'ND' in licence:return 'discovery_only_nd'
    if source.get('machineTranslated'):return 'machine_evidence_only'
    if source['id'].startswith('wordnet-') or source['id']=='kenet-tr' or 'Source licenses' in licence:return 'component_review'
    if licence in {'CC BY-SA 4.0','CC BY-SA 3.0','CC BY 4.0','MIT'}:return 'compatible'
    return 'licence_review'

def source_family(source):
    if is_derived_crow_source(source):return 'derived-crow-family'
    sid=source['id']
    if sid.startswith('wordnet-') or sid=='kenet-tr':return 'wordnet-cow'
    if 'wiktionary' in sid or sid.startswith('wikdict-'):return 'wiktionary-family'
    if sid in {'cedict','cc-cidict','cvdict','cedict-es-ai'}:return 'cedict-family'
    if sid=='handedict-en-ai':return 'handedict'
    return sid

def load_json(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as f:return json.load(f)

def data_path(web_root, file):
    root=(Path(web_root)/'public/data').resolve(); path=(Path(web_root)/'public'/file.removeprefix('/')).resolve()
    if not path.is_relative_to(root):raise ValueError('Catalog source path is outside public/data')
    return path

def write_jsonl(path, records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.pending')
    raw=temp.open('wb')
    try:
        writer=gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) if path.suffix=='.gz' else raw
        try:
            for record in records:writer.write((canonical_json(record)+'\n').encode())
        finally:
            if writer is not raw:writer.close()
    finally:raw.close()
    temp.replace(path)
