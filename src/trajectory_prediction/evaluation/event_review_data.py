"""Full-source review tasks and mutable notes; no human-label import or scoring."""
import csv
import io
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.event_sources import read_json
from trajectory_prediction.utils.provenance import sha256_file

LOCKED = ('task_id','task_kind','role','instance','block_id','event_id','diagnostic_id')
EDITABLE = ('review_status','reviewer','notes')
COLUMNS = (*LOCKED, *EDITABLE)
STATUSES = ('pending','in_progress','reviewed')
MAX_CELL_CODEPOINTS = 8192
REVIEW_WHITESPACE = ''.join(map(chr,[9,10,11,12,13,28,29,30,31,32,133,160,5760,
                                    *range(8192,8203),8232,8233,8239,8287,12288,65279]))


def csv_text(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator='\r\n')
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue()


def validate_working(text, baseline):
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff'), newline=''), strict=True)
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError('malformed working CSV') from error
    if reader.fieldnames != list(COLUMNS):
        raise ValueError('working CSV columns mismatch')
    expected = {r['task_id']:r for r in baseline}
    if len(expected) != len(baseline):
        raise ValueError('duplicate baseline task')
    seen, counts = set(), Counter()
    for row in rows:
        key = row.get('task_id')
        if (set(row) != set(COLUMNS) or any(v is None for v in row.values())
                or key not in expected or key in seen
                or any(row[k] != expected[key][k] for k in LOCKED)):
            raise ValueError('working CSV task identity/coverage changed')
        if row['review_status'] not in STATUSES:
            raise ValueError('unknown working review status')
        if any(len(value) > MAX_CELL_CODEPOINTS for value in row.values()):
            raise ValueError('working CSV cell exceeds 8192 Unicode code points')
        if row['review_status'] == 'reviewed' and (not row['reviewer'].strip(REVIEW_WHITESPACE)
                                                 or not row['notes'].strip(REVIEW_WHITESPACE)):
            raise ValueError('reviewed reading record needs reviewer and notes')
        seen.add(key); counts[row['review_status']] += 1
    if seen != set(expected):
        raise ValueError('working CSV missing tasks')
    return {k:counts[k] for k in STATUSES}


def _unique(rows, field):
    result = {r[field]:r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate review source '+field)
    return result


def _npz(path, references):
    path = path.resolve()
    if str(path) not in references or sha256_file(path) != references[str(path)]:
        raise ValueError('unbound/changed GT source')
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def assemble_review(parent, references):
    parent = Path(parent).resolve()
    run = read_json(parent/'run.json')
    b0 = Path(run['binding']['parent']).resolve()
    blocks = read_json(parent/'blocks.json'); block_map = _unique(blocks,'block_id')
    events = read_json(parent/'events.json'); _unique(events,'event_id')
    diagnostics = read_json(parent/'diagnostics.json'); _unique(diagnostics,'diagnostic_id')
    events_by_block, diagnostics_by_block, orphan = defaultdict(list), defaultdict(list), []
    for e in events:
        b = block_map.get(e['block_id'])
        if b is None or any(e[k] != b[k] for k in ('role','instance','category','scene','log')):
            raise ValueError('event/block review foreign key mismatch')
        events_by_block[b['block_id']].append(e['event_id'])
    for d in diagnostics:
        if d.get('block_id') is None:
            if d['reason'] != 'temporal_resolution_gap':
                raise ValueError('unsupported orphan diagnostic')
            orphan.append(d)
        else:
            b = block_map.get(d['block_id'])
            if b is None or any(d[k] != b[k] for k in ('role','instance','category','scene','log')):
                raise ValueError('diagnostic/block review foreign key mismatch')
            diagnostics_by_block[b['block_id']].append(d['diagnostic_id'])
    enriched = []
    for role in ('independent_fit','independent_select'):
        points = _npz(b0/(role+'-tracks.npz'), references)
        velocity = _npz(parent/(role+'-kinematics.npz'), references)
        tokens = points['annotation_token']; times = points['timestamp_us']; xy = points['world_xy']
        if (tokens.dtype.kind != 'U' or times.dtype != np.int64 or xy.dtype != np.float64
                or times.shape != tokens.shape or xy.shape != (len(tokens),2)
                or len(set(tokens)) != len(tokens) or not np.isfinite(xy).all()):
            raise ValueError('invalid GT source shape/dtype/tokens')
        positions = {str(t):i for i,t in enumerate(tokens)}
        interval_rows = defaultdict(list)
        for i,key in enumerate(velocity['block_id']):
            if velocity['valid'][i]: interval_rows[str(key)].append(i)
        for b in (b for b in blocks if b['role'] == role):
            left, right = positions.get(b['first_annotation']), positions.get(b['last_annotation'])
            if left is None or right is None or left > right:
                raise ValueError('missing/reversed review GT endpoints')
            sl = slice(left,right+1)
            if (any(str(x) != b['instance'] for x in points['instance_token'][sl])
                    or any(str(x) != b['segment_id'] for x in points['segment_id'][sl])
                    or int(times[left]) != b['start_us'] or int(times[right]) != b['end_us']
                    or np.any(times[sl] <= 0) or np.any(np.diff(times[sl]) <= 0)):
                raise ValueError('GT block identity/clock mismatch')
            indices = sorted(interval_rows[b['block_id']],key=lambda i:int(velocity['start_us'][i]))
            if len(indices) != b['interval_count'] or len(indices) != right-left:
                raise ValueError('review point/interval conservation mismatch')
            for offset,i in enumerate(indices):
                if (str(velocity['left_token'][i]) != str(tokens[left+offset])
                        or str(velocity['right_token'][i]) != str(tokens[left+offset+1])
                        or int(velocity['start_us'][i]) != int(times[left+offset])
                        or int(velocity['end_us'][i]) != int(times[left+offset+1])):
                    raise ValueError('review velocity/point token mismatch')
            speeds = velocity['speed_mps'][indices]
            if speeds.dtype != np.float64 or not np.isfinite(speeds).all() or np.any(speeds < 0):
                raise ValueError('invalid review speeds')
            enriched.append({**b,'time_us':times[sl].tolist(),'xy':xy[sl].tolist(),
                             'annotation_tokens':tokens[sl].tolist(),'sample_tokens':points['sample_token'][sl].tolist(),
                             'speed_mps':speeds.tolist(),'event_ids':sorted(events_by_block[b['block_id']]),
                             'diagnostic_ids':sorted(diagnostics_by_block[b['block_id']])})
    if len(enriched) != len(blocks):
        raise ValueError('unsupported role in review blocks')
    tasks = []
    def task(kind, record, source_id, block_id='', event_id='', diagnostic_id=''):
        tasks.append({'task_id':digest({'schema':'review_task_v1','parent_identity':run['identity'],
                                       'kind':kind,'source_id':source_id}),
                      'task_kind':kind,'role':record['role'],'instance':record['instance'],
                      'block_id':block_id,'event_id':event_id,'diagnostic_id':diagnostic_id,
                      'review_status':'pending','reviewer':'','notes':''})
    for b in blocks: task('block_scan',b,b['block_id'],block_id=b['block_id'])
    for e in events: task('event_check',e,e['event_id'],block_id=e['block_id'],event_id=e['event_id'])
    for d in orphan: task('diagnostic_check',d,d['diagnostic_id'],diagnostic_id=d['diagnostic_id'])
    tasks.sort(key=lambda r:(r['role'],r['instance'],r['block_id'],r['task_kind'],r['task_id']))
    summary = {'status':'review_materials_ready','block_scan_tasks':len(blocks),'event_check_tasks':len(events),
               'orphan_diagnostic_tasks':len(orphan),'task_rows':len(tasks),'diagnostic_rows':len(diagnostics),
               'blocks_without_candidates':sum(not b['event_ids'] for b in enriched),
               'role_task_counts':dict(sorted(Counter(t['role'] for t in tasks).items())),
               'initial_working_status_counts':{'pending':len(tasks),'in_progress':0,'reviewed':0},
               'manual_completed':0,'human_event_labels_imported':False,'new_prediction_scores_computed':False,
               'neural_training':False,'holdout_evaluated':False,'coverage_reused_without_new_anchor_selection':True}
    return {'parent_identity':run['identity'],'blocks':sorted(enriched,key=lambda b:(b['role'],b['instance'],b['start_us'])),
            'events':events,'diagnostics':diagnostics,'window_links':read_json(parent/'window_event_links.json'),
            'coverage':read_json(parent/'summary.json')['coverage'],'queue':tasks,'summary':summary}
