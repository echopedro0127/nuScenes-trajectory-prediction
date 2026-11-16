import json
import re
import shutil
import subprocess

import pytest
from test_event_review_data import review_fixture as review_fixture  # noqa: PLC0414


def test_offline_html_embeds_all_sources_and_escapes_script(review_fixture):
    from trajectory_prediction.evaluation.event_review_data import assemble_review
    from trajectory_prediction.evaluation.event_review_html import render_html
    data=assemble_review(*review_fixture)
    data['blocks'][0]['scene']='</script><script>alert(1)</script>'
    html=render_html(data)
    assert '</script><script>alert(1)' not in html
    payload=re.search(r'id="review-data">(.*?)</script>',html,re.DOTALL).group(1)
    assert json.loads(payload)==data
    assert '\u003c/script\u003e' in html
    assert '<script src=' not in html and 'fetch(' not in html
    assert 'review-data' in html and 'import-working' in html and 'export-working' in html
    assert "default-src 'none'" in html


def test_node_csv_and_locked_identity_behavior(review_fixture,tmp_path):
    from trajectory_prediction.evaluation.event_review_data import assemble_review, csv_text
    from trajectory_prediction.evaluation.event_review_html import script_source
    node=shutil.which('node')
    if node is None: pytest.skip('Node is unavailable; browser script contract not executed')
    baseline=assemble_review(*review_fixture)['queue']
    path=tmp_path/'review.js';path.write_text(script_source(),encoding='utf-8')
    command="""
const fs=require('fs'),a=require('assert'),m=require(process.argv[1]);
const d=JSON.parse(fs.readFileSync(0,'utf8')),rows=d.baseline.map(r=>({...r}));
a.equal(m.csvText(rows),d.csv);
rows.reverse();rows[0].review_status='reviewed';rows[0].reviewer='reader';rows[0].notes='comma, quote "\\nand newline';
let csv=m.csvText(rows);a.deepEqual(m.validateRows(m.parseCsv('\\ufeff'+csv),d.baseline),rows);
for(const change of [r=>r.instance='wrong',r=>r.review_status='confirmed',r=>{r.review_status='reviewed';r.reviewer='';}]){
 const bad=rows.map(r=>({...r}));change(bad[0]);a.throws(()=>m.validateRows(m.parseCsv(m.csvText(bad)),d.baseline));
}
a.throws(()=>m.parseCsv('"unterminated'));
a.throws(()=>m.validateRows(m.parseCsv(m.csvText(rows.slice(1))),d.baseline));
console.log('Node CSV roundtrip and identity checks passed');
"""
    result=subprocess.run([node,'-e',command,str(path)],input=json.dumps({'baseline':baseline,'csv':csv_text(baseline)}),
                          capture_output=True,text=True,encoding='utf-8',check=False)
    assert result.returncode==0,result.stderr
