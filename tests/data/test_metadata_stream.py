import json

import pytest

from trajectory_prediction.data.metadata_stream import iter_json_objects


@pytest.mark.parametrize('chunk_size', [1, 2, 7, 1024])
def test_stream_crosses_objects_unicode_and_escapes(tmp_path, chunk_size):
    rows = [{'token': 'a', 'name': '中文\\"x', 'nested': {'x': [1, True, None]}}, {'token': 'b'}]
    path = tmp_path / 'table.json'
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    assert list(iter_json_objects(path, chunk_size=chunk_size)) == rows


def test_empty_array_and_whitespace(tmp_path):
    path = tmp_path / 'empty.json'
    path.write_text(' \n [ \t ] \n')
    assert list(iter_json_objects(path, chunk_size=1)) == []


@pytest.mark.parametrize('text', ['', '{}', '[{}', '[{},]', '[{},,{}]', '[{} {}]', '[1]',
                                  '[null]', '[] true', '[{"a":1,"a":2}]', '[{"x":NaN}]',
                                  '[{"x":Infinity}]', '[{"x":-Infinity}]', '[{"x":1e400}]',
                                  '[{"x":"unterminated}]'])
def test_malformed_input_is_never_skipped(tmp_path, text):
    path = tmp_path / 'bad.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        list(iter_json_objects(path, chunk_size=2))


def test_bounded_record_and_validated_limits(tmp_path):
    path = tmp_path / 'large.json'
    path.write_text(json.dumps([{'x': 'a' * 100}]))
    with pytest.raises(ValueError, match='record'):
        list(iter_json_objects(path, chunk_size=3, max_record_chars=30))
    for invalid in (0, -1, True, 1.2):
        with pytest.raises(ValueError):
            list(iter_json_objects(path, chunk_size=invalid))


def test_can_stop_without_consuming_later_invalid_record(tmp_path):
    path = tmp_path / 'lazy.json'
    path.write_text('[{"token":"first"},bad]')
    rows = iter_json_objects(path, chunk_size=1)
    assert next(rows) == {'token': 'first'}
    rows.close()
