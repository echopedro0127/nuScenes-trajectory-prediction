"""Strict, bounded, lazy iteration over JSON metadata object arrays."""
import json
import math


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON object field')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite JSON number: ' + value)


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('JSON number exceeds finite float64 range')
    return result


def iter_json_objects(path, chunk_size=1048576, max_record_chars=1048576):
    """Yield complete dictionaries without accepting damaged table syntax.

    Limits count decoded characters; UTF-8 decoding is delegated to TextIO.
    Memory is bounded by the chunk plus a single bounded record, not table size.
    """
    if any(type(v) is not int or v <= 0 for v in (chunk_size, max_record_chars)):
        raise ValueError('positive integer chunk/record limits required')
    decoder = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)
    with open(path, encoding='utf-8', newline='') as stream:
        buffer, index, eof = '', 0, False

        def fill():
            nonlocal buffer, index, eof
            chunk = stream.read(chunk_size)
            buffer, index = buffer[index:] + chunk, 0
            eof = not chunk

        def peek():
            nonlocal index
            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index < len(buffer):
                    return buffer[index]
                if eof:
                    return None
                fill()

        if peek() != '[':
            raise ValueError('top-level JSON object array required')
        index += 1
        first = True
        while True:
            char = peek()
            if char == ']' and first:
                index += 1
                break
            if char is None:
                raise ValueError('truncated JSON array')
            while True:
                try:
                    row, end = decoder.raw_decode(buffer, index)
                except json.JSONDecodeError as error:
                    if len(buffer) - index > max_record_chars:
                        raise ValueError('metadata record exceeds character limit') from error
                    if eof:
                        raise ValueError('invalid/truncated metadata record') from error
                    fill()
                    continue
                if end - index > max_record_chars:
                    raise ValueError('metadata record exceeds character limit')
                if not isinstance(row, dict):
                    raise ValueError('metadata array elements must be objects')  # noqa: TRY004
                index = end
                yield row
                break
            first = False
            char = peek()
            if char == ']':
                index += 1
                break
            if char != ',':
                raise ValueError('comma or array terminator required')
            index += 1
            if peek() == ']':
                raise ValueError('trailing comma in metadata array')
        if peek() is not None:
            raise ValueError('trailing data after metadata array')
