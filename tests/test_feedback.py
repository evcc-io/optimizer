import pytest

from optimizer import explain, feedback
from optimizer.app import app, settings


class FakeTable:
    """The two table calls the module makes, on a list."""

    def __init__(self):
        self.rows = []

    def create_entity(self, entity):
        self.rows.append(dict(entity))

    def list_entities(self):
        return list(self.rows)


@pytest.fixture
def store(monkeypatch):
    fake = FakeTable()
    monkeypatch.setattr(settings, 'feedback_connection_string', 'UseDevelopmentStorage=true')
    monkeypatch.setattr(feedback, 'table', lambda _settings: fake)
    return fake


@pytest.fixture
def voted(payload):
    return {**payload, 'explanation': 'because', 'vote': 'down', 'comment': 'wrong time'}


def test_pack_round_trips_long_digests():
    text = 'row | 1.0 | 2.0\n' * 20000
    packed = feedback.pack(text)

    assert packed['digestParts'] >= 1
    assert all(len(packed[f'digest{i}']) <= feedback.CHUNK for i in range(packed['digestParts']))
    assert feedback.unpack(packed) == text
    assert feedback.unpack(feedback.pack('')) == ''


def test_feedback_stores_the_vote_with_the_plan_and_explanation(voted, store):
    client = app.test_client()
    response = client.post('/optimize/explain/feedback', json=voted)
    assert response.status_code == 200, response.json

    row, = store.rows
    assert (row['vote'], row['comment'], row['answer']) == ('down', 'wrong time', 'because')
    assert row['language'] == 'de'
    assert row['promptHash'] == feedback.prompt_hash(explain.SYSTEM_PROMPT)
    assert row['status'] == 'Optimal'
    assert (row['steps'], row['horizonHours'], row['vehicles'], row['batteries']) == (5, 2, 0, 1)
    assert row['PartitionKey'] == row['created'].strftime('%Y-%m')
    assert 'today 10:30 | 15' in feedback.unpack(row)

    # a second vote on the same explanation is another row
    assert client.post('/optimize/explain/feedback', json={**voted, 'vote': 'up', 'comment': ''}).status_code == 200
    assert [(r['vote'], r['comment']) for r in feedback.rows(settings)] == [('down', 'wrong time'), ('up', '')]


@pytest.mark.parametrize('change', [{'vote': 'meh'}, {'vote': None}, {'explanation': None}, {'res': {}}])
def test_feedback_rejects_incomplete_votes(voted, store, change):
    assert app.test_client().post('/optimize/explain/feedback', json={**voted, **change}).status_code == 400
    assert store.rows == []


def test_explain_stores_nothing(payload, configured, store, monkeypatch):
    monkeypatch.setattr(explain, 'chat', lambda _settings, _messages: ('because', {}))
    response = app.test_client().post('/optimize/explain', json=payload)

    assert response.status_code == 200
    assert response.json == {'explanation': 'because'}
    assert store.rows == []


def test_feedback_without_store_is_503(voted):
    assert app.test_client().post('/optimize/explain/feedback', json=voted).status_code == 503


def test_store_failure_is_502(voted, store, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError('table offline')

    monkeypatch.setattr(feedback, 'store', broken)
    response = app.test_client().post('/optimize/explain/feedback', json=voted)

    assert response.status_code == 502
    assert 'table offline' in response.json['message']
