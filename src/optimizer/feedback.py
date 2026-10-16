"""Explanations a user voted on, in an Azure table, the raw material for tuning the prompt.

Nothing is stored for an explanation by itself: the service stays stateless until the user votes.
The vote brings the explain payload and the answer back, and one entity keeps what the model saw
(the digest), which prompt and model, the answer, the vote and a few plan features to group by.
PartitionKey is the month of the vote.
"""
import gzip
import hashlib
import uuid
from datetime import datetime, timezone

from azure.data.tables import TableClient
from azure.identity import DefaultAzureCredential

TABLE = 'explanations'

# a string or binary property holds 64 KiB at most, the digest of a long horizon is larger
CHUNK = 60_000


def configured(settings):
    return bool(settings.feedback_connection_string or settings.feedback_table_endpoint)


def table(settings):
    if settings.feedback_connection_string:
        return TableClient.from_connection_string(settings.feedback_connection_string, TABLE)
    return TableClient(settings.feedback_table_endpoint, TABLE, credential=DefaultAzureCredential())


def subject_hash(subject):
    """Account pseudonym: lets one voter's rows be grouped or removed without naming the account."""
    return hashlib.sha256(subject.encode()).hexdigest()[:16] if subject else None


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def pack(text):
    """gzip the text and split it into properties digest0..digestN below the property size limit."""
    data = gzip.compress(text.encode())
    parts = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)] or [b'']
    return {'digestParts': len(parts), **{f'digest{i}': part for i, part in enumerate(parts)}}


def unpack(entity):
    data = b''.join(bytes(entity[f'digest{i}']) for i in range(entity.get('digestParts', 0)))
    return gzip.decompress(data).decode() if data else ''


def features(payload):
    """Plan features worth grouping votes by."""
    req, res = payload['req'], payload['res']
    details = payload.get('details') or {}
    kinds = [d.get('type') for d in details.get('batteryDetails') or []]
    return {
        'status': res.get('status'),
        'steps': len(req['time_series']['dt']),
        'horizonHours': round(sum(req['time_series']['dt']) / 3600),
        'vehicles': kinds.count('vehicle'),
        'batteries': kinds.count('battery'),
        'strategy': (req.get('strategy') or {}).get('charging_strategy'),
        'departure': any(b.get('r_departure') for b in req['batteries']),
        'goals': any(any(v > 0 for v in b.get('s_goal') or []) for b in req['batteries']),
        'demands': any(any(v > 0 for v in b.get('p_demand') or []) for b in req['batteries']),
    }


def store(settings, payload, digest, prompt, subject=None):
    """Store a vote with the explanation it was given on. payload is the explain payload plus
    explanation, vote and comment."""
    now = datetime.now(timezone.utc)
    entity = {
        'PartitionKey': f'{now:%Y-%m}',
        'RowKey': uuid.uuid4().hex,
        'created': now,
        'language': payload.get('language') or 'en',
        'promptHash': prompt_hash(prompt),
        'model': settings.azure_openai_deployment,
        'answer': payload['explanation'],
        'vote': payload['vote'],
        'comment': (payload.get('comment') or '')[:4000],
        'subject': subject_hash(subject),
        **features(payload),
        **pack(digest),
    }
    table(settings).create_entity(entity)


def rows(settings, since=None):
    """All votes, oldest first, with the digest unpacked. since limits to months from YYYY-MM on."""
    query = f"PartitionKey ge '{since}'" if since else None
    for entity in table(settings).query_entities(query) if query else table(settings).list_entities():
        row = {k: v for k, v in entity.items() if not k.startswith('digest')}
        row['digest'] = unpack(entity)
        yield row
