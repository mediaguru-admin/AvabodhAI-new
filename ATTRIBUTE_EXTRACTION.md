# Chunk attribute extraction

Attribute definitions are stored in MongoDB and are resolved per tenant. The
repository's existing organisation identifier is named `org_unit_id` and is
carried by `X-Org-Unit-ID`. If it is supplied, its active definition overrides
the tenant definition with the same key; tenant definitions fill missing keys. No Jev call
is made when no effective definitions exist or `JEV_API_KEY` is empty.

## Definition API

All requests require `X-Tenant-ID`. `X-Org-Unit-ID` is optional for these
routes.

```text
GET    /attributes
PUT    /attributes/{key}
DELETE /attributes/{key}
GET    /attributes/chunks/{chunk_id}
```

Example definition (Jev `choice` requires predefined values):

```json
{
  "key": "place",
  "description": "Place explicitly mentioned in this chunk",
  "type": "choice",
  "allowed_values": ["Mumbai", "Delhi", "Bengaluru"],
  "active": true
}
```

`score` definitions use `allowed_values` as criteria and `noul` definitions
return a typed yes/no probability. Values below `JEV_MIN_CONFIDENCE` and the
`__NOT_FOUND__` choice are not persisted.

## Configuration

Set `MONGODB_URL`, `MONGODB_DATABASE`, and `JEV_API_KEY` in the environment.
The uploader remains fully functional without MongoDB or Jev: extraction is a
safe no-op and vector ingestion continues. Jev is a hosted API, not a local
Ollama model.
