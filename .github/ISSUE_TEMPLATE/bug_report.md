---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

<!-- A clear and concise description of what the bug is -->

## To Reproduce

Steps to reproduce the behavior:

1. Configure '...'
2. Run command '...'
3. Send request '...'
4. See error

## Expected Behavior

<!-- A clear and concise description of what you expected to happen -->

## Actual Behavior

<!-- What actually happened -->

## Error Message

```
Paste the full error message and stack trace here
```

## Environment

- **ml-api version**: [e.g., 0.1.0]
- **Python version**: [e.g., 3.10.5]
- **OS**: [e.g., Ubuntu 22.04, macOS 14.0]
- **Installation method**: [pip, uv, docker]
- **Database**: [PostgreSQL version]
- **Redis**: [Redis version]

## Configuration

```yaml
# Relevant configuration (redact sensitive info)
GCS_BUCKET: my-bucket
DATABASE_URL: postgresql+asyncpg://...
```

## Code Sample

```python
# Minimal code to reproduce the issue
from ml_api.services.split_service import SplitService

# Your code here
```

## Logs

```
Paste relevant logs here
```

## Request/Response

<!-- If API-related, provide the request and response -->

**Request:**
```bash
curl -X POST http://localhost:8000/v1/splits \
  -H "Content-Type: application/json" \
  -d '{...}'
```

**Response:**
```json
{
  "error": "..."
}
```

## Screenshots

<!-- If applicable, add screenshots to help explain your problem -->

## Additional Context

<!-- Add any other context about the problem here -->

## Possible Solution

<!-- If you have suggestions on how to fix the bug -->

## Checklist

- [ ] I've searched existing issues to ensure this bug hasn't been reported
- [ ] I've included all relevant information above
- [ ] I've tested with the latest version
- [ ] I've provided a minimal reproducible example
