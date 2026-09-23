## Troubleshooting

When running ADK Web in GitHub Codespaces, use the forwarded host and allow the Codespaces origin:

```bash
adk web --host 0.0.0.0 --port 8001 --allow_origins="*"
```

This fixes `403 Forbidden` responses for `/dev-ui` assets when opening the forwarded port.
