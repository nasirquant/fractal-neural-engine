# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |

## Reporting a Vulnerability

Please report security vulnerabilities to **security@fnse.dev** (or create a private GitHub Security Advisory).

Do **not** open public issues for security vulnerabilities.

We will acknowledge receipt within 48 hours and provide a timeline for fix.

## Security Considerations

This project runs user-generated code via the **SkillCompiler** subsystem. Key security measures include:

- **Sandboxed execution** - Skills run in isolated environments with restricted imports
- **Keyword blocking** - Dangerous keywords (`eval`, `exec`, `os.`, `subprocess`, etc.) are blocked
- **Resource limits** - CPU time, memory, and recursion depth are enforced
- **Circuit breakers** - Automatic intervention on anomalous agent behavior
- **Audit logging** - All skill compilations and executions are logged

### For Deployments

- Always run behind authentication (the API includes auth middleware)
- Use a dedicated Redis instance with authentication enabled
- Rotate API keys regularly
- Monitor Grafana dashboards for anomalous activity
- Keep dependencies updated (Dependabot alerts enabled)