# Browser-Use Local Deployment with Docker Compose

Complete Docker Compose configuration for browser-use local deployment with 100% local-only architecture, Chrome/Chromium support, and AIgent/Bytebot integration.

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- Port availability: 9242, 9222, 5900, 5432, 6379, 9090, 3000

### Basic Deployment

1. **Clone and Configure**
   ```bash
   cd browser-use/
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

2. **Start Services**
   ```bash
   # Standalone browser-use deployment
   docker-compose up -d
   
   # Or integrated with Bytebot platform
   cd ..
   docker-compose -f docker-compose.integrated.yml up -d
   ```

3. **Access Services**
   - Browser-Use API: http://localhost:9242
   - Chrome DevTools: http://localhost:9222
   - VNC Browser View: http://localhost:5900 (password: browseruse)
   - Prometheus Metrics: http://localhost:9090
   - Grafana Dashboards: http://localhost:3000 (admin/admin)

## 📋 Architecture Overview

### Standalone Configuration (`docker-compose.yml`)

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Browser-Use   │    │      XVFB       │    │   PostgreSQL    │
│   Main Service  │◄──►│  Display Server │    │    Database     │
│   Port: 9242    │    │   Port: 5900    │    │   Port: 5433    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Chrome      │    │     Redis       │    │   Prometheus    │
│ DevTools (CDP)  │    │     Cache       │    │   Monitoring    │
│   Port: 9222    │    │   Port: 6380    │    │   Port: 9090    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │     Grafana     │
                                               │   Dashboards    │
                                               │   Port: 3000    │
                                               └─────────────────┘
```

### Integrated Configuration (`docker-compose.integrated.yml`)

```
┌───────────────────────────────────────────────────────────────┐
│                    AIgent/Bytebot Platform                    │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Bytebot-Agent  │   Bytebot-UI    │      Bytebot-Desktop        │
│   Port: 9991    │   Port: 9992    │       Port: 9990            │
└─────────────────┴─────────────────┴─────────────────────────────┘
         │                │                        │
         ▼                ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Browser-Use Service                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Browser-Use   │  │      XVFB       │  │   MCP Server    │ │
│  │  Port: 9242     │  │   Port: 5901    │  │   Port: 8100    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │      Redis      │    │   Monitoring    │
│  (Shared DB)    │    │  (Shared Cache) │    │ Stack (Shared)  │
│   Port: 5432    │    │   Port: 6379    │    │ Prometheus/Graf │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# AI API Keys
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# Local Deployment
LOCAL_DEPLOYMENT=true
BYTEBOT_INTEGRATION=true  # For integrated mode

# Browser Settings
BROWSER_USE_HEADLESS=false
CHROME_DEBUG_PORT=9222

# Security
LOCAL_SECRETS_ENCRYPTION_KEY=your_secret_key
```

### Service Configuration

Browser-use configuration is managed in `config/browser-use.yaml`:

```yaml
app:
  name: "browser-use-local"
  debug: false

browser:
  executable_path: "/usr/bin/chromium-browser"
  headless: false
  sandbox: false

ai:
  default_provider: "anthropic"
  providers:
    anthropic:
      model: "claude-3-5-sonnet-20241022"
```

## 🌐 Service Access

### API Endpoints

- **Health Check**: `GET http://localhost:9242/health`
- **Browser Tasks**: `POST http://localhost:9242/api/v1/tasks`
- **Sessions**: `GET http://localhost:9242/api/v1/sessions`
- **Screenshots**: `POST http://localhost:9242/api/v1/sessions/{id}/screenshot`

### Browser Visualization

- **VNC Access**: Connect to `localhost:5900` with VNC client
- **Chrome DevTools**: Open `http://localhost:9222` in browser

### Monitoring

- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin/admin)

## 💾 Data Persistence

### Volume Mounts

```bash
# Browser-use data
./browser-use/data:/data                    # Main data directory
./browser-use/logs:/app/logs               # Application logs
./browser-use/config:/app/config           # Configuration files
./browser-use/secrets:/app/secrets         # Encrypted secrets

# Database data
postgres_data:/var/lib/postgresql/data     # PostgreSQL data
redis_data:/data                           # Redis persistence
```

### Backup Strategy

```bash
# Backup database
docker-compose exec postgres pg_dump -U browseruse browseruse > backup.sql

# Backup browser data
tar -czf browser-data-backup.tar.gz browser-use/data/

# Backup configuration
tar -czf config-backup.tar.gz browser-use/config/ browser-use/secrets/
```

## 🔒 Security Features

### Local-Only Architecture
- ✅ No cloud dependencies (except AI APIs)
- ✅ Local file-based secrets management
- ✅ Local database storage
- ✅ Local monitoring stack

### Access Control
- Role-based API access (admin/operator/viewer)
- Local JWT token authentication
- Encrypted secrets storage
- Network isolation via Docker networks

### Chrome Security
- Sandboxing disabled for local development
- Remote debugging enabled for CDP access
- Proper user permissions and isolation

## 🔧 Maintenance

### Log Management

```bash
# View service logs
docker-compose logs browser-use
docker-compose logs -f postgres

# Log rotation is configured automatically
# Logs stored in: ./browser-use/logs/
```

### Health Monitoring

```bash
# Check service health
curl http://localhost:9242/health

# View metrics
curl http://localhost:9242/metrics

# Database status
docker-compose exec postgres pg_isready -U browseruse
```

### Updates

```bash
# Update browser-use image
docker-compose pull browser-use
docker-compose up -d browser-use

# Update all services
docker-compose pull
docker-compose up -d
```

## 🐛 Troubleshooting

### Common Issues

1. **Chrome fails to start**
   ```bash
   # Check Chrome process
   docker-compose exec browser-use ps aux | grep chrome
   
   # Verify display server
   docker-compose exec xvfb xset -display :99 q
   ```

2. **Permission issues**
   ```bash
   # Fix data directory permissions
   sudo chown -R 911:911 browser-use/data/
   sudo chmod -R 755 browser-use/data/
   ```

3. **Port conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :9242
   
   # Modify ports in docker-compose.yml or .env
   ```

### Debug Mode

```bash
# Enable debug logging
echo "BROWSER_USE_DEBUG=true" >> .env
docker-compose up -d

# Access container for debugging
docker-compose exec browser-use bash
```

## 🔗 Integration

### Bytebot Integration

When using the integrated configuration:

1. Browser-use APIs are available to Bytebot agents
2. Shared database for cross-service data
3. Unified monitoring and logging
4. MCP server for Claude integration

### MCP Server

The MCP server provides Claude-compatible tools:

```bash
# Access MCP server
curl http://localhost:8100/tools

# Available tools: browser_navigate, browser_click, browser_type, etc.
```

## 📚 API Documentation

### Create Browser Task

```bash
curl -X POST http://localhost:9242/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Web Search Task",
    "description": "Search for information on a website",
    "url": "https://example.com",
    "actions": [
      {"type": "navigate", "url": "https://example.com"},
      {"type": "screenshot"},
      {"type": "extract", "selector": "h1, p"}
    ]
  }'
```

### Create Browser Session

```bash
curl -X POST http://localhost:9242/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Browser Session",
    "profile": "default",
    "headless": false
  }'
```

For complete API documentation, visit the browser-use service at `/docs` endpoint once running.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test with Docker Compose
5. Submit pull request

## 📞 Support

- GitHub Issues: Create an issue for bugs or feature requests
- Documentation: Check the browser-use official documentation
- Community: Join the browser-use Discord server