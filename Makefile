# Browser-Use Local Deployment Makefile
# Simplified commands for managing the Docker Compose deployment

.PHONY: help setup start stop restart logs status clean health test

# Default target
help:
	@echo "Browser-Use Local Deployment Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup and Deployment:"
	@echo "  make setup      - Initial setup (create directories, copy env file)"
	@echo "  make start      - Start all services"
	@echo "  make stop       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo ""
	@echo "Monitoring:"
	@echo "  make logs       - Show service logs"
	@echo "  make status     - Show service status"
	@echo "  make health     - Check service health"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Stop services and remove containers/volumes"
	@echo "  make update     - Pull latest images and restart"
	@echo "  make backup     - Create backup of data and config"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Run basic functionality tests"
	@echo ""
	@echo "Integration:"
	@echo "  make start-integrated  - Start with Bytebot integration"
	@echo "  make stop-integrated   - Stop integrated deployment"

# Setup commands
setup:
	@echo "Setting up browser-use local deployment..."
	@mkdir -p data/{profiles,downloads} logs secrets config
	@chmod 700 secrets
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file - please configure your API keys"; fi
	@echo "Setup completed. Please edit .env file with your configuration."

# Docker Compose commands
start: setup
	@echo "Starting browser-use services..."
	@docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 30
	@make health
	@echo "Services started successfully!"
	@echo ""
	@echo "Access URLs:"
	@echo "  Browser-Use API: http://localhost:9242"
	@echo "  Chrome DevTools: http://localhost:9222"
	@echo "  VNC Browser:     http://localhost:5900"
	@echo "  Prometheus:      http://localhost:9090"
	@echo "  Grafana:         http://localhost:3000"

stop:
	@echo "Stopping browser-use services..."
	@docker-compose down
	@echo "Services stopped."

restart: stop start

# Integrated deployment
start-integrated: setup
	@echo "Starting integrated AIgent/Bytebot deployment..."
	@cd .. && docker-compose -f docker-compose.integrated.yml up -d
	@echo "Waiting for services to be ready..."
	@sleep 45
	@make health
	@echo "Integrated services started successfully!"

stop-integrated:
	@echo "Stopping integrated deployment..."
	@cd .. && docker-compose -f docker-compose.integrated.yml down
	@echo "Integrated services stopped."

# Monitoring commands
logs:
	@docker-compose logs -f browser-use

logs-all:
	@docker-compose logs -f

status:
	@docker-compose ps

health:
	@echo "Checking service health..."
	@curl -f -s http://localhost:9242/health > /dev/null && echo "✅ Browser-Use: Healthy" || echo "❌ Browser-Use: Unhealthy"
	@docker-compose exec -T postgres pg_isready -q -U browseruse && echo "✅ PostgreSQL: Healthy" || echo "❌ PostgreSQL: Unhealthy"
	@docker-compose exec -T redis redis-cli ping > /dev/null && echo "✅ Redis: Healthy" || echo "❌ Redis: Unhealthy"

# Maintenance commands
clean:
	@echo "WARNING: This will remove all containers, volumes, and data!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read
	@docker-compose down -v --remove-orphans
	@docker system prune -f
	@echo "Cleanup completed."

update:
	@echo "Updating browser-use images..."
	@docker-compose pull
	@docker-compose up -d
	@echo "Update completed."

backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@docker-compose exec postgres pg_dump -U browseruse browseruse > backups/database-$(shell date +%Y%m%d_%H%M%S).sql
	@tar -czf backups/browser-data-$(shell date +%Y%m%d_%H%M%S).tar.gz data/
	@tar -czf backups/config-$(shell date +%Y%m%d_%H%M%S).tar.gz config/ secrets/ .env
	@echo "Backup completed in backups/ directory"

# Development commands
dev-logs:
	@docker-compose logs -f browser-use postgres redis

dev-shell:
	@docker-compose exec browser-use bash

dev-db:
	@docker-compose exec postgres psql -U browseruse -d browseruse

# Testing commands
test:
	@echo "Running basic functionality tests..."
	@echo "Testing browser-use API health..."
	@curl -f http://localhost:9242/health || (echo "❌ Health check failed" && exit 1)
	@echo "✅ Health check passed"
	@echo "Testing browser session creation..."
	@curl -X POST http://localhost:9242/api/v1/sessions \
		-H "Content-Type: application/json" \
		-d '{"name": "test-session", "profile": "default"}' || (echo "❌ Session creation failed" && exit 1)
	@echo "✅ Session creation test passed"
	@echo "All tests passed!"

# Debugging commands
debug-browser:
	@echo "Browser-use container logs:"
	@docker-compose logs --tail=100 browser-use

debug-chrome:
	@echo "Checking Chrome processes in container:"
	@docker-compose exec browser-use ps aux | grep chrome || echo "No Chrome processes found"

debug-display:
	@echo "Checking display server:"
	@docker-compose exec xvfb xset -display :99 q || echo "Display server not responding"

debug-permissions:
	@echo "Checking file permissions:"
	@docker-compose exec browser-use ls -la /data
	@ls -la data/

# Port checking
check-ports:
	@echo "Checking port availability..."
	@for port in 9242 9222 5900 5432 6379 9090 3000; do \
		if netstat -tuln 2>/dev/null | grep -q ":$$port " || ss -tuln 2>/dev/null | grep -q ":$$port "; then \
			echo "❌ Port $$port is in use"; \
		else \
			echo "✅ Port $$port is available"; \
		fi; \
	done

# Resource monitoring
monitor:
	@echo "Resource usage:"
	@docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Quick commands
quick-start: setup
	@docker-compose up -d browser-use postgres
	@echo "Quick start completed (core services only)"

quick-stop:
	@docker-compose stop browser-use postgres
	@echo "Core services stopped"