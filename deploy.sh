#!/bin/bash

# Browser-Use Local Deployment Script
# Automated setup for browser-use Docker Compose deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIRED_DOCKER_VERSION="20.10"
REQUIRED_COMPOSE_VERSION="2.0"
DEPLOYMENT_MODE="standalone" # standalone or integrated

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Docker version
check_docker_version() {
    if ! command_exists docker; then
        log_error "Docker is not installed. Please install Docker Engine $REQUIRED_DOCKER_VERSION or higher."
        exit 1
    fi
    
    DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    log_info "Docker version: $DOCKER_VERSION"
    
    # Note: This is a simple version check, in production you might want more robust version comparison
    if [[ $(echo "$DOCKER_VERSION $REQUIRED_DOCKER_VERSION" | tr " " "\n" | sort -V | head -n1) != "$REQUIRED_DOCKER_VERSION" ]]; then
        log_warning "Docker version $DOCKER_VERSION detected. Recommended: $REQUIRED_DOCKER_VERSION or higher"
    fi
}

# Check Docker Compose version
check_compose_version() {
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose is not installed. Please install Docker Compose $REQUIRED_COMPOSE_VERSION or higher."
        exit 1
    fi
    
    if command_exists docker-compose; then
        COMPOSE_VERSION=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_VERSION=$(docker compose version --short)
        COMPOSE_CMD="docker compose"
    fi
    
    log_info "Docker Compose version: $COMPOSE_VERSION"
    log_info "Using command: $COMPOSE_CMD"
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check available memory
    if command_exists free; then
        AVAILABLE_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
        if [[ $AVAILABLE_MEM_GB -lt 4 ]]; then
            log_warning "Available memory: ${AVAILABLE_MEM_GB}GB. Recommended: 4GB or more"
        else
            log_success "Available memory: ${AVAILABLE_MEM_GB}GB"
        fi
    fi
    
    # Check disk space
    if command_exists df; then
        AVAILABLE_DISK_GB=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
        if [[ $AVAILABLE_DISK_GB -lt 10 ]]; then
            log_warning "Available disk space: ${AVAILABLE_DISK_GB}GB. Recommended: 10GB or more"
        else
            log_success "Available disk space: ${AVAILABLE_DISK_GB}GB"
        fi
    fi
}

# Check port availability
check_ports() {
    log_info "Checking port availability..."
    
    PORTS_TO_CHECK=(9242 9222 5900 5432 6379 9090 3000)
    PORTS_IN_USE=()
    
    for port in "${PORTS_TO_CHECK[@]}"; do
        if command_exists netstat && netstat -tuln | grep -q ":$port "; then
            PORTS_IN_USE+=($port)
        elif command_exists ss && ss -tuln | grep -q ":$port "; then
            PORTS_IN_USE+=($port)
        elif command_exists lsof && lsof -i ":$port" >/dev/null 2>&1; then
            PORTS_IN_USE+=($port)
        fi
    done
    
    if [[ ${#PORTS_IN_USE[@]} -gt 0 ]]; then
        log_warning "The following ports are already in use: ${PORTS_IN_USE[*]}"
        log_warning "You may need to modify port mappings in docker-compose.yml or .env"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    else
        log_success "All required ports are available"
    fi
}

# Setup environment file
setup_env_file() {
    log_info "Setting up environment configuration..."
    
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
            log_success "Created .env file from .env.example"
        else
            log_error ".env.example file not found!"
            exit 1
        fi
    else
        log_info ".env file already exists, skipping creation"
    fi
    
    log_warning "Please edit .env file with your API keys and configuration before proceeding"
    log_info "Required API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY"
    
    read -p "Have you configured the .env file with your API keys? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Please configure .env file and run this script again"
        exit 0
    fi
}

# Create required directories
setup_directories() {
    log_info "Creating required directories..."
    
    DIRECTORIES=(
        "$SCRIPT_DIR/data"
        "$SCRIPT_DIR/data/profiles"
        "$SCRIPT_DIR/data/downloads"
        "$SCRIPT_DIR/logs"
        "$SCRIPT_DIR/secrets"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            chmod 755 "$dir"
            log_success "Created directory: $dir"
        fi
    done
    
    # Set proper permissions for secrets directory
    chmod 700 "$SCRIPT_DIR/secrets"
}

# Pull Docker images
pull_images() {
    log_info "Pulling Docker images..."
    
    if [[ "$DEPLOYMENT_MODE" == "integrated" ]]; then
        cd "$SCRIPT_DIR/.."
        $COMPOSE_CMD -f docker-compose.integrated.yml pull
    else
        cd "$SCRIPT_DIR"
        $COMPOSE_CMD pull
    fi
    
    log_success "Docker images pulled successfully"
}

# Start services
start_services() {
    log_info "Starting browser-use services..."
    
    if [[ "$DEPLOYMENT_MODE" == "integrated" ]]; then
        cd "$SCRIPT_DIR/.."
        $COMPOSE_CMD -f docker-compose.integrated.yml up -d
    else
        cd "$SCRIPT_DIR"
        $COMPOSE_CMD up -d
    fi
    
    log_success "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for browser-use to be healthy
    RETRIES=0
    MAX_RETRIES=30
    
    while [[ $RETRIES -lt $MAX_RETRIES ]]; do
        if curl -f -s http://localhost:9242/health >/dev/null 2>&1; then
            log_success "Browser-use service is ready"
            break
        fi
        
        RETRIES=$((RETRIES + 1))
        if [[ $RETRIES -eq $MAX_RETRIES ]]; then
            log_error "Browser-use service failed to start within expected time"
            show_logs
            exit 1
        fi
        
        log_info "Waiting for browser-use service... ($RETRIES/$MAX_RETRIES)"
        sleep 10
    done
}

# Show service status
show_status() {
    log_info "Service status:"
    
    if [[ "$DEPLOYMENT_MODE" == "integrated" ]]; then
        cd "$SCRIPT_DIR/.."
        $COMPOSE_CMD -f docker-compose.integrated.yml ps
    else
        cd "$SCRIPT_DIR"
        $COMPOSE_CMD ps
    fi
}

# Show logs
show_logs() {
    log_info "Recent service logs:"
    
    if [[ "$DEPLOYMENT_MODE" == "integrated" ]]; then
        cd "$SCRIPT_DIR/.."
        $COMPOSE_CMD -f docker-compose.integrated.yml logs --tail=50 browser-use
    else
        cd "$SCRIPT_DIR"
        $COMPOSE_CMD logs --tail=50 browser-use
    fi
}

# Display access information
show_access_info() {
    log_success "Browser-use deployment completed successfully!"
    echo
    log_info "Service Access URLs:"
    echo "  🌐 Browser-Use API:    http://localhost:9242"
    echo "  🔧 Chrome DevTools:    http://localhost:9222"
    echo "  🖥️  VNC Browser View:   http://localhost:5900 (password: browseruse)"
    echo "  📊 Prometheus Metrics: http://localhost:9090"
    echo "  📈 Grafana Dashboards: http://localhost:3000 (admin/admin)"
    echo
    log_info "Health Check:"
    echo "  curl http://localhost:9242/health"
    echo
    log_info "View Logs:"
    echo "  $COMPOSE_CMD logs -f browser-use"
    echo
    log_info "Stop Services:"
    if [[ "$DEPLOYMENT_MODE" == "integrated" ]]; then
        echo "  $COMPOSE_CMD -f ../docker-compose.integrated.yml down"
    else
        echo "  $COMPOSE_CMD down"
    fi
}

# Cleanup function
cleanup() {
    if [[ "$1" == "stop" ]]; then
        log_info "Stopping browser-use services..."
        
        if [[ "$DEPLOYMENT_MODE" == "integrated" ]]; then
            cd "$SCRIPT_DIR/.."
            $COMPOSE_CMD -f docker-compose.integrated.yml down
        else
            cd "$SCRIPT_DIR"
            $COMPOSE_CMD down
        fi
        
        log_success "Services stopped"
    elif [[ "$1" == "clean" ]]; then
        log_warning "This will remove all containers, volumes, and data!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [[ "$DEPLOYMENT_MODE" == "integrated" ]]; then
                cd "$SCRIPT_DIR/.."
                $COMPOSE_CMD -f docker-compose.integrated.yml down -v --remove-orphans
            else
                cd "$SCRIPT_DIR"
                $COMPOSE_CMD down -v --remove-orphans
            fi
            
            log_success "Services and data cleaned up"
        fi
    fi
}

# Main deployment function
main() {
    log_info "Browser-Use Local Deployment Script"
    echo "===================================="
    echo
    
    # Parse command line arguments
    case "${1:-}" in
        "stop")
            cleanup "stop"
            exit 0
            ;;
        "clean")
            cleanup "clean"
            exit 0
            ;;
        "logs")
            show_logs
            exit 0
            ;;
        "status")
            show_status
            exit 0
            ;;
        "integrated")
            DEPLOYMENT_MODE="integrated"
            log_info "Using integrated deployment mode"
            ;;
        "standalone"|"")
            DEPLOYMENT_MODE="standalone"
            log_info "Using standalone deployment mode"
            ;;
        *)
            echo "Usage: $0 [standalone|integrated|stop|clean|logs|status]"
            echo
            echo "  standalone  - Deploy browser-use only (default)"
            echo "  integrated  - Deploy with full Bytebot integration"
            echo "  stop        - Stop running services"
            echo "  clean       - Stop services and remove all data"
            echo "  logs        - Show service logs"
            echo "  status      - Show service status"
            exit 1
            ;;
    esac
    
    # Run checks and setup
    check_docker_version
    check_compose_version
    check_system_requirements
    check_ports
    setup_env_file
    setup_directories
    
    # Deploy services
    pull_images
    start_services
    wait_for_services
    
    # Show results
    show_status
    show_access_info
}

# Run main function with all arguments
main "$@"