#!/bin/bash
# BI-IDE v8 Disaster Recovery Script
# Restores from backups with point-in-time recovery capability

set -euo pipefail

# Configuration
BACKUP_DIR="/backups/bi-ide-v8"
S3_BUCKET="s3://bi-ide-v8-backups"
RESTORE_DIR="/tmp/restore_${TIMESTAMP}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"; }
error() { echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"; }
info() { echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"; }

# Usage
usage() {
    cat << EOF
BI-IDE v8 Disaster Recovery Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    list              List available backups
    restore-db        Restore PostgreSQL database
    restore-redis     Restore Redis
    restore-uploads   Restore uploaded files
    restore-full      Full system restore
    pitr              Point-in-time recovery

Options:
    -b, --backup      Specific backup file to restore
    -t, --time        Point-in-time for PITR (YYYY-MM-DD HH:MM:SS)
    -f, --force       Skip confirmation prompts
    -h, --help        Show this help message

Examples:
    $0 list
    $0 restore-db --backup bi_ide_v8_20240223_120000.sql.gz
    $0 pitr --time "2024-02-23 14:30:00"
    $0 restore-full --force

EOF
    exit 0
}

# List available backups
list_backups() {
    info "Available PostgreSQL backups:"
    aws s3 ls "${S3_BUCKET}/postgres/" | tail -20
    
    info "\nAvailable Redis backups:"
    aws s3 ls "${S3_BUCKET}/redis/" | tail -10
    
    info "\nAvailable Upload backups:"
    aws s3 ls "${S3_BUCKET}/uploads/" | tail -10
}

# Download backup from S3
download_backup() {
    local backup_type=$1
    local backup_file=$2
    
    mkdir -p "${RESTORE_DIR}"
    
    log "Downloading ${backup_file} from S3..."
    aws s3 cp "${S3_BUCKET}/${backup_type}/${backup_file}" "${RESTORE_DIR}/"
    
    # Download checksum
    if aws s3 ls "${S3_BUCKET}/${backup_type}/${backup_file}.sha256" > /dev/null 2>&1; then
        aws s3 cp "${S3_BUCKET}/${backup_type}/${backup_file}.sha256" "${RESTORE_DIR}/"
        
        # Verify checksum
        log "Verifying backup integrity..."
        if ! (cd "${RESTORE_DIR}" && sha256sum -c "${backup_file}.sha256"); then
            error "Backup verification failed!"
            exit 1
        fi
        log "Backup verification passed!"
    fi
    
    echo "${RESTORE_DIR}/${backup_file}"
}

# Restore PostgreSQL
restore_postgres() {
    local backup_file=${1:-""}
    
    if [ -z "${backup_file}" ]; then
        # Get latest backup
        backup_file=$(aws s3 ls "${S3_BUCKET}/postgres/" | grep '\.sql\.gz$' | tail -1 | awk '{print $4}')
        if [ -z "${backup_file}" ]; then
            error "No PostgreSQL backups found!"
            exit 1
        fi
    fi
    
    warn "WARNING: This will DESTROY the current database and restore from backup!"
    if [ "${FORCE:-false}" != "true" ]; then
        read -p "Are you sure? (yes/no): " confirm
        if [ "${confirm}" != "yes" ]; then
            log "Restore cancelled."
            exit 0
        fi
    fi
    
    local local_backup=$(download_backup "postgres" "${backup_file}")
    
    log "Stopping application services..."
    kubectl scale deployment bi-ide-api -n bi-ide-v8 --replicas=0
    kubectl scale deployment bi-ide-worker -n bi-ide-v8 --replicas=0
    
    log "Creating pre-restore backup..."
    backup_postgres_pre="${BACKUP_DIR}/postgres/pre_restore_${TIMESTAMP}.sql.gz"
    PGPASSWORD="${DB_PASSWORD}" pg_dump -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" | gzip > "${backup_postgres_pre}"
    
    log "Dropping existing database..."
    PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -U "${DB_USER}" -d postgres -c "DROP DATABASE IF EXISTS ${DB_NAME};"
    
    log "Creating new database..."
    PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -U "${DB_USER}" -d postgres -c "CREATE DATABASE ${DB_NAME};"
    
    log "Restoring from backup..."
    gunzip -c "${local_backup}" | PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}"
    
    if [ $? -eq 0 ]; then
        log "PostgreSQL restore completed successfully!"
    else
        error "PostgreSQL restore failed!"
        exit 1
    fi
    
    log "Restarting application services..."
    kubectl scale deployment bi-ide-api -n bi-ide-v8 --replicas=3
    kubectl scale deployment bi-ide-worker -n bi-ide-v8 --replicas=2
    
    # Cleanup
    rm -rf "${RESTORE_DIR}"
}

# Point-in-Time Recovery
restore_pitr() {
    local target_time=$1
    
    if [ -z "${target_time}" ]; then
        error "Target time required for PITR!"
        usage
    fi
    
    warn "WARNING: PITR will restore database to ${target_time}"
    warn "All data after this time will be LOST!"
    
    if [ "${FORCE:-false}" != "true" ]; then
        read -p "Are you absolutely sure? (yes/no): " confirm
        if [ "${confirm}" != "yes" ]; then
            log "PITR cancelled."
            exit 0
        fi
    fi
    
    # Find base backup before target time
    local base_backup=$(aws s3 ls "${S3_BUCKET}/postgres/" | grep '\.sql\.gz$' | \
        awk -v target="${target_time}" '
        {
            # Parse filename timestamp (bi_ide_v8_YYYYMMDD_HHMMSS.sql.gz)
            match($4, /([0-9]{8})_([0-9]{6})/, arr);
            ts = arr[1] " " arr[2];
            gsub(/_/, " ", ts);
            if (ts <= target) print $4;
        }' | tail -1)
    
    if [ -z "${base_backup}" ]; then
        error "No suitable base backup found for PITR!"
        exit 1
    fi
    
    log "Using base backup: ${base_backup}"
    
    # Download and restore base backup
    local local_backup=$(download_backup "postgres" "${base_backup}")
    
    log "Stopping PostgreSQL..."
    kubectl exec -it postgres-0 -n bi-ide-v8 -- pg_ctl stop -D /var/lib/postgresql/data
    
    log "Clearing data directory..."
    kubectl exec -it postgres-0 -n bi-ide-v8 -- rm -rf /var/lib/postgresql/data/*
    
    log "Restoring base backup..."
    kubectl cp "${local_backup}" bi-ide-v8/postgres-0:/tmp/base_backup.sql.gz
    kubectl exec -it postgres-0 -n bi-ide-v8 -- bash -c "
        gunzip -c /tmp/base_backup.sql.gz | psql -U postgres
    "
    
    log "Configuring recovery..."
    kubectl exec -it postgres-0 -n bi-ide-v8 -- bash -c "cat > /var/lib/postgresql/data/recovery.conf << EOF
restore_command = 'aws s3 cp ${S3_BUCKET}/wal/%f %p'
recovery_target_time = '${target_time}'
recovery_target_inclusive = true
EOF"
    
    log "Starting PostgreSQL with recovery..."
    kubectl exec -it postgres-0 -n bi-ide-v8 -- pg_ctl start -D /var/lib/postgresql/data
    
    # Wait for recovery to complete
    log "Waiting for PITR recovery to complete..."
    while kubectl exec -it postgres-0 -n bi-ide-v8 -- psql -U postgres -c "SELECT pg_is_in_recovery();" | grep -q "t"; do
        sleep 5
        echo -n "."
    done
    echo ""
    
    log "PITR completed successfully! Database restored to ${target_time}"
    
    # Cleanup
    rm -rf "${RESTORE_DIR}"
}

# Restore Redis
restore_redis() {
    local backup_file=${1:-""}
    
    if [ -z "${backup_file}" ]; then
        backup_file=$(aws s3 ls "${S3_BUCKET}/redis/" | grep '\.rdb\.gz$' | tail -1 | awk '{print $4}')
    fi
    
    local local_backup=$(download_backup "redis" "${backup_file}")
    
    log "Stopping Redis..."
    redis-cli -h "${REDIS_HOST}" -a "${REDIS_PASSWORD}" SHUTDOWN SAVE
    
    log "Restoring Redis data..."
    gunzip -c "${local_backup}" > /data/redis/dump.rdb
    
    log "Starting Redis..."
    redis-server /etc/redis/redis.conf
    
    log "Redis restore completed!"
}

# Restore uploads
restore_uploads() {
    local backup_file=${1:-""}
    
    if [ -z "${backup_file}" ]; then
        backup_file=$(aws s3 ls "${S3_BUCKET}/uploads/" | grep '\.tar\.gz$' | tail -1 | awk '{print $4}')
    fi
    
    local local_backup=$(download_backup "uploads" "${backup_file}")
    
    warn "This will overwrite current uploads!"
    if [ "${FORCE:-false}" != "true" ]; then
        read -p "Continue? (yes/no): " confirm
        if [ "${confirm}" != "yes" ]; then
            exit 0
        fi
    fi
    
    log "Restoring uploads..."
    tar -xzf "${local_backup}" -C /data/uploads
    
    log "Uploads restore completed!"
}

# Full system restore
restore_full() {
    warn "FULL SYSTEM RESTORE - ALL DATA WILL BE REPLACED!"
    
    if [ "${FORCE:-false}" != "true" ]; then
        read -p "Type 'RESTORE' to confirm: " confirm
        if [ "${confirm}" != "RESTORE" ]; then
            log "Full restore cancelled."
            exit 0
        fi
    fi
    
    log "Starting full system restore..."
    
    # Stop all services
    log "Stopping all services..."
    kubectl scale deployment bi-ide-api bi-ide-ui bi-ide-worker -n bi-ide-v8 --replicas=0
    
    # Restore components
    restore_postgres
    restore_redis
    restore_uploads
    
    # Start services
    log "Starting all services..."
    kubectl scale deployment bi-ide-api -n bi-ide-v8 --replicas=3
    kubectl scale deployment bi-ide-ui -n bi-ide-v8 --replicas=3
    kubectl scale deployment bi-ide-worker -n bi-ide-v8 --replicas=2
    
    log "Full system restore completed!"
}

# Parse arguments
BACKUP_FILE=""
TARGET_TIME=""
FORCE=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        list|restore-db|restore-redis|restore-uploads|restore-full|pitr)
            COMMAND="$1"
            shift
            ;;
        -b|--backup)
            BACKUP_FILE="$2"
            shift 2
            ;;
        -t|--time)
            TARGET_TIME="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            usage
            ;;
    esac
done

# Execute command
case "${COMMAND}" in
    list)
        list_backups
        ;;
    restore-db)
        restore_postgres "${BACKUP_FILE}"
        ;;
    restore-redis)
        restore_redis "${BACKUP_FILE}"
        ;;
    restore-uploads)
        restore_uploads "${BACKUP_FILE}"
        ;;
    restore-full)
        restore_full
        ;;
    pitr)
        restore_pitr "${TARGET_TIME}"
        ;;
    *)
        usage
        ;;
esac
