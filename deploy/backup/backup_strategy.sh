#!/bin/bash
# BI-IDE v8 Automated Backup Strategy
# Production-ready backup automation with multiple strategies

set -euo pipefail

# Configuration
BACKUP_DIR="/backups/bi-ide-v8"
S3_BUCKET="s3://bi-ide-v8-backups"
RETENTION_DAYS=30
RETENTION_WEEKS=12
RETENTION_MONTHS=12
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y-%m-%d)
DAY_OF_WEEK=$(date +%u)
DAY_OF_MONTH=$(date +%d)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Create backup directories
init() {
    log "Initializing backup directories..."
    mkdir -p "${BACKUP_DIR}"/{daily,weekly,monthly,postgres,redis,uploads,configs}
    mkdir -p "${BACKUP_DIR}/logs"
}

# PostgreSQL Backup
backup_postgres() {
    log "Starting PostgreSQL backup..."
    local backup_file="${BACKUP_DIR}/postgres/bi_ide_v8_${TIMESTAMP}.sql.gz"
    
    # Full database dump with compression
    PGPASSWORD="${DB_PASSWORD}" pg_dump \
        -h "${DB_HOST:-postgres}" \
        -U "${DB_USER:-bi_ide}" \
        -d "${DB_NAME:-bi_ide_v8}" \
        --verbose \
        --no-owner \
        --no-acl \
        --format=custom \
        | gzip > "${backup_file}"
    
    if [ $? -eq 0 ]; then
        log "PostgreSQL backup completed: ${backup_file}"
        # Calculate checksum
        sha256sum "${backup_file}" > "${backup_file}.sha256"
    else
        error "PostgreSQL backup failed!"
        return 1
    fi
    
    # Backup to S3
    aws s3 cp "${backup_file}" "${S3_BUCKET}/postgres/" --storage-class STANDARD_IA
    aws s3 cp "${backup_file}.sha256" "${S3_BUCKET}/postgres/"
}

# Redis Backup
backup_redis() {
    log "Starting Redis backup..."
    local backup_file="${BACKUP_DIR}/redis/redis_${TIMESTAMP}.rdb"
    
    # Trigger Redis BGSAVE
    redis-cli -h "${REDIS_HOST:-redis}" -a "${REDIS_PASSWORD:-}" BGSAVE
    
    # Wait for save to complete
    sleep 5
    while redis-cli -h "${REDIS_HOST:-redis}" -a "${REDIS_PASSWORD:-}" LASTSAVE | grep -q "bgsave_in_progress:1"; do
        sleep 1
    done
    
    # Copy RDB file
    redis-cli -h "${REDIS_HOST:-redis}" -a "${REDIS_PASSWORD:-}" --rdb "${backup_file}"
    
    gzip "${backup_file}"
    
    if [ $? -eq 0 ]; then
        log "Redis backup completed: ${backup_file}.gz"
        sha256sum "${backup_file}.gz" > "${backup_file}.gz.sha256"
    else
        error "Redis backup failed!"
        return 1
    fi
    
    aws s3 cp "${backup_file}.gz" "${S3_BUCKET}/redis/" --storage-class STANDARD_IA
}

# Application Data Backup
backup_uploads() {
    log "Starting uploads backup..."
    local backup_file="${BACKUP_DIR}/uploads/uploads_${TIMESTAMP}.tar.gz"
    
    tar -czf "${backup_file}" -C /data/uploads .
    
    if [ $? -eq 0 ]; then
        log "Uploads backup completed: ${backup_file}"
        sha256sum "${backup_file}" > "${backup_file}.sha256"
    else
        error "Uploads backup failed!"
        return 1
    fi
    
    aws s3 cp "${backup_file}" "${S3_BUCKET}/uploads/" --storage-class STANDARD_IA
}

# Kubernetes Configs Backup
backup_configs() {
    log "Starting Kubernetes configs backup..."
    local backup_file="${BACKUP_DIR}/configs/k8s_${TIMESTAMP}.tar.gz"
    
    # Export all K8s resources
    kubectl get all -n bi-ide-v8 -o yaml > "${BACKUP_DIR}/configs/resources_${TIMESTAMP}.yaml"
    kubectl get configmaps -n bi-ide-v8 -o yaml > "${BACKUP_DIR}/configs/configmaps_${TIMESTAMP}.yaml"
    kubectl get secrets -n bi-ide-v8 -o yaml > "${BACKUP_DIR}/configs/secrets_${TIMESTAMP}.yaml" 2>/dev/null || true
    kubectl get ingress -n bi-ide-v8 -o yaml > "${BACKUP_DIR}/configs/ingress_${TIMESTAMP}.yaml"
    
    tar -czf "${backup_file}" -C "${BACKUP_DIR}/configs" .
    
    # Clean up temporary files
    rm -f "${BACKUP_DIR}/configs"/*_${TIMESTAMP}.yaml
    
    log "Kubernetes configs backup completed: ${backup_file}"
    aws s3 cp "${backup_file}" "${S3_BUCKET}/configs/"
}

# Point-in-Time Recovery Setup
setup_pitr() {
    log "Configuring PostgreSQL PITR (Point-in-Time Recovery)..."
    
    # Ensure WAL archiving is enabled
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST:-postgres}" \
        -U "${DB_USER:-bi_ide}" \
        -d "${DB_NAME:-bi_ide_v8}" \
        -c "SELECT pg_switch_wal();" 2>/dev/null || true
    
    # Archive WAL files to S3
    aws s3 sync /var/lib/postgresql/data/pg_wal/ "${S3_BUCKET}/wal/" --storage-class GLACIER
}

# Retention Policy Cleanup
cleanup_old_backups() {
    log "Running retention policy cleanup..."
    
    # Daily backups - keep 30 days
    find "${BACKUP_DIR}/postgres" -name "*.sql.gz" -mtime +${RETENTION_DAYS} -delete
    find "${BACKUP_DIR}/redis" -name "*.rdb.gz" -mtime +${RETENTION_DAYS} -delete
    find "${BACKUP_DIR}/uploads" -name "*.tar.gz" -mtime +${RETENTION_DAYS} -delete
    
    # S3 lifecycle management
    aws s3api put-bucket-lifecycle-configuration \
        --bucket bi-ide-v8-backups \
        --lifecycle-configuration file://deploy/backup/lifecycle.json || true
}

# Backup verification
verify_backup() {
    log "Verifying latest backup..."
    local latest_backup=$(ls -t ${BACKUP_DIR}/postgres/*.sql.gz 2>/dev/null | head -1)
    
    if [ -n "${latest_backup}" ]; then
        # Verify checksum
        if sha256sum -c "${latest_backup}.sha256"; then
            log "Backup verification passed: ${latest_backup}"
            
            # Test restore to temporary database
            log "Testing restore to temporary database..."
            createdb -h "${DB_HOST:-postgres}" -U "${DB_USER:-bi_ide}" "test_restore_${TIMESTAMP}" 2>/dev/null || true
            
            if gunzip -c "${latest_backup}" | psql -h "${DB_HOST:-postgres}" -U "${DB_USER:-bi_ide}" "test_restore_${TIMESTAMP}" -q; then
                log "Restore test passed!"
                dropdb -h "${DB_HOST:-postgres}" -U "${DB_USER:-bi_ide}" "test_restore_${TIMESTAMP}" 2>/dev/null || true
            else
                error "Restore test failed!"
            fi
        else
            error "Backup verification failed: ${latest_backup}"
        fi
    fi
}

# Send notification
notify() {
    local status=$1
    local message=$2
    
    # Slack notification
    if [ -n "${SLACK_WEBHOOK:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"BI-IDE v8 Backup ${status}: ${message}\"}" \
            "${SLACK_WEBHOOK}"
    fi
    
    # Email notification
    if [ -n "${EMAIL_TO:-}" ]; then
        echo "${message}" | mail -s "BI-IDE v8 Backup ${status}" "${EMAIL_TO}"
    fi
}

# Main backup execution
main() {
    log "Starting BI-IDE v8 backup process..."
    
    init
    
    # Determine backup type based on schedule
    local backup_type="daily"
    if [ "${DAY_OF_WEEK}" == "7" ]; then
        backup_type="weekly"
    fi
    if [ "${DAY_OF_MONTH}" == "01" ]; then
        backup_type="monthly"
    fi
    
    log "Running ${backup_type} backup..."
    
    # Execute backups
    if backup_postgres && backup_redis && backup_uploads && backup_configs; then
        log "All backups completed successfully!"
        setup_pitr
        cleanup_old_backups
        verify_backup
        notify "SUCCESS" "${backup_type} backup completed successfully at ${TIMESTAMP}"
    else
        error "Backup process failed!"
        notify "FAILED" "Backup failed at ${TIMESTAMP}"
        exit 1
    fi
    
    log "Backup process finished."
}

# Execute main function
main "$@"
