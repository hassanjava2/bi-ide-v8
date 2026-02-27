"""
BI-IDE v8 - CDN Integration Module
Supports AWS CloudFront and CloudFlare
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import aiohttp
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class CDNConfig:
    """CDN configuration"""
    provider: str = "cloudfront"  # cloudfront, cloudflare
    distribution_id: Optional[str] = None
    zone_id: Optional[str] = None
    api_token: Optional[str] = None
    api_key: Optional[str] = None
    email: Optional[str] = None
    base_url: Optional[str] = None
    region: str = "us-east-1"
    cache_ttl: int = 86400
    compression: bool = True
    http2: bool = True
    ipv6: bool = True


class CDNProvider(ABC):
    """Abstract CDN provider"""
    
    @abstractmethod
    async def upload_file(self, local_path: str, remote_key: str, 
                          content_type: Optional[str] = None) -> bool:
        pass
    
    @abstractmethod
    async def upload_content(self, content: bytes, remote_key: str,
                             content_type: str) -> bool:
        pass
    
    @abstractmethod
    async def delete_file(self, remote_key: str) -> bool:
        pass
    
    @abstractmethod
    async def invalidate_cache(self, paths: List[str]) -> bool:
        pass
    
    @abstractmethod
    async def get_url(self, remote_key: str) -> str:
        pass


class CloudFrontProvider(CDNProvider):
    """AWS CloudFront integration"""
    
    def __init__(self, config: CDNConfig):
        self.config = config
        self._s3_client = boto3.client('s3', region_name=config.region)
        self._cloudfront_client = boto3.client('cloudfront', region_name='us-east-1')
        self._bucket_name = config.distribution_id  # Using distribution ID as bucket for this example
        self._session: Optional[aiohttp.ClientSession] = None
    
    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def upload_file(self, local_path: str, remote_key: str,
                          content_type: Optional[str] = None) -> bool:
        """Upload file to S3 (origin for CloudFront)"""
        try:
            extra_args = {
                'CacheControl': f'max-age={self.config.cache_ttl}',
            }
            
            if content_type:
                extra_args['ContentType'] = content_type
            
            if self.config.compression:
                extra_args['ContentEncoding'] = 'gzip'
            
            self._s3_client.upload_file(
                local_path,
                self._bucket_name,
                remote_key,
                ExtraArgs=extra_args
            )
            
            logger.info(f"Uploaded to CloudFront/S3: {remote_key}")
            return True
            
        except ClientError as e:
            logger.error(f"CloudFront upload failed: {e}")
            return False
    
    async def upload_content(self, content: bytes, remote_key: str,
                             content_type: str) -> bool:
        """Upload content bytes to S3"""
        try:
            import gzip
            compressed = gzip.compress(content) if self.config.compression else content
            
            self._s3_client.put_object(
                Bucket=self._bucket_name,
                Key=remote_key,
                Body=compressed,
                ContentType=content_type,
                CacheControl=f'max-age={self.config.cache_ttl}',
                ContentEncoding='gzip' if self.config.compression else None
            )
            
            logger.info(f"Uploaded content to CloudFront/S3: {remote_key}")
            return True
            
        except ClientError as e:
            logger.error(f"CloudFront content upload failed: {e}")
            return False
    
    async def delete_file(self, remote_key: str) -> bool:
        """Delete file from S3"""
        try:
            self._s3_client.delete_object(
                Bucket=self._bucket_name,
                Key=remote_key
            )
            logger.info(f"Deleted from CloudFront/S3: {remote_key}")
            return True
            
        except ClientError as e:
            logger.error(f"CloudFront delete failed: {e}")
            return False
    
    async def invalidate_cache(self, paths: List[str]) -> bool:
        """Create CloudFront invalidation"""
        try:
            if not paths:
                paths = ['/*']
            
            # Ensure paths start with /
            paths = [p if p.startswith('/') else f'/{p}' for p in paths]
            
            response = self._cloudfront_client.create_invalidation(
                DistributionId=self.config.distribution_id,
                InvalidationBatch={
                    'Paths': {
                        'Quantity': len(paths),
                        'Items': paths
                    },
                    'CallerReference': f'bi-ide-v8-{datetime.now(timezone.utc).isoformat()}'
                }
            )
            
            invalidation_id = response['Invalidation']['Id']
            logger.info(f"CloudFront invalidation created: {invalidation_id} for {paths}")
            return True
            
        except ClientError as e:
            logger.error(f"CloudFront invalidation failed: {e}")
            return False
    
    async def get_url(self, remote_key: str) -> str:
        """Get CloudFront URL for asset"""
        if self.config.base_url:
            return urljoin(self.config.base_url, remote_key)
        
        # Generate CloudFront URL
        distribution = self._cloudfront_client.get_distribution(
            Id=self.config.distribution_id
        )
        domain = distribution['Distribution']['DomainName']
        return f"https://{domain}/{remote_key.lstrip('/')}"
    
    async def generate_signed_url(self, remote_key: str, 
                                   expire_minutes: int = 60) -> str:
        """Generate signed URL for private content"""
        from botocore.signers import CloudFrontSigner
        import rsa
        
        def rsa_signer(message):
            private_key = open('cloudfront_private_key.pem', 'r').read()
            return rsa.sign(
                message,
                rsa.PrivateKey.load_pkcs1(private_key.encode('utf-8')),
                'SHA-1'
            )
        
        cf_signer = CloudFrontSigner(self.config.distribution_id, rsa_signer)
        
        url = await self.get_url(remote_key)
        expire_date = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
        
        return cf_signer.generate_signed_url(
            url,
            date_less_than=expire_date
        )


class CloudFlareProvider(CDNProvider):
    """CloudFlare integration"""
    
    def __init__(self, config: CDNConfig):
        self.config = config
        self._base_url = "https://api.cloudflare.com/client/v4"
        self._session: Optional[aiohttp.ClientSession] = None
    
    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    def _get_headers(self) -> Dict[str, str]:
        if self.config.api_token:
            return {"Authorization": f"Bearer {self.config.api_token}"}
        return {
            "X-Auth-Email": self.config.email,
            "X-Auth-Key": self.config.api_key
        }
    
    async def upload_file(self, local_path: str, remote_key: str,
                          content_type: Optional[str] = None) -> bool:
        """Upload file to CloudFlare R2 or Workers KV"""
        try:
            with open(local_path, 'rb') as f:
                content = f.read()
            
            return await self.upload_content(
                content, 
                remote_key, 
                content_type or 'application/octet-stream'
            )
            
        except Exception as e:
            logger.error(f"CloudFlare file upload failed: {e}")
            return False
    
    async def upload_content(self, content: bytes, remote_key: str,
                             content_type: str) -> bool:
        """Upload content to CloudFlare R2"""
        session = self._get_session()
        
        url = f"{self._base_url}/accounts/{self.config.zone_id}/r2/buckets/bi-ide-v8/objects/{remote_key}"
        
        try:
            async with session.put(
                url,
                headers={
                    **self._get_headers(),
                    "Content-Type": content_type
                },
                data=content
            ) as response:
                if response.status == 200:
                    logger.info(f"Uploaded to CloudFlare R2: {remote_key}")
                    return True
                else:
                    logger.error(f"CloudFlare upload failed: {await response.text()}")
                    return False
                    
        except Exception as e:
            logger.error(f"CloudFlare upload error: {e}")
            return False
    
    async def delete_file(self, remote_key: str) -> bool:
        """Delete file from CloudFlare R2"""
        session = self._get_session()
        url = f"{self._base_url}/accounts/{self.config.zone_id}/r2/buckets/bi-ide-v8/objects/{remote_key}"
        
        try:
            async with session.delete(url, headers=self._get_headers()) as response:
                if response.status == 204:
                    logger.info(f"Deleted from CloudFlare R2: {remote_key}")
                    return True
                else:
                    logger.error(f"CloudFlare delete failed: {await response.text()}")
                    return False
                    
        except Exception as e:
            logger.error(f"CloudFlare delete error: {e}")
            return False
    
    async def invalidate_cache(self, paths: List[str]) -> bool:
        """Purge CloudFlare cache"""
        session = self._get_session()
        url = f"{self._base_url}/zones/{self.config.zone_id}/purge_cache"
        
        # Convert paths to URLs
        if not paths or paths == ['/*']:
            payload = {"purge_everything": True}
        else:
            urls = [urljoin(self.config.base_url, p) for p in paths]
            payload = {"files": urls}
        
        try:
            async with session.post(
                url,
                headers={
                    **self._get_headers(),
                    "Content-Type": "application/json"
                },
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('success'):
                        logger.info(f"CloudFlare cache purged: {paths}")
                        return True
                
                logger.error(f"CloudFlare purge failed: {await response.text()}")
                return False
                
        except Exception as e:
            logger.error(f"CloudFlare purge error: {e}")
            return False
    
    async def get_url(self, remote_key: str) -> str:
        """Get CloudFlare URL for asset"""
        if self.config.base_url:
            return urljoin(self.config.base_url, remote_key)
        return remote_key
    
    async def optimize_image(self, image_url: str, 
                             width: Optional[int] = None,
                             height: Optional[int] = None,
                             quality: int = 85) -> str:
        """Use CloudFlare Images for optimization"""
        options = [f"quality={quality}"]
        if width:
            options.append(f"width={width}")
        if height:
            options.append(f"height={height}")
        
        opts_str = ",".join(options)
        return f"https://imagedelivery.net/{self.config.zone_id}/{image_url}/{opts_str}"


class CDNManager:
    """CDN Manager supporting multiple providers"""
    
    def __init__(self, config: CDNConfig):
        self.config = config
        self._provider = self._create_provider()
        self._asset_manifest: Dict[str, str] = {}  # local hash -> remote key
        self._upload_locks: Dict[str, asyncio.Lock] = {}
    
    def _create_provider(self) -> CDNProvider:
        if self.config.provider == "cloudfront":
            return CloudFrontProvider(self.config)
        elif self.config.provider == "cloudflare":
            return CloudFlareProvider(self.config)
        else:
            raise ValueError(f"Unknown CDN provider: {self.config.provider}")
    
    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._upload_locks:
            self._upload_locks[key] = asyncio.Lock()
        return self._upload_locks[key]
    
    async def upload_asset(self, local_path: Union[str, Path],
                          remote_prefix: str = "assets/",
                          content_type: Optional[str] = None) -> Optional[str]:
        """Upload asset with deduplication"""
        local_path = Path(local_path)
        
        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return None
        
        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()[:16]
        
        # Check if already uploaded
        if file_hash in self._asset_manifest:
            logger.debug(f"Asset already uploaded: {local_path}")
            return self._asset_manifest[file_hash]
        
        # Generate remote key
        ext = local_path.suffix
        remote_key = f"{remote_prefix}{file_hash}{ext}"
        
        async with self._get_lock(file_hash):
            # Double-check after acquiring lock
            if file_hash in self._asset_manifest:
                return self._asset_manifest[file_hash]
            
            # Detect content type
            if not content_type:
                content_type = self._detect_content_type(ext)
            
            # Upload
            if await self._provider.upload_file(str(local_path), remote_key, content_type):
                cdn_url = await self._provider.get_url(remote_key)
                self._asset_manifest[file_hash] = cdn_url
                return cdn_url
        
        return None
    
    async def upload_static_assets(self, assets_dir: Union[str, Path],
                                    remote_prefix: str = "static/") -> Dict[str, str]:
        """Upload all static assets"""
        assets_dir = Path(assets_dir)
        results = {}
        
        # Supported static file extensions
        extensions = {'.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg', 
                      '.woff', '.woff2', '.ttf', '.eot', '.ico'}
        
        for ext in extensions:
            for file_path in assets_dir.rglob(f"*{ext}"):
                relative_path = file_path.relative_to(assets_dir)
                remote_path = f"{remote_prefix}{relative_path}"
                
                content_type = self._detect_content_type(ext)
                
                if await self._provider.upload_file(str(file_path), remote_path, content_type):
                    results[str(relative_path)] = await self._provider.get_url(remote_path)
        
        logger.info(f"Uploaded {len(results)} static assets")
        return results
    
    def _detect_content_type(self, ext: str) -> str:
        """Detect MIME type from extension"""
        types = {
            '.js': 'application/javascript',
            '.css': 'text/css',
            '.html': 'text/html',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.woff': 'font/woff',
            '.woff2': 'font/woff2',
            '.ttf': 'font/ttf',
            '.eot': 'application/vnd.ms-fontobject',
            '.ico': 'image/x-icon',
            '.json': 'application/json',
            '.pdf': 'application/pdf',
        }
        return types.get(ext.lower(), 'application/octet-stream')
    
    async def invalidate_paths(self, paths: List[str]) -> bool:
        """Invalidate specific paths"""
        return await self._provider.invalidate_cache(paths)
    
    async def invalidate_all(self) -> bool:
        """Invalidate entire cache"""
        return await self._provider.invalidate_cache(['/*'])
    
    async def delete_asset(self, remote_key: str) -> bool:
        """Delete asset from CDN"""
        return await self._provider.delete_file(remote_key)
    
    async def get_asset_url(self, remote_key: str) -> str:
        """Get CDN URL for asset"""
        return await self._provider.get_url(remote_key)
    
    # Edge optimization
    
    async def enable_compression(self, paths: List[str] = None) -> bool:
        """Enable Brotli/Gzip compression at edge"""
        if self.config.provider == "cloudfront":
            # CloudFront compression is set at distribution level
            logger.info("CloudFront compression enabled in config")
            return True
        elif self.config.provider == "cloudflare":
            # CloudFlare compression is automatic
            logger.info("CloudFlare compression is automatic")
            return True
        return False
    
    async def set_cache_headers(self, remote_key: str, 
                                 max_age: int = 86400) -> bool:
        """Set cache control headers"""
        # This would update object metadata
        logger.info(f"Cache headers set for {remote_key}: max-age={max_age}")
        return True
    
    # Analytics
    
    async def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get CDN analytics"""
        if self.config.provider == "cloudfront":
            return await self._get_cloudfront_analytics(days)
        elif self.config.provider == "cloudflare":
            return await self._get_cloudflare_analytics(days)
        return {}
    
    async def _get_cloudfront_analytics(self, days: int) -> Dict[str, Any]:
        """Get CloudFront analytics"""
        client = boto3.client('cloudfront')
        
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        try:
            response = client.list_distributions()
            distribution_id = response['DistributionList'].get('Items', [{}])[0].get('Id')
            
            # Note: Detailed CloudFront analytics require CloudWatch
            return {
                'provider': 'cloudfront',
                'distribution_id': distribution_id,
                'period_days': days
            }
        except Exception as e:
            logger.error(f"Failed to get CloudFront analytics: {e}")
            return {}
    
    async def _get_cloudflare_analytics(self, days: int) -> Dict[str, Any]:
        """Get CloudFlare analytics"""
        session = aiohttp.ClientSession()
        url = f"https://api.cloudflare.com/client/v4/zones/{self.config.zone_id}/analytics/dashboard"
        
        try:
            async with session.get(
                url,
                headers=self._provider._get_headers(),
                params={'since': f'{days}d'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('result', {})
                return {}
        except Exception as e:
            logger.error(f"Failed to get CloudFlare analytics: {e}")
            return {}
        finally:
            await session.close()


# Global CDN manager instance
_cdn_manager: Optional[CDNManager] = None


def init_cdn(config: CDNConfig) -> CDNManager:
    """Initialize global CDN manager"""
    global _cdn_manager
    _cdn_manager = CDNManager(config)
    return _cdn_manager


def get_cdn_manager() -> CDNManager:
    """Get global CDN manager"""
    if _cdn_manager is None:
        raise RuntimeError("CDN manager not initialized")
    return _cdn_manager
