"""
Tests for new routers (RTX5090, Network, Brain)
"""

import pytest


class TestRTX5090Router:
    """Testing RTX 5090 management endpoints"""

    def test_rtx_health(self, client):
        """RTX health check should return status (online/offline)"""
        response = client.get("/api/v1/rtx5090/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] in ("online", "offline")
        assert "rtx_url" in data

    def test_rtx_status(self, client):
        """RTX status should return detailed info or offline"""
        response = client.get("/api/v1/rtx5090/status")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data

    def test_rtx_models(self, client):
        """RTX models list should return array or offline"""
        response = client.get("/api/v1/rtx5090/models")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data

    def test_rtx_config(self, client):
        """RTX config should show connection details"""
        response = client.get("/api/v1/rtx5090/config")
        assert response.status_code == 200

        data = response.json()
        assert "host" in data
        assert "port" in data
        assert "base_url" in data
        assert isinstance(data["port"], int)


class TestNetworkRouter:
    """Testing network management endpoints"""

    def test_network_full_status(self, client):
        """Network status should return service overview"""
        response = client.get("/api/v1/network/status")
        assert response.status_code == 200

        data = response.json()
        assert "overall_status" in data
        assert data["overall_status"] in ("healthy", "degraded", "critical")
        assert "services" in data
        assert "total_services" in data
        assert "services_up" in data

    def test_network_ping_known_service(self, client):
        """Pinging a known service should return result"""
        response = client.get("/api/v1/network/ping/rtx5090")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert data["service"] == "rtx5090"
        assert "status" in data

    def test_network_ping_unknown_service(self, client):
        """Pinging an unknown service should list available ones"""
        response = client.get("/api/v1/network/ping/nonexistent")
        assert response.status_code == 200

        data = response.json()
        assert "error" in data
        assert "available_services" in data
        assert isinstance(data["available_services"], list)

    def test_network_topology(self, client):
        """Network topology should return architecture info"""
        response = client.get("/api/v1/network/topology")
        assert response.status_code == 200

        data = response.json()
        assert "architecture" in data
        assert "nodes" in data
        assert "connections" in data
        assert "failover_chain" in data


class TestBrainRouter:
    """Testing brain management endpoints"""

    def test_brain_status(self, client):
        """Brain status should return running state"""
        response = client.get("/api/v1/brain/status")
        assert response.status_code == 200

        data = response.json()
        assert "is_running" in data
        assert "scheduler" in data
        assert "config" in data

    def test_brain_jobs_list(self, client):
        """Brain jobs should return list"""
        response = client.get("/api/v1/brain/jobs")
        assert response.status_code == 200

        data = response.json()
        assert "jobs" in data
        assert "count" in data
        assert isinstance(data["jobs"], list)

    def test_brain_evaluations_list(self, client):
        """Brain evaluations should return list"""
        response = client.get("/api/v1/brain/evaluations")
        assert response.status_code == 200

        data = response.json()
        assert "evaluations" in data
        assert "count" in data
        assert isinstance(data["evaluations"], list)
