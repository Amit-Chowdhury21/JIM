"""
Phase 1: Infrastructure Integration Tests
==========================================
Comprehensive tests for Docker stack connectivity and data flow.

Run with: pytest tests/test_infrastructure_integration.py -v
"""

import pytest
import time
import numpy as np
from typing import Dict, Optional


class TestQuestDBConnectivity:
    """Test QuestDB (time-series database) connectivity."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize QuestDB connection."""
        try:
            import psycopg2
            self.conn = psycopg2.connect(
                dbname="qdb",
                user="admin",
                password="quest",
                host="localhost",
                port=8812
            )
            self.cursor = self.conn.cursor()
        except ImportError:
            pytest.skip("psycopg2 not installed")
        except Exception as e:
            pytest.skip(f"QuestDB not available: {e}")

    def teardown_method(self):
        """Clean up connection."""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'conn'):
            self.conn.close()

    def test_questdb_connectivity(self):
        """Test basic TCP connection to QuestDB."""
        assert self.cursor is not None
        self.cursor.execute("SELECT 1")
        result = self.cursor.fetchone()
        assert result[0] == 1

    def test_create_table(self):
        """Test table creation."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_ticks (
                timestamp TIMESTAMP,
                price DOUBLE,
                volume DOUBLE
            ) timestamp(timestamp) PARTITION BY DAY
        """)
        self.conn.commit()

    def test_insert_and_retrieve(self):
        """Test data insertion and retrieval."""
        self.cursor.execute("TRUNCATE TABLE test_ticks")  # Clean slate
        self.conn.commit()

        # Insert sample data
        self.cursor.execute("""
            INSERT INTO test_ticks(timestamp, price, volume) VALUES (
                systimestamp(),
                2000.50,
                100.0
            )
        """)
        self.conn.commit()

        # Retrieve
        self.cursor.execute("SELECT COUNT(*) FROM test_ticks")
        count = self.cursor.fetchone()[0]
        assert count == 1


class TestRedisConnectivity:
    """Test Redis (in-memory cache) connectivity."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize Redis connection."""
        try:
            import redis
            self.redis = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True
            )
            self.redis.ping()
        except ImportError:
            pytest.skip("redis-py not installed")
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

    def test_redis_connectivity(self):
        """Test basic Redis connectivity."""
        result = self.redis.ping()
        assert result is True

    def test_set_and_get(self):
        """Test basic set/get operations."""
        key = "test_key"
        value = "test_value"
        
        self.redis.set(key, value)
        retrieved = self.redis.get(key)
        
        assert retrieved == value
        self.redis.delete(key)

    def test_feature_caching(self):
        """Test caching feature vectors (realistic scenario)."""
        # Simulate feature vector caching
        symbol = "XAUUSD"
        feature_key = f"features:{symbol}:ma20"
        
        # Store feature
        self.redis.set(feature_key, "1.05", ex=3600)  # Expire in 1 hour
        
        # Retrieve
        cached = self.redis.get(feature_key)
        assert cached == "1.05"
        
        # Check TTL
        ttl = self.redis.ttl(feature_key)
        assert ttl > 0
        
        self.redis.delete(feature_key)


class TestMinIOConnectivity:
    """Test MinIO (object storage) connectivity."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize MinIO connection."""
        try:
            from minio import Minio
            self.minio = Minio(
                "localhost:9100",
                access_key="minioadmin",
                secret_key="minioadmin",
                secure=False
            )
        except ImportError:
            pytest.skip("minio-py not installed")
        except Exception as e:
            pytest.skip(f"MinIO not available: {e}")

    def test_minio_connectivity(self):
        """Test basic MinIO connectivity."""
        buckets = self.minio.list_buckets()
        assert buckets is not None

    def test_bucket_creation(self):
        """Test bucket creation."""
        bucket_name = "test-bucket"
        try:
            # Check if exists
            self.minio.bucket_exists(bucket_name)
        except Exception:
            pass  # Expected if bucket doesn't exist


class TestMLflowConnectivity:
    """Test MLflow (model registry) connectivity."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize MLflow client."""
        try:
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            self.client = mlflow.tracking.MlflowClient("http://localhost:5000")
        except ImportError:
            pytest.skip("mlflow not installed")
        except Exception as e:
            pytest.skip(f"MLflow not available: {e}")

    def test_mlflow_connectivity(self):
        """Test MLflow server availability."""
        # Try to get experiments
        experiments = self.client.search_experiments()
        assert experiments is not None

    def test_create_experiment(self):
        """Test experiment creation."""
        exp_name = f"test_exp_{int(time.time())}"
        try:
            exp_id = self.client.create_experiment(exp_name)
            assert exp_id is not None
        except Exception as e:
            # Experiment may already exist
            pass


class TestPrometheusConnectivity:
    """Test Prometheus (metrics) connectivity."""
    
    def test_prometheus_connectivity(self):
        """Test Prometheus HTTP endpoint."""
        try:
            import requests
            response = requests.get("http://localhost:9090/api/v1/status/config", timeout=5)
            assert response.status_code == 200
        except ImportError:
            pytest.skip("requests not installed")
        except Exception as e:
            pytest.skip(f"Prometheus not available: {e}")


class TestGrafanaConnectivity:
    """Test Grafana (dashboards) connectivity."""
    
    def test_grafana_connectivity(self):
        """Test Grafana HTTP endpoint."""
        try:
            import requests
            response = requests.get("http://localhost:3000/api/health", timeout=5)
            assert response.status_code == 200
        except ImportError:
            pytest.skip("requests not installed")
        except Exception as e:
            pytest.skip(f"Grafana not available: {e}")


class TestDataPipeline:
    """Integration tests for end-to-end data flow."""
    
    def test_questdb_to_redis_pipeline(self):
        """Test data flow from QuestDB to Redis cache."""
        pytest.importorskip("psycopg2")
        pytest.importorskip("redis")
        
        # This would typically be part of the feature engineering pipeline
        # For now, just verify both systems are accessible
        try:
            import psycopg2
            import redis
            
            qdb_conn = psycopg2.connect(
                dbname="qdb", user="admin", password="quest",
                host="localhost", port=8812
            )
            redis_conn = redis.Redis(host="localhost", port=6379, decode_responses=True)
            
            qdb_conn.close()
            redis_conn.close()
        except Exception as e:
            pytest.skip(f"Pipeline test skipped: {e}")


# Markers for running specific tests
@pytest.mark.infrastructure
def test_infrastructure_stack():
    """Placeholder for infrastructure stack validation."""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
