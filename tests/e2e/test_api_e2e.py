"""
End-to-end tests for FastAPI server.

Tests the complete API workflow from request to response.
"""
import pytest
import asyncio
import base64
import numpy as np
from httpx import AsyncClient, ASGITransport
from PIL import Image
import io

from robo_vla.server.app import app


@pytest.fixture
def sample_request_data():
    """Create sample request data."""
    # Create a simple test image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)

    # Convert to base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "rgb_image": img_base64,
        "depth_image": img_base64,  # Using same for simplicity
        "text_command": "Pick the red box",
        "robot_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "safety_constraints": {
            "max_velocity": 1.0,
            "max_acceleration": 0.5
        }
    }


@pytest.mark.asyncio
class TestAPIEndpoints:
    """Test API endpoints end-to-end."""

    async def test_health_check(self):
        """Test health check endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data

    async def test_ready_check(self):
        """Test readiness endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data

    async def test_execute_command_endpoint(self, sample_request_data):
        """Test main execute command endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/execute", json=sample_request_data, timeout=30.0)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "joint_commands" in data
        assert "gripper_command" in data
        assert "success_probability" in data
        assert "execution_time" in data
        assert "detected_objects" in data
        assert "reasoning" in data

        # Verify data types and ranges
        assert len(data["joint_commands"]) == 7
        assert 0 <= data["gripper_command"] <= 1
        assert 0 <= data["success_probability"] <= 1
        assert data["execution_time"] > 0

    async def test_execute_command_invalid_input(self):
        """Test execute command with invalid input."""
        invalid_data = {
            "rgb_image": "invalid_base64",
            "depth_image": "invalid_base64",
            "text_command": "",
            "robot_state": [0.0, 0.0]  # Wrong size
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/execute", json=invalid_data, timeout=30.0)

        # Should return error
        assert response.status_code in [400, 422, 500]

    async def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")

        assert response.status_code == 200
        # Check for Prometheus format
        assert "# TYPE" in response.text or "# HELP" in response.text

    async def test_multiple_sequential_requests(self, sample_request_data):
        """Test multiple sequential requests."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            responses = []
            for i in range(5):
                # Modify command for each request
                data = sample_request_data.copy()
                data["text_command"] = f"Pick object {i}"

                response = await client.post("/execute", json=data, timeout=30.0)
                responses.append(response)

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            assert "joint_commands" in response.json()

    async def test_concurrent_requests(self, sample_request_data):
        """Test concurrent request handling."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Send 3 concurrent requests
            tasks = [
                client.post("/execute", json=sample_request_data, timeout=30.0)
                for _ in range(3)
            ]

            responses = await asyncio.gather(*tasks)

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "joint_commands" in data


@pytest.mark.asyncio
class TestWebSocketAPI:
    """Test WebSocket real-time control."""

    async def test_websocket_connection(self):
        """Test WebSocket connection establishment."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.websocket_connect("/realtime_control") as websocket:
                # Connection should be established
                assert websocket is not None

    async def test_websocket_bidirectional_communication(self, sample_request_data):
        """Test bidirectional WebSocket communication."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.websocket_connect("/realtime_control") as websocket:
                # Send request
                await websocket.send_json(sample_request_data)

                # Receive response
                response = await websocket.receive_json()

                # Verify response
                assert "joint_commands" in response
                assert "gripper_command" in response

    async def test_websocket_multiple_messages(self, sample_request_data):
        """Test multiple messages over WebSocket."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.websocket_connect("/realtime_control") as websocket:
                # Send and receive 5 messages
                for i in range(5):
                    data = sample_request_data.copy()
                    data["text_command"] = f"Command {i}"

                    await websocket.send_json(data)
                    response = await websocket.receive_json()

                    assert "joint_commands" in response


@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for API."""

    @pytest.mark.asyncio
    async def test_api_latency(self, sample_request_data):
        """Test API response latency."""
        import time

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Warm up
            for _ in range(3):
                await client.post("/execute", json=sample_request_data, timeout=30.0)

            # Measure latency
            latencies = []
            for _ in range(20):
                start = time.time()
                response = await client.post("/execute", json=sample_request_data, timeout=30.0)
                latency = (time.time() - start) * 1000  # ms
                latencies.append(latency)

                assert response.status_code == 200

            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            print(f"API Latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms")

            # Should meet latency SLAs
            assert avg_latency < 1000, f"Average latency too high: {avg_latency:.2f}ms"
            assert p95_latency < 1500, f"P95 latency too high: {p95_latency:.2f}ms"

    @pytest.mark.asyncio
    async def test_api_throughput(self, sample_request_data):
        """Test API throughput under load."""
        import time

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            num_requests = 50
            start = time.time()

            # Send requests concurrently in batches
            batch_size = 10
            for i in range(0, num_requests, batch_size):
                tasks = [
                    client.post("/execute", json=sample_request_data, timeout=30.0)
                    for _ in range(min(batch_size, num_requests - i))
                ]
                responses = await asyncio.gather(*tasks)

                # Verify all succeeded
                for response in responses:
                    assert response.status_code == 200

            elapsed = time.time() - start
            throughput = num_requests / elapsed

            print(f"API Throughput: {throughput:.2f} requests/second")

            # Should achieve reasonable throughput
            assert throughput > 5, f"Throughput too low: {throughput:.2f} req/s"


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test API with missing required fields."""
        incomplete_data = {
            "text_command": "Pick the object"
            # Missing rgb_image, depth_image, robot_state
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/execute", json=incomplete_data, timeout=30.0)

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_robot_state(self, sample_request_data):
        """Test API with invalid robot state."""
        data = sample_request_data.copy()
        data["robot_state"] = [999.0] * 8  # Invalid joint angles

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/execute", json=data, timeout=30.0)

        # Should handle gracefully
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_empty_command(self, sample_request_data):
        """Test API with empty text command."""
        data = sample_request_data.copy()
        data["text_command"] = ""

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/execute", json=data, timeout=30.0)

        # Should return error or default behavior
        assert response.status_code in [200, 400, 422]


@pytest.mark.integration
class TestFullSystemE2E:
    """Full system end-to-end tests."""

    @pytest.mark.asyncio
    async def test_complete_pick_and_place_workflow(self, sample_request_data):
        """Test complete pick and place workflow."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Step 1: Detect objects
            detect_data = sample_request_data.copy()
            detect_data["text_command"] = "Detect all objects"
            detect_response = await client.post("/execute", json=detect_data, timeout=30.0)
            assert detect_response.status_code == 200

            detected_objects = detect_response.json()["detected_objects"]
            assert len(detected_objects) >= 0

            # Step 2: Pick object
            pick_data = sample_request_data.copy()
            pick_data["text_command"] = "Pick the first object"
            pick_response = await client.post("/execute", json=pick_data, timeout=30.0)
            assert pick_response.status_code == 200

            pick_result = pick_response.json()
            assert "joint_commands" in pick_result
            assert pick_result["gripper_command"] < 0.5  # Gripper should close

            # Step 3: Place object
            place_data = sample_request_data.copy()
            place_data["text_command"] = "Place the object in the bin"
            place_data["robot_state"] = pick_result["joint_commands"] + [pick_result["gripper_command"]]
            place_response = await client.post("/execute", json=place_data, timeout=30.0)
            assert place_response.status_code == 200

            place_result = place_response.json()
            assert place_result["gripper_command"] > 0.5  # Gripper should open
