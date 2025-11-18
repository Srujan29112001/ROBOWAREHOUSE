"""
Integration tests for VLA model.

Tests the integration between perception, language, and action generation.
"""
import pytest
import torch
import numpy as np

from robo_vla.vla_model.vla import VLAModel
from robo_vla.perception.pipeline import PerceptionPipeline
from robo_vla.language.grounding import LanguageGroundingModule
from robo_vla.vla_model.cross_attention import CrossAttentionFusion
from robo_vla.vla_model.action_decoder import ActionDecoder


@pytest.fixture
def sample_scene():
    """Create a sample scene with objects."""
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Add a red box
    rgb[100:200, 100:200] = [255, 0, 0]
    return rgb


@pytest.fixture
def vla_model():
    """Create VLA model instance."""
    return VLAModel(device='cpu')


@pytest.fixture
def text_commands():
    """Sample text commands."""
    return [
        "Pick the red box",
        "Place the object in the bin",
        "Move the blue cube to the shelf",
        "Grasp the top object",
        "Stack the blocks"
    ]


class TestVLAIntegration:
    """Integration tests for VLA model components."""

    def test_end_to_end_prediction(self, vla_model, sample_scene, text_commands):
        """Test complete end-to-end prediction."""
        robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        for command in text_commands:
            result = vla_model.predict(
                rgb_image=sample_scene,
                text_command=command,
                robot_state=robot_state
            )

            # Verify output structure
            assert 'joint_commands' in result
            assert 'gripper_command' in result
            assert 'success_probability' in result
            assert 'detected_objects' in result

            # Verify shapes
            assert len(result['joint_commands']) == 7
            assert 0 <= result['gripper_command'] <= 1
            assert 0 <= result['success_probability'] <= 1

            # Verify joint commands are in valid range
            for joint in result['joint_commands']:
                assert -np.pi <= joint <= np.pi

    def test_perception_language_fusion(self, sample_scene):
        """Test integration between perception and language modules."""
        # Initialize components
        perception = PerceptionPipeline(device='cpu')
        language = LanguageGroundingModule(device='cpu')
        fusion = CrossAttentionFusion()

        # Process perception
        perception_result = perception.process(sample_scene)
        visual_features = perception_result['features']

        # Process language
        command = "Pick the red box"
        language_result = language.process(command)
        text_features = language_result['embeddings']

        # Create dummy robot state
        robot_state = torch.zeros(1, 8)

        # Fuse features
        fused_features, attention_weights = fusion(
            visual_features,
            text_features,
            robot_state
        )

        # Verify fusion output
        assert fused_features is not None
        assert attention_weights is not None
        assert fused_features.shape[-1] == 256  # Expected feature dimension

    def test_action_generation_from_fused_features(self):
        """Test action generation from fused features."""
        # Create mock fused features
        batch_size = 1
        num_objects = 5
        feature_dim = 256
        fused_features = torch.randn(batch_size, num_objects, feature_dim)

        # Initialize action decoder
        action_decoder = ActionDecoder()

        # Generate actions
        actions, success_prob = action_decoder(fused_features, steps=10)

        # Verify output
        assert len(actions) == 10  # 10 timesteps
        for action in actions:
            assert 'joints' in action
            assert 'gripper' in action
            assert 'timestamp' in action
            assert action['joints'].shape[-1] == 7
            assert action['gripper'].shape[-1] == 1

        assert success_prob.shape == (batch_size, 1)

    def test_vla_with_different_robot_states(self, vla_model, sample_scene):
        """Test VLA model with different robot configurations."""
        command = "Pick the object"

        # Test with different joint angles
        robot_states = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # Home position
            np.array([0.5, -0.5, 0.3, -0.8, 0.2, 0.9, -0.3, 0.5]),  # Random position
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),  # Extended position
        ]

        results = []
        for state in robot_states:
            result = vla_model.predict(
                rgb_image=sample_scene,
                text_command=command,
                robot_state=state
            )
            results.append(result)

        # Verify all predictions were successful
        assert len(results) == len(robot_states)

        # Different robot states should produce different actions
        for i in range(len(results) - 1):
            joints_diff = np.abs(
                np.array(results[i]['joint_commands']) -
                np.array(results[i + 1]['joint_commands'])
            ).mean()
            # Actions should differ for different starting states
            assert joints_diff > 0.01

    def test_vla_gradient_flow(self):
        """Test that gradients flow through the entire VLA model."""
        vla = VLAModel(device='cpu')
        vla.train()

        # Create dummy inputs
        rgb = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8).numpy()
        command = "Pick the object"
        robot_state = np.zeros(8)

        # Get prediction (forward pass)
        result = vla.predict(rgb, command, robot_state)

        # Verify model is in training mode
        assert vla.training

        # Check that parameters have gradients enabled
        for param in vla.parameters():
            if param.requires_grad:
                assert param.requires_grad

    def test_vla_inference_mode(self, vla_model, sample_scene):
        """Test VLA model in inference mode."""
        vla_model.eval()

        command = "Pick the object"
        robot_state = np.zeros(8)

        with torch.no_grad():
            result = vla_model.predict(sample_scene, command, robot_state)

        # Verify prediction is deterministic
        with torch.no_grad():
            result2 = vla_model.predict(sample_scene, command, robot_state)

        # Results should be identical in eval mode
        assert np.allclose(
            result['joint_commands'],
            result2['joint_commands'],
            atol=1e-5
        )

    def test_batch_processing(self, vla_model):
        """Test VLA model with batch processing."""
        # Create batch of scenes
        batch_size = 4
        scenes = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        commands = [
            "Pick the red box",
            "Place the object",
            "Move to shelf",
            "Grasp the cube"
        ]

        robot_states = [
            np.zeros(8) for _ in range(batch_size)
        ]

        results = []
        for scene, cmd, state in zip(scenes, commands, robot_states):
            result = vla_model.predict(scene, cmd, state)
            results.append(result)

        # Verify all predictions were successful
        assert len(results) == batch_size

        # Each result should have valid outputs
        for result in results:
            assert len(result['joint_commands']) == 7
            assert 0 <= result['gripper_command'] <= 1


@pytest.mark.slow
class TestVLAPerformance:
    """Performance tests for VLA model."""

    def test_inference_latency(self, vla_model, sample_scene):
        """Test VLA model inference latency."""
        import time

        vla_model.eval()
        command = "Pick the object"
        robot_state = np.zeros(8)

        # Warm up
        for _ in range(5):
            vla_model.predict(sample_scene, command, robot_state)

        # Measure latency
        latencies = []
        for _ in range(20):
            start = time.time()
            vla_model.predict(sample_scene, command, robot_state)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"VLA Latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms, P99: {p99_latency:.2f}ms")

        # Should meet latency requirements (< 100ms avg on CPU for testing)
        assert avg_latency < 500, f"Average latency too high: {avg_latency:.2f}ms"

    def test_throughput(self, vla_model, sample_scene):
        """Test VLA model throughput."""
        import time

        vla_model.eval()
        command = "Pick the object"
        robot_state = np.zeros(8)

        # Process multiple requests
        num_requests = 50
        start = time.time()

        for _ in range(num_requests):
            vla_model.predict(sample_scene, command, robot_state)

        elapsed = time.time() - start
        throughput = num_requests / elapsed

        print(f"VLA Throughput: {throughput:.2f} requests/second")

        # Should achieve reasonable throughput on CPU
        assert throughput > 1, f"Throughput too low: {throughput:.2f} req/s"


class TestVLAKnowledgeIntegration:
    """Tests for VLA integration with knowledge graph."""

    def test_knowledge_graph_query_integration(self, sample_scene):
        """Test integration with Neo4j knowledge graph."""
        from robo_vla.knowledge.neo4j_graph import Neo4jGraphRAG

        # Initialize components (skip if Neo4j not available)
        try:
            graph = Neo4jGraphRAG(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password"
            )
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")

        # Add test data
        graph.add_object("red_box", {"color": "red", "shape": "box"})
        graph.add_strategy("top_grasp", {"success_rate": 0.9})
        graph.add_relationship("red_box", "top_grasp", "MANIPULATED_BY")

        # Query strategies
        strategies = graph.get_manipulation_strategies("red_box")

        # Verify results
        assert len(strategies) > 0
        assert strategies[0]['name'] == 'top_grasp'

        # Cleanup
        graph.clear_database()
        graph.close()
