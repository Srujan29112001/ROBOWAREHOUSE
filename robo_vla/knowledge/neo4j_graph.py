"""Neo4j graph database for robot knowledge."""

import logging
from typing import Dict, List, Optional

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class Neo4jGraphDB:
    """
    Neo4j graph database for storing robot manipulation knowledge.

    Schema:
    - Objects: Physical objects in the environment
    - Strategies: Manipulation strategies
    - Locations: Spatial locations
    - Properties: Object properties
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "robot_knowledge",
    ):
        """
        Initialize Neo4j connection.

        Args:
            uri: Neo4j URI
            username: Database username
            password: Database password
            database: Database name
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database

        logger.info(f"Connected to Neo4j at {uri}")

        # Initialize schema
        self._init_schema()

    def close(self):
        """Close database connection."""
        self.driver.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self.driver.session(database=self.database) as session:
            # Create constraints
            session.run(
                "CREATE CONSTRAINT object_id IF NOT EXISTS "
                "FOR (o:Object) REQUIRE o.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT strategy_id IF NOT EXISTS "
                "FOR (s:Strategy) REQUIRE s.id IS UNIQUE"
            )

        logger.info("Schema initialized")

    def add_object(
        self,
        object_id: str,
        name: str,
        properties: Dict,
    ) -> None:
        """
        Add object to knowledge graph.

        Args:
            object_id: Unique object ID
            name: Object name
            properties: Object properties (color, shape, weight, etc.)
        """
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (o:Object {id: $object_id})
                SET o.name = $name,
                    o += $properties
                """,
                object_id=object_id,
                name=name,
                properties=properties,
            )

        logger.debug(f"Added object: {name}")

    def add_strategy(
        self,
        strategy_id: str,
        name: str,
        applicable_to: List[str],
        parameters: Dict,
    ) -> None:
        """
        Add manipulation strategy.

        Args:
            strategy_id: Strategy ID
            name: Strategy name
            applicable_to: List of object types this applies to
            parameters: Strategy parameters
        """
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (s:Strategy {id: $strategy_id})
                SET s.name = $name,
                    s.applicable_to = $applicable_to,
                    s += $parameters
                """,
                strategy_id=strategy_id,
                name=name,
                applicable_to=applicable_to,
                parameters=parameters,
            )

        logger.debug(f"Added strategy: {name}")

    def link_object_strategy(
        self,
        object_id: str,
        strategy_id: str,
        success_rate: float = 0.0,
    ) -> None:
        """Link object to manipulation strategy."""
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (o:Object {id: $object_id})
                MATCH (s:Strategy {id: $strategy_id})
                MERGE (o)-[r:CAN_BE_MANIPULATED_WITH]->(s)
                SET r.success_rate = $success_rate
                """,
                object_id=object_id,
                strategy_id=strategy_id,
                success_rate=success_rate,
            )

    def find_strategies_for_object(
        self,
        object_name: str,
        min_success_rate: float = 0.7,
    ) -> List[Dict]:
        """
        Find manipulation strategies for an object.

        Args:
            object_name: Name of the object
            min_success_rate: Minimum success rate threshold

        Returns:
            List of strategies with parameters
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (o:Object {name: $object_name})
                      -[r:CAN_BE_MANIPULATED_WITH]->(s:Strategy)
                WHERE r.success_rate >= $min_success_rate
                RETURN s ORDER BY r.success_rate DESC
                """,
                object_name=object_name,
                min_success_rate=min_success_rate,
            )

            strategies = [record["s"] for record in result]
            return strategies

    def find_similar_objects(
        self,
        object_id: str,
        similarity_threshold: float = 0.8,
    ) -> List[Dict]:
        """Find similar objects."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (o1:Object {id: $object_id})
                      -[sim:SIMILAR_TO]->(o2:Object)
                WHERE sim.similarity_score >= $threshold
                RETURN o2, sim.similarity_score as score
                ORDER BY score DESC
                """,
                object_id=object_id,
                threshold=similarity_threshold,
            )

            similar = [
                {"object": record["o2"], "score": record["score"]}
                for record in result
            ]
            return similar

    def get_object_location(self, object_id: str) -> Optional[Dict]:
        """Get object location."""
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (o:Object {id: $object_id})-[:LOCATED_AT]->(loc:Location)
                RETURN loc
                """,
                object_id=object_id,
            )

            record = result.single()
            return record["loc"] if record else None

    def update_success_rate(
        self,
        object_id: str,
        strategy_id: str,
        success: bool,
    ) -> None:
        """Update strategy success rate based on execution result."""
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (o:Object {id: $object_id})
                      -[r:CAN_BE_MANIPULATED_WITH]->(s:Strategy {id: $strategy_id})
                SET r.attempts = COALESCE(r.attempts, 0) + 1,
                    r.successes = COALESCE(r.successes, 0) + CASE WHEN $success THEN 1 ELSE 0 END,
                    r.success_rate = (COALESCE(r.successes, 0) + CASE WHEN $success THEN 1 ELSE 0 END) * 1.0 / (COALESCE(r.attempts, 0) + 1)
                """,
                object_id=object_id,
                strategy_id=strategy_id,
                success=success,
            )

        logger.debug(f"Updated success rate for {object_id} with {strategy_id}")

    def query_cypher(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute custom Cypher query."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
