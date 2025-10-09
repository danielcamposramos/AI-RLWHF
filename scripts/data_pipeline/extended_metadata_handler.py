from datetime import datetime
from typing import Dict, Any

class ExtendedMetadataHandler:
    """
    Extends the metadata of RLWHF tuples with additional fields for
    longitudinal tracking and deeper analysis.
    """
    EXTENDED_SCHEMA = {
        "iteration_count": int,      # Track chain iterations
        "consensus_score": float,    # From Deep Seek's builder
        "hardware_profile": str,     # For variance logging
        "update_timestamp": str
    }

    def extend_metadata(self, existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds or updates extended metadata fields to an existing metadata dictionary.

        Args:
            existing_metadata: The original metadata dictionary from an RLWHF tuple.

        Returns:
            The metadata dictionary extended with new tracking fields.
        """
        extended = existing_metadata.copy()

        # Increment iteration count or initialize it
        extended["iteration_count"] = existing_metadata.get("iteration_count", 0) + 1

        # Add other extended fields with default values
        extended.setdefault("consensus_score", 1.0)
        extended.setdefault("hardware_profile", "unknown")

        # Always update the timestamp
        extended["update_timestamp"] = datetime.now().isoformat()

        return extended

    def validate_extended(self, metadata: Dict[str, Any]) -> bool:
        """
        Validates if the metadata contains all the required extended fields.

        Args:
            metadata: The metadata dictionary to validate.

        Returns:
            True if all extended schema fields are present, False otherwise.
        """
        return all(key in metadata for key in self.EXTENDED_SCHEMA)