"""P-ONE-specific `Detector` class(es)."""

from torch_geometric.data import Data

from graphnet.models.detector.detector import Detector
from graphnet.data.constants import FEATURES

class POne(Detector):
    """`Detector` class for P-One prototype"""

    features = FEATURES.PONE

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.
        Args:
            data: Input graph data.
        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # TODO: Implement Preprocessing

        return data