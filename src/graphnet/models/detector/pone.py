"""P-ONE-specific `Detector` class(es)."""

from torch_geometric.data import Data

from graphnet.models.detector.detector import Detector

class POne(Detector):
    """`Detector` class for P-One prototype"""

    features = [
        'pmt_x',
        'pmt_y',
        'pmt_z',
        'pmt_azimuth',
        'pmt_zenith',
        'time',
    ]

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