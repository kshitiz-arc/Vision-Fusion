"""VisionFusion AI — core perception modules."""

from .preprocessing   import Preprocessor
from .edge_detection  import EdgeDetector
from .face_detection  import FaceDetector, FaceDetection
from .motion_detection import MotionDetector, MotionRegion
from .contour_analysis import ContourAnalyzer, ContourShape
from .object_detection import ObjectDetector, Detection
from .tracking        import CentroidTracker, Track, SingleObjectTracker
from .visualization   import Visualizer
from .cnn_classifier  import CNNClassifier, ClassificationResult

__all__ = [
    "Preprocessor",
    "EdgeDetector",
    "FaceDetector", "FaceDetection",
    "MotionDetector", "MotionRegion",
    "ContourAnalyzer", "ContourShape",
    "ObjectDetector", "Detection",
    "CentroidTracker", "Track", "SingleObjectTracker",
    "Visualizer",
    "CNNClassifier", "ClassificationResult",
]
