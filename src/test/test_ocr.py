import unittest
from vision.ocr import PokerOCR

class TestPokerOCR(unittest.TestCase):
    def setUp(self):
        self.ocr = PokerOCR()

    def test_ocr_initialization(self):
        self.assertIsNotNone(self.ocr)

 