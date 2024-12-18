# -*- coding: utf-8 -*-
"""
Exemple d'utilisation de unittest
"""

import ExempleFonctionSurfaceExceptionNeg
import unittest

class TestFonctionSurface(unittest.TestCase):

    def test_1(self):
        self.assertEqual(ExempleFonctionSurfaceExceptionNeg.surface(3,5), 15)

    def test_2(self):
        self.assertEqual(ExempleFonctionSurfaceExceptionNeg.surface(2.5,4), 10)
        
    def test_negatif(self):
        with self.assertRaises(Exception):
            ExempleFonctionSurfaceExceptionNeg.surface(-3,4)        

unittest.main()
