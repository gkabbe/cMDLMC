from unittest import TestCase

from mdlmc.misc.tools import chunk


class TestTools(TestCase):
    def test_chunk(self):

        simple_range = range(100)
        range_chunk = chunk(simple_range, 3)
        for start, stop, chk in range_chunk:
            self.assertEqual(simple_range[start: stop], chk)

        self.assertEqual(chk[-1], 99)

        some_list = list("abcdefghijklmnopqrstuvwxyz")

        for start, stop, chk in chunk(some_list, 7):
            self.assertEqual(some_list[start: stop], chk)

        alphabet = "abcdefghijklmnopqrstuvwxyz"

        for start, stop, chk in chunk(alphabet, 11):
            self.assertEqual(alphabet[start: stop], chk)
