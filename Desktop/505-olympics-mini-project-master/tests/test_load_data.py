from unittest import TestCase


class TestLoad_data(TestCase):
    def test_load_data(self):
        from build import load_data

        res = load_data()

        self.assertEqual(13, res['# Summer']['Afghanistan'])
        self.assertEqual(152, res['Silver']['Australia'])

    def test_first_country(self):
        from build import first_country, load_data
        res = load_data()
        res = first_country(res)

        self.assertEqual(13, res["# Summer"])

    def test_gold_medal(self):
        from build import gold_medal, load_data
        res = load_data()
        res = gold_medal(res)

        self.assertEqual('United States', res)

    def test_biggest_difference_in_gold_medal(self):
        from build import biggest_difference_in_gold_medal, load_data
        res = load_data()
        res = biggest_difference_in_gold_medal(res)

        self.assertEqual('United States', res)

    def test_get_points(self):
        from build import get_points, load_data
        res = load_data()
        res = get_points(res)

        self.assertEqual(2, res['Afghanistan'])
        self.assertEqual(130, res['Argentina'])
        self.assertEqual(923, res['Australia'])
        self.assertEqual(569, res['Austria'])
        self.assertEqual(276, res['Belgium'])