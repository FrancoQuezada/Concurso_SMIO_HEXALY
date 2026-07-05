from smio_clrp.core.distance import distance
from smio_clrp.io.instance_reader import read_instance


def test_euclidean_distance_is_rounded_to_one_decimal():
    instance = read_instance("data/samples/tiny_coords.txt")

    assert distance(instance, ("depot", 1), ("customer", 101)) == 1.4
    assert distance(instance, ("customer", 101), ("customer", 105)) == 5.0


def test_full_matrix_distances_may_be_asymmetric():
    instance = read_instance("data/samples/tiny_full_matrix.txt")

    assert distance(instance, ("depot", 1), ("depot", 2)) == 10
    assert distance(instance, ("depot", 2), ("depot", 1)) == 12
