from smio_clrp.io.instance_reader import read_instance


def test_read_coords_instance():
    instance = read_instance("data/samples/tiny_coords.txt")

    assert instance.name == "tiny_coords"
    assert instance.distance_format == "COORDS"
    assert len(instance.depots) == 2
    assert len(instance.customers) == 5
    assert instance.vehicle_capacity == 10
    assert instance.depots_by_id[1].capacity == 18
    assert instance.customers_by_id[101].demand == 4


def test_read_full_matrix_instance():
    instance = read_instance("data/samples/tiny_full_matrix.txt")

    assert instance.name == "tiny_full_matrix"
    assert instance.distance_format == "FULL_MATRIX"
    assert instance.distance_matrix.shape == (7, 7)
    assert instance.node_index[("depot", 1)] == 0
    assert instance.node_index[("customer", 101)] == 2
