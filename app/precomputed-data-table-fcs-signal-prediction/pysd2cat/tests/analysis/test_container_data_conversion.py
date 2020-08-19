import pickle
import pysd2cat.analysis.container_data_conversion as cdc
import typing
import os
from container_dict import real_container_dict
from dataclasses import dataclass


def test_container_well_idx_name():
    assert cdc.container_well_idx_name(12, 0) == "a1"
    assert cdc.container_well_idx_name(12, 1) == "a2"
    assert cdc.container_well_idx_name(12, 12) == "b1"


def test_column_dict():
    assert cdc.column_dict(12, [0, 1, 12, 13]) == {
        "col1": ["a1", "b1"],
        "col2": ["a2", "b2"]
    }


def test_aliquot_dict():
    assert cdc.aliquot_dict({0: "UWBF_24926"}, None, 0) == {
        "strain": "UWBF_24926"
    }


@dataclass
class MockContainer:
    col_count: int
    well_map: dict
    # Not used currently. Should be a pandas dataframe, if used.
    aliquots: typing.Any

    @property
    def attributes(self):
        return {"container_type": {"col_count": self.col_count}}


def test_convert_container_to_dict():
    """
    Test converting a made up mock container object to a dictionary
    """
    col_count = 12
    well_map = {0: "strain1", 12: "strain1",
                1: "strain2", 13: "strain2"}
    container = MockContainer(col_count=col_count,
                              well_map=well_map,
                              aliquots=None)
    assert container.attributes["container_type"]["col_count"] == col_count

    container_dict = cdc.container_to_dict(container)

    assert "aliquots" in container_dict
    assert "columns" in container_dict

    columns = container_dict["columns"]
    aliquots = container_dict["aliquots"]

    assert set(columns.keys()) == {"col1", "col2"}
    assert set(columns["col1"]) == {"a1", "b1"}
    assert set(columns["col2"]) == {"a2", "b2"}

    assert set(aliquots.keys()) == {"a1", "a2", "b1", "b2"}
    assert aliquots["a1"] == aliquots["b1"]
    assert aliquots["a1"]["strain"] == "strain1"

    assert aliquots["a2"] == aliquots["b2"]
    assert aliquots["a2"]["strain"] == "strain2"


def test_pickled_container():
    """
    Load a pickled container that was retrieved from strateos
    and test the conversion on a real container object.
    """
    pickle_file = os.path.join(os.path.dirname(__file__),
                               "container-object-ct1dqvp6nm67z9b.pickle")
    with open(pickle_file, "rb") as f:
        container = pickle.load(f)

    container_dict = cdc.container_to_dict(container)

    assert "aliquots" in container_dict
    assert "columns" in container_dict

    assert container_dict == real_container_dict


# Commented out test below just because it actually connects to strateos
# test_pickled_container is the same test using the data retrieved from
# strateos pickled locally without needing to connect.

# def test_from_source():
#     """
#     Test connecting to strateos and pulling the container object directly
#     and converting it to a dictionary
#     """
#     from transcriptic.jupyter import objects
#     from transcriptic.config import Connection
#     conn = Connection.from_file("~/.transcriptic")
#     container_id = "ct1dqvp6nm67z9b"
#     container = objects.Container(container_id)

#     assert cdc.container_to_dict(container) == real_container_dict
