.PHONY: test parse-samples solve-samples

test:
	pytest -q

parse-samples:
	PYTHONPATH=src clrp parse data/samples/tiny_coords.txt
	PYTHONPATH=src clrp parse data/samples/tiny_full_matrix.txt

solve-samples:
	PYTHONPATH=src clrp solve data/samples/tiny_coords.txt --output solutions/tiny_coords.sol --seed 1
	PYTHONPATH=src clrp solve data/samples/tiny_full_matrix.txt --output solutions/tiny_full_matrix.sol --seed 1
