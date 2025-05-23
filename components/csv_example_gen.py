from tfx.v1.components import CsvExampleGen

def create_csv_example_gen(
    input_base: str,
) -> CsvExampleGen:
    """
    Creates and returns a CsvExampleGen component for reading CSV input files.

    Args:
      input_data: Path to the directory (or file pattern) containing CSV data.

    Returns:
      A CsvExampleGen component ready to be added to your TFX pipeline.
    """
    return CsvExampleGen(input_base=input_base)
