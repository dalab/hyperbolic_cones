from smart_open import smart_open
import sys
import csv

class Relations(object):
    """Class to stream relations from a tsv-like file."""

    def __init__(self, file_path, encoding='utf8', delimiter='\t'):
        """Initialize instance from file containing a pair of nodes (a relation) per line.

        Parameters
        ----------
        file_path : str
            Path to file containing a pair of nodes (a relation) per line, separated by `delimiter`.
        encoding : str, optional
            Character encoding of the input file.
        delimiter : str, optional
            Delimiter character for each relation.
        """

        self.file_path = file_path
        self.encoding = encoding
        self.delimiter = delimiter

    def __iter__(self):
        """Streams relations from self.file_path decoded into unicode strings.

        Yields
        -------
        2-tuple (unicode, unicode)
            Relation from input file.
        """
        with smart_open(self.file_path) as file_obj:
            if sys.version_info[0] < 3:
                lines = file_obj
            else:
                lines = (l.decode(self.encoding) for l in file_obj)
            # csv.reader requires bytestring input in python2, unicode input in python3
            reader = csv.reader(lines, delimiter=self.delimiter)
            for row in reader:
                if sys.version_info[0] < 3:
                    row = [value.decode(self.encoding) for value in row]
                (v,u) = tuple(row) # Swap line in the csv file because we want the correct edge direction.
                assert u != v
                yield (u,v)

