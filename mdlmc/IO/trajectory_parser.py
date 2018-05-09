import logging
import warnings
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from io import IOBase
from typing import Iterator, Union, Tuple, IO, Container

try:
    import tables
except ImportError:
    warnings.warn("Pytables and/or h5py were not found, but are needed for use of HDF5 files.")
import h5py
import numpy as np

from ..atoms.numpy_atom import dtype_xyz
from ..misc.tools import chunk_trajectory

logger = logging.getLogger(__name__)


@contextmanager
def as_file(file_or_string):
    """Allows handling of filenames or files in the same way."""
    need_to_close = False
    # If the input is no file, try to open it
    if not isinstance(file_or_string, IOBase):
        logger.debug("Try to open filename %s", file_or_string)
        file = open(file_or_string, "r")
        need_to_close = True
    else:
        file = file_or_string
    yield file

    if need_to_close:
        file.close()


class Frame:
    """Wrapper around structured array to ease selection by name or index"""
    def __init__(self, names, positions, *, time=None):
        self._names = names
        self._positions = positions
        self._time = time

    @classmethod
    def from_recarray(cls, array: np.ndarray, *, time=None):
        names = array["name"]
        positions = array["pos"]
        return cls(names, positions, time=time)

    def _extract_array(self, selection: Union[str, list, np.ndarray]):
        """
        Extract a selection of the frame as numpy array.
        Parameters
        ----------
        selection:
            Either indices of selected atoms, or string with desired atom type selection

        Returns
        -------
        result: np.ndarray
        """

        if isinstance(selection, str):
            logger.debug("Select atoms of type %s", selection)
            result = self._names[self._names == selection], self._positions[self._names == selection]
        elif isinstance(selection, (list, np.ndarray)):
            logger.debug("Select atoms with indices %s", selection)
            result = self._names[selection], self._positions[selection]
        else:
            raise ValueError(f"Selection {selection} not understood")
        return result

    def __getitem__(self, item):
        result = self._extract_array(item)
        return Frame(*result, time=self._time)

    def __repr__(self):
        lines = "\n".join([f"{atomname}    {atompos[0]:20.10f} {atompos[1]:20.10f} "
                           f"{atompos[2]:20.10f}"
                           for atomname, atompos in zip(self.atom_names, self.atom_positions)])
        repr = f"{self.atom_number}\n\n{lines}"""
        return repr

    def append(self, f2: "Frame"):
        new_frame = Frame(np.hstack([self.atom_names, f2.atom_names]),
                          np.vstack([self.atom_positions, f2.atom_positions]))
        return new_frame

    @property
    def atom_names(self):
        return self._names

    @atom_names.setter
    def atom_names(self, name):
        self._names[:] = name

    @property
    def atom_positions(self):
        return self._positions

    @property
    def atom_number(self):
        return self._names.size

    @property
    def time(self):
        return self._time


class Trajectory(metaclass=ABCMeta):
    """Abstract Trajectory class which should be inherited when defining
    a custom trajectory class"""

    @abstractmethod
    def __iter__(self):
        pass

    @property
    @abstractmethod
    def current_frame_number(self):
        """Return the current frame."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the length (i.e. number of frames) of the trajectory"""
        pass


def parse_xyz(f, frame_len, selection=None, no_of_frames=None):
    def filter_lines(f, frame_len):
        for i, line in enumerate(f):
            if i % frame_len not in (0, 1):
                yield line
                if (i + 1) // frame_len == no_of_frames:
                    break

    if selection is not None:
        def filter_selection(f, s):
            for i, line in enumerate(f):
                if i % (frame_len - 2) in s:
                    yield line

        filter_ = filter_selection(filter_lines(f, frame_len), selection)
        output_shape = (-1, len(selection))
    else:
        filter_ = filter_lines(f, frame_len)
        output_shape = (-1, frame_len - 2)

    data = np.genfromtxt(filter_, dtype=dtype_xyz)
    return data.reshape(output_shape)


def filter_lines(f, frame_len, no_of_frames=None):
    for i, line in enumerate(f):
        if i % frame_len not in (0, 1):
            yield line
            if (i + 1) // frame_len == no_of_frames:
                break


def filter_selection(f, s, frame_len):
    for i, line in enumerate(f):
        if i % frame_len in s:
            yield line


class XYZTrajectory(Trajectory):
    def __init__(self, filename: Union[str, IO],
                 *,
                 time_step: float,
                 number_of_atoms: int = None,
                 selection: Union[Container, str, Tuple[str]] = None,
                 repeat: bool = False) -> None:
        """
        Parameters
        ----------
        filename:
            Path to the xyz file
        time_step:
            time between two frames
        number_of_atoms:
            total number of atoms in the xyz file
        selection: str or array of indices
            subset of atoms which should be selected
        repeat: bool
            repeat the trajectory when iterating
        """

        self.filename = filename
        self._number_of_atoms = number_of_atoms
        self.selection = selection
        self.repeat = repeat
        self._current_frame_number = 0
        self.time_step = time_step
        self._first_frame = None

        # Check if atom number is specified
        # If not, try to read it from file
        if not self._number_of_atoms:
            logger.info("Number of atoms not specified. Will try to read it from xyz file")
            with open(self.filename, "r") as f:
                try:
                    self._number_of_atoms = int(f.readline())
                except((ValueError, TypeError)):
                    logger.error("Could not read atom number from %s", self.filename)
                    raise

    def __iter__(self) -> Iterator[Frame]:

        time_step = self.time_step
        frame_len = self._number_of_atoms + 2

        if self.selection is not None:
            if isinstance(self.selection, str):
                self.selection = get_xyz_selection_from_atomname(self.filename, self.selection)
            elif isinstance(self.selection, tuple):
                if isinstance(self.selection[0], str):
                    self.selection = get_xyz_selection_from_atomname(self.filename, *self.selection)

            def filter_(f):
                yield from filter_selection(filter_lines(f, frame_len, no_of_frames=1),
                                            self.selection, self._number_of_atoms)
        else:
            def filter_(f):
                yield from filter_lines(f, frame_len, no_of_frames=1)

        while True:
            with warnings.catch_warnings(), as_file(self.filename) as f:
                np.warnings.filterwarnings("error", message="genfromtxt: Empty input file")
                while True:
                    if self._current_frame_number % 100 == 0:
                        logger.debug("Reading xyz frame %i", self._current_frame_number)
                    try:
                        data = np.genfromtxt(filter_(f), dtype=dtype_xyz)
                    except Warning as w:
                        logger.debug(w)
                        logger.info("Reached end of file")
                        break
                    yield Frame.from_recarray(data, time=self._current_frame_number*time_step)
                    self._current_frame_number += 1

            if not self.repeat:
                break

    @property
    def current_frame_number(self):
        return self._current_frame_number

    def __len__(self):
        logger.info("Determining length (number of frames) of xyz trajectory")
        logger.info("This may take a while")

        counter = 0
        with open(self.filename, "r") as f:
            for _ in f:
                counter += 1
                if counter % 1000 == 0:
                    logger.debug("Line %i", counter)

        return counter // (self._number_of_atoms + 2)


def get_xyz_selection_from_atomname(xyz_filename, *atomnames):
    """Determine the indices of a set of atom names in an xyz trajectory."""
    with as_file(xyz_filename) as f:
        frame_len = int(f.readline())
        selection = []
        f.readline()
        for i, line in enumerate(f):
            if line.split()[0] in atomnames:
                selection.append(i)
            if i == frame_len:
                break
        # If xyz_filename is actually a file,
        # it should be rewinded to the beginning
        f.seek(0)
    logger.debug("Selection: %s", selection)
    return np.array(selection)


class HDF5Trajectory(Trajectory):
    def __init__(self,
                 filename: str,
                 time_step: float,
                 selection: Union[Container, str, Tuple[str]] = None,
                 repeat: bool = False,
                 chunk_size: int = 1000) -> None:
        self.filename = filename
        self.time_step = time_step
        # TODO: Implement selection
        if selection:
            warnings.warn("Selection is not implemented yet!")
        self.selection = selection
        self.repeat = repeat
        self._chunk_size = chunk_size
        self._current_frame_number = None

        self._atomnames_key = "atom_names"
        self._trajectory_key = "trajectory"

        with h5py.File(filename, "r") as f:
            self.atom_names = f[self._atomnames_key][:].astype("<U2")

    def __iter__(self):
        time_step = self.time_step
        atom_names = self.atom_names
        chunk_size = self._chunk_size
        with h5py.File(self.filename, "r") as h5file:
            traj = h5file[self._trajectory_key]

            step = 0
            while True:
                for _, _, frames in chunk_trajectory(traj, chunk_size=chunk_size):
                    for frame in frames:
                        yield Frame(atom_names, frame.astype(float), time=step*time_step)
                step += 1

                if not self.repeat:
                    break

    def __len__(self):
        with h5py.File(self.filename, "r") as h5file:
            traj = h5file[self._trajectory_key]
            return traj.shape[0]

    @property
    def current_frame_number(self):
        return self._current_frame_number
