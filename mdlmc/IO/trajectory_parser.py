#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
import types
from abc import ABCMeta, abstractproperty, abstractmethod
from contextlib import contextmanager
from functools import reduce
from io import IOBase
from typing import Iterator, Union, Tuple, IO, Container
import warnings

import h5py
import mdtraj
import numpy as np

from ..atoms import numpy_atom as npa
from ..atoms.numpy_atom import dtype_xyz, dtype_xyz_bytes
from ..misc.tools import argparse_compatible, chunk


warnings.filterwarnings('error')


logger = logging.getLogger(__name__)


@contextmanager
def as_file(file_or_string):
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
    def __init__(self, array):
        self._array = array

    def get_selection(self, selection):
        pass

    def __getattr__(self, selection):
        if isinstance(selection, str):
            result = self._array["pos"][self._array["name"] == selection]
        elif isinstance(selection, list):
            result = self._array["pos"][selection]
        else:
            raise ValueError("Selection not understood")
        return result


class Trajectory(metaclass=ABCMeta):
    """Abstract Trajectory class which should be inherited when defining
    a custom trajectory class"""

    @property
    @abstractmethod
    def current_frame_number(self):
        """Return the current frame."""
        pass

    @abstractmethod
    def __iter__(self):
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

    data = np.genfromtxt(filter_, dtype=dtype_xyz_bytes)
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
                 repeat: bool = False):

        self.filename = filename
        self.number_of_atoms = number_of_atoms
        self.selection = selection
        self.repeat = repeat
        self._current_frame_number = 0
        self.time_step = time_step

        # Check if atom number is specified
        # If not, try to read it from file
        if not self.number_of_atoms:
            logger.info("Number of atoms not specified. Will try to read it from xyz file")
            with open(self.filename, "r") as f:
                try:
                    self.number_of_atoms = int(f.readline())
                except(("ValueError", "TypeError")):
                    logger.error("Could not read atom number from %s", self.filename)
                    raise

    @property
    def current_frame_number(self):
        return self._current_frame_number

    def __iter__(self) -> Iterator[np.array]:
        """

        Parameters
        ----------
        filename
        number_of_atoms:
            Number of atoms per frame
        selection:
            List of atom indices or string or tuple of strings

        Returns
        -------

        """


        frame_len = self.number_of_atoms + 2

        if self.selection is not None:
            if isinstance(self.selection, str):
                self.selection = get_xyz_selection_from_atomname(self.filename, self.selection)
            elif isinstance(self.selection, tuple):
                if isinstance(self.selection[0], str):
                    self.selection = get_xyz_selection_from_atomname(self.filename, *self.selection)

            def filter_(f):
                yield from filter_selection(filter_lines(f, frame_len, no_of_frames=1),
                                            self.selection, self.number_of_atoms)
        else:
            def filter_(f):
                yield from filter_lines(f, frame_len, no_of_frames=1)

        while True:
            with warnings.catch_warnings(), as_file(self.filename) as f:
                while True:
                    logger.debug("Reading xyz frame %i", self._current_frame_number)
                    try:
                        data = np.genfromtxt(filter_(f), dtype=dtype_xyz_bytes)
                    except Warning:
                        logger.info("Reached end of file")
                        break
                    yield data
                    self._current_frame_number += 1

            if not self.repeat:
                break


def get_xyz_selection_from_atomname(xyz_filename, *atomnames):
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


def get_hdf5_selection_from_atomname(hdf5_filename, *atomnames):
    """Expects a hdf5 trajectory created with mdconvert (from package mdtraj)"""
    trajgen = mdtraj.iterload(hdf5_filename, chunk=1)
    frame = next(trajgen)
    if atomnames:
        selections = [frame.topology.select("type {}".format(atomname)) for atomname in atomnames]
        selection = np.sort(np.hstack(selections))
        name_array = np.zeros(frame.n_atoms, dtype=str)
        for name, indices in zip(atomnames, selections):
            name_array[indices] = name
        name_array = name_array[name_array != ""]
    else:
        selection = frame.topology.select("all")
        name_array = np.array([atom.name for r in frame.topology.residues for atom in r.atoms])
    return selection, name_array


@argparse_compatible
def save_trajectory_to_hdf5(xyz_fname, hdf5_fname=None, chunk=1000, *, remove_com_movement=False,
                            verbose=False, file=sys.stdout, dataset_name="trajectory",
                            selection=None):

    with open(xyz_fname, "rb") as f:
        frame_len = int(f.readline()) + 2
        f.seek(0)
        first_frame, = parse_xyz(f, frame_len, no_of_frames=1, selection=selection)
        atom_names = first_frame["name"]
        frame_shape = first_frame["pos"].shape

    if not hdf5_fname:
        hdf5_fname = os.path.splitext(xyz_fname)[0] + ".hdf5"

    # Use a set as a data structure to optimize selection speed
    if selection is not None:
        selection = set(selection)

    with h5py.File(hdf5_fname, "w") as hdf5_file:
        # Use blosc compression (needs tables import and code 32001)
        traj_atomnames = hdf5_file.create_dataset("atom_names", atom_names.shape, dtype="2S")
        traj_atomnames[:] = atom_names
        traj = hdf5_file.create_dataset(dataset_name, shape=(10 * chunk, *frame_shape),
                                        dtype=np.float32, maxshape=(None, *frame_shape),
                                        compression=32001)

        with open(xyz_fname, "rb") as f:
            counter = 0
            start_time = time.time()
            while True:
                frames = parse_xyz(f, frame_len, no_of_frames=chunk,
                                   selection=selection).astype(dtype_xyz)

                if frames.shape[0] == 0:
                    break

                if remove_com_movement:
                    npa.remove_center_of_mass_movement(frames)

                # Resize trajectory if necessary
                if counter + frames.shape[0] > traj.shape[0]:
                    logger.debug("Need to resize trajectory hdf5 dataset")
                    traj.resize(2 * traj.shape[0], axis=0)

                traj[counter:counter + frames.shape[0]] = frames["pos"]
                counter += frames.shape[0]
                logger.info("Parsed frames: {: 6d}. {:.2f} fps".format(
                    counter, counter / (time.time() - start_time)))
        traj.resize(counter, axis=0)


def load_trajectory_from_hdf5(hdf5_fname, *atom_names, hdf5_key="trajectory", clip=None,
                              verbose=False, file=sys.stdout):
    """Loads hdf5 trajectory.
    Be careful when specifying clip or atom_names! In this case, the trajectory slice will be loaded
    into memory.
    """
    f = h5py.File(hdf5_fname, "r")
    traj_atom_names = f["atom_names"].value.astype("U")
    if atom_names:
        if verbose:
            print("# Will select atoms", *atom_names, file=file)
        selection = reduce(np.logical_or, [traj_atom_names == name for name in atom_names])
    else:
        selection = slice(None)

    slice_ = slice(0, clip)

    try:
        if clip or atom_names:
            print("np array!!!!")
            trajectories = f[hdf5_key][slice_, selection]
        else:
            trajectories = f[hdf5_key]
    except KeyError as e:
        raise KeyError("Could not find key {} in hdf5 file. "
                       "Available keys: {}".format(hdf5_key, list(f.keys()))) from e
    return traj_atom_names[selection], trajectories


def create_dataset_from_hdf5_trajectory(hdf5_file, trajectory_dataset, dataset_name, selection,
                                        chunk_size, verbose=False, file=sys.stdout, overwrite=False,
                                        dtype=float):
    """Given a hdf5 trajectory, this function creates a new data set inside the hdf5 file with the
    specified dataset_name according to the supplied selection.
    If it already exists, it will be returned directly.

    Parameters
    ----------
    hdf5_file: HDF5 file
        The new dataset will be saved in this file

    trajectory_dataset: array_like
        The array or HDF5 dataset containing the atom trajectory
        or the data on which the selection will operate

    dataset_name: str or tuple of str
        Name of the new dataset(s).
        If a tuple of dataset names is given, a selection function is expected that
        returns tuples of data.

    selection: function or array_like
        Will be used to select items from trajectory_dataset.
        Can be an array of indices or a function that creates a new array out of
        trajectory_dataset

    chunk_size: int
        The size of the data processed in one step

        verbose: bool
        Verbosity

    file: file_like

    overwrite: bool
        If set True, an already existing dataset will be overwritten

    dtype: type or tuple of types
        Specifies the dtype(s) used for the dataset(s)

    Returns
    -------
    new_dataset: HDF5 dataset or tuple of HDF5 datasets"""

    # If selection is a function, leave it as it is
    if type(selection) in (types.FunctionType, types.MethodType, types.BuiltinFunctionType,
                           types.BuiltinMethodType):
        selection_fct = selection
    # If not, make a function out of it
    else:
        if type(dataset_name) is tuple:
            raise TypeError("If a tuple of dataset names is given, selection needs to be a function"
                            "returning a tuple of data")

        def selection_fct(arr):
            return np.take(arr, selection, axis=1)

    # If selection_fct does not return tuples, make it do that
    if type(dataset_name) not in (tuple, list):
        dataset_names = (dataset_name,)

        def selection_fct_tup(arr):
            return (selection_fct(arr),)

    else:
        dataset_names = dataset_name
        selection_fct_tup = selection_fct

    logger.debug("Looking for data under {}".format(dataset_names))

    if type(dtype) is type:
        dtypes = tuple([dtype for ds in dataset_names])
    elif type(dtype) is tuple:
        dtypes = dtype
        assert len(dtypes) == len(dataset_names), "Number of dtypes and number of datasets is not" \
                                                  " equal."
    else:
        raise TypeError("dtype must be of type \"type\" or tuple of types")

    logger.debug("Got the following dtypes: {}".format(dtypes))

    first_frame = trajectory_dataset[:1]
    one_frame_shapes = [ff.shape for ff in selection_fct_tup(first_frame)]

    logger.debug("Got the following shapes: {}".format(one_frame_shapes))

    new_datasets = []
    for dsn, ofs, dt in zip(dataset_names, one_frame_shapes, dtypes):
        if dsn not in hdf5_file.keys():
            print("# Did not find data set", dsn, "in HDF5 file")
            new_dataset = hdf5_file.create_dataset(dsn, shape=(trajectory_dataset.shape[0],
                                                   *ofs[1:]), dtype=dt, compression=32001)
            # Set last frame to nan, to identify the data set as unfinished
            new_dataset[-1] = np.nan

        else:
            new_dataset = hdf5_file[dsn]
        new_datasets.append(new_dataset)

    # Check if any dataset has not been completely written
    is_unfinished = any([np.isnan(hdf5_file[dsn][-1]).any() for dsn in dataset_names])
    if is_unfinished or overwrite:
        if is_unfinished:
            logger.info(
                "It looks like data set {} has not been written (completely) to hdf5 yet.".format(
                    dataset_name))
            logger.info("Will do it now")
        if not is_unfinished and overwrite:
            logger.info("File has been written before, but overwrite = True")
            logger.info("Will overwrite now")
        start_time = time.time()
        for start, stop, traj_chunk in chunk(trajectory_dataset, chunk_size):
            chunk_of_datasets = selection_fct_tup(traj_chunk)
            for ds, chnk in zip(new_datasets, chunk_of_datasets):
                ds[start:stop] = chnk
            print("# Parsed frames: {: 6d}. {:.2f} fps".format(
                  stop, stop / (time.time() - start_time)), end="\r", flush=True, file=sys.stderr)
        print(file=sys.stderr)
    if len(new_datasets) == 1:
        return new_datasets[0]

    return new_datasets


def save_trajectory_to_npz(xyz_fname, npz_fname=None, remove_com_movement=False,
                           verbose=False, selection=None):
    with open(xyz_fname, "rb") as f:
        frame_length = int(f.readline()) + 2
        f.seek(0)

        logger.debug("Frame length: {}".format(frame_length))

        chunk_size = 1000
        counter = 0
        trajectory = []
        start_time = time.time()
        data = parse_xyz(f, frame_length, no_of_frames=chunk_size)
        while data.shape[0] > 0:
            counter += chunk_size
            fps = counter / (time.time() - start_time)
            print("# {:6d} ({:.2f} fps)".format(counter, fps), end="\r", flush=True)
            trajectory.append(data)
            data = parse_xyz(f, frame_length, no_of_frames=chunk_size, selection=selection)
        print("")
        trajectory = np.vstack(trajectory)

        if remove_com_movement:
            logger.info("Removing center of mass movement...")
            npa.remove_center_of_mass_movement(trajectory)

        if not npz_fname:
            npz_fname = os.path.splitext(xyz_fname)[0] + ".npy"
        np.savez(npz_fname, trajectory=trajectory)


def load_trajectory_from_npz(npz_fname, *atom_names, clip=None, verbose=False):
    file_content = np.load(npz_fname)
    if type(file_content) == np.ndarray:
        trajectory = file_content
    else:
        if "trajectory" in file_content.keys():
            trajectory = file_content["trajectory"]
        else:
            trajectory = file_content.items()[0]

    if clip:
        logger.info("Clipping trajectory to the first {} frames".format(clip))
        trajectory = trajectory[:clip]
    if len(atom_names) > 0:
        single_atom_trajs = []
        for atom in atom_names:
            selection_indices, = np.where(trajectory[0]["name"].astype(str) == atom)
            atom_traj = trajectory[:, selection_indices]
            atom_traj = np.array(atom_traj["pos"], order="C")
            single_atom_trajs.append(atom_traj)
        return single_atom_trajs
    else:
        return trajectory


def load_atoms(filename, *atom_names, auxiliary_file=None, verbose=False, clip=None, hdf5=False):

    if auxiliary_file:
        # User explicitly specified auxiliary file
        logger.info("Both xyz file and auxiliary npz/npy file specified.")
        logger.info("Will try to load from auxiliary file {}".format(auxiliary_file))
    else:
        # Look for auxiliary file with same name as xyz file, but different extension
        if hdf5:
            auxiliary_file = os.path.splitext(filename)[0] + ".hdf5"
        else:
            auxiliary_file = os.path.splitext(filename)[0] + ".npz"

    logger.info("Looking for auxiliary file {}".format(auxiliary_file))

    if os.path.exists(auxiliary_file):
        logger.info("Found it")

    else:

        logger.info("No auxiliary file found.")
        logger.info("Will create it now...")
        if hdf5:
            save_trajectory_to_hdf5(filename, hdf5_fname=auxiliary_file,
                                    remove_com_movement=True, verbose=verbose)
        else:
            save_trajectory_to_npz(filename, npz_fname=auxiliary_file,
                                   remove_com_movement=True, verbose=verbose)

    if hdf5:
        return load_trajectory_from_hdf5(auxiliary_file, *atom_names, clip=clip, verbose=verbose)
    else:
        return load_trajectory_from_npz(auxiliary_file, *atom_names, clip=clip, verbose=verbose)


def save_to_hdf5_cmd(*args):
    parser = argparse.ArgumentParser(
        description="Save trajectory to HDF5", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("xyz_fname", help="XYZ trajectory file name")
    parser.add_argument("--hdf5_fname", help="HDF5 file name")
    parser.add_argument("--chunk", "-c", type=int, default=1000, help="Chunk size")
    parser.add_argument("--remove_com", "-r", action="store_true", help="Remove center of mass movement")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity")
    parser.set_defaults(func=save_trajectory_to_hdf5)
    args = parser.parse_args()
    args.func(args)


def print_hdf5(*args):
    parser = argparse.ArgumentParser(description="Print HDF5 trajectory to command line in xyz "
                                                 "format",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hdf5_fname", help="HDF5 file name")
    parser.add_argument("--atom_names", "-a", nargs="*", default=[],
                        help="Which atoms should be printed?")
    parser.add_argument("--clip", "-c", type=int, help="Clip trajectory after number of frames")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity")
    args = parser.parse_args()
    atom_names, trajectory = load_trajectory_from_hdf5(args.hdf5_fname, *args.atom_names,
                                                       clip=args.clip, verbose=args.verbose)

    for frame in trajectory:
        print(frame.shape[0])
        print()
        for name, pos in zip(atom_names, frame):
            print(name, " ".join(map(str, pos)))


def print_hdf5_mdtraj():
    parser = argparse.ArgumentParser(description="Print HDF5 trajectory to command line in xyz "
                                                 "format",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hdf5_fname", help="HDF5 file name")
    parser.add_argument("--atom_names", "-a", nargs="*", default=[],
                        help="Which atoms should be printed?")
    parser.add_argument("--clip", "-c", type=int, help="Clip trajectory after number of frames")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbosity")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    selection, names = get_hdf5_selection_from_atomname(args.hdf5_fname, *args.atom_names)
    trajectory = mdtraj.iterload(args.hdf5_fname, atom_indices=selection)

    total_number_of_frames = 0
    for frames in trajectory:
        logger.info("{: 8.2f} ps".format(frames.time[0]))
        logger.debug("Frames: {}".format(total_number_of_frames))
        for frame, time in zip(frames.xyz, frames.time):
            print(frames.n_atoms)
            print(time)
            for name, coord in zip(names, frame):
                print("{} {:>10.2f} {:>10.2f} {:>10.2f}".format(name, *(10 * coord)))
        total_number_of_frames += frames.n_frames
        if total_number_of_frames >= args.clip:
            break
