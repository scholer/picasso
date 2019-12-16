"""
    picasso.io
    ~~~~~~~~~~

    General purpose library for handling input and output of files

    :author: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""
import os.path as _ospath
import numpy as _np
import yaml as _yaml
import glob as _glob
import h5py as _h5py
import re as _re
import struct as _struct
import json as _json
import os as _os
import threading as _threading
from PyQt5.QtWidgets import QMessageBox as _QMessageBox
from . import lib as _lib


class NoMetadataFileError(FileNotFoundError):
    pass


def _user_settings_filename():
    home = _ospath.expanduser("~")
    return _ospath.join(home, ".picasso", "settings.yaml")


def load_raw(path, prompt_info=None):
    try:
        info = load_info(path)
    except FileNotFoundError as error:
        if prompt_info is None:
            raise error
        else:
            result = prompt_info()
            if result is None:
                return
            else:
                info, save = result
                info = [info]
                if save:
                    base, ext = _ospath.splitext(path)
                    info_path = base + ".yaml"
                    save_info(info_path, info)
    dtype = _np.dtype(info[0]["Data Type"])
    shape = (info[0]["Frames"], info[0]["Height"], info[0]["Width"])
    movie = _np.memmap(path, dtype, "r", shape=shape)
    if info[0]["Byte Order"] != "<":
        movie = movie.byteswap()
        info[0]["Byte Order"] = "<"
    return movie, info


def save_config(CONFIG):
    this_file = _ospath.abspath(__file__)
    this_directory = _ospath.dirname(this_file)
    with open(_ospath.join(this_directory, "config.yaml"), "w") as config_file:
        _yaml.dump(CONFIG, config_file, width=1000)


def save_raw(path, movie, info):
    movie.tofile(path)
    info_path = _ospath.splitext(path)[0] + ".yaml"
    save_info(info_path, info)


def multiple_filenames(path, index):
    base, ext = _ospath.splitext(path)
    filename = base + "_" + str(index) + ext
    return filename


def load_tif(path):
    movie = TiffMultiMap(path, memmap_frames=False)
    info = movie.info()
    return movie, [info]


def load_movie(path, prompt_info=None):
    base, ext = _ospath.splitext(path)
    ext = ext.lower()
    if ext == ".raw":
        return load_raw(path, prompt_info=prompt_info)
    elif ext == ".tif":
        return load_tif(path)


def load_info(path, qt_parent=None):
    path_base, path_extension = _ospath.splitext(path)
    filename = path_base + ".yaml"
    try:
        with open(filename, "r") as info_file:
            info = list(_yaml.load_all(info_file, Loader=_yaml.FullLoader))
    except FileNotFoundError as e:
        print(
            "\nAn error occured. Could not find metadata file:\n{}".format(
                filename
            )
        )
        if qt_parent is not None:
            _QMessageBox.critical(
                qt_parent,
                "An error occured",
                "Could not find metadata file:\n{}".format(filename),
            )
        raise NoMetadataFileError(e)
    return info


def load_user_settings():
    settings_filename = _user_settings_filename()
    settings = None
    try:
        settings_file = open(settings_filename, "r")
    except FileNotFoundError:
        return _lib.AutoDict()
    try:
        settings = _yaml.load(settings_file, Loader=_yaml.FullLoader)
        settings_file.close()
    except Exception as e:
        print(e)
        print("Error reading user settings, Reset.")
    if not settings:
        return _lib.AutoDict()
    return _lib.AutoDict(settings)


def save_info(path, info, default_flow_style=False):
    with open(path, "w") as file:
        _yaml.dump_all(info, file, default_flow_style=default_flow_style)


def _to_dict_walk(node):
    """ Converts mapping objects (subclassed from dict)
    to actual dict objects, including nested ones
    """
    node = dict(node)
    for key, val in node.items():
        if isinstance(val, dict):
            node[key] = _to_dict_walk(val)
    return node


def save_user_settings(settings):
    settings = _to_dict_walk(settings)
    settings_filename = _user_settings_filename()
    _os.makedirs(_ospath.dirname(settings_filename), exist_ok=True)
    with open(settings_filename, "w") as settings_file:
        _yaml.dump(dict(settings), settings_file, default_flow_style=False)


class TiffMap:

    TIFF_TYPES = {
        # 0 is invalid TIFF data type
        1: "B",  # byte, 8-bit unsigned integer
        2: "c",  # ascii character, 8-bit bytes w/ last byte null
        3: "H",  # 16-bit unsigned integer
        4: "L",  # 32-bit unsigned integer
        5: "RATIONAL",  # 64-bit unsigned fraction
        # 6 is !8-bit signed integer
        16: "Q",  # BigTIFF 64-bit unsigned integer
        17: "q",  # BigTIFF 64-bit signed integer
        18: "Q",  # BigTIFF 64-bit unsigned integer (offset)
    }
    TYPE_SIZES = {
        "c": 1,  # char, string of length 1.
        "B": 1,  # unsigned char, maps to python int.
        "h": 2,  # short, 16-bit signed int from 0-65535
        "H": 2,  # unsigned short, 16-bit signed int,
        "i": 4,  # 32-bit int, signed.
        "I": 4,  # 32-bit int, unsigned.
        "L": 4,  # Same as "I", 32-bit unsigned int. Unsigned long int.
        "Q": 8,  # 64-bit int, unsigned. Aka "unsigned long long int"
        "RATIONAL": 8,
    }
    UNSIGNED_BYTESIZE_TO_TYPE = {1: "B", 2: "H", 4: "L", 8: "Q"}
    TIFF_VERSIONS = {42: 1, 43: 2}  # 42/1 is standard TIFF, 43/2 is BigTIFF.

    def __init__(self, path, verbose=False):
        if verbose:
            print("Reading info from {}".format(path))
        self.path = _ospath.abspath(path)
        self.file = open(self.path, "rb")
        self.verbose = verbose
        # Standard TIFF specification:
        # Byte 0-1: Byte order - b"II"/0x4D4D/77 is little, b"MM"/0x4949/73 is big.
        self._tif_byte_order = {b"II": "<", b"MM": ">"}[self.file.read(2)]
        # Byte 2-3: TIFF version - 42=0x002A=b"\x00*" is standard TIFF, 43=0x002B=b"\x00+" is BigTIFF.
        # If using little endian,  then 42 is written b"*\x00"/0x2A00, and BigTIFF 0x002B=b"+\x00".
        self._tiff_version_tag = self.read('H', 1)  # Read a single 2-byte/16-bit unsigned short from current position.
        self._tiff_version_num = self.TIFF_VERSIONS[self._tiff_version_tag]
        self._first_ifd_offset_type = {42: "L", 43: "Q"}[self._tiff_version_tag]
        self._total_ifd_count_bytetype = {42: "H", 43: "Q"}[self._tiff_version_tag]
        self._tiff_version_str = {42: "standard TIFF", 43: "BigTIFF"}[self._tiff_version_tag]
        # In standard tiff, each IFD tag-entry is 12 bytes; in BigTIFF they are 20 bytes:
        self._idf_nbytes_per_entry = {42: 12, 43: 20}[self._tiff_version_tag]
        if self._tiff_version_tag == 42:  # 42 means "Standard TIFF"
            # Standard TIFF uses 16-bit int for the number of IFDs:
            self._offset_for_first_ifd_offset = 4
            self._total_ifd_count_bytetype = "H"
            self._total_ifd_count_bytesize = 2
            self._idf_entry_tag_bytetype = "H"  # 16-bit
            self._idf_entry_datatype_bytetype = "H"  # 16-bit
            self._idf_entry_element_count_bytetype = "L"  # 32-bit, the length of this tag's data.
            self._idf_entry_tagdata_or_offset_bytetype = "L"  # 32-bit for standard TIFF
            self._idf_entry_tagdata_bytesize_before_offset = 4  # More than 4 bytes => use offset
            self._idf_nbytes_per_entry = 12  # Each IFD tag-entry is described by 12 bytes
            # Standard TIFF always uses 32-bit offset sizes:
            assert self.file.tell() == 4
            self._ifd_offsets_bytesize = 4
            self._ifd_offsets_bytetype = "L"
            # Find offset for first IFD:
            self.first_ifd_offset = self.read("L")  # Standard TIFF uses 32-bit int to specify first ifd offset.

        elif self._tiff_version_tag == 43:  # 43 means "BigTIFF"
            # BigTIFF uses 64-bit int for the number of IFDs:
            self._offset_for_first_ifd_offset = 8
            self._total_ifd_count_bytetype = "Q"
            self._total_ifd_count_bytesize = 8
            self._idf_entry_tag_bytetype = "H"  # 16-bit
            self._idf_entry_datatype_bytetype = "H"  # 16-bit
            self._idf_entry_element_count_bytetype = "Q"  # 64-bit for BigTIFF
            self._idf_entry_tagdata_or_offset_bytetype = "Q"  # 64-bit for BigTIFF
            self._idf_entry_tagdata_bytesize_before_offset = 8  # More than 8 bytes => use offset
            self._idf_nbytes_per_entry = 20  # Each BigTIFF IFD tag-entry is described by 20 bytes

            # BigTIFF can use variable-sized ints to specify offsets, defined by bytes 4-5:
            assert self.file.tell() == 4
            self._ifd_offsets_bytesize = self.read("H")
            assert self._ifd_offsets_bytesize == 8  # I think offset sizes are always 64-bit (=8 bytes).
            self._ifd_offsets_bytetype = self.UNSIGNED_BYTESIZE_TO_TYPE[self._ifd_offsets_bytesize]
            assert self._ifd_offsets_bytetype == "Q"  # byte-size of 8 is 64-bit int (unsigned long long)
            # BigTIFF bytes 6-7 should be null:
            _this_should_be_null = self.read("H")
            assert _this_should_be_null == 0

            # BigTIFF bytes 8-15 is a 64-bit int specifying offset to first directory:
            # (OBS: I'm not sure if this is actually *always* 64-bit, or if it can vary
            #  according to the bytesize specified by bytes 4-5)
            assert self.file.tell() == 8
            self.first_ifd_offset = self.read(self._ifd_offsets_bytetype)  # Should

        if verbose:
            print(" - byte order:", self._tif_byte_order)
            print(" - tif version tag:", self._tiff_version_tag, f"({self._tiff_version_str} format)")

        # The TIFF file header contains IFDs, each IFDs contain tags/entries.
        # For instance, the first IFD may contain tag-entries that describe
        # width (tag=256), height (tag=257), bits-per-sample (tag=258),
        # and most importantly, one or more tags specifying where to find the image data.
        # e.g. tag=273 (0x0111) is StripOffsets and specifies the byte offset to for this "strip",
        # where strip in our case is an image.
        # Each IFD is a sequence of entries, each entry is 12 bytes (standard)
        # or 20 bytes (bigtiff) long and describes:
        # * entry tag (2 bytes)
        # * entry data type (2 bytes - Standard TIFF defines 14 data types).
        # * entry element count - the number of elements for this entry
        #   * entry-element-count is 4 bytes for Standard TIFF ("L")
        #   * entry-element-count is 8 bytes for BigTIFF ("Q")
        # * entry-data or entry-data-offset
        #   * This is 4 bytes in standard tiff and 8 bytes in BigTiff.
        #   * Whether the data can fit within these 4 or 8 bytes is determined by the number of
        #   * entry-data is (we use an offset, if data is more than 4 bytes).
        # After all entries have been listed (so at byte 12*n_entries bytes for standard tiff,
        # or 20*n_entries bytes for BigTIFF), we write the offset to the next IFD.
        # If there are no more IDFs, then we write zero/null.
        # (*) In standard tiff, entry/tag can contain 2*32 elements (e.g. a string with 4 mio. characters):
        #     In

        # Read info from first IFD (Image File Directory)
        self.file.seek(self.first_ifd_offset)
        n_entries = self.read(self._total_ifd_count_bytetype)  # entries in THIS IFD
        self.first_idf_tag_entries = []
        for i in range(n_entries):
            entry_offset = (self.first_ifd_offset
                            + self._total_ifd_count_bytesize
                            + i * self._idf_nbytes_per_entry)
            self.file.seek(entry_offset)
            print(f"Entry #{i} offset:", entry_offset)
            tag = self.read(self._idf_entry_tag_bytetype)
            print(f" - tag: {tag}")
            type = self.TIFF_TYPES[self.read(self._idf_entry_datatype_bytetype)]
            print(f" - type: {type}")
            count = self.read(self._idf_entry_element_count_bytetype)
            print(f" - count: {count}")
            # value_or_offset = self.read(type, count)
            print(f"TiffMap init, first IFD, tag/entry #{i}, tag={tag} (type={type} x count={count}):")
            tag_dict = {'tag': tag, 'data-type': type, 'data-length': count}
            if count * self.TYPE_SIZES[type] > self._idf_entry_tagdata_bytesize_before_offset:
                # We don't want to read tag value if we need to seek to a new offset.
                tag_data_offset = self.read(self._idf_entry_tagdata_or_offset_bytetype)
                tag_value = 0
                tag_dict['data-offset'] = tag_data_offset
                print(" - tag data at offset", tag_data_offset)
            else:
                tag_value = self.read(type, count)
                tag_dict['value'] = tag_value
                print(" - tag data value", tag_value)
            self.first_idf_tag_entries.append(tag_dict)
            if tag == 256:
                self.width = tag_value
            elif tag == 257:
                self.height = tag_value
            elif tag == 258:
                bits_per_sample = tag_value
                dtype_str = "u" + str(int(bits_per_sample / 8))
                # Picasso uses internally exclusively little endian byte order
                self.dtype = _np.dtype(dtype_str)
                # the tif byte order might be different
                # so we also store the file dtype
                self._tif_dtype = _np.dtype(self._tif_byte_order + dtype_str)
        self.frame_shape = (self.height, self.width)
        self.frame_size = self.height * self.width

        # Collect image offsets
        self.image_offsets = []
        offset_for_next_idf_offset = self._offset_for_first_ifd_offset
        next_ifd_offset = self.first_ifd_offset
        ifd_number = 0
        print("\nCollecting image offsets...")
        while next_ifd_offset != 0:
            if ifd_number % 10 == 0:
                print(".", end="")
                # print(" - next_idf_offset:", next_ifd_offset)
            self.file.seek(next_ifd_offset)
            n_entries = self.read_numbers(self._total_ifd_count_bytetype)
            if n_entries is None:
                # Some MM files have trailing nonsense bytes
                break
            for i in range(n_entries):
                self.file.seek(next_ifd_offset
                               + self._total_ifd_count_bytesize
                               + i * self._idf_nbytes_per_entry)
                tag = self.read(self._idf_entry_tag_bytetype)
                if tag == 273:
                    type = self.TIFF_TYPES[self.read(self._idf_entry_datatype_bytetype)]
                    count = self.read(self._idf_entry_element_count_bytetype)
                    image_strip_offset = self.read(type, count)
                    if ifd_number % 10 == 0:
                        print(",", end="")
                        # print(f" -- TiffMap init, ifd #{ifd_number} image strip offset = {image_strip_offset} "
                        #       f"(found in tag entry #{i}, tag={tag}, type={type}, count={count})")
                    self.image_offsets.append(image_strip_offset)
                    break

            offset_for_next_idf_offset = (
                    next_ifd_offset + self._total_ifd_count_bytesize + n_entries * self._idf_nbytes_per_entry)
            self.file.seek(offset_for_next_idf_offset)
            next_ifd_offset = self.read(self._ifd_offsets_bytetype)
            ifd_number += 1

        print(" - Finished collecting image offsets:", self.image_offsets)
        self.n_frames = len(self.image_offsets)
        self.last_ifd_offset = offset_for_next_idf_offset
        self.lock = _threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):

        with self.lock:  # for reading frames from multiple threads
            if isinstance(it, tuple):
                if isinstance(it, int) or _np.issubdtype(it[0], _np.integer):
                    return self[it[0]][it[1:]]
                elif isinstance(it[0], slice):
                    indices = range(*it[0].indices(self.n_frames))
                    stack = _np.array([self.get_frame(_) for _ in indices])
                    if len(indices) == 0:
                        return stack
                    else:
                        if len(it) == 2:
                            return stack[:, it[1]]
                        elif len(it) == 3:
                            return stack[:, it[1], it[2]]
                        else:
                            raise IndexError
                elif it[0] == Ellipsis:
                    stack = self[it[0]]
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            elif isinstance(it, slice):
                indices = range(*it.indices(self.n_frames))
                return _np.array([self.get_frame(_) for _ in indices])
            elif it == Ellipsis:
                return _np.array(
                    [self.get_frame(_) for _ in range(self.n_frames)]
                )
            elif isinstance(it, int) or _np.issubdtype(it, _np.integer):
                return self.get_frame(it)
            raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def info(self):
        info = {
            "Byte Order": self._tif_byte_order,
            "File": self.path,
            "Height": self.height,
            "Width": self.width,
            "Data Type": self.dtype.name,
            "Frames": self.n_frames,
        }
        # The following block is MM-specific
        self.file.seek(self.first_ifd_offset)
        tag_entries = []
        n_entries = self.read("H")
        for i in range(n_entries):
            self.file.seek(self.first_ifd_offset
                           + self._total_ifd_count_bytesize
                           + i * self._idf_nbytes_per_entry)
            tag = self.read(self._idf_entry_tag_bytetype)
            type = self.TIFF_TYPES[self.read(self._idf_entry_datatype_bytetype)]
            count = self.read(self._idf_entry_element_count_bytetype)
            if count * self.TYPE_SIZES[type] > self._idf_entry_tagdata_bytesize_before_offset:
                entry_data_offset = self.read(self._idf_entry_tagdata_or_offset_bytetype)
                self.file.seek(entry_data_offset)
                entry_value = self.read(type, count)
            else:
                entry_data_offset = 0
                entry_value = self.read(type, count)
            tag_entries.append({'tag': tag, 'data-type': type, 'data-length': count,
                                'data-offset': entry_data_offset, 'value': entry_value})
            print(f"info() Entry {i}, tag={tag}, type={type}, count={count}.")
            if tag == 51123:
                # This is the Micro-Manager tag
                # We generate an info dict that contains any info we need.
                readout = entry_value.strip(
                    b"\0"
                )  # Strip null bytes which MM 1.4.22 adds
                mm_info_raw = _json.loads(readout.decode())
                # Convert to ensure compatbility with MM 2.0
                mm_info = {}
                for key in mm_info_raw.keys():
                    if key != "scopeDataKeys":
                        try:
                            mm_info[key] = mm_info_raw[key].get("PropVal")
                        except AttributeError:
                            mm_info[key] = mm_info_raw[key]

                info["Micro-Manager Metadata"] = mm_info
                if "Camera" in mm_info.keys():
                    info["Camera"] = mm_info["Camera"]
                else:
                    info["Camera"] = "None"
        # Acquistion comments
        self.file.seek(self.last_ifd_offset)
        comments = ""
        offset = 0
        while True:  # Fin the block with the summary
            line = self.file.readline()
            if "Summary" in str(line):
                break
            if not line:
                break
            offset += len(line)

        if line:
            for i in range(len(line)):
                self.file.seek(self.last_ifd_offset + offset + i)
                readout = self.read("L")
                if readout == 84720485:  # Acquisition comments
                    count = self.read("L")
                    readout = self.file.read(4 * count).strip(b"\0")
                    comments = _json.loads(readout.decode())["Summary"].split(
                        "\n"
                    )
                    break

        info["Micro-Manager Acquisiton Comments"] = comments

        return info

    def get_frame(self, index, array=None):
        self.file.seek(self.image_offsets[index])
        frame = _np.reshape(
            _np.fromfile(
                self.file, dtype=self._tif_dtype, count=self.frame_size
            ),
            self.frame_shape,
        )
        # We only want to deal with little endian byte order downstream:
        if self._tif_byte_order == ">":
            frame.byteswap(True)
            frame = frame.newbyteorder("<")
        return frame

    def read(self, type, count=1):
        if type == "c":
            return self.file.read(count)
        elif type == "RATIONAL":
            return self.read_numbers("L") / self.read_numbers("L")
        else:
            return self.read_numbers(type, count)

    def read_numbers(self, type, count=1):
        size = self.TYPE_SIZES[type]
        fmt = self._tif_byte_order + count * type
        try:
            return _struct.unpack(fmt, self.file.read(count * size))[0]
        except _struct.error:
            print("_struct.error encountered; returning None.")
            return None

    def close(self):
        self.file.close()

    def tofile(self, file_handle, byte_order=None):
        do_byteswap = byte_order != self.byte_order
        for image in self:
            if do_byteswap:
                image = image.byteswap()
            image.tofile(file_handle)


class TiffMultiMap:
    def __init__(self, path, memmap_frames=False, verbose=False):
        self.path = _ospath.abspath(path)
        self.dir = _ospath.dirname(self.path)
        base, ext = _ospath.splitext(
            _ospath.splitext(self.path)[0]
        )  # split two extensions as in .ome.tif
        base = _re.escape(base)
        pattern = _re.compile(
            base + r"_(\d*).ome.tif"
        )  # This matches the basename + an appendix of the file number
        entries = [_.path for _ in _os.scandir(self.dir) if _.is_file()]
        matches = [_re.match(pattern, _) for _ in entries]
        matches = [_ for _ in matches if _ is not None]
        paths_indices = [(int(_.group(1)), _.group(0)) for _ in matches]
        self.paths = [self.path] + [
            path for index, path in sorted(paths_indices)
        ]
        self.maps = [TiffMap(path, verbose=verbose) for path in self.paths]
        self.n_maps = len(self.maps)
        self.n_frames_per_map = [_.n_frames for _ in self.maps]
        self.n_frames = sum(self.n_frames_per_map)
        self.cum_n_frames = _np.insert(_np.cumsum(self.n_frames_per_map), 0, 0)
        self.dtype = self.maps[0].dtype
        self.height = self.maps[0].height
        self.width = self.maps[0].width
        self.shape = (self.n_frames, self.height, self.width)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, it):
        if isinstance(it, tuple):
            if it[0] == Ellipsis:
                stack = self[it[0]]
                if len(it) == 2:
                    return stack[:, it[1]]
                elif len(it) == 3:
                    return stack[:, it[1], it[2]]
                else:
                    raise IndexError
            elif isinstance(it[0], slice):
                indices = range(*it[0].indices(self.n_frames))
                stack = _np.array([self.get_frame(_) for _ in indices])
                if len(indices) == 0:
                    return stack
                else:
                    if len(it) == 2:
                        return stack[:, it[1]]
                    elif len(it) == 3:
                        return stack[:, it[1], it[2]]
                    else:
                        raise IndexError
            if isinstance(it[0], int) or _np.issubdtype(it[0], _np.integer):
                return self[it[0]][it[1:]]
        elif isinstance(it, slice):
            indices = range(*it.indices(self.n_frames))
            return _np.array([self.get_frame(_) for _ in indices])
        elif it == Ellipsis:
            return _np.array([self.get_frame(_) for _ in range(self.n_frames)])
        elif isinstance(it, int) or _np.issubdtype(it, _np.integer):
            return self.get_frame(it)
        raise TypeError

    def __iter__(self):
        for i in range(self.n_frames):
            yield self[i]

    def __len__(self):
        return self.n_frames

    def close(self):
        for map in self.maps:
            map.close()

    def get_frame(self, index):
        # TODO deal with negative numbers
        for i in range(self.n_maps):
            if self.cum_n_frames[i] <= index < self.cum_n_frames[i + 1]:
                break
        else:
            raise IndexError
        return self.maps[i][index - self.cum_n_frames[i]]

    def info(self):
        info = self.maps[0].info()
        info["Frames"] = self.n_frames
        return info

    def tofile(self, file_handle, byte_order=None):
        for map in self.maps:
            map.tofile(file_handle, byte_order)


def to_raw_combined(basename, paths):
    raw_file_name = basename + ".ome.raw"
    with open(raw_file_name, "wb") as file_handle:
        with TiffMap(paths[0]) as tif:
            tif.tofile(file_handle, "<")
            info = tif.info()
        for path in paths[1:]:
            with TiffMap(path) as tif:
                info_ = tif.info()
                info["Frames"] += info_["Frames"]
                if "Comments" in info_:
                    info["Comments"] = info_["Comments"]
                tif.tofile(file_handle, "<")
        info["Generated by"] = "Picasso ToRaw"
        info["Byte Order"] = "<"
        info["Original File"] = _ospath.basename(info.pop("File"))
        info["Raw File"] = _ospath.basename(raw_file_name)
        save_info(basename + ".ome.yaml", [info])


def get_movie_groups(paths):
    groups = {}
    if len(paths) > 0:
        pattern = _re.compile(
            r"(.*?)(_(\d*))?.ome.tif"
        )  # This matches the basename + an opt appendix of the file number
        matches = [_re.match(pattern, path) for path in paths]
        match_infos = [
            {"path": _.group(), "base": _.group(1), "index": _.group(3)}
            for _ in matches
        ]
        for match_info in match_infos:
            if match_info["index"] is None:
                match_info["index"] = 0
            else:
                match_info["index"] = int(match_info["index"])
        basenames = set([_["base"] for _ in match_infos])
        for basename in basenames:
            match_infos_group = [
                _ for _ in match_infos if _["base"] == basename
            ]
            group = [_["path"] for _ in match_infos_group]
            indices = [_["index"] for _ in match_infos_group]
            group = [path for (index, path) in sorted(zip(indices, group))]
            groups[basename] = group
    return groups


def to_raw(path, verbose=True):
    paths = _glob.glob(path)
    groups = get_movie_groups(paths)
    n_groups = len(groups)
    if n_groups:
        for i, (basename, group) in enumerate(groups.items()):
            if verbose:
                print(
                    "Converting movie {}/{}...".format(i + 1, n_groups),
                    end="\r",
                )
            to_raw_combined(basename, group)
        if verbose:
            print()
    else:
        if verbose:
            print("No files matching {}".format(path))


def save_datasets(path, info, **kwargs):
    with _h5py.File(path, "w") as hdf:
        for key, val in kwargs.items():
            hdf.create_dataset(key, data=val)
    base, ext = _ospath.splitext(path)
    info_path = base + ".yaml"
    save_info(info_path, info)


def save_locs(path, locs, info):
    locs = _lib.ensure_sanity(locs, info)
    with _h5py.File(path, "w") as locs_file:
        locs_file.create_dataset("locs", data=locs)
    base, ext = _ospath.splitext(path)
    info_path = base + ".yaml"
    save_info(info_path, info)


def load_locs(path, qt_parent=None):
    with _h5py.File(path, "r") as locs_file:
        locs = locs_file["locs"][...]
    locs = _np.rec.array(
        locs, dtype=locs.dtype
    )  # Convert to rec array with fields as attributes
    info = load_info(path, qt_parent=qt_parent)
    return locs, info


def load_clusters(path, qt_parent=None):
    with _h5py.File(path, "r") as cluster_file:
        clusters = cluster_file["clusters"][...]
    clusters = _np.rec.array(
        clusters, dtype=clusters.dtype
    )  # Convert to rec array with fields as attributes
    return clusters


def load_filter(path, qt_parent=None):
    with _h5py.File(path, "r") as locs_file:
        try:
            locs = locs_file["locs"][...]
            info = load_info(path, qt_parent=qt_parent)
        except KeyError:
            try:
                locs = locs_file["groups"][...]
                info = load_info(path, qt_parent=qt_parent)
            except KeyError:
                locs = locs_file["clusters"][...]
                info = []

    locs = _np.rec.array(
        locs, dtype=locs.dtype
    )  # Convert to rec array with fields as attributes
    return locs, info
