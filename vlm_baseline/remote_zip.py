"""
vlm_baseline/remote_zip.py  —  read single files out of a remote zip, over HTTP range requests
=============================================================================================

CLEVRER's validation videos are one 6.2 GB zip, and a VLM baseline needs about 500 of its
5,000 videos. The server supports byte ranges, so we read the zip's index once and then fetch
each video with a single request, which is roughly 0.8 GB instead of 6.2 GB.

    python vlm_baseline/remote_zip.py            # list what is inside, no video downloaded

Used by the Kaggle job; also runs on a laptop.
"""

import io
import os
import struct
import subprocess
import time
import zipfile
import zlib

URL = "https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip"


def _curl(url, byte_range=None, tries=4):
    """One ranged GET. The server drops an occasional request, so retry with a pause."""
    args = ["curl", "-sL", "-m", "300", "--retry", "2", "--retry-delay", "1"]
    if byte_range:
        args += ["-r", byte_range]
    last = b""
    for attempt in range(tries):
        out = subprocess.run(args + [url], capture_output=True)
        if out.returncode == 0 and (byte_range is None or out.stdout):
            return out.stdout
        last = out.stderr
        time.sleep(1.5 * (attempt + 1))
    raise IOError("range %s failed after %d tries: %s" % (byte_range, tries, last[:200]))


class HttpFile(io.RawIOBase):
    """Minimal seekable file over HTTP range requests, used only to read the zip index."""

    def __init__(self, url, size=None):
        self.url = url
        self._pos = 0
        self.size = size if size is not None else self._length()

    def _length(self):
        args = ["curl", "-sL", "-m", "120", "-r", "0-0", "-D", "-", "-o", os.devnull, self.url]
        head = subprocess.run(args, capture_output=True).stdout.decode("latin-1")
        for line in head.splitlines():
            if line.lower().startswith("content-range:"):
                return int(line.split("/")[-1].strip())
        raise IOError("server did not report a size; ranges unsupported")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, min(self.size, base + offset))
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0:
            return b""
        data = _curl(self.url, "%d-%d" % (self._pos, self._pos + n - 1))
        self._pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


class RemoteZip:
    """The zip's index, plus one-request reads of whole members."""

    def __init__(self, url=URL):
        self.url = url
        self.zf = zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 20))

    def namelist(self):
        return self.zf.namelist()

    def getinfo(self, name):
        return self.zf.getinfo(name)

    def videos(self):
        """video id -> member name, for entries that look like video_<id>.mp4"""
        out = {}
        for name in self.zf.namelist():
            base = os.path.basename(name)
            if base.startswith("video_") and base.endswith(".mp4"):
                try:
                    out[int(base[len("video_"):-len(".mp4")])] = name
                except ValueError:
                    pass
        return out

    def read(self, name):
        """Fetch one member with a single ranged request."""
        info = self.zf.getinfo(name)
        head = _curl(self.url, "%d-%d" % (info.header_offset, info.header_offset + 29))
        if len(head) < 30 or head[:4] != b"PK\x03\x04":
            raise IOError("bad local header for %s" % name)
        name_len, extra_len = struct.unpack("<HH", head[26:30])
        start = info.header_offset + 30 + name_len + extra_len
        raw = _curl(self.url, "%d-%d" % (start, start + info.compress_size - 1))
        if len(raw) != info.compress_size:
            raise IOError("short read for %s: %d of %d" % (name, len(raw), info.compress_size))
        if info.compress_type == zipfile.ZIP_STORED:
            return raw
        return zlib.decompress(raw, -15)


# kept for older call sites
def open_zip(url=URL):
    return RemoteZip(url)


def member_map(rz):
    return rz.videos()


if __name__ == "__main__":
    rz = RemoteZip()
    names = rz.namelist()
    print("entries:", len(names))
    vids = rz.videos()
    ids = sorted(vids)
    print("videos:", len(vids), "| id range:", ids[0], "to", ids[-1])
    info = rz.getinfo(vids[ids[0]])
    print("one video: %s  %.2f MB (%s)" %
          (vids[ids[0]], info.file_size / 1e6,
           "stored" if info.compress_type == zipfile.ZIP_STORED else "deflated"))
    print("estimated transfer for 500 videos: %.2f GB" % (500 * info.compress_size / 1e9))
