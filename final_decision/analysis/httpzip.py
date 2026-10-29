"""Read one member of a remote zip with HTTP range requests (no full download)."""
import io, sys, urllib.request, zipfile

class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        self.pos += len(data)
        return data
    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)

if __name__ == "__main__":
    url, member, out = sys.argv[1], sys.argv[2], sys.argv[3]
    f = RangeFile(url)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
    names = z.namelist()
    print("archive bytes", f.size, "members", len(names))
    info = z.getinfo(member)
    print("member", member, "compressed", info.compress_size, "size", info.file_size)
    with z.open(member) as src, open(out, "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    print("wrote", out)
