# C Code

## Extending PALP

### Prerequisites
Install `libarrow` (using `sudo`):

```console
apt update
apt install -y -V ca-certificates lsb-release wget
wget https://packages.apache.org/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt update
apt install -y -V libarrow-dev libarrow-glib-dev
```

Install GNU Debugger via `apt install gbd`.

Arrow files are located in:
`/root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data___polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/`

The command `ls -l /root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data__
_polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/
total 161019080` should yield (excerpt):

File Size (MB)  | File Name
-----------------------------------------------------------
      0.01 MB     | dataset_info.json
    591.71 MB     | polytopes-4d-full-00000-of-00237.arrow
    619.61 MB     | polytopes-4d-full-00001-of-00237.arrow
    665.45 MB     | polytopes-4d-full-00002-of-00237.arrow
    604.63 MB     | polytopes-4d-full-00003-of-00237.arrow
    481.78 MB     | polytopes-4d-full-00004-of-00237.arrow
    481.78 MB     | polytopes-4d-full-00005-of-00237.arrow
    481.78 MB     | polytopes-4d-full-00006-of-00237.arrow
    481.78 MB     | polytopes-4d-full-00007-of-00237.arrow
    481.78 MB     | polytopes-4d-full-00008-of-00237.arrow
    481.78 MB     | polytopes-4d-full-00009-of-00237.arrow
    481.78 MB     | polytopes-4d-full-00010-of-00237.arrow
    481.78 MB     | polytopes-4d-full-00011-of-00237.arrow
    666.48 MB     | polytopes-4d-full-00012-of-00237.arrow
    519.92 MB     | polytopes-4d-full-00013-of-00237.arrow
    519.92 MB     | polytopes-4d-full-00014-of-00237.arrow

### Compile
Compile the project via:

```console
make -f Makefile clean
make -f Makefile
```
### Run
Run `collect_ws5` via:

```console
./collect_ws5.x /root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data___polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/ ws5.txt 5000 14
```

The arguments are:

* argv[1] = Dataset folder
* argv[2] = Output file for `WS5`
* argv[3] = maximum number of rows read
* argv[4] = Filer `vertex_count` (optional)

Run `build_cws_all_min` via:

```console
./build_cws_all_min.x ws5.txt cws_min.txt 50000
```

### Debug
Debug `collect_ws5` via:

```console
gdb --args ./collect_ws5.x /root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data___polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/ ws5.txt 5000 14
run
```

Debug `build_cws_all_min` via:

```console
gdb --args ./build_cws_all_min.x ws5.txt cws_min.txt 200000
run
```

### All in one quick go

```console
make -f Makefile clean
make -f Makefile
./collect_ws5.x /root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data___polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/ ws5.txt 5000 14

ulimit -s unlimited
./build_cws_all_min.x ws5.txt cws_min.txt 200000
```
