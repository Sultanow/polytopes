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
** /root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data___polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/ **



### Compile
Compile the project via:

```console
make -f Makefile clean
make -f Makefile
make -f Makefile collect_ws5.x
```
### Run
Run it via:

```console
./collect_ws5.x /root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data___polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/ ws5.txt 5000 14
```

The arguments are:

* argv[1] = Dataset folder
* argv[2] = Output file for `WS5`
* argv[3] = maximum number of rows read
* argv[4] = Filer `vertex_count` (optional)

### Debug
Debug it via:

```console
gdb --args ./collect_ws5.x /root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data___polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/ ws5.txt 5000 14
run
```
