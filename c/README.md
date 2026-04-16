# C Code

## Extending PALP

### Prerequisites

#### Libraries and Tools
Install `PALP`:

```console
apt install make build-essential
mkdir ~/palp
cd ~/palp/
wget http://hep.itp.tuwien.ac.at/~kreuzer/CY/palp/palp-2.21.tar.gz
cd palp-2.21/
make
```

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

#### Data

Arrow files are located in:
`/root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data___polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/`

The command `ls -l /root/data/calabi-yau-data___polytopes-4d/hf_cache_home/datasets/calabi-yau-data__
_polytopes-4d/default/0.0.0/60c0e119a03608418df538191f65da3f43b5b819/
total 161019080` should yield (excerpt):

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

We need the files containing Weights. Maximilian Kreuzer and Harald Skarke provide this data. Download, unzip and merge them via

```console
cd ~/data

wget http://hep.itp.tuwien.ac.at/%7Ekreuzer/CY/W/w5.ip.gz
gunzip w5.ip.gz

wget http://hep.itp.tuwien.ac.at/%7Ekreuzer/CY/W/w33.ip

wget http://hep.itp.tuwien.ac.at/%7Ekreuzer/CY/W/w44.ip.gz
gunzip w44.ip.gz

wget http://hep.itp.tuwien.ac.at/%7Ekreuzer/CY/W/w34.ip.gz
gunzip w34.ip.gz

wget http://hep.itp.tuwien.ac.at/%7Ekreuzer/CY/W/w333.ip

cat w5.ip w33.ip w44.ip w34.ip w333.ip > ~/data/cws4.ip

wget http://hep.itp.tuwien.ac.at/%7Ekreuzer/CY/W/wK3.ip
```

### Compile
Compile the project via:

```console
make -f Makefile cleanall
make -f Makefile
```

### Run
Run the program via `./cws.x -c5 ~/data/w5.ip ~/data/cws4.ip ~/data/wK3.ip > cws55_reflexive_minimal.txt`.

The arguments are:

### Debug
Run `gdb ./cws.x` and then in gdb call `run -c5 ~/data/w5.ip ~/data/cws4.ip ~/data/wK3.ip`.
