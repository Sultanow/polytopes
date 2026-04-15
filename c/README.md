# C Code

## Extending PALP

### Prerequisites

#### Libraries and Tools
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

The file `~/data/w5.ip` contains 184,026 4D Single Weights. Download and unzip them via

```console
cd ~/data
wget http://hep.itp.tuwien.ac.at/%7Ekreuzer/CY/W/w5.ip.gz
gunzip w5.ip.gz
```

The content should look as follows (excerpt):

    5 1 1 1 1 1
    6 1 1 1 1 2
    7 1 1 1 2 2
    7 1 1 1 1 3
    8 1 1 2 2 2
    8 1 1 1 2 3
    8 1 1 1 1 4
    9 1 1 2 2 3
    9 1 1 1 3 3
    9 1 1 1 2 4
    10 1 2 2 2 3
    10 1 1 2 3 3
    10 1 1 2 2 4
    10 1 1 1 3 4
    10 1 1 1 2 5
    11 1 2 2 3 3
    11 1 1 2 3 4

### Compile
Compile the project via:

```console
make -f Makefile cleanall
make -f Makefile
```

### Run
Create the file `~/data/w5_6d.ip` containing the 5D Single Weights via `./cws.x -w6 1 500 -r > ~/data/w5_6d.ip`.

The arguments are:
- `-w6` generate IP weight systems with exactly **6 weights** per system, which correspond to single weight systems for **5-dimensional** polytopes (since a 5D simplex has 6 vertices, i.e. `N = dim + 1 = 6`)
- `1 500` scan degrees from **1 to 500** (the highest degree occurring in 5D reflexive single weight systems is below 500, analogous to the 4D case)
- `-r` keep only **reflexive** weight systems, i.e. those whose associated Newton polytope is reflexive; this pre-filters the list and keeps the file size manageable

Then run `./cws.x -c5 ~/data/w5.ip ~/data/w5_6d.ip > cws55_reflexive_minimal.txt` to generate the 5D CWS.

The arguments are:
- `-c5` generate **combined weight systems for 5-dimensional** polytopes; unlike `-c4` and below (which are fully self-contained), `-c5` requires two external weight files as input
- `~/data/w5.ip` the **4D weight file** (N=5 weights per line, format `d w1 w2 w3 w4 w5`), used as the building block for the lower-dimensional components of combined weight systems; downloaded from the Kreuzer website
- `~/data/w5_6d.ip` the **5D weight file** generated in the previous step (N=6 weights per line, format `d w1 w2 w3 w4 w5 w6`), used for the 5-dimensional simplex components

The program iterates over all geometrically distinct combination types (11 two-weight-system types, around 25 three-weight-system types, and the five-fold product `2×2×2×2×2`) building candidate CWS for each pair or triple of input weights, and outputs only those that are *reflexive and minimal*.
