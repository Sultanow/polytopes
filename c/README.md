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

##### Arrow files (not yet needed)
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

##### Weight files (strictly required)
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

To extend all existing d=3 and d=4 weight systems to d=5 by appending (1,1) weight rows, we use the following input files, which are structured as follows:

| File | N | nw | dim | after x(1,1) → dim |
|-------|---|----|-----|-------------------|
| `w5.ip` | 5 | 1 | 4 | N=7, nw=2, dim=5 ✓ |
| `w33.ip` | 6 | 2 | 4 | N=8, nw=3, dim=5 ✓ |
| `w44.ip` | 6 | 2 | 4 | N=8, nw=3, dim=5 ✓ |
| `w34.ip` | 6 | 2 | 4 | N=8, nw=3, dim=5 ✓ |
| `w333.ip` | 7 | 3 | 4 | N=9, nw=4, dim=5 ✓ |

### Compile
Compile the project via:

```console
make -f Makefile cleanall
make -f Makefile
```

### Run
First remove the default 8 MB stack size limit for the current shell session via `ulimit -s unlimited`. This is required because `Poly_Min_check` uses a deep mutual recursion between `Find_Ref_Subpoly` and `Find_RSP_Drop_and_Keep`, allocating ~62 KB per recursion level. 5D polytopes with many vertices exceed the default limit and cause a segfault. Then run the program via:

```console
ulimit -s unlimited
./cws.x -c5 ~/data/w5.ip > cws55_typeB.txt
wc -l cws55_typeB.txt
```

The arguments are:

- `c5` generates combined weight systems (CWS) for 5-dimensional reflexive polytopes (Type B only). Unlike -c4 and below (which are self-contained), -c5 requires one external input file. Type A (trivial extension of d=4 CWS via appending a (1,1) row) is deliberately excluded: it violates the KS construction (math/0001106, Lemma 1) because the appended simplex shares no vertices with the existing structure, producing a degenerate product polytope rather than a genuine minimal 5D polytope.
- `~/data/w5.ip` contains the 4D single weight systems (l=5, format d w1 w2 w3 w4 w5), downloaded from the Kreuzer website (w5.ip.gz, 184,026 entries). Used as building blocks for all Type B combination types: per Lemma 1 of math/0001106, every component of a combined 5D CWS must be a proper sub-simplex of the 5D structure with dim ≤ 4, hence at most 5 weights. No w6.ip is needed or available — the KS classification was only completed for n ≤ 4 (Lemma 8: IP implies reflexive only holds for n ≤ 4).
- `> cws55_typeB.txt` is the output file containing all reflexive minimal 5D CWS of Type B, one per line in PALP format with point counts M:p v N:p v appended.

### Debug
Run `gdb ./cws.x` and then in gdb call `run -c5 ~/data/w5.ip > cws55_typeB.txt`.
