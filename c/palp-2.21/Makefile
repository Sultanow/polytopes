# M A K E  F I L E

#   main programs:	  class.c  cws.c  poly.c  nef.c  mori.c

SOURCES= Coord.c Rat.c Vertex.c Polynf.c 
OBJECTS= $(SOURCES:.c=.o)

CLASS_SRC= Subpoly.c Subadd.c Subdb.c
CLASS_OBJ= $(CLASS_SRC:.c=.o)

NEF_SRC= E_Poly.c Nefpart.c LG.c
NEF_OBJ= $(NEF_SRC:.c=.o)

MORI_SRC= Moricone.c
MORI_OBJ= $(MORI_SRC:.c=.o)

# Compiler
CC=cc
CXX=g++

# Flags
CFLAGS=-g -O0 -D POLY_Dmax=6
CXXFLAGS=-g -O0 -std=c++17 -D POLY_Dmax=6

# Arrow (ggf. anpassen!)
ARROW_INC=-I/usr/include
ARROW_LIB=-L/usr/lib -larrow

#   targets : dependencies ; command

all: poly class cws nef

clean:
	rm -f *.o

cleanall:
	rm -f *.o *.x core


# ========================
# PALP ORIGINAL TARGETS
# ========================

poly:	poly.o  $(OBJECTS) LG.o  Global.h LG.h
	$(CC) $(CFLAGS) -o poly.x poly.o $(OBJECTS) LG.o

class:	class.o $(OBJECTS) $(CLASS_OBJ) Global.h Subpoly.h
	$(CC) $(CFLAGS) -o class.x class.o $(OBJECTS) $(CLASS_OBJ)

cws:    cws.o $(OBJECTS) $(CLASS_OBJ) LG.o Global.h LG.h Subpoly.h
	$(CC) $(CFLAGS) -o cws.x cws.o $(OBJECTS) $(CLASS_OBJ) LG.o

nef:    nef.o   $(OBJECTS) $(NEF_OBJ) Global.h 
	$(CC) $(CFLAGS) -o nef.x nef.o $(OBJECTS) $(NEF_OBJ)

mori:   mori.o $(OBJECTS) $(MORI_OBJ) Global.h
	$(CC) $(CFLAGS) -o mori.x mori.o $(OBJECTS) $(MORI_OBJ)

# ========================
# ARROW BRIDGE
# ========================

arrow_bridge.o:   arrow_bridge.cc arrow_bridge.h
	$(CXX) $(CXXFLAGS) $(ARROW_INC) -c arrow_bridge.cc

# ========================
# Collect Normalized 5-Weight Systems
# ========================

collect_ws5.o: collect_ws5.c
	$(CC) $(CFLAGS) -c collect_ws5.c

collect_ws5.x: collect_ws5.o arrow_bridge.o $(OBJECTS) $(CLASS_OBJ) LG.o
	$(CXX) $(CXXFLAGS) -o collect_ws5.x \
	collect_ws5.o arrow_bridge.o \
	$(OBJECTS) $(CLASS_OBJ) LG.o \
	$(ARROW_LIB)

# ========================
# DEPENDENCIES
# ========================

Coord.o:        Global.h Rat.h 
Polynf.o:	Global.h Rat.h
Rat.o:          Global.h Rat.h 
Subpoly.o:      Global.h Rat.h Subpoly.h  
Subadd.o:	Global.h Subpoly.h
Vertex.o:       Global.h Rat.h  
Subdb.o:	Global.h Subpoly.h
LG.o:           Global.h Rat.h LG.h

E_Poly.o:       Global.h Nef.h Rat.h
Nefpart.o:	Global.h Nef.h

poly.o:         Global.h LG.h
class.o:	Global.h Subpoly.h
cws.o:		Global.h LG.h Rat.h Subpoly.h
cws_core5.o:	Global.h LG.h Rat.h Subpoly.h
nef.o:          Global.h Nef.h LG.h
mori.o:     Global.h Rat.h LG.h