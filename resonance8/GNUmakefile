# ===========================================================================
#  Makefile resonance                                 Chun Shen Apr. 9, 2013
# ===========================================================================
##
##  Environments :	MAIN	= 	main sourcefile	
##
##  Usage : 	(g)make	[all]		compile the whole project		
##			install	make all and copy binary to $INSTPATH
##			clean		remove objectfiles in obj_$TYPE 
##			distclean	remove all objectsfiles and binaries
##  
CC:=`which nvcc`
# chnaged from CFLAGS= -O3 -Wall
CFLAGS= -O3

RM		=	rm -f
O               =       .o
LDFLAGS         =       $(CFLAGS)
SYSTEMFILES     =       $(SRCGNU)

# --------------- Files involved ------------------

ifeq "$(MAIN)" ""
MAIN		=	resonance.e
endif

#SRC             =       decay.cu reso.cu functions.cu int.cu tools.cu \
#                        Table.cu arsenal.cu
#removed int.cu and tools.cu since I've moved everythign into CUDA_decay.cu

SRC		=	decay.cu reso.cu functions.cu\
			Table.cu arsenal.cu
#changed everything to .cu
INC		= 	decay.h reso.h functions.h\
                  Table.h arsenal.h

# -------------------------------------------------

OBJDIR		=	obj
SRCFILES 	= 	$(SRC) $(INC) GNUmakefile
OBJECTS		=	$(addprefix $(OBJDIR)/, $(addsuffix $O, \
			$(basename $(SRC))))
TARGET		=	$(MAIN)
INSTPATH	=	../

# --------------- Pattern rules -------------------

$(OBJDIR)/%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.cu:
	if [ -f $@ ] ; then touch $@ ; else false ; fi

# -------------------------------------------------

.PHONY:		all mkobjdir clean distclean install

all:		mkobjdir $(TARGET)

help:
		@grep '^##' GNUmakefile

mkobjdir:	
		-@mkdir -p $(OBJDIR)

$(TARGET):	$(OBJECTS)	
		$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
#		strip $(TARGET)

clean:		
		-rm $(OBJECTS)

distclean:	
		-rm $(TARGET)
		-rm -r obj

install:	$(TARGET)
		cp $(TARGET) $(INSTPATH)

# --------------- Dependencies -------------------
./reso.cpp: reso.h functions.h decay.h
./decay.cpp: decay.h int.h
./functions.cpp: Table.h functions.h tools.h
#./int.cpp: int.h
#./tools.cpp: tools.h
./arsenal.cpp: arsenal.h
./Table.cpp: Table.h arsenal.h

