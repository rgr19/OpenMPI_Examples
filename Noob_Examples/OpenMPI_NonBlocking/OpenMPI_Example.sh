#!/bin/bash

set -e

OPAL_PREFIX=$PWD/extern/OpenMPI LD_LIBRARY_PATH=extern/OpenMPI/lib extern/OpenMPI/bin/mpirun -n 4 OpenMPI_Example
OPAL_PREFIX=$PWD/extern/OpenMPI LD_LIBRARY_PATH=extern/OpenMPI/lib extern/OpenMPI/bin/mpirun --oversubscribe -n 8 OpenMPI_Example


/home/a/W/W.priv/Extern/CMake/bin/cmake -H. -Bcmake-build
make -C cmake-build

OPAL_PREFIX=$PWD/extern/OpenMPI LD_LIBRARY_PATH=extern/OpenMPI/lib extern/OpenMPI/bin/mpirun -n 4 cmake-build/OpenMPI_NonBlocking
OPAL_PREFIX=$PWD/extern/OpenMPI LD_LIBRARY_PATH=extern/OpenMPI/lib extern/OpenMPI/bin/mpirun --oversubscribe -n 8 cmake-build/OpenMPI_NonBlocking

set +e