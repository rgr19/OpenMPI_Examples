
$HOME/W/W.priv/Extern/CMake/bin/cmake -H. -Bcmake-build
make -C cmake-build
OPAL_PREFIX=lib/OpenMPI LD_LIBRARY_PATH=lib/OpenMPI/lib lib/OpenMPI/bin/mpirun -n 4 OpenMPI_Example 
