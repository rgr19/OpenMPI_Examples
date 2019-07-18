# CMake generated Testfile for 
# Source directory: /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking
# Build directory: /home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build-debug
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(OpenMPI_NonBlocking_test_000 "/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/extern/OpenMPI/bin/mpiexec" "-n" "1" "/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build-debug/OpenMPI_NonBlocking")
add_test(OpenMPI_NonBlocking_test_001 "/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/extern/OpenMPI/bin/mpiexec" "-n" "4" "/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build-debug/OpenMPI_NonBlocking")
add_test(OpenMPI_NonBlocking_test_002 "/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/extern/OpenMPI/bin/mpiexec" "-n" "8" "--oversubscribe" "/home/a/W/W.priv/Examples/OpenMPI_Examples/OpenMPI_NonBlocking/cmake-build-debug/OpenMPI_NonBlocking")
