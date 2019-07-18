file(REMOVE_RECURSE
  "libcompute_lib.pdb"
  "libcompute_lib.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/compute_lib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
